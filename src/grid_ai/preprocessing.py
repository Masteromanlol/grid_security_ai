"""Data preprocessing module."""

import os
import torch
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from torch_geometric.data import Data, Dataset
import pandapower as pp

from . import utils

logger = logging.getLogger(__name__)

def extract_features(grid_state: Dict[str, pd.DataFrame]) -> torch.Tensor:
    """Extract relevant features from grid state.
    
    Args:
        grid_state: Dictionary containing pandapower results dataframes
        
    Returns:
        torch.Tensor: Feature matrix of shape [num_nodes, num_features]
        
    Raises:
        ValueError: If required features are missing or contain invalid values
    """
    bus_features = [
        'vm_pu',          # Voltage magnitude (p.u.)
        'va_degree',      # Voltage angle (degrees)
        'p_mw',          # Active power injection
        'q_mvar'         # Reactive power injection
    ]
    
    if 'bus_results' not in grid_state:
        raise ValueError("grid_state missing 'bus_results'")
    
    # Check for required features
    missing_features = [f for f in bus_features if f not in grid_state['bus_results']]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Extract and check for NaN values
    features = []
    for feature in bus_features:
        values = grid_state['bus_results'][feature].values
        if np.any(np.isnan(values)):
            raise ValueError(f"NaN values found in feature {feature}")
        features.append(values)
    
    # Stack features into matrix
    x = torch.tensor(np.stack(features, axis=1), dtype=torch.float)
    
    # Add sanity checks for values
    if torch.any(torch.isnan(x)):
        raise ValueError("NaN values found in feature tensor")
    if torch.any(torch.isinf(x)):
        raise ValueError("Infinite values found in feature tensor")
        
    return x

def get_edge_data(net: pp.pandapowerNet) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract graph connectivity and edge weights from pandapower network.
    
    Args:
        net: pandapower network object
        
    Returns:
        Tuple of:
            torch.Tensor: Edge index matrix of shape [2, num_edges]
            torch.Tensor: Edge weights/features of shape [num_edges, num_features]
    """
    # Get line connectivity and features
    from_bus = net.line['from_bus'].values
    to_bus = net.line['to_bus'].values
    line_r = net.line['r_ohm_per_km'].values * net.line['length_km'].values
    line_x = net.line['x_ohm_per_km'].values * net.line['length_km'].values
    line_c = net.line['c_nf_per_km'].values * net.line['length_km'].values
    line_imax = net.line['max_i_ka'].values
    
    # Get transformer connectivity and features
    from_bus_t = net.trafo['hv_bus'].values
    to_bus_t = net.trafo['lv_bus'].values
    trafo_r = net.trafo['vk_percent'].values * net.trafo['sn_mva'].values / 100
    trafo_x = np.sqrt(net.trafo['vk_percent'].values**2 - net.trafo['vkr_percent'].values**2) * net.trafo['sn_mva'].values / 100
    trafo_ratio = net.trafo['vn_hv_kv'].values / net.trafo['vn_lv_kv'].values
    trafo_sn = net.trafo['sn_mva'].values
    
    # Combine edges
    edges = np.concatenate([
        np.stack([from_bus, to_bus]),
        np.stack([to_bus, from_bus]),  # Add reverse edges
        np.stack([from_bus_t, to_bus_t]),
        np.stack([to_bus_t, from_bus_t])  # Add reverse edges
    ], axis=1)
    
    # Combine edge features [r, x, c, imax, ratio, sn]
    # For lines: use actual values, set ratio=1, sn=None
    # For transformers: use r/x from vk, set c=0, imax from sn
    edge_features = np.concatenate([
        # Lines forward
        np.stack([
            line_r, line_x, line_c,
            line_imax, np.ones_like(line_r),
            np.zeros_like(line_r)
        ], axis=1),
        # Lines backward (same features)
        np.stack([
            line_r, line_x, line_c,
            line_imax, np.ones_like(line_r),
            np.zeros_like(line_r)
        ], axis=1),
        # Transformers forward
        np.stack([
            trafo_r, trafo_x, np.zeros_like(trafo_r),
            trafo_sn, trafo_ratio, trafo_sn
        ], axis=1),
        # Transformers backward (inverse ratio)
        np.stack([
            trafo_r, trafo_x, np.zeros_like(trafo_r),
            trafo_sn, 1/trafo_ratio, trafo_sn
        ], axis=1)
    ], axis=0)
    
    return (
        torch.tensor(edges, dtype=torch.long),
        torch.tensor(edge_features, dtype=torch.float)
    )

def normalize_features(x: torch.Tensor, mean: torch.Tensor = None, std: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize features to zero mean and unit variance.
    
    Args:
        x: Feature matrix [num_samples, num_features]
        mean: Optional pre-computed mean
        std: Optional pre-computed standard deviation
        
    Returns:
        Tuple of (normalized features, mean, std)
    """
    if mean is None:
        mean = x.mean(dim=0, keepdim=True)
    if std is None:
        std = x.std(dim=0, keepdim=True)
        std[std == 0] = 1  # Prevent division by zero
        
    x_norm = (x - mean) / std
    return x_norm, mean, std

def create_graph_dataset(config_path: str):
    """Create PyG dataset from simulation results.
    
    Args:
        config_path: Path to configuration file
    """
    # Load config and setup logging
    config = utils.load_config(config_path)
    logger = utils.setup_logging(config['log_dir'], 'preprocessing')
    
    # Validate config
    required_keys = ['raw_simulation_dir', 'processed_dataset_file', 'normalization_params_file']
    missing_keys = [k for k in required_keys if k not in config]
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")
    
    # Load base network and get topology with edge features
    base_net = utils.get_pandapower_net(config)
    edge_index, edge_attr = get_edge_data(base_net)
    num_buses = len(base_net.bus)
    
    # Run base case for reference
    try:
        pp.runpp(base_net)
    except Exception as e:
        logger.error(f"Base case power flow failed: {str(e)}")
        raise
        
    try:
        base_features = extract_features({
            'bus_results': base_net.res_bus,
            'line_results': base_net.res_line,
            'trafo_results': base_net.res_trafo
        })
    except Exception as e:
        logger.error(f"Failed to extract base case features: {str(e)}")
        raise
    
    # Process simulation results in chunks to manage memory
    dataset = []
    all_features = []  # For computing normalization parameters
    raw_dir = config['raw_simulation_dir']
    
    # Count total files first
    pkl_files = [f for f in os.listdir(raw_dir) if f.endswith('.pkl')]
    total_files = len(pkl_files)
    
    if total_files == 0:
        raise ValueError(f"No .pkl files found in {raw_dir}")
    
    logger.info(f"Processing {total_files} result files")
    
    # Process in chunks of 1000 files
    chunk_size = 1000
    for i in range(0, total_files, chunk_size):
        chunk_files = pkl_files[i:i + chunk_size]
        
        for filename in chunk_files:
            try:
                # Load simulation result
                with open(os.path.join(raw_dir, filename), 'rb') as f:
                    result = pd.read_pickle(f)
                
                if not result['success']:
                    logger.warning(f"Skipping failed simulation {filename}")
                    continue
                
                # Extract features
                try:
                    x = extract_features(result)
                except Exception as e:
                    logger.warning(f"Failed to extract features from {filename}: {str(e)}")
                    continue
                
                # Validate feature dimensions
                if x.shape[0] != num_buses:
                    logger.warning(f"Feature dimension mismatch in {filename}: {x.shape[0]} != {num_buses}")
                    continue
                
                # Create contingency encoding (one-hot)
                cont_type = torch.zeros(num_buses, 2)  # [line, transformer]
                if result['contingency']['type'] == 'line':
                    cont_type[:, 0] = 1
                elif result['contingency']['type'] == 'transformer':
                    cont_type[:, 1] = 1
                else:
                    logger.warning(f"Unknown contingency type in {filename}: {result['contingency']['type']}")
                    continue
                
                # Combine features
                x = torch.cat([base_features, x, cont_type], dim=1)
                all_features.append(x)
                
                # Extract target
                try:
                    y = torch.tensor(
                        result['bus_results'][['vm_pu', 'va_degree']].values,
                        dtype=torch.float
                    )
                except Exception as e:
                    logger.warning(f"Failed to extract target from {filename}: {str(e)}")
                    continue
                
                # Create and add Data object with edge features
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
                dataset.append(data)
                
            except Exception as e:
                logger.warning(f"Error processing {filename}: {str(e)}")
                continue
        
        logger.info(f"Processed {len(dataset)} samples out of {i + len(chunk_files)} files")
    
    if len(dataset) == 0:
        raise ValueError("No valid samples could be created")
    
    # Compute and save normalization parameters
    all_features = torch.cat([x.x for x in dataset], dim=0)
    _, mean, std = normalize_features(all_features)
    
    norm_params = {
        'mean': mean,
        'std': std,
        'feature_dim': all_features.shape[1]
    }
    
    torch.save(norm_params, config['normalization_params_file'])
    logger.info(f"Saved normalization parameters to {config['normalization_params_file']}")
    
    # Save dataset
    os.makedirs(os.path.dirname(config['processed_dataset_file']), exist_ok=True)
    torch.save(dataset, config['processed_dataset_file'])
    logger.info(f"Saved dataset with {len(dataset)} samples to {config['processed_dataset_file']}")

def preprocess_data(config_path: str):
    """Main preprocessing function."""
    create_graph_dataset(config_path)