"""Data preprocessing module."""

import os
import torch
import logging
import numpy as np
import pandas as pd
import pickle # Use pickle directly to load the list
from typing import Dict, List, Tuple
from torch_geometric.data import Data, Dataset
import pandapower as pp

from . import utils

logger = logging.getLogger(__name__)

# --- Keep extract_features, get_edge_data, normalize_features functions as they are ---
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
        # Ensure bus_results is a DataFrame
        if not isinstance(grid_state['bus_results'], pd.DataFrame):
             raise TypeError(f"Expected bus_results to be a pandas DataFrame, got {type(grid_state['bus_results'])}")
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
    # Calculate R and X based on vk_percent and vkr_percent if available
    # vk_percent = sqrt(vkr_percent^2 + vkx_percent^2)
    # R = vkr_percent / 100 * Sn / (Vn^2)  (approximation)
    # X = vkx_percent / 100 * Sn / (Vn^2)  (approximation)
    # For simplicity here, approximate R from vkr_percent, X from remaining vk_percent
    # Note: Pandapower impedance calculation might be more complex.
    # Using rated voltage squared V^2/Sn_base as base impedance
    # Assuming Sn_mva is the base power
    base_kv_hv = net.trafo['vn_hv_kv'].values
    base_kv_lv = net.trafo['vn_lv_kv'].values
    sn_mva = net.trafo['sn_mva'].values

    # Using HV side as reference for impedance calculation base
    z_base_hv = (base_kv_hv ** 2) / sn_mva # Base impedance on HV side in Ohms

    trafo_r = (net.trafo['vkr_percent'].values / 100) * z_base_hv
    vkx_percent_squared = net.trafo['vk_percent'].values**2 - net.trafo['vkr_percent'].values**2
    # Handle potential negative values due to floating point inaccuracies
    vkx_percent = np.sqrt(np.maximum(0, vkx_percent_squared))
    trafo_x = (vkx_percent / 100) * z_base_hv

    trafo_ratio = net.trafo['vn_hv_kv'].values / net.trafo['vn_lv_kv'].values
    trafo_sn = net.trafo['sn_mva'].values
    # Estimate max current based on rated power and voltage (approximation)
    # I_max = S_n / (sqrt(3) * V_n) -- Assuming 3-phase for KA calculation
    # Using HV side voltage for max current reference
    trafo_imax = trafo_sn / (np.sqrt(3) * base_kv_hv) if np.sqrt(3) > 0 and np.all(base_kv_hv > 0) else np.zeros_like(trafo_sn)


    # Combine edges (ensure they are integers)
    edges = np.concatenate([
        np.stack([from_bus.astype(int), to_bus.astype(int)]),
        np.stack([to_bus.astype(int), from_bus.astype(int)]),  # Add reverse edges for lines
        np.stack([from_bus_t.astype(int), to_bus_t.astype(int)]),
        np.stack([to_bus_t.astype(int), from_bus_t.astype(int)])  # Add reverse edges for trafos
    ], axis=1)

    # Combine edge features [r, x, c, imax, ratio, sn]
    # For lines: use actual values, set ratio=1, sn=0 (or NaN/placeholder if needed)
    # For transformers: use calculated r/x, set c=0, use estimated imax, actual ratio, actual sn
    num_lines = len(line_r)
    num_trafos = len(trafo_r)

    edge_features = np.concatenate([
        # Lines forward
        np.stack([
            line_r, line_x, line_c, line_imax,
            np.ones(num_lines), np.zeros(num_lines) # ratio=1, sn=0 for lines
        ], axis=1),
        # Lines backward (same features)
        np.stack([
            line_r, line_x, line_c, line_imax,
            np.ones(num_lines), np.zeros(num_lines)
        ], axis=1),
        # Transformers forward
        np.stack([
            trafo_r, trafo_x, np.zeros(num_trafos), trafo_imax, # c=0 for trafos
            trafo_ratio, trafo_sn
        ], axis=1),
        # Transformers backward (inverse ratio)
        np.stack([
            trafo_r, trafo_x, np.zeros(num_trafos), trafo_imax,
            1/trafo_ratio, trafo_sn # Use 1/ratio for reverse direction
        ], axis=1)
    ], axis=0)

    # Handle potential NaNs or infs in edge features (e.g., from division by zero)
    edge_features = np.nan_to_num(edge_features, nan=0.0, posinf=0.0, neginf=0.0)


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
        # Ensure stddev is not zero before division
        std = torch.where(std == 0, torch.ones_like(std), std)

    x_norm = (x - mean) / std
    return x_norm, mean, std


def process_single_result(result: dict, base_features: torch.Tensor, edge_index: torch.Tensor,
                     edge_attr: torch.Tensor, num_buses: int, logger: logging.Logger) -> tuple:
    """Process a single simulation result dictionary. (Moved logic from process_result_file)

    Args:
        result: A single simulation result dictionary.
        base_features: Base network features
        edge_index: Edge connectivity
        edge_attr: Edge features
        num_buses: Number of buses in network
        logger: Logger instance

    Returns:
        Tuple of (Data object, features tensor) or (None, None) if processing fails
    """
    if not isinstance(result, dict):
        logger.warning(f"Skipping invalid result item (expected dict, got {type(result)}).")
        return None, None

    if not result.get('success'): # Use .get for safer access
        cont_info = result.get('contingency', {})
        cont_label = f"{cont_info.get('type','unknown')} {cont_info.get('id','unknown')}"
        logger.warning(f"Skipping failed simulation result for contingency {cont_label}.")
        return None, None

    try:
        # Extract features
        x_contingency = extract_features(result)

        # Validate feature dimensions
        if x_contingency.shape[0] != num_buses:
            logger.warning(f"Feature dimension mismatch for contingency {result.get('contingency')}: {x_contingency.shape[0]} != {num_buses}")
            return None, None

        # Create contingency encoding (one-hot)
        cont_type_tensor = torch.zeros(num_buses, 2)  # [line, transformer]
        contingency_info = result.get('contingency', {})
        cont_type = contingency_info.get('type')

        if cont_type == 'line':
            cont_type_tensor[:, 0] = 1
        elif cont_type == 'transformer':
            cont_type_tensor[:, 1] = 1
        else:
            logger.warning(f"Unknown contingency type {cont_type} in result: {contingency_info}")
            # Decide how to handle: skip or use default encoding (e.g., all zeros)
            # Skipping for now:
            return None, None

        # Combine features: base state + contingency state + contingency type encoding
        # Ensure base_features and x_contingency have the same number of rows (num_buses)
        if base_features.shape[0] != x_contingency.shape[0]:
             logger.error(f"Base feature row count {base_features.shape[0]} differs from contingency feature row count {x_contingency.shape[0]}. Skipping.")
             return None, None

        combined_x = torch.cat([base_features, x_contingency, cont_type_tensor], dim=1)


        # Extract target (voltage magnitude and angle from the contingency result)
        if 'bus_results' not in result or not isinstance(result['bus_results'], pd.DataFrame):
             logger.warning(f"Missing or invalid 'bus_results' DataFrame for contingency {contingency_info}. Skipping.")
             return None, None

        target_columns = ['vm_pu', 'va_degree']
        if not all(col in result['bus_results'].columns for col in target_columns):
            logger.warning(f"Missing target columns {target_columns} in bus_results for contingency {contingency_info}. Skipping.")
            return None, None

        y = torch.tensor(
            result['bus_results'][target_columns].values,
            dtype=torch.float
        )
        if y.shape[0] != num_buses:
            logger.warning(f"Target dimension mismatch for contingency {contingency_info}: {y.shape[0]} != {num_buses}. Skipping.")
            return None, None


        # Create Data object with edge features
        data = Data(x=combined_x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        return data, combined_x # Return combined_x for normalization calculation

    except Exception as e:
        cont_info = result.get('contingency', {})
        cont_label = f"{cont_info.get('type','unknown')} {cont_info.get('id','unknown')}"
        logger.error(f"Error processing single result for contingency {cont_label}: {str(e)}", exc_info=True) # Added exc_info for traceback
        return None, None


def create_graph_dataset(config_path: str):
    """Create PyG dataset from simulation results.

    Args:
        config_path: Path to configuration file
    """
    # Load config and setup logging
    config = utils.load_config(config_path)
    logger = utils.setup_logging(config['log_dir'], 'preprocessing')

    # Validate config
    required_keys = ['processed_dataset_file', 'normalization_params_file']
    missing_keys = [k for k in required_keys if k not in config]
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")

    if 'raw_simulation_dir' not in config and 'raw_simulation_dirs' not in config:
        raise ValueError("Either raw_simulation_dir or raw_simulation_dirs must be provided")

    # Load base network and get topology with edge features
    base_net = utils.get_pandapower_net(config)
    edge_index, edge_attr = get_edge_data(base_net)
    num_buses = len(base_net.bus)
    logger.info(f"Base network loaded: {num_buses} buses.")

    # Run base case power flow for reference features
    try:
        # Use AC as default if not specified, matching simulation logic
        method = config.get('simulation', {}).get('method', 'ac')
        if method == 'ac':
            pp.runpp(base_net)
        else:
            pp.rundcpp(base_net)
        logger.info(f"Base case {method.upper()} power flow successful.")
    except Exception as e:
        logger.error(f"Base case power flow failed: {str(e)}")
        raise

    try:
        base_results_dict = {
            'bus_results': base_net.res_bus,
            'line_results': base_net.res_line,
            'trafo_results': base_net.res_trafo
        }
        base_features = extract_features(base_results_dict)
        logger.info(f"Base features extracted, shape: {base_features.shape}")
    except Exception as e:
        logger.error(f"Failed to extract base case features: {str(e)}")
        raise

    # Process simulation results
    dataset = []
    all_combined_features = []  # Store combined features for normalization

    # Handle either single directory or list of directories
    raw_dirs = config.get('raw_simulation_dirs', [])
    if isinstance(config.get('raw_simulation_dir'), str):
         # If single dir is specified, add it to the list
         raw_dirs.append(config['raw_simulation_dir'])
    if not raw_dirs:
         raise ValueError("No raw simulation directories specified in config.")


    total_files_found = 0
    valid_files_processed = 0
    total_results_processed = 0

    for raw_dir in raw_dirs:
        if not os.path.exists(raw_dir):
            logger.warning(f"Directory not found: {raw_dir}, skipping...")
            continue

        logger.info(f"Scanning directory: {raw_dir}")
        try:
            pkl_files = [f for f in os.listdir(raw_dir) if f.endswith('.pkl')]
        except OSError as e:
            logger.error(f"Could not list files in directory {raw_dir}: {e}")
            continue

        if not pkl_files:
            logger.warning(f"No .pkl files found in {raw_dir}, skipping...")
            continue

        total_files_found += len(pkl_files)
        logger.info(f"Found {len(pkl_files)} .pkl files in {raw_dir}")

        for filename in pkl_files:
            filepath = os.path.join(raw_dir, filename)
            try:
                with open(filepath, 'rb') as f:
                    # Load the list of results from the pickle file
                    results_list = pickle.load(f)

                if not isinstance(results_list, list):
                    logger.warning(f"Skipping file {filepath}: Expected a list of results, found {type(results_list)}.")
                    continue

                valid_files_processed += 1
                results_in_file = 0
                # Iterate through each result dictionary in the loaded list
                for single_result_dict in results_list:
                    processed_item = process_single_result(
                        single_result_dict, base_features, edge_index,
                        edge_attr, num_buses, logger
                    )

                    if processed_item is not None:
                        data_object, combined_feature_vector = processed_item
                        if data_object is not None:
                            dataset.append(data_object)
                            all_combined_features.append(combined_feature_vector)
                            results_in_file += 1
                total_results_processed += results_in_file
                if results_in_file > 0:
                     logger.debug(f"Successfully processed {results_in_file} results from {filepath}. Total dataset size: {len(dataset)}")
                else:
                     logger.warning(f"Processed 0 valid results from {filepath}.")


            except pickle.UnpicklingError as e:
                logger.error(f"Error unpickling file {filepath}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error processing file {filepath}: {e}", exc_info=True)

    logger.info(f"Finished processing directories. Found {total_files_found} files, processed {valid_files_processed} valid files, created {len(dataset)} dataset samples.")

    if not dataset:
        raise ValueError("No valid samples could be created from the simulation results.")

    # Compute and save normalization parameters based on COMBINED features
    if not all_combined_features:
         raise ValueError("No features were collected for normalization.")

    # Stack all combined feature tensors (num_samples * num_nodes, num_features)
    # Need to handle potential variations if feature extraction failed for some nodes/samples
    # Safest: stack tensors individually first, then concatenate samples if shapes match
    try:
        stacked_features = torch.cat(all_combined_features, dim=0) # Concatenate along the node dimension
        logger.info(f"Calculating normalization over tensor shape: {stacked_features.shape}")
        _, mean, std = normalize_features(stacked_features) # Calculate mean/std over all nodes in dataset

        # Check the dimensions - mean/std should have shape [1, num_combined_features]
        expected_feature_dim = base_features.shape[1] + base_features.shape[1] + 2 # base + contingency + encoding
        if mean.shape[1] != expected_feature_dim or std.shape[1] != expected_feature_dim:
             logger.error(f"Normalization parameter dimension mismatch. Expected {expected_feature_dim}, got mean={mean.shape}, std={std.shape}")
             # Fallback or raise error - using fallback mean=0, std=1 for now
             mean = torch.zeros(1, expected_feature_dim)
             std = torch.ones(1, expected_feature_dim)
             logger.warning("Using fallback normalization (mean=0, std=1).")


        norm_params_dict = {
            'mean': mean,
            'std': std,
             # Store the combined feature dimension for validation later
            'feature_dim': combined_x.shape[1] if 'combined_x' in locals() else expected_feature_dim
        }

        # Ensure output directory exists
        output_dir = os.path.dirname(config['normalization_params_file'])
        os.makedirs(output_dir, exist_ok=True)

        torch.save(norm_params_dict, config['normalization_params_file'])
        logger.info(f"Saved normalization parameters to {config['normalization_params_file']}")

    except Exception as e:
         logger.error(f"Failed to compute or save normalization parameters: {e}", exc_info=True)
         # Decide if you want to proceed without normalization or raise error
         raise # Raising error as normalization is usually critical


    # Save dataset
    output_dir = os.path.dirname(config['processed_dataset_file'])
    os.makedirs(output_dir, exist_ok=True)
    try:
        torch.save(dataset, config['processed_dataset_file'])
        logger.info(f"Saved final dataset with {len(dataset)} samples to {config['processed_dataset_file']}")
    except Exception as e:
        logger.error(f"Failed to save the processed dataset: {e}", exc_info=True)
        raise


def preprocess_data(config_path: str):
    """Main preprocessing function."""
    create_graph_dataset(config_path)