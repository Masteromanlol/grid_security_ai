"""Power grid simulation module."""

import copy
import logging
import pandas as pd
import pandapower as pp
import networkx as nx
from typing import Dict, Any, Optional

from . import utils

logger = logging.getLogger(__name__)

def find_isolated_buses(net: pp.pandapowerNet) -> set:
    """Find isolated buses in a pandapower network.
    
    Args:
        net: pandapower network
        
    Returns:
        Set of isolated bus indices
    """
    # Create graph from lines
    edges = [(int(row['from_bus']), int(row['to_bus'])) 
            for _, row in net.line.iterrows() if row['in_service']]
    
    # Add transformer connections
    edges.extend([(int(row['hv_bus']), int(row['lv_bus']))
                for _, row in net.trafo.iterrows() if row['in_service']])
    
    # Create graph
    G = nx.Graph(edges)
    
    # Add isolated nodes (buses with no connections)
    G.add_nodes_from(range(len(net.bus)))
    
    # Find connected components
    components = list(nx.connected_components(G))
    
    # The largest component is the main grid
    main_grid = max(components, key=len)
    
    # All other buses are isolated
    isolated = set(range(len(net.bus))) - set(main_grid)
    
    return isolated
def run_single_contingency(base_net: pp.pandapowerNet, contingency: Dict[str, Any], method: str = 'ac') -> Dict[str, Any]:
    """Run a single contingency simulation.
    
    Args:
        base_net: Base pandapower network (will be deep copied)
        contingency: Dictionary with 'type' and 'id' keys
        method: Power flow method ('ac' or 'dc')
        
    Returns:
        Dictionary containing simulation results and status
    """
    # Deep copy to avoid modifying original network
    net = copy.deepcopy(base_net)
    
    try:
        # Apply contingency based on type
        if contingency['type'] == 'line':
            if contingency['id'] >= len(net.line):
                raise ValueError(f"Line ID {contingency['id']} out of range")
            pp.drop_lines(net, [contingency['id']])
        elif contingency['type'] == 'transformer':
            if contingency['id'] >= len(net.trafo):
                raise ValueError(f"Transformer ID {contingency['id']} out of range")
            pp.drop_trafos(net, [contingency['id']])
        else:
            raise ValueError(f"Unknown contingency type: {contingency['type']}")
        
        # Check for isolated buses
        isolated = find_isolated_buses(net)
        if isolated:
            return {
                'success': False,
                'contingency': contingency,
                'error': 'IsolatedBuses',
                'isolated_buses': list(isolated),
                'message': f"Network has {len(isolated)} isolated buses after contingency"
            }
        
        # Try to run power flow with specified method
        if method == 'ac':
            pp.runpp(net, enforce_q_lims=True, init='dc')  # Use DC solution as initialization
        elif method == 'dc':
            pp.rundcpp(net)
        else:
            raise ValueError(f"Unknown power flow method: {method}")
        
        # Extract results if successful
        results = {
            'success': True,
            'contingency': contingency,
            'bus_results': net.res_bus.copy(),
            'line_results': net.res_line.copy(),
            'trafo_results': net.res_trafo.copy(),
            'converged': net._ppc['success'],
            'isolated_buses': []
        }
        
    except Exception as e:
        error_str = str(e)
        results = {
            'success': False,
            'contingency': contingency,
            'error': 'LoadflowNotConverged' if 'LoadflowNotConverged' in error_str else error_str,
            'isolated_buses': []
        }
    
    return results

def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If required parameters are missing or invalid
    """
    required_keys = ['log_dir', 'contingency_file', 'output_dir', 'pandapower_module', 'simulation']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
            
    sim_config = config['simulation']
    if 'method' in sim_config and sim_config['method'] not in ['ac', 'dc']:
        raise ValueError("Simulation method must be 'ac' or 'dc'")

def run_simulation(config_path: str, task_id: Optional[int] = None):
    """Run complete contingency analysis based on configuration.
    
    Args:
        config_path: Path to YAML configuration file
        task_id: Optional SLURM array task ID for parallel processing
    """
    import os
    import pickle
    
    # Load and validate config
    config = utils.load_config(config_path)
    validate_config(config)
    
    # Set up logging with task-specific file
    log_name = f'simulation_task_{task_id}' if task_id else 'simulation'
    logger = utils.setup_logging(config['log_dir'], log_name)
    
    # Load base network
    base_net = utils.get_pandapower_net(config)
    logger.info(f"Loaded network with {len(base_net.bus)} buses")
    
    # Run base case with specified method
    method = config['simulation'].get('method', 'ac')
    try:
        if method == 'ac':
            pp.runpp(base_net)
        else:
            pp.rundcpp(base_net)
        logger.info(f"Base case {method.upper()} power flow successful")
    except Exception as e:
        logger.error(f"Base case power flow failed: {str(e)}")
        raise
    
    # Load and validate contingency list
    contingencies = pd.read_csv(config['contingency_file'], delimiter=',')
    if 'type' not in contingencies.columns or 'id' not in contingencies.columns:
        raise ValueError("Contingency file must have 'type' and 'id' columns")
    
    # Determine contingencies for this task
    total_contingencies = len(contingencies)
    if task_id is not None:
        contingencies_per_task = 100
        start_idx = (task_id - 1) * contingencies_per_task
        end_idx = min(start_idx + contingencies_per_task, total_contingencies)
        contingencies = contingencies.iloc[start_idx:end_idx]
    
    logger.info(f"Processing {len(contingencies)} contingencies")
    
    # Process contingencies
    results = []
    for _, cont in contingencies.iterrows():
        try:
            result = run_single_contingency(
                base_net,
                {'type': cont['type'], 'id': cont['id']},
                method=method
            )
            results.append(result)
            
            if result['success']:
                logger.info(f"Contingency {cont['type']} {cont['id']} successful")
            else:
                logger.warning(f"Contingency {cont['type']} {cont['id']} failed: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"Error processing contingency {cont['type']} {cont['id']}: {str(e)}")
            results.append({
                'success': False,
                'contingency': {'type': cont['type'], 'id': cont['id']},
                'error': str(e)
            })
    
    # Save results
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(
        output_dir,
        f"results_task_{task_id}.pkl" if task_id else "results.pkl"
    )
    
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"Saved {len(results)} results to {output_file}")
    return results

def check_security_constraints(grid_state: Dict[str, pd.DataFrame]) -> Dict[str, bool]:
    """Check if grid state satisfies security constraints.
    
    Args:
        grid_state: Dictionary with bus_results, line_results, etc.
        
    Returns:
        Dictionary of constraint checks and their status
    """
    checks = {}
    
    # Voltage magnitude constraints (typically 0.95-1.05 p.u.)
    v_max = grid_state['bus_results']['vm_pu'].max()
    v_min = grid_state['bus_results']['vm_pu'].min()
    checks['voltage_in_limits'] = (0.95 <= v_min) and (v_max <= 1.05)
    
    # Line loading constraints
    checks['lines_not_overloaded'] = (grid_state['line_results']['loading_percent'] <= 100.0).all()
    
    # Transformer loading constraints
    checks['trafos_not_overloaded'] = (grid_state['trafo_results']['loading_percent'] <= 100.0).all()
    
    return checks