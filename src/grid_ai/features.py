"""Feature engineering for power grid security analysis."""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Any, List, Tuple, Optional, Union
from numpy.typing import NDArray

from . import time_series
from . import contingency as cont
from . import components as comp

def extract_network_features(net_results: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """Extract network-level features from power flow results.
    
    Args:
        net_results: Dictionary containing bus_results, line_results, trafo_results DataFrames
        
    Returns:
        Dictionary of network-level features
    """
    features = {}
    
    # Voltage stability metrics
    bus_results = net_results['bus_results']
    features['voltage_mean'] = bus_results['vm_pu'].mean()
    features['voltage_std'] = bus_results['vm_pu'].std()
    features['voltage_min'] = bus_results['vm_pu'].min()
    features['voltage_max'] = bus_results['vm_pu'].max()
    features['voltage_violations'] = ((bus_results['vm_pu'] < 0.95) | 
                                   (bus_results['vm_pu'] > 1.05)).sum()
    
    # Loading metrics
    line_results = net_results['line_results']
    features['line_loading_mean'] = line_results['loading_percent'].mean()
    features['line_loading_std'] = line_results['loading_percent'].std()
    features['line_loading_max'] = line_results['loading_percent'].max()
    features['line_overload_count'] = (line_results['loading_percent'] > 100).sum()
    
    # Power flow metrics
    features['total_p_mw'] = line_results['p_from_mw'].abs().sum()
    features['total_q_mvar'] = line_results['q_from_mvar'].abs().sum()
    features['losses_mw'] = line_results['pl_mw'].sum()
    
    # Transformer metrics
    trafo_results = net_results['trafo_results']
    if not trafo_results.empty:
        features['trafo_loading_mean'] = trafo_results['loading_percent'].mean()
        features['trafo_loading_max'] = trafo_results['loading_percent'].max()
        features['trafo_overload_count'] = (trafo_results['loading_percent'] > 100).sum()
    
    return features

def extract_contingency_features(contingency: Dict[str, Any], net_structure: Dict[str, Any]) -> Dict[str, float]:
    """Extract features specific to the contingency.
    
    Args:
        contingency: Dictionary with 'type' and 'id' of the contingency
        net_structure: Dictionary containing network structural information
        
    Returns:
        Dictionary of contingency-specific features
    """
    features = {}
    
    # One-hot encode contingency type
    features['is_line'] = int(contingency['type'] == 'line')
    features['is_trafo'] = int(contingency['type'] == 'transformer')
    
    # Add normalized ID (as percentage of total components)
    if contingency['type'] == 'line':
        features['component_id_normalized'] = contingency['id'] / net_structure['n_lines']
    else:
        features['component_id_normalized'] = contingency['id'] / net_structure['n_trafos']
    
    return features

def compute_graph_metrics(edges: List[Tuple[int, int]], n_buses: int, 
                        voltage_angles: Optional[Union[np.ndarray, List[float]]] = None) -> Dict[str, float]:
    """Compute graph-theoretic and electrical metrics for the power network.
    
    Args:
        edges: List of (from_bus, to_bus) tuples
        n_buses: Total number of buses in the network
        voltage_angles: Optional array of voltage angles for electrical distance
        
    Returns:
        Dictionary of graph metrics including:
        - Basic connectivity (components, density)
        - Centrality measures (degree, eigenvector, betweenness)
        - Clustering metrics (coefficient distribution)
        - Path metrics (shortest paths, diameter)
        - Electrical metrics (if voltage angles provided)
    """
    G = nx.Graph(edges)
    
    # Add isolated nodes
    G.add_nodes_from(range(n_buses))
    
    metrics = {}
    
    # Basic graph metrics
    metrics['n_components'] = nx.number_connected_components(G)
    metrics['largest_component_size'] = len(max(nx.connected_components(G), key=len))
    metrics['isolated_nodes'] = n_buses - G.number_of_edges()
    
    # Graph density and sparsity
    metrics['density'] = nx.density(G)
    degree_values = [d for _, d in G.degree()]
    metrics['avg_degree'] = np.mean(degree_values) if degree_values else 0
    
    # Get largest connected component for complex metrics
    largest_cc = G.subgraph(max(nx.connected_components(G), key=len))
    
    # Centrality metrics
    try:
        eigenvector_centrality = nx.eigenvector_centrality(largest_cc)
        metrics['eigenvector_centrality_mean'] = np.mean(list(eigenvector_centrality.values()))
        metrics['eigenvector_centrality_std'] = np.std(list(eigenvector_centrality.values()))
    except:
        metrics['eigenvector_centrality_mean'] = -1
        metrics['eigenvector_centrality_std'] = -1
    
    try:
        betweenness_centrality = nx.betweenness_centrality(largest_cc)
        metrics['betweenness_centrality_mean'] = np.mean(list(betweenness_centrality.values()))
        metrics['betweenness_centrality_std'] = np.std(list(betweenness_centrality.values()))
    except:
        metrics['betweenness_centrality_mean'] = -1
        metrics['betweenness_centrality_std'] = -1
    
    # Clustering metrics
    clustering_dict = nx.clustering(largest_cc)
    clustering_coeffs = [v for v in clustering_dict.values()]
    if clustering_coeffs:
        metrics['clustering_coeff_mean'] = float(np.mean(clustering_coeffs))
        metrics['clustering_coeff_std'] = float(np.std(clustering_coeffs))
    else:
        metrics['clustering_coeff_mean'] = 0.0
        metrics['clustering_coeff_std'] = 0.0
    
    # Path length distribution
    try:
        path_lengths = []
        for c in nx.connected_components(G):
            subgraph = G.subgraph(c)
            if len(c) > 1:
                lengths = []
                for node1 in subgraph.nodes():
                    for node2 in subgraph.nodes():
                        if node1 < node2:  # Avoid counting paths twice
                            lengths.append(nx.shortest_path_length(subgraph, node1, node2))
                if lengths:
                    path_lengths.extend(lengths)
        
        if path_lengths:
            metrics['path_length_mean'] = np.mean(path_lengths)
            metrics['path_length_std'] = np.std(path_lengths)
            metrics['path_length_max'] = max(path_lengths)  # Diameter
        else:
            metrics['path_length_mean'] = -1
            metrics['path_length_std'] = -1
            metrics['path_length_max'] = -1
    except:
        metrics['path_length_mean'] = -1
        metrics['path_length_std'] = -1
        metrics['path_length_max'] = -1
    
    # Electrical distance metrics if voltage angles provided
    if voltage_angles is not None:
        # Calculate angle differences across edges
        angle_diffs = []
        for edge in G.edges():
            angle_diffs.append(abs(voltage_angles[edge[0]] - voltage_angles[edge[1]]))
        
        if angle_diffs:
            metrics['voltage_angle_diff_mean'] = np.mean(angle_diffs)
            metrics['voltage_angle_diff_std'] = np.std(angle_diffs)
            metrics['voltage_angle_diff_max'] = max(angle_diffs)
        else:
            metrics['voltage_angle_diff_mean'] = 0
            metrics['voltage_angle_diff_std'] = 0
            metrics['voltage_angle_diff_max'] = 0
    
    return metrics

def extract_all_features(result: Dict[str, Any], net_structure: Dict[str, Any]) -> Dict[str, float]:
    """Combine all features for a single simulation result.
    
    Args:
        result: Dictionary containing simulation results
        net_structure: Dictionary with network structural information
        
    Returns:
        Dictionary of all features
    """
    features = {}
    
    # Build NetworkX graph
    G = nx.Graph(net_structure['edges'])
    G.add_nodes_from(range(net_structure['n_buses']))
    
    # Only extract detailed features if simulation was successful
    if result['success']:
        # Network features from power flow results
        features.update(extract_network_features({
            'bus_results': result['bus_results'],
            'line_results': result['line_results'],
            'trafo_results': result['trafo_results']
        }))
        
        # Graph metrics if we have network structure
        if 'edges' in net_structure:
            features.update(compute_graph_metrics(
                net_structure['edges'],
                net_structure['n_buses']
            ))
            
        # Time series features if available
        if 'time_series' in result:
            ts_data = result['time_series']
            if ('voltage' in ts_data and 'current' in ts_data and 
                'frequency' in ts_data and 'sampling_rate' in ts_data):
                features.update(time_series.extract_all_temporal_features(
                    ts_data['voltage'],
                    ts_data['current'],
                    ts_data['frequency'],
                    ts_data['sampling_rate']
                ))
                
        # Extract system state for contingency analysis
        bus_results = result['bus_results']
        line_results = result['line_results']
        
        system_state = {
            'voltage_profile': bus_results['vm_pu'].values,
            'angle_profile': bus_results['va_degree'].values,
            'power_flows': line_results['p_from_mw'].values,
            'load_distribution': bus_results['p_mw'].values,
            'capacity_distribution': line_results['i_ka'].values  # Use actual current instead of max
        }
        
        # Analyze contingency impact
        impact_features = cont.analyze_contingency_impact(
            G, result['contingency'], system_state
        )
        features.update(impact_features)
        
        # Extract component-level features
        component_features = comp.extract_component_features(result)
        features.update(component_features)
    
    # Contingency features (always available)
    features.update(extract_contingency_features(
        result['contingency'],
        net_structure
    ))
    
    # Add failure information
    features['failed'] = not result['success']
    if not result['success']:
        features['failure_type'] = result.get('error', 'unknown')
        features['has_isolated_buses'] = bool(result.get('isolated_buses', []))
    else:
        features['failure_type'] = 'none'
        features['has_isolated_buses'] = False
    
    return features

def prepare_feature_matrix(results: List[Dict[str, Any]], net_structure: Dict[str, Any]) -> pd.DataFrame:
    """Convert a list of simulation results into a feature matrix.
    
    Args:
        results: List of simulation result dictionaries
        net_structure: Dictionary with network structural information
        
    Returns:
        DataFrame with all features
    """
    # Extract features for each result
    feature_dicts = [extract_all_features(result, net_structure) 
                    for result in results]
    
    # Convert to DataFrame
    df = pd.DataFrame(feature_dicts)
    
    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=['failure_type'])
    
    return df