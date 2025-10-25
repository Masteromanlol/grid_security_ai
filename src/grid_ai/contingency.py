"""Contingency impact analysis module for power grid security."""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Any, List, Tuple, Optional, Union
from numpy.typing import NDArray

def calculate_cascade_probability(G: nx.Graph,
                               initial_failures: List[Tuple[str, int]],
                               load_distribution: NDArray[np.float64],
                               capacity_distribution: NDArray[np.float64]) -> Dict[str, float]:
    """Calculate cascade probability metrics using a probabilistic model.
    
    Args:
        G: NetworkX graph representing the power grid
        initial_failures: List of (component_type, component_id) failures
        load_distribution: Array of component loadings
        capacity_distribution: Array of component capacities
    
    Returns:
        Dictionary containing:
        - cascade_probability: Overall probability of cascade
        - expected_cascade_depth: Expected number of cascade stages
        - max_cascade_probability: Probability of maximum cascade
    """
    metrics = {}
    
    # Calculate overload probability for each component
    # Ensure arrays have the same length and handle division by zero
    min_len = min(len(load_distribution), len(capacity_distribution))
    capacity_safe = np.where(capacity_distribution[:min_len] == 0, 1e-6, capacity_distribution[:min_len])
    overload_prob = np.clip(load_distribution[:min_len] / capacity_safe, 0, 1)
    
    # Initialize cascade stage probabilities
    n_components = len(load_distribution)
    stage_probs = np.zeros(n_components)
    stage_probs[0] = 1.0  # Initial failure probability
    
    # Identify critical components using centrality
    # For large graphs, use approximation to avoid computational issues
    if G.number_of_nodes() > 1000:
        # Use degree centrality as proxy for large graphs
        betweenness = nx.degree_centrality(G)
    else:
        betweenness = nx.betweenness_centrality(G)
    critical_nodes = sorted(betweenness, key=betweenness.get, reverse=True)[:10]
    
    # Calculate cascade probability based on network topology and loading
    cascade_paths = []
    max_cascade_prob = 0.0
    
    for start in initial_failures:
        component_id = start[1]
        
        # Find potential cascade paths
        for target in critical_nodes:
            try:
                path = nx.shortest_path(G, component_id, target)
                if len(path) > 1:  # Exclude self-loops
                    path_prob = 1.0
                    for i in range(len(path)-1):
                        edge_idx = G.edges[(path[i], path[i+1])]['idx']
                        path_prob *= overload_prob[edge_idx]
                    cascade_paths.append((path, path_prob))
                    max_cascade_prob = max(max_cascade_prob, path_prob)
            except nx.NetworkXNoPath:
                continue
    
    # Calculate overall cascade probability
    if cascade_paths:
        # Use independent cascade model
        cascade_prob = 1.0 - np.prod([1.0 - p for _, p in cascade_paths])
        
        # Calculate expected cascade depth
        path_lengths = np.array([len(path) for path, prob in cascade_paths])
        path_probs = np.array([prob for _, prob in cascade_paths])
        expected_depth = np.sum(path_lengths * path_probs) / np.sum(path_probs)
    else:
        cascade_prob = 0.0
        expected_depth = 0.0
    
    metrics['cascade_probability'] = float(cascade_prob)
    metrics['expected_cascade_depth'] = float(expected_depth)
    metrics['max_cascade_probability'] = float(max_cascade_prob)
    
    return metrics

def predict_load_loss(G: nx.Graph,
                     load_distribution: NDArray[np.float64],
                     failed_components: List[Tuple[str, int]]) -> Dict[str, float]:
    """Predict potential load loss due to contingency.
    
    Args:
        G: NetworkX graph representing the power grid
        load_distribution: Array of node loads
        failed_components: List of (component_type, component_id) failures
    
    Returns:
        Dictionary containing:
        - expected_load_loss: Expected MW of load loss
        - load_loss_percentage: Percentage of total load lost
        - critical_load_impact: Impact on critical loads
    """
    metrics = {}
    
    total_load = np.sum(load_distribution)
    
    # Find affected nodes after failures
    affected_nodes = set()
    failed_edges = []
    for component_type, component_id in failed_components:
        if component_type == 'line':
            # For lines, find the edge with matching idx and get end nodes
            for u, v, data in G.edges(data=True):
                if data.get('idx') == component_id:
                    affected_nodes.update([u, v])
                    failed_edges.append((u, v))
                    break
        else:  # transformer or bus
            affected_nodes.add(component_id)
    
    # Calculate direct load loss
    direct_loss = np.sum(load_distribution[list(affected_nodes)])
    
    # Calculate potential additional load loss from islanding
    remaining_graph = G.copy()
    for _, component_id in failed_components:
        if remaining_graph.has_node(component_id):
            remaining_graph.remove_node(component_id)
    
    # Find islands
    islands = list(nx.connected_components(remaining_graph))
    
    # Calculate load loss from islanding
    islanding_loss = 0.0
    for island in islands:
        island_load = np.sum(load_distribution[list(island)])
        # Small islands are likely to collapse
        if len(island) < 3:  # This threshold can be adjusted
            islanding_loss += island_load
    
    total_loss = direct_loss + islanding_loss
    
    metrics['expected_load_loss'] = float(total_loss)
    metrics['load_loss_percentage'] = float(total_loss / total_load if total_load > 0 else 0)
    metrics['islanding_loss'] = float(islanding_loss)
    
    return metrics

def analyze_islanding_risk(G: nx.Graph,
                         initial_failures: List[Tuple[str, int]]) -> Dict[str, float]:
    """Analyze risk and characteristics of potential islanding.
    
    Args:
        G: NetworkX graph representing the power grid
        initial_failures: List of (component_type, component_id) failures
    
    Returns:
        Dictionary containing:
        - islanding_probability: Probability of islanding
        - expected_islands: Expected number of islands
        - largest_island_size: Size of largest potential island
    """
    metrics = {}
    
    # Create copy of graph for analysis
    H = G.copy()
    
    # Remove failed components
    for _, component_id in initial_failures:
        if H.has_node(component_id):
            H.remove_node(component_id)
    
    # Analyze connectivity
    components = list(nx.connected_components(H))
    n_components = len(components)
    
    # Calculate islanding metrics
    if n_components > 1:
        metrics['islanding_probability'] = 1.0
        metrics['expected_islands'] = float(n_components)
        metrics['largest_island_size'] = float(max(len(c) for c in components))
        
        # Calculate isolation risk for each component
        isolation_risk = []
        original_size = G.number_of_nodes()
        
        for comp in components:
            comp_size = len(comp)
            # Small components have higher risk
            risk = 1.0 - (comp_size / original_size)
            isolation_risk.append(risk)
        
        metrics['mean_isolation_risk'] = float(np.mean(isolation_risk))
        metrics['max_isolation_risk'] = float(max(isolation_risk))
    else:
        metrics['islanding_probability'] = 0.0
        metrics['expected_islands'] = 1.0
        metrics['largest_island_size'] = float(G.number_of_nodes())
        metrics['mean_isolation_risk'] = 0.0
        metrics['max_isolation_risk'] = 0.0
    
    return metrics

def compute_stability_indices(voltage_profile: NDArray[np.float64],
                            angle_profile: NDArray[np.float64],
                            power_flows: NDArray[np.float64]) -> Dict[str, float]:
    """Compute power system stability indices.
    
    Args:
        voltage_profile: Array of bus voltage magnitudes
        angle_profile: Array of voltage angles
        power_flows: Array of branch power flows
        
    Returns:
        Dictionary containing various stability indices
    """
    metrics = {}
    
    # Voltage Stability Index (VSI)
    v_mean = np.mean(voltage_profile)
    v_min = np.min(voltage_profile)
    metrics['voltage_stability_index'] = float(v_min / v_mean if v_mean > 0 else 0)
    
    # Angle Stability Index (ASI)
    angle_diff = np.max(angle_profile) - np.min(angle_profile)
    metrics['angle_stability_index'] = float(min(1.0, 180.0 / angle_diff) if angle_diff > 0 else 1.0)
    
    # Power Flow Stability Index (PFSI)
    rated_flows = np.ones_like(power_flows)  # This should be replaced with actual ratings
    flow_margins = rated_flows - np.abs(power_flows)
    metrics['power_flow_stability_index'] = float(np.mean(flow_margins / rated_flows))
    
    # Combined Stability Index
    metrics['combined_stability_index'] = float(np.mean([
        metrics['voltage_stability_index'],
        metrics['angle_stability_index'],
        metrics['power_flow_stability_index']
    ]))
    
    return metrics

def calculate_vulnerability_scores(G: nx.Graph,
                                component_data: Dict[str, Any]) -> Dict[str, float]:
    """Calculate vulnerability scores for grid components.
    
    Args:
        G: NetworkX graph representing the power grid
        component_data: Dictionary containing component characteristics
        
    Returns:
        Dictionary containing vulnerability metrics
    """
    metrics = {}
    
    # Calculate centrality-based vulnerability
    # For large graphs, use approximation to avoid computational issues
    if G.number_of_nodes() > 1000:
        # Use degree centrality as proxy for large graphs
        betweenness = nx.degree_centrality(G)
        eigenvector = nx.degree_centrality(G)
    else:
        betweenness = nx.betweenness_centrality(G)
        try:
            eigenvector = nx.eigenvector_centrality(G)
        except nx.AmbiguousSolution:
            # Fallback to degree centrality for disconnected graphs
            eigenvector = nx.degree_centrality(G)
    
    # Identify critical components
    critical_nodes = set(sorted(betweenness, key=betweenness.get, reverse=True)[:5])
    critical_nodes.update(sorted(eigenvector, key=eigenvector.get, reverse=True)[:5])
    
    # Calculate topological vulnerability
    metrics['topological_vulnerability'] = float(np.mean(list(betweenness.values())))
    
    # Calculate load-based vulnerability
    if 'load' in component_data and 'capacity' in component_data:
        load = component_data['load']
        capacity = component_data['capacity']
        if len(load) == len(capacity):
            load_factors = load / np.where(capacity == 0, 1e-6, capacity)
            metrics['load_vulnerability'] = float(np.mean(load_factors))
        else:
            # Shapes don't match, skip calculation
            metrics['load_vulnerability'] = 0.0
    else:
        metrics['load_vulnerability'] = 0.0
    
    # Calculate connectivity vulnerability
    articulation_points = list(nx.articulation_points(G))
    metrics['connectivity_vulnerability'] = float(len(articulation_points) / G.number_of_nodes())
    
    # Calculate overall vulnerability score
    weights = {
        'topological': 0.4,
        'load': 0.4,
        'connectivity': 0.2
    }
    
    metrics['overall_vulnerability'] = float(
        weights['topological'] * metrics['topological_vulnerability'] +
        weights['load'] * metrics['load_vulnerability'] +
        weights['connectivity'] * metrics['connectivity_vulnerability']
    )
    
    return metrics

def analyze_contingency_impact(G: nx.Graph,
                             contingency: Dict[str, Any],
                             system_state: Dict[str, Any]) -> Dict[str, float]:
    """Analyze complete impact of a contingency.
    
    Args:
        G: NetworkX graph representing the power grid
        contingency: Dictionary describing the contingency
        system_state: Dictionary containing current system state
        
    Returns:
        Dictionary containing all impact metrics
    """
    metrics = {}
    
    # Extract data
    load_dist = system_state.get('load_distribution', np.ones(G.number_of_nodes()))
    capacity_dist = system_state.get('capacity_distribution', np.ones(G.number_of_nodes()))
    voltage_profile = system_state.get('voltage_profile', np.ones(G.number_of_nodes()))
    angle_profile = system_state.get('angle_profile', np.zeros(G.number_of_nodes()))
    power_flows = system_state.get('power_flows', np.zeros(G.number_of_edges()))
    
    # Calculate cascade probabilities
    # Convert contingency dict to tuple format expected by the function
    contingency_tuple = (contingency['type'], contingency['id'])
    cascade_metrics = calculate_cascade_probability(
        G, [contingency_tuple], load_dist, capacity_dist
    )
    metrics.update(cascade_metrics)
    
    # Predict load loss
    load_metrics = predict_load_loss(
        G, load_dist, [contingency_tuple]
    )
    metrics.update(load_metrics)

    # Analyze islanding risk
    islanding_metrics = analyze_islanding_risk(
        G, [contingency_tuple]
    )
    metrics.update(islanding_metrics)
    
    # Compute stability indices
    stability_metrics = compute_stability_indices(
        voltage_profile, angle_profile, power_flows
    )
    metrics.update(stability_metrics)
    
    # Calculate vulnerability scores
    vulnerability_metrics = calculate_vulnerability_scores(
        G, {'load': load_dist, 'capacity': capacity_dist}
    )
    metrics.update(vulnerability_metrics)
    
    return metrics