"""Component-level feature extraction module for power grid analysis."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from numpy.typing import NDArray

def analyze_generator_dynamics(gen_results: pd.DataFrame) -> Dict[str, float]:
    """Analyze generator dynamic characteristics.
    
    Args:
        gen_results: DataFrame containing generator simulation results
        
    Returns:
        Dictionary of generator dynamics metrics including:
        - Power factor
        - Reactive power capability
        - Response characteristics
    """
    metrics = {}
    
    # Power factor analysis
    p_mw = gen_results['p_mw']
    q_mvar = gen_results['q_mvar']
    s_mva = np.sqrt(p_mw**2 + q_mvar**2)
    
    metrics['power_factor_mean'] = float(np.mean(p_mw / s_mva))
    metrics['power_factor_std'] = float(np.std(p_mw / s_mva))
    
    # Reactive power capability
    q_capacity = gen_results['max_q_mvar'] - gen_results['min_q_mvar']
    q_utilization = (gen_results['q_mvar'] - gen_results['min_q_mvar']) / q_capacity
    
    metrics['q_capacity_mean'] = float(np.mean(q_capacity))
    metrics['q_utilization_mean'] = float(np.mean(q_utilization))
    metrics['q_margin_min'] = float(np.min(1 - q_utilization))
    
    # Operating point analysis
    metrics['loading_percent_mean'] = float(np.mean(gen_results['loading_percent']))
    metrics['loading_percent_max'] = float(np.max(gen_results['loading_percent']))
    
    # Voltage control
    if 'vm_pu' in gen_results.columns:
        metrics['voltage_deviation'] = float(np.std(gen_results['vm_pu']))
        metrics['voltage_setpoint_error'] = float(
            np.mean(np.abs(gen_results['vm_pu'] - gen_results['vm_set_pu']))
        )
    
    return metrics

def analyze_line_thermal_limits(line_results: pd.DataFrame,
                              ambient_temp: Optional[float] = 25.0) -> Dict[str, float]:
    """Analyze line thermal loading and limits.
    
    Args:
        line_results: DataFrame containing line simulation results
        ambient_temp: Ambient temperature in Celsius
        
    Returns:
        Dictionary of thermal metrics including:
        - Loading margins
        - Temperature estimates
        - Ampacity limits
    """
    metrics = {}
    
    # Basic loading analysis
    loading_percent = line_results['loading_percent']
    metrics['loading_mean'] = float(np.mean(loading_percent))
    metrics['loading_std'] = float(np.std(loading_percent))
    metrics['loading_max'] = float(np.max(loading_percent))
    
    # Thermal loading margins
    margins = 100 - loading_percent
    metrics['thermal_margin_mean'] = float(np.mean(margins))
    metrics['thermal_margin_min'] = float(np.min(margins))
    metrics['overload_duration'] = float(np.sum(loading_percent > 100) / len(loading_percent))
    
    # Temperature estimation (simplified)
    # Using IEEE standard temperature rise equation
    rated_temp_rise = 50  # Standard temperature rise at rated current
    actual_temp_rise = rated_temp_rise * (loading_percent / 100)**2
    conductor_temp = ambient_temp + actual_temp_rise
    
    metrics['temp_mean'] = float(np.mean(conductor_temp))
    metrics['temp_max'] = float(np.max(conductor_temp))
    metrics['temp_margin'] = float(np.min(90 - conductor_temp))  # Assuming 90°C limit
    
    # Dynamic ampacity (simplified)
    if 'i_ka' in line_results.columns and 'max_i_ka' in line_results.columns:
        current_ratio = line_results['i_ka'] / line_results['max_i_ka']
        metrics['current_ratio_mean'] = float(np.mean(current_ratio))
        metrics['current_ratio_max'] = float(np.max(current_ratio))
        metrics['ampacity_margin'] = float(np.min(1 - current_ratio))
    
    return metrics

def analyze_transformer_aging(trafo_results: pd.DataFrame,
                            ambient_temp: Optional[float] = 25.0) -> Dict[str, float]:
    """Analyze transformer aging factors.
    
    Args:
        trafo_results: DataFrame containing transformer simulation results
        ambient_temp: Ambient temperature in Celsius
        
    Returns:
        Dictionary of aging metrics including:
        - Hot-spot temperature
        - Loss of life
        - Loading capability
    """
    metrics = {}
    
    # Loading analysis
    loading = trafo_results['loading_percent']
    metrics['loading_mean'] = float(np.mean(loading))
    metrics['loading_std'] = float(np.std(loading))
    metrics['loading_max'] = float(np.max(loading))
    
    # Hot-spot temperature calculation (IEEE C57.91)
    rated_hst_rise = 80  # Rated hot-spot temperature rise
    actual_hst_rise = rated_hst_rise * (loading / 100)**2
    hot_spot_temp = ambient_temp + actual_hst_rise
    
    metrics['hot_spot_temp_mean'] = float(np.mean(hot_spot_temp))
    metrics['hot_spot_temp_max'] = float(np.max(hot_spot_temp))
    
    # Aging acceleration factor
    aging_factor = 2**((hot_spot_temp - 110) / 6)  # IEEE equation
    metrics['aging_factor_mean'] = float(np.mean(aging_factor))
    metrics['aging_factor_max'] = float(np.max(aging_factor))
    
    # Loss of life estimation
    equivalent_hours = np.mean(aging_factor) * len(loading)
    metrics['loss_of_life_hours'] = float(equivalent_hours)
    
    # Loading capability
    if 's_mva' in trafo_results.columns:
        loading_ratio = trafo_results['s_mva'] / trafo_results['sn_mva']
        metrics['loading_ratio_mean'] = float(np.mean(loading_ratio))
        metrics['loading_ratio_peak'] = float(np.max(loading_ratio))
        metrics['overload_duration'] = float(
            np.sum(loading_ratio > 1.0) / len(loading_ratio)
        )
    
    return metrics

def analyze_bus_voltage_sensitivity(bus_results: pd.DataFrame,
                                  power_injection: pd.DataFrame) -> Dict[str, float]:
    """Analyze bus voltage sensitivity to power changes.
    
    Args:
        bus_results: DataFrame containing bus voltage results
        power_injection: DataFrame containing power injection data
        
    Returns:
        Dictionary of voltage sensitivity metrics
    """
    metrics = {}
    
    # Voltage stability analysis
    voltage = bus_results['vm_pu']
    metrics['voltage_mean'] = float(np.mean(voltage))
    metrics['voltage_std'] = float(np.std(voltage))
    metrics['voltage_min'] = float(np.min(voltage))
    
    # Voltage sensitivity calculation
    if 'p_mw' in power_injection.columns:
        # dV/dP sensitivity
        voltage_diff = np.diff(voltage)
        power_diff = np.diff(power_injection['p_mw'])
        
        # Avoid division by zero
        valid_idx = power_diff != 0
        if np.any(valid_idx):
            dvdp = voltage_diff[valid_idx] / power_diff[valid_idx]
            metrics['dvdp_mean'] = float(np.mean(dvdp))
            metrics['dvdp_max'] = float(np.max(np.abs(dvdp)))
        else:
            metrics['dvdp_mean'] = 0.0
            metrics['dvdp_max'] = 0.0
    
    # Voltage violation analysis
    violations_high = voltage > 1.05
    violations_low = voltage < 0.95
    
    metrics['violation_duration_high'] = float(np.mean(violations_high))
    metrics['violation_duration_low'] = float(np.mean(violations_low))
    metrics['violation_magnitude_max'] = float(
        max(np.max(voltage[violations_high]) - 1.05 if np.any(violations_high) else 0,
            0.95 - np.min(voltage[violations_low]) if np.any(violations_low) else 0)
    )
    
    return metrics

def analyze_protection_metrics(relay_data: Dict[str, Any]) -> Dict[str, float]:
    """Analyze protection system metrics.
    
    Args:
        relay_data: Dictionary containing protection system data
        
    Returns:
        Dictionary of protection metrics including:
        - Coordination margins
        - Coverage analysis
        - Response times
    """
    metrics = {}
    
    # Protection coverage
    if 'zones' in relay_data:
        zones = relay_data['zones']
        metrics['protection_zones'] = float(len(zones))
        metrics['primary_coverage'] = float(
            len([z for z in zones if z['type'] == 'primary']) / len(zones)
        )
        metrics['backup_coverage'] = float(
            len([z for z in zones if z['type'] == 'backup']) / len(zones)
        )
    
    # Coordination timing
    if 'coordination_times' in relay_data:
        coord_times = relay_data['coordination_times']
        metrics['coord_margin_min'] = float(np.min(coord_times))
        metrics['coord_margin_mean'] = float(np.mean(coord_times))
        
    # Operating times
    if 'response_times' in relay_data:
        response_times = relay_data['response_times']
        metrics['response_time_mean'] = float(np.mean(response_times))
        metrics['response_time_max'] = float(np.max(response_times))
        metrics['response_time_std'] = float(np.std(response_times))
    
    # Reliability metrics
    if 'reliability' in relay_data:
        rel_data = relay_data['reliability']
        metrics['dependability'] = float(rel_data.get('dependability', 1.0))
        metrics['security'] = float(rel_data.get('security', 1.0))
        
    return metrics

def extract_component_features(result: Dict[str, Any]) -> Dict[str, float]:
    """Extract all component-level features from simulation results.
    
    Args:
        result: Dictionary containing simulation results
        
    Returns:
        Dictionary containing all component-level features
    """
    features = {}
    
    if not result.get('success', False):
        return features
    
    # Generator features
    if 'gen_results' in result:
        gen_features = analyze_generator_dynamics(result['gen_results'])
        features.update({f'gen_{k}': v for k, v in gen_features.items()})
    
    # Line thermal features
    if 'line_results' in result:
        line_features = analyze_line_thermal_limits(result['line_results'])
        features.update({f'line_{k}': v for k, v in line_features.items()})
    
    # Transformer features
    if 'trafo_results' in result:
        trafo_features = analyze_transformer_aging(result['trafo_results'])
        features.update({f'trafo_{k}': v for k, v in trafo_features.items()})
    
    # Bus voltage sensitivity
    if 'bus_results' in result and 'power_injection' in result:
        bus_features = analyze_bus_voltage_sensitivity(
            result['bus_results'],
            result['power_injection']
        )
        features.update({f'bus_{k}': v for k, v in bus_features.items()})
    
    # Protection system metrics
    if 'protection_data' in result:
        prot_features = analyze_protection_metrics(result['protection_data'])
        features.update({f'prot_{k}': v for k, v in prot_features.items()})
    
    return features