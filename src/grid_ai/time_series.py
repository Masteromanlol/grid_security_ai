"""Time series feature extraction module for power grid analysis."""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from scipy import signal, stats
from numpy.typing import NDArray

def compute_voltage_stability_indices(voltage_series: NDArray[np.float64]) -> Dict[str, float]:
    """Compute voltage stability indices from time series data.
    
    Args:
        voltage_series: Array of voltage magnitudes over time
        
    Returns:
        Dictionary of voltage stability indices including:
        - L-index: Indicates proximity to voltage collapse
        - VSI: Voltage stability index
        - Critical rate of change
    """
    indices = {}
    
    # Calculate L-index (normalized voltage deviation)
    v_mean = np.mean(voltage_series)
    v_std = np.std(voltage_series)
    indices['l_index'] = v_std / v_mean if v_mean > 0 else float('inf')
    
    # Voltage Stability Index (VSI)
    v_min = np.min(voltage_series)
    v_max = np.max(voltage_series)
    indices['vsi'] = (v_max - v_min) / v_mean if v_mean > 0 else float('inf')
    
    # Critical rate of change
    diff = np.diff(voltage_series)
    indices['max_roc'] = np.max(np.abs(diff))
    indices['mean_roc'] = np.mean(np.abs(diff))
    
    return indices

def extract_oscillation_features(signal_series: NDArray[np.float64], 
                               sampling_rate: float) -> Dict[str, float]:
    """Extract oscillation characteristics from time series data.
    
    Args:
        signal_series: Array of signal values over time
        sampling_rate: Sampling rate of the signal in Hz
        
    Returns:
        Dictionary of oscillation features including:
        - Dominant frequencies
        - Damping ratios
        - Power spectral density statistics
    """
    features = {}
    
    # Compute power spectral density
    freqs, psd = signal.welch(signal_series, fs=sampling_rate)
    
    # Find dominant frequencies (peaks in PSD)
    peaks, _ = signal.find_peaks(psd)
    if len(peaks) > 0:
        dom_freq_idx = peaks[np.argmax(psd[peaks])]
        features['dominant_frequency'] = freqs[dom_freq_idx]
        features['dominant_power'] = psd[dom_freq_idx]
    else:
        features['dominant_frequency'] = 0.0
        features['dominant_power'] = 0.0
    
    # Calculate damping ratio using logarithmic decrement method
    try:
        # Find peaks for damping calculation
        peaks, _ = signal.find_peaks(signal_series)
        if len(peaks) >= 2:
            # Calculate logarithmic decrement
            peak_values = signal_series[peaks]
            log_dec = np.mean(np.log(peak_values[:-1] / peak_values[1:]))
            # Convert to damping ratio
            features['damping_ratio'] = log_dec / np.sqrt(4 * np.pi**2 + log_dec**2)
        else:
            features['damping_ratio'] = 1.0  # Critically damped
    except:
        features['damping_ratio'] = 1.0
    
    # Power spectral density statistics
    features['psd_mean'] = np.mean(psd)
    features['psd_std'] = np.std(psd)
    features['psd_max'] = np.max(psd)
    
    return features

def compute_time_domain_features(time_series: NDArray[np.float64]) -> Dict[str, float]:
    """Compute statistical features from time series data.
    
    Args:
        time_series: Array of values over time
        
    Returns:
        Dictionary of time domain features including:
        - Basic statistics (mean, std, etc.)
        - Shape features (skewness, kurtosis)
        - Rate of change statistics
    """
    features = {}
    
    # Basic statistics
    features['mean'] = np.mean(time_series)
    features['std'] = np.std(time_series)
    features['min'] = np.min(time_series)
    features['max'] = np.max(time_series)
    features['range'] = features['max'] - features['min']
    
    # Shape features
    features['skewness'] = float(stats.skew(time_series))
    features['kurtosis'] = float(stats.kurtosis(time_series))
    
    # Rate of change statistics
    diff = np.diff(time_series)
    features['roc_mean'] = np.mean(np.abs(diff))
    features['roc_std'] = np.std(diff)
    features['roc_max'] = np.max(np.abs(diff))
    
    # Zero crossings and peaks
    zero_crossings = np.where(np.diff(np.signbit(time_series - np.mean(time_series))))[0]
    features['zero_crossing_rate'] = len(zero_crossings) / len(time_series)
    
    peaks, _ = signal.find_peaks(time_series)
    features['peak_rate'] = len(peaks) / len(time_series)
    
    return features

def extract_frequency_features(time_series: NDArray[np.float64], 
                             sampling_rate: float) -> Dict[str, float]:
    """Extract frequency domain features from time series data.
    
    Args:
        time_series: Array of values over time
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Dictionary of frequency domain features including:
        - Frequency band power
        - Spectral entropy
        - Frequency ratio metrics
    """
    features = {}
    
    # Compute FFT
    n = len(time_series)
    fft = np.fft.fft(time_series)
    freqs = np.fft.fftfreq(n, 1/sampling_rate)
    
    # Get positive frequencies only
    pos_freqs = freqs[1:n//2]
    pos_fft = np.abs(fft[1:n//2])
    
    # Frequency bands (example for power systems - adjust as needed)
    bands = {
        'low': (0.1, 1.0),    # Sub-synchronous
        'med': (1.0, 5.0),    # Near synchronous
        'high': (5.0, 20.0)   # Super synchronous
    }
    
    # Calculate band powers
    total_power = np.sum(pos_fft**2)
    for band_name, (low, high) in bands.items():
        mask = (pos_freqs >= low) & (pos_freqs < high)
        band_power = np.sum(pos_fft[mask]**2)
        features[f'{band_name}_band_power'] = band_power / total_power if total_power > 0 else 0
    
    # Spectral entropy
    if total_power > 0:
        psd_norm = pos_fft**2 / total_power
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        features['spectral_entropy'] = spectral_entropy
    else:
        features['spectral_entropy'] = 0
    
    # Frequency ratios
    if len(pos_fft) > 0:
        features['freq_centroid'] = np.sum(pos_freqs * pos_fft) / np.sum(pos_fft)
        features['freq_variance'] = np.sum(((pos_freqs - features['freq_centroid'])**2) * pos_fft) / np.sum(pos_fft)
    else:
        features['freq_centroid'] = 0
        features['freq_variance'] = 0
    
    return features

def detect_temporal_patterns(time_series: NDArray[np.float64], 
                           window_size: int) -> Dict[str, float]:
    """Detect and quantify temporal patterns in time series data.
    
    Args:
        time_series: Array of values over time
        window_size: Size of sliding window for pattern detection
        
    Returns:
        Dictionary of pattern features including:
        - Autocorrelation statistics
        - Trend strength
        - Seasonality metrics
        - Change point statistics
    """
    features = {}
    
    # Compute autocorrelation
    acf = np.correlate(time_series - np.mean(time_series), 
                      time_series - np.mean(time_series), 
                      mode='full')[len(time_series)-1:]
    acf = acf / acf[0]  # Normalize
    
    # Autocorrelation features
    features['acf_1'] = acf[1] if len(acf) > 1 else 0
    features['acf_2'] = acf[2] if len(acf) > 2 else 0
    features['acf_3'] = acf[3] if len(acf) > 3 else 0
    
    # Trend strength using rolling statistics
    rolling_mean = np.convolve(time_series, 
                             np.ones(window_size)/window_size, 
                             mode='valid')
    trend_strength = 1 - np.var(time_series[window_size-1:] - rolling_mean) / np.var(time_series)
    features['trend_strength'] = max(0, trend_strength)  # Ensure non-negative
    
    # Seasonality detection using FFT
    fft = np.fft.fft(time_series)
    freqs = np.fft.fftfreq(len(time_series))
    pos_mask = freqs > 0
    peaks, _ = signal.find_peaks(np.abs(fft[pos_mask]))
    
    if len(peaks) > 0:
        features['seasonal_strength'] = np.max(np.abs(fft[pos_mask][peaks])) / len(time_series)
        features['n_seasonal_peaks'] = len(peaks)
    else:
        features['seasonal_strength'] = 0
        features['n_seasonal_peaks'] = 0
    
    # Change point detection using rolling statistics
    rolling_std = np.std(np.lib.stride_tricks.sliding_window_view(time_series, window_size), axis=1)
    threshold = np.mean(rolling_std) + 2 * np.std(rolling_std)
    change_points = np.where(rolling_std > threshold)[0]
    
    features['n_change_points'] = len(change_points)
    features['change_point_density'] = len(change_points) / len(time_series)
    
    return features

def extract_all_temporal_features(voltage_series: NDArray[np.float64],
                                current_series: NDArray[np.float64],
                                frequency_series: NDArray[np.float64],
                                sampling_rate: float,
                                window_size: Optional[int] = None) -> Dict[str, float]:
    """Extract all temporal features from multiple time series.
    
    Args:
        voltage_series: Array of voltage magnitudes over time
        current_series: Array of current magnitudes over time
        frequency_series: Array of frequency values over time
        sampling_rate: Sampling rate in Hz
        window_size: Optional window size for pattern detection
        
    Returns:
        Dictionary containing all temporal features
    """
    if window_size is None:
        window_size = min(100, len(voltage_series) // 10)
    
    features = {}
    
    # Voltage stability features
    voltage_indices = compute_voltage_stability_indices(voltage_series)
    features.update({f'voltage_{k}': v for k, v in voltage_indices.items()})
    
    # Oscillation features for each signal
    for name, series in [('voltage', voltage_series), 
                        ('current', current_series), 
                        ('frequency', frequency_series)]:
        osc_features = extract_oscillation_features(series, sampling_rate)
        features.update({f'{name}_{k}': v for k, v in osc_features.items()})
    
    # Time domain features for each signal
    for name, series in [('voltage', voltage_series), 
                        ('current', current_series), 
                        ('frequency', frequency_series)]:
        time_features = compute_time_domain_features(series)
        features.update({f'{name}_{k}': v for k, v in time_features.items()})
    
    # Frequency domain features for each signal
    for name, series in [('voltage', voltage_series), 
                        ('current', current_series), 
                        ('frequency', frequency_series)]:
        freq_features = extract_frequency_features(series, sampling_rate)
        features.update({f'{name}_{k}': v for k, v in freq_features.items()})
    
    # Pattern detection for each signal
    for name, series in [('voltage', voltage_series), 
                        ('current', current_series), 
                        ('frequency', frequency_series)]:
        pattern_features = detect_temporal_patterns(series, window_size)
        features.update({f'{name}_{k}': v for k, v in pattern_features.items()})
    
    return features