"""Synthetic data generator for preprocessing smoke test."""
import os
import numpy as np
import torch
from grid_ai.data.schema import GridFeatures, ProcessedDataset, NormalizationParams

def generate_synthetic_grid(n_buses=10):
    """Generate a synthetic grid for testing."""
    n_lines = n_buses - 1  # Tree structure
    n_transformers = n_buses // 4
    n_generators = n_buses // 3
    n_loads = n_buses // 2
    
    # Create grid features
    features = GridFeatures(
        n_buses=n_buses,
        n_lines=n_lines,
        n_transformers=n_transformers,
        n_generators=n_generators,
        n_loads=n_loads,
        
        # Create tree-structured grid
        adjacency_matrix=np.eye(n_buses) + np.eye(n_buses, k=1),
        edge_index=np.array([
            np.arange(n_lines),
            np.arange(1, n_lines + 1)
        ]),
        edge_attr=np.random.randn(n_lines, 3),  # 3 edge features
        
        # Node features
        bus_features=np.random.randn(n_buses, 5),  # 5 bus features
        voltage_pu=1.0 + np.random.randn(n_buses) * 0.05,
        voltage_angle=np.random.randn(n_buses) * 0.1,
        
        # Branch features
        line_loading=np.random.uniform(50, 90, n_lines),
        transformer_loading=np.random.uniform(40, 80, n_transformers)
    )
    
    return features

def main():
    """Generate synthetic dataset and save for training."""
    n_samples = 100
    n_buses = 10
    
    # Generate samples
    features_list = []
    labels = np.zeros((n_samples, 2))  # 2 output features
    
    for i in range(n_samples):
        features = generate_synthetic_grid(n_buses)
        features_list.append(features)
        
        # Generate synthetic labels (voltage magnitude and angle predictions)
        labels[i] = np.array([
            np.mean(features.voltage_pu),
            np.mean(features.voltage_angle)
        ])
    
    # Calculate normalization parameters
    norm_params = NormalizationParams(
        mean={
            'bus_features': np.mean([f.bus_features for f in features_list], axis=0),
            'voltage_pu': np.mean([f.voltage_pu for f in features_list]),
            'voltage_angle': np.mean([f.voltage_angle for f in features_list]),
            'line_loading': np.mean([f.line_loading for f in features_list]),
            'transformer_loading': np.mean([f.transformer_loading for f in features_list])
        },
        std={
            'bus_features': np.std([f.bus_features for f in features_list], axis=0),
            'voltage_pu': np.std([f.voltage_pu for f in features_list]),
            'voltage_angle': np.std([f.voltage_angle for f in features_list]),
            'line_loading': np.std([f.line_loading for f in features_list]),
            'transformer_loading': np.std([f.transformer_loading for f in features_list])
        },
        min={},  # Not used
        max={}   # Not used
    )
    
    # Create processed dataset
    dataset = ProcessedDataset(
        features=features_list,
        labels=labels,
        normalization_params=norm_params,
        metadata={'n_samples': n_samples, 'n_buses': n_buses}
    )
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    torch.save(dataset, 'data/processed/dataset.pt')
    torch.save(norm_params, 'data/processed/normalization.pt')
    print("Saved synthetic dataset and normalization parameters")

if __name__ == '__main__':
    main()