"""Tests for data schema and typing."""
import numpy as np
import pytest
from grid_ai.data.schema import GridFeatures, NormalizationParams, ProcessedDataset


def test_grid_features():
    """Test creating and manipulating GridFeatures."""
    n_buses = 10
    n_edges = 15
    
    features = GridFeatures(
        n_buses=n_buses,
        n_lines=8,
        n_transformers=2,
        n_generators=3,
        n_loads=5,
        adjacency_matrix=np.zeros((n_buses, n_buses)),
        edge_index=np.random.randint(0, n_buses, (2, n_edges)),
        edge_attr=np.random.randn(n_edges, 3),
        bus_features=np.random.randn(n_buses, 5),
        voltage_pu=np.ones(n_buses),
        voltage_angle=np.zeros(n_buses),
        line_loading=np.random.uniform(0, 100, 8),
        transformer_loading=np.random.uniform(0, 100, 2)
    )
    
    # Test conversion to PyG Data
    data = features.to_pyg_data()
    assert data.num_nodes == n_buses
    assert data.edge_index.shape == (2, n_edges)
    assert data.x.shape == (n_buses, 5)
    assert data.voltage.shape == (n_buses,)
    assert data.line_load.shape == (8,)


def test_normalization():
    """Test feature normalization and denormalization."""
    # Create sample features
    n_buses = 5
    features = GridFeatures(
        n_buses=n_buses,
        n_lines=3,
        n_transformers=1,
        n_generators=2,
        n_loads=3,
        adjacency_matrix=np.zeros((n_buses, n_buses)),
        edge_index=np.array([[0, 1, 2], [1, 2, 3]]),
        edge_attr=np.random.randn(3, 2),
        bus_features=np.random.randn(n_buses, 4),
        voltage_pu=np.ones(n_buses) + np.random.randn(n_buses) * 0.1,
        voltage_angle=np.random.randn(n_buses) * 0.2,
        line_loading=np.random.uniform(0, 100, 3),
        transformer_loading=np.array([60.0])
    )
    
    # Create normalization params
    norm_params = NormalizationParams(
        mean={
            'bus_features': np.mean(features.bus_features, axis=0),
            'voltage_pu': np.mean(features.voltage_pu),
            'voltage_angle': 0.0,
            'line_loading': 50.0,
            'transformer_loading': 50.0
        },
        std={
            'bus_features': np.std(features.bus_features, axis=0),
            'voltage_pu': 0.1,
            'voltage_angle': 0.2,
            'line_loading': 25.0,
            'transformer_loading': 25.0
        },
        min={},  # Not used in current implementation
        max={}
    )
    
    # Test normalization
    normalized = norm_params.normalize(features)
    assert np.allclose(
        normalized.voltage_pu, 
        (features.voltage_pu - norm_params.mean['voltage_pu']) / norm_params.std['voltage_pu']
    )
    
    # Test denormalization recovers original
    denormalized = norm_params.denormalize(normalized)
    assert np.allclose(denormalized.voltage_pu, features.voltage_pu)
    assert np.allclose(denormalized.bus_features, features.bus_features)


def test_processed_dataset():
    """Test ProcessedDataset container."""
    # Create minimal dataset
    n_samples = 3
    n_buses = 4
    features = []
    
    for _ in range(n_samples):
        feat = GridFeatures(
            n_buses=n_buses,
            n_lines=2,
            n_transformers=1,
            n_generators=1,
            n_loads=2,
            adjacency_matrix=np.zeros((n_buses, n_buses)),
            edge_index=np.array([[0, 1], [1, 2]]),
            edge_attr=np.random.randn(2, 2),
            bus_features=np.random.randn(n_buses, 3),
            voltage_pu=np.ones(n_buses),
            voltage_angle=np.zeros(n_buses),
            line_loading=np.random.uniform(0, 100, 2),
            transformer_loading=np.array([50.0])
        )
        features.append(feat)
    
    labels = np.array([0, 1, 0])
    
    norm_params = NormalizationParams(
        mean={'bus_features': 0.0},
        std={'bus_features': 1.0},
        min={},
        max={}
    )
    
    dataset = ProcessedDataset(
        features=features,
        labels=labels,
        normalization_params=norm_params,
        metadata={'train_idx': [0, 1], 'val_idx': [2]}
    )
    
    assert len(dataset) == n_samples
    
    # Test getting PyG data
    data = dataset.get_pyg_data(0)
    assert data.y == 0
    assert data.num_nodes == n_buses