"""Data schema and typing for grid security data."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
import numpy as np
import torch
from torch_geometric.data import Data


@dataclass
class GridFeatures:
    """Features extracted from power grid simulations."""
    
    # Network-level features
    n_buses: int
    n_lines: int
    n_transformers: int
    n_generators: int
    n_loads: int
    
    # Topology features
    adjacency_matrix: np.ndarray  # (n_buses, n_buses) sparse adjacency
    edge_index: np.ndarray  # (2, n_edges) COO format edge list
    edge_attr: np.ndarray  # (n_edges, n_edge_features)
    
    # Node features
    bus_features: np.ndarray  # (n_buses, n_bus_features)
    voltage_pu: np.ndarray  # (n_buses,) per-unit voltages
    voltage_angle: np.ndarray  # (n_buses,) voltage angles
    
    # Branch features  
    line_loading: np.ndarray  # (n_lines,) line loading percentages
    transformer_loading: np.ndarray  # (n_transformers,) transformer loading
    
    # Time series features (optional)
    voltage_history: Optional[np.ndarray] = None  # (n_timesteps, n_buses)
    frequency_history: Optional[np.ndarray] = None  # (n_timesteps, n_buses)
    
    # Contingency info (optional)
    contingency_type: Optional[str] = None
    failed_components: Optional[List[int]] = None
    
    def to_pyg_data(self) -> Data:
        """Convert features to PyG Data object for GNN training."""
        return Data(
            x=torch.from_numpy(self.bus_features).float(),
            edge_index=torch.from_numpy(self.edge_index).long(),
            edge_attr=torch.from_numpy(self.edge_attr).float(),
            voltage=torch.from_numpy(self.voltage_pu).float(),
            angle=torch.from_numpy(self.voltage_angle).float(),
            line_load=torch.from_numpy(self.line_loading).float(),
            num_nodes=self.n_buses
        )


@dataclass
class NormalizationParams:
    """Parameters for feature normalization."""
    
    # Per-feature statistics
    mean: Dict[str, Union[float, np.ndarray]]
    std: Dict[str, Union[float, np.ndarray]]
    min: Dict[str, Union[float, np.ndarray]]
    max: Dict[str, Union[float, np.ndarray]]
    
    def normalize(self, features: GridFeatures) -> GridFeatures:
        """Apply normalization to features using stored parameters."""
        # Deep copy to avoid modifying original
        import copy
        normalized = copy.deepcopy(features)
        
        # Normalize numerical arrays
        for field in ['bus_features', 'edge_attr', 'voltage_pu', 
                     'voltage_angle', 'line_loading', 'transformer_loading']:
            if hasattr(normalized, field):
                arr = getattr(normalized, field)
                if arr is not None and field in self.mean:
                    arr = (arr - self.mean[field]) / (self.std[field] + 1e-8)
                    setattr(normalized, field, arr)
        
        return normalized
    
    def denormalize(self, features: GridFeatures) -> GridFeatures:
        """Reverse normalization using stored parameters."""
        import copy
        denormalized = copy.deepcopy(features)
        
        for field in ['bus_features', 'edge_attr', 'voltage_pu',
                     'voltage_angle', 'line_loading', 'transformer_loading']:
            if hasattr(denormalized, field):
                arr = getattr(denormalized, field)
                if arr is not None and field in self.mean:
                    arr = arr * (self.std[field] + 1e-8) + self.mean[field]
                    setattr(denormalized, field, arr)
        
        return denormalized


@dataclass
class ProcessedDataset:
    """Container for processed dataset ready for training."""
    
    features: List[GridFeatures]
    labels: np.ndarray
    normalization_params: NormalizationParams
    metadata: Dict[str, Any]  # Additional info like data split indices
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Data:
        """Get normalized PyG Data object at given index."""
        normalized = self.normalization_params.normalize(self.features[idx])
        data = normalized.to_pyg_data()
        data.y = torch.tensor(self.labels[idx])
        return data
    
    def get_pyg_data(self, idx: int) -> Data:
        """Get normalized PyG Data object at given index."""
        return self[idx]