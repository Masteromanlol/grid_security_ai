"""Test cases for preprocessing module."""

import os
import unittest
import tempfile
import pandas as pd
import numpy as np
import torch
from grid_ai.preprocessing import normalize_features, extract_features, get_edge_index
import pandapower as pp

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        """Create test network and data."""
        # Create simple network
        self.net = pp.create_empty_network()
        
        # Create buses
        b1 = pp.create_bus(self.net, vn_kv=110)
        b2 = pp.create_bus(self.net, vn_kv=110)
        b3 = pp.create_bus(self.net, vn_kv=110)
        
        # Create lines
        pp.create_line(self.net, b1, b2, length_km=10, std_type="NAYY 4x50 SE")
        pp.create_line(self.net, b2, b3, length_km=10, std_type="NAYY 4x50 SE")
        
        # Run power flow
        pp.create_gen(self.net, b1, p_mw=100, vm_pu=1.0)
        pp.create_load(self.net, b3, p_mw=100)
        pp.runpp(self.net)
        
        # Create dummy grid state
        self.grid_state = {
            'bus_results': pd.DataFrame({
                'vm_pu': [1.0, 0.98, 0.97],
                'va_degree': [0.0, -1.0, -2.0],
                'p_mw': [100, 0, -100],
                'q_mvar': [10, 0, -10]
            }),
            'line_results': self.net.res_line.copy(),
            'trafo_results': pd.DataFrame()  # Empty transformer results
        }
    
    def test_normalize_features(self):
        """Test feature normalization."""
        # Create sample features
        x = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        
        # Test normalization
        x_norm, mean, std = normalize_features(x)
        
        # Check shapes
        self.assertEqual(x_norm.shape, x.shape)
        self.assertEqual(mean.shape, (1, 3))
        self.assertEqual(std.shape, (1, 3))
        
        # Check normalization results
        self.assertTrue(torch.allclose(x_norm.mean(dim=0), torch.zeros(3), atol=1e-6))
        self.assertTrue(torch.allclose(x_norm.std(dim=0), torch.ones(3), atol=1e-6))
        
        # Test with pre-computed statistics
        x_norm2, _, _ = normalize_features(x, mean, std)
        self.assertTrue(torch.allclose(x_norm, x_norm2))
        
        # Test with zero standard deviation
        x_const = torch.ones(3, 2)  # Constant column
        x_norm, _, _ = normalize_features(x_const)
        self.assertTrue(torch.allclose(x_norm, torch.zeros_like(x_const)))
    
    def test_extract_features(self):
        """Test feature extraction from grid state."""
        features = extract_features(self.grid_state)
        
        # Check shape
        self.assertEqual(features.shape, (3, 4))  # 3 buses, 4 features per bus
        
        # Check feature values
        self.assertTrue(torch.allclose(
            features[:, 0],  # vm_pu
            torch.tensor([1.0, 0.98, 0.97])
        ))
        
        self.assertTrue(torch.allclose(
            features[:, 1],  # va_degree
            torch.tensor([0.0, -1.0, -2.0])
        ))
    
    def test_get_edge_index(self):
        """Test edge index extraction from network."""
        edge_index = get_edge_index(self.net)
        
        # Check shape
        self.assertEqual(edge_index.shape[0], 2)  # Source and target nodes
        self.assertEqual(edge_index.shape[1], 4)  # 2 lines * 2 (bidirectional)
        
        # Check edge index values (should include both directions)
        expected_edges = torch.tensor([
            [0, 1, 1, 2],  # From nodes
            [1, 0, 2, 1]   # To nodes
        ])
        self.assertTrue(torch.equal(edge_index, expected_edges))