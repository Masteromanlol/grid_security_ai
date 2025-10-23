"""Test cases for model module."""

import unittest
import torch
from torch_geometric.data import Data
from grid_ai.model import GridSecurityModel

class TestModel(unittest.TestCase):
    def setUp(self):
        """Initialize test configuration and model."""
        # Model configuration
        self.config = {
            'model': {
                'input_dim': 6,      # 4 node features + 2 contingency encoding
                'hidden_dim': 32,
                'output_dim': 2,     # vm_pu and va_degree predictions
                'num_layers': 3,
                'dropout': 0.1,
                'gnn_type': 'gcn'
            }
        }
        
        # Create model
        self.model = GridSecurityModel(self.config)
        
        # Create dummy data
        self.num_nodes = 5
        self.x = torch.randn(self.num_nodes, self.config['model']['input_dim'])
        self.edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 4],  # From nodes
            [1, 0, 2, 1, 3, 2, 4, 3]   # To nodes
        ], dtype=torch.long)
        
        self.data = Data(
            x=self.x,
            edge_index=self.edge_index
        )
    
    def test_model_initialization(self):
        """Test model initialization with different configurations."""
        # Test GCN initialization
        gcn_config = self.config.copy()
        gcn_config['model']['gnn_type'] = 'gcn'
        gcn_model = GridSecurityModel(gcn_config)
        self.assertEqual(len(gcn_model.convs), gcn_config['model']['num_layers'])
        
        # Test GAT initialization
        gat_config = self.config.copy()
        gat_config['model']['gnn_type'] = 'gat'
        gat_model = GridSecurityModel(gat_config)
        self.assertEqual(len(gat_model.convs), gat_config['model']['num_layers'])
        
        # Test invalid GNN type
        invalid_config = self.config.copy()
        invalid_config['model']['gnn_type'] = 'invalid'
        with self.assertRaises(ValueError):
            GridSecurityModel(invalid_config)
    
    def test_model_output_shape(self):
        """Test model output dimensions."""
        # Test single graph
        output = self.model(self.data)
        self.assertEqual(
            output.shape,
            (self.num_nodes, self.config['model']['output_dim'])
        )
        
        # Test batched graphs
        batch_size = 3
        batch_x = torch.cat([self.x for _ in range(batch_size)], dim=0)
        batch_edge_index = torch.cat([
            self.edge_index + i * self.num_nodes
            for i in range(batch_size)
        ], dim=1)
        batch_data = Data(
            x=batch_x,
            edge_index=batch_edge_index,
            batch=torch.repeat_interleave(
                torch.arange(batch_size),
                torch.tensor([self.num_nodes] * batch_size)
            )
        )
        
        batch_output = self.model(batch_data)
        self.assertEqual(
            batch_output.shape,
            (batch_size * self.num_nodes, self.config['model']['output_dim'])
        )
    
    def test_model_forward_backward(self):
        """Test model forward and backward passes."""
        self.model.train()
        
        # Forward pass
        output = self.model(self.data)
        
        # Create dummy target
        target = torch.randn_like(output)
        
        # Compute loss
        loss = torch.nn.MSELoss()(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check that all parameters have gradients
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient")