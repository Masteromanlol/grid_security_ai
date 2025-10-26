"""Neural network model definition."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.utils import add_self_loops

class GridSecurityModel(nn.Module):
    """Graph Neural Network for grid security assessment."""
    
    def __init__(self, config):
        """Initialize the model based on config parameters.
        
        Args:
            config: Dictionary containing model configuration
                - input_dim: Number of node features
                - hidden_dim: Size of hidden layers
                - num_layers: Number of GNN layers
                - dropout: Dropout probability
                - output_dim: Number of output features per node
                - gnn_type: Type of GNN layer ('gcn' or 'gat')
                - use_edge_weights: Whether to use edge weights
                - residual: Whether to use residual connections
        """
        super().__init__()
        
        # Validate config
        model_config = config.get('model', {})
        required_params = ['input_dim', 'hidden_dim', 'num_layers', 'dropout', 'output_dim', 'gnn_type']
        missing_params = [p for p in required_params if p not in model_config]
        if missing_params:
            raise ValueError(f"Missing required model parameters: {missing_params}")
            
        # Extract parameters
        self.input_dim = model_config['input_dim']
        self.hidden_dim = model_config['hidden_dim']
        self.num_layers = model_config['num_layers']
        self.dropout = model_config['dropout']
        self.output_dim = model_config['output_dim']
        self.gnn_type = model_config['gnn_type']
        self.use_edge_weights = model_config.get('use_edge_weights', False)
        self.use_residual = model_config.get('residual', True)
        
        if self.num_layers < 2:
            raise ValueError("num_layers must be at least 2")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")
            
        # Create GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.linear_transforms = nn.ModuleList() if self.use_residual else None
        
        # Input layer
        if self.gnn_type == 'gcn':
            self.convs.append(GCNConv(self.input_dim, self.hidden_dim))
        elif self.gnn_type == 'gat':
            self.convs.append(GATConv(self.input_dim, self.hidden_dim))
        else:
            raise ValueError(f"Unknown GNN type: {self.gnn_type}")
        
        self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim))
        if self.use_residual:
            self.linear_transforms.append(nn.Linear(self.input_dim, self.hidden_dim))
        
        # Hidden layers
        for _ in range(self.num_layers - 2):
            if self.gnn_type == 'gcn':
                self.convs.append(GCNConv(self.hidden_dim, self.hidden_dim))
            else:
                self.convs.append(GATConv(self.hidden_dim, self.hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim))
            if self.use_residual:
                self.linear_transforms.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        
        # Output layer
        if self.gnn_type == 'gcn':
            self.convs.append(GCNConv(self.hidden_dim, self.output_dim))
        else:
            self.convs.append(GATConv(self.hidden_dim, self.output_dim))
            
        # Initialize weights
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize model parameters using Xavier initialization."""
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, GCNConv):
                # GCNConv has lin.weight and lin.bias
                if hasattr(m, 'lin') and hasattr(m.lin, 'weight'):
                    torch.nn.init.xavier_uniform_(m.lin.weight)
                if hasattr(m, 'lin') and hasattr(m.lin, 'bias') and m.lin.bias is not None:
                    torch.nn.init.zeros_(m.lin.bias)
            elif isinstance(m, GATConv):
                if hasattr(m, 'lin'):
                    torch.nn.init.xavier_uniform_(m.lin.weight)
                if hasattr(m, 'att') and hasattr(m.att, 'weight'):
                    torch.nn.init.xavier_uniform_(m.att.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        self.apply(init_weights)
    
    def forward(self, data):
        """Forward pass through the network.

        Args:
            data: PyG Data object containing:
                - x: Node features [num_nodes, input_dim]
                - edge_index: Graph connectivity [2, num_edges]
                - edge_attr: Optional edge features/weights [num_edges]
                - batch: Batch assignment for multiple graphs

        Returns:
            Graph-level predictions [batch_size, output_dim]
        """
        x, edge_index = data.x, data.edge_index
        edge_weight = data.edge_attr if hasattr(data, 'edge_attr') and self.use_edge_weights else None
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long)

        # Add self-loops to stabilize message passing
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=x.size(0))

        # Initial layer
        identity = x
        if edge_weight is not None:
            x = self.convs[0](x, edge_index, edge_weight)
        else:
            x = self.convs[0](x, edge_index)
        x = self.batch_norms[0](x)
        x = F.relu(x)
        if self.use_residual:
            x = x + self.linear_transforms[0](identity)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Hidden layers
        for i in range(1, self.num_layers - 1):
            identity = x
            if edge_weight is not None:
                x = self.convs[i](x, edge_index, edge_weight)
            else:
                x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            if self.use_residual:
                x = x + self.linear_transforms[i](identity)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Output layer
        if edge_weight is not None:
            x = self.convs[-1](x, edge_index, edge_weight)
        else:
            x = self.convs[-1](x, edge_index)

        # Global pooling (graph-level)
        x = global_mean_pool(x, batch)

        return x