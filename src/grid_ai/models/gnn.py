"""Graph Neural Network model for grid security prediction."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool

class GridSecurityGNN(nn.Module):
    """Graph Neural Network for grid security prediction.
    
    Args:
        input_dim: Number of input features per node
        hidden_dim: Number of hidden units
        output_dim: Number of output features per node
        num_layers: Number of GNN layers
        dropout: Dropout rate
        gnn_type: Type of GNN layer ('gat' or 'gcn')
        use_edge_weights: Whether to use edge weights
        residual: Whether to use residual connections
    """
    def __init__(
        self, 
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        dropout: float = 0.2,
        gnn_type: str = "gat",
        use_edge_weights: bool = True,
        residual: bool = True
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        
        # Input layer
        self.input_project = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            if gnn_type == "gat":
                self.convs.append(
                    GATConv(
                        hidden_dim, 
                        hidden_dim, 
                        heads=4,
                        dropout=dropout,
                        add_self_loops=True
                    )
                )
            else:  # gcn
                self.convs.append(
                    GCNConv(
                        hidden_dim,
                        hidden_dim,
                        improved=True
                    )
                )
        
        # Batch normalization layers
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        # Output layers
        self.global_pool = global_mean_pool
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, data):
        """Forward pass.
        
        Args:
            data: PyG Data object containing:
                - x: Node features [num_nodes, input_dim]
                - edge_index: Edge indices [2, num_edges]
                - edge_attr: Edge features [num_edges, edge_dim] (optional)
                - batch: Batch indices [num_nodes]
                
        Returns:
            Node-level or graph-level predictions
        """
        x, edge_index = data.x, data.edge_index
        
        # Initial feature projection
        x = self.input_project(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GNN layers with residual connections
        for i in range(self.num_layers):
            identity = x
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.residual:
                x = x + identity
        
        # Global pooling
        x = self.global_pool(x, data.batch)
        
        # MLP head
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        
        return x