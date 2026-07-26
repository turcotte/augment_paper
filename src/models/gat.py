import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, Set2Set

class GATRegression(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_channels=128, heads=4, processing_steps=5):
        super().__init__()
        self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=heads, edge_dim=edge_dim, dropout=0.1)
        self.gat2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=edge_dim, dropout=0.1)
        self.gat3 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, concat=False, edge_dim=edge_dim)
        self.dropout = nn.Dropout(0.2)
        self.pool = Set2Set(hidden_channels, processing_steps=processing_steps)
        self.regressor = nn.Linear(2 * hidden_channels, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        x = F.dropout(F.relu(self.gat1(x, edge_index, edge_attr)), p=0.1, training=self.training)
        x = F.dropout(F.relu(self.gat2(x, edge_index, edge_attr)), p=0.1, training=self.training)
        x = self.dropout(F.relu(self.gat3(x, edge_index, edge_attr)))
        
        latent = self.pool(x, batch)
        pred = self.regressor(latent)
        
        return latent, pred
