import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class TangCNNRegressor(nn.Module):
    """
    1D Convolutional Neural Network for predicting Mean Ribosome Load (MRL) 
    from variable-length nucleotide sequences (default 50).
    """
    def __init__(
        self, 
        sequence_length: int = 50, 
        num_filters: int = 160, 
        filter_length: int = 8, 
        latent_dim: int = 80
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.num_filters = num_filters
        self.filter_length = filter_length
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv1d(4, num_filters, filter_length, padding="same")
        self.conv2 = nn.Conv1d(num_filters, num_filters, filter_length, padding="same")
        self.bn1 = nn.BatchNorm1d(num_filters)

        self.conv3 = nn.Conv1d(num_filters, num_filters, filter_length, padding="same")
        self.bn2 = nn.BatchNorm1d(num_filters)

        reduced_filters = num_filters // 2
        self.conv4 = nn.Conv1d(num_filters, reduced_filters, filter_length, padding="same")
        self.bn3 = nn.BatchNorm1d(reduced_filters)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(sequence_length * reduced_filters, latent_dim)
        self.regressor = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x is (batch_size, sequence_length, channels)
        # Convert to (batch_size, channels, sequence_length) for Conv1d
        x = x.permute(0, 2, 1) 

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn1(x)
        x = F.dropout(x, 0.2, self.training)

        x = F.relu(self.conv3(x))
        x = self.bn2(x)
        x = F.dropout(x, 0.4, self.training)

        x = F.relu(self.conv4(x))
        x = self.bn3(x)
        x = F.dropout(x, 0.2, self.training)

        x = self.flatten(x)
        latent = F.relu(self.fc(x))
        prediction = self.regressor(latent)
        return latent, prediction
