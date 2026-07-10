"""
model.py
PyTorch feedforward neural network for cricket match outcome prediction.
Architecture: input -> [Linear -> BatchNorm -> ReLU -> Dropout] x N -> Linear(2)
"""

import torch
import torch.nn as nn


class CricketOutcomeNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list = None, dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
