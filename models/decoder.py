import torch
from torch import nn


class MatchDecoder(nn.Module):
    """Scores a pair from symmetric combinations, so score(a, b) == score(b, a)."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z, edge_label_index):
        z_a, z_b = z[edge_label_index[0]], z[edge_label_index[1]]
        return self.mlp(torch.cat([z_a * z_b, (z_a - z_b).abs()], dim=-1)).squeeze(-1)
