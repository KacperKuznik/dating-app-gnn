import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv

HEADS = 4


class GATEncoder(nn.Module):
    """Like SAGEEncoder, but each person weighs their matches with learned attention
    instead of a plain mean. Self-loops keep cold-start people (no matches) scorable."""

    def __init__(self, in_dim, hidden_dim, heads=HEADS):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim // heads, heads=heads, dropout=0.3)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=1, dropout=0.3)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        return self.conv2(x, edge_index)
