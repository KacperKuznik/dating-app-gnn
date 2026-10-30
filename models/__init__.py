"""Match models: a GNN encoder (GraphSAGE or GAT) turns people into embeddings,
a shared decoder scores pairs of embeddings."""
from torch import nn

from models.decoder import MatchDecoder
from models.gat import GATEncoder
from models.sage import SAGEEncoder

HIDDEN_DIM = 64
ENCODERS = {"sage": SAGEEncoder, "gat": GATEncoder}


class MatchGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=HIDDEN_DIM, arch="sage"):
        super().__init__()
        self.encoder = ENCODERS[arch](in_dim, hidden_dim)
        self.decoder = MatchDecoder(hidden_dim)

    def forward(self, x, edge_index, edge_label_index):
        return self.decoder(self.encoder(x, edge_index), edge_label_index)
