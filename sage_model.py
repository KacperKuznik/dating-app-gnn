import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling

from generate_dataset import EDUCATION_LEVELS, GENDERS, INTERESTS, RELATIONSHIP_GOALS, generate_people
from generate_matches import generate_matches

SEED = 42
HIDDEN_DIM = 64
EPOCHS = 300
LR = 0.005
# Share of people whose matches are hidden from message passing each epoch,
# so the model also learns to score brand-new users who have no matches yet
COLD_START_RATIO = 0.2


def build_node_features(people, reference=None):
    """Encode each person as a fixed-length numeric vector.

    Age and height are standardised with the stats of `reference` (defaults to
    `people`), so new users are scaled exactly like the training data.
    """
    ref = people if reference is None else reference
    age = (people["age"] - ref["age"].mean()) / ref["age"].std()
    height = (people["height_cm"] - ref["height_cm"].mean()) / ref["height_cm"].std()

    gender = pd.get_dummies(pd.Categorical(people["gender"], categories=GENDERS))
    goal = pd.get_dummies(pd.Categorical(people["relationship_goal"], categories=RELATIONSHIP_GOALS))
    education = pd.get_dummies(pd.Categorical(people["education"], categories=EDUCATION_LEVELS))

    interest_idx = {name: i for i, name in enumerate(INTERESTS)}
    likes = np.zeros((len(people), len(INTERESTS)))
    dislikes = np.zeros((len(people), len(INTERESTS)))
    for row, (liked, disliked) in enumerate(zip(people["likes"], people["dislikes"])):
        likes[row, [interest_idx[x] for x in liked]] = 1
        dislikes[row, [interest_idx[x] for x in disliked]] = 1

    features = np.hstack([
        age.to_numpy()[:, None],
        height.to_numpy()[:, None],
        gender.to_numpy(dtype=float),
        goal.to_numpy(dtype=float),
        education.to_numpy(dtype=float),
        likes,
        dislikes,
    ])
    return torch.tensor(features, dtype=torch.float)


def build_graph(people, matches):
    x = build_node_features(people)
    # Person ids are 1-based, node indices are 0-based
    src = torch.tensor(matches["source"].to_numpy() - 1)
    dst = torch.tensor(matches["target"].to_numpy() - 1)
    edge_index = torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], dim=1)
    return Data(x=x, edge_index=edge_index)


class SAGEEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        return self.conv2(x, edge_index)


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


class MatchSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.encoder = SAGEEncoder(in_dim, hidden_dim)
        self.decoder = MatchDecoder(hidden_dim)

    def forward(self, x, edge_index, edge_label_index):
        return self.decoder(self.encoder(x, edge_index), edge_label_index)


def train_step(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    # Fresh random non-matches every epoch
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.edge_label_index.size(1),
    )
    edge_label_index = torch.cat([data.edge_label_index, neg_edge_index], dim=1)
    edge_label = torch.cat([data.edge_label, torch.zeros(neg_edge_index.size(1))])

    cold = torch.rand(data.num_nodes) < COLD_START_RATIO
    keep = ~(cold[data.edge_index[0]] | cold[data.edge_index[1]])
    logits = model(data.x, data.edge_index[:, keep], edge_label_index)
    loss = F.binary_cross_entropy_with_logits(logits, edge_label)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    probs = torch.sigmoid(model(data.x, data.edge_index, data.edge_label_index)).numpy()
    labels = data.edge_label.numpy()
    return {
        "auc": roc_auc_score(labels, probs),
        "ap": average_precision_score(labels, probs),
        "acc": accuracy_score(labels, probs > 0.5),
    }


@torch.no_grad()
def match_probability(model, data, person_a, person_b):
    """Probability that two people (1-based ids) match."""
    model.eval()
    pair = torch.tensor([[person_a - 1], [person_b - 1]])
    return torch.sigmoid(model(data.x, data.edge_index, pair)).item()


def train(verbose=True):
    """Train on the generated dataset; returns everything needed to inspect the run."""
    # PyG's negative_sampling draws from Python's `random`, so seed every RNG
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    people = generate_people()
    pairs = generate_matches(people)
    matches = pairs[pairs["match"] == 1]
    data = build_graph(people, matches)
    if verbose:
        print(f"graph: {data.num_nodes} people, {matches.shape[0]} matches, {data.num_node_features} features")

    # Val/test hold out real matches plus an equal number of non-matches.
    # Train negatives are sampled on the fly in train_step.
    # disjoint_train_ratio keeps supervision edges out of message passing,
    # otherwise the model learns "edge exists in graph" instead of compatibility.
    train_data, val_data, test_data = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=False,
        disjoint_train_ratio=0.3,
    )(data)

    model = MatchSAGE(data.num_node_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)

    best_val_auc, best_state = 0.0, None
    history = []
    for epoch in range(1, EPOCHS + 1):
        loss = train_step(model, optimizer, train_data)
        val = evaluate(model, val_data)
        history.append({"epoch": epoch, "loss": loss, "val_auc": val["auc"], "val_ap": val["ap"]})
        if val["auc"] > best_val_auc:
            best_val_auc = val["auc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and epoch % 25 == 0:
            print(f"epoch {epoch:3d}  loss {loss:.4f}  val auc {val['auc']:.4f}  val ap {val['ap']:.4f}")

    model.load_state_dict(best_state)
    return {
        "model": model,
        "people": people,
        "data": data,
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "history": pd.DataFrame(history),
        "best_val_auc": best_val_auc,
    }


def main():
    run = train()
    model, test_data, best_val_auc = run["model"], run["test_data"], run["best_val_auc"]
    test = evaluate(model, test_data)
    print(f"\nbest val auc {best_val_auc:.4f}")
    print(f"test auc {test['auc']:.4f}  test ap {test['ap']:.4f}  test acc {test['acc']:.4f}")

    torch.save(model.state_dict(), "match_sage.pt")

    # Example: score a few held-out test pairs
    print("\nsample test predictions:")
    for k in range(5):
        a, b = (test_data.edge_label_index[:, k] + 1).tolist()
        p = match_probability(model, test_data, a, b)
        print(f"  person {a:3d} & person {b:3d}: p(match) = {p:.3f}  actual = {int(test_data.edge_label[k])}")


if __name__ == "__main__":
    main()
