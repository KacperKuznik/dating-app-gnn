import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from torch import nn
from torch_geometric.nn import SAGEConv

from speed_dating import (
    FIELDS, GENDERS, GOALS, HOBBIES, OTHER_NUMERIC, PREFERENCES, RACES, SELF_RATINGS,
    load_pairs, load_people,
)

SEED = 42
HIDDEN_DIM = 64
EPOCHS = 300
LR = 0.005
# Share of people whose matches are hidden from message passing each epoch,
# so the model also learns to score brand-new users who have no matches yet
COLD_START_RATIO = 0.2
# Share of training dates used as supervision each epoch; their match edges are
# removed from message passing, otherwise the model learns "edge exists" instead of compatibility
SUPERVISION_RATIO = 0.3
SPLIT_SHARES = {"train": 0.7, "val": 0.15, "test": 0.15}

NUMERIC = OTHER_NUMERIC + HOBBIES + PREFERENCES + SELF_RATINGS
CATEGORICAL = {
    "gender": list(GENDERS.values()),
    "goal": list(GOALS.values()),
    "field": list(FIELDS.values()),
    "race": list(RACES.values()),
}


def build_node_features(people, reference=None):
    """Encode each person as a fixed-length numeric vector.

    Numeric answers are imputed and standardised with the stats of `reference`
    (defaults to `people`), so new users are scaled exactly like the training data.
    A missing category becomes an all-zero one-hot block.
    """
    ref = people if reference is None else reference
    numeric = people[NUMERIC].astype(float).fillna(ref[NUMERIC].median())
    numeric = (numeric - ref[NUMERIC].mean()) / ref[NUMERIC].std()
    one_hots = [
        pd.get_dummies(pd.Categorical(people[col], categories=cats)).to_numpy(dtype=float)
        for col, cats in CATEGORICAL.items()
    ]
    return torch.tensor(np.hstack([numeric.to_numpy()] + one_hots), dtype=torch.float)


def undirected(edge_pairs):
    """[2, E] pairs -> [2, 2E] edge_index with both directions."""
    return torch.cat([edge_pairs, edge_pairs.flip(0)], dim=1)


def split_people(n, seed=SEED):
    """Assign every person to train / val / test. Val and test people are unseen during training."""
    order = np.random.default_rng(seed).permutation(n)
    n_train = int(SPLIT_SHARES["train"] * n)
    n_val = int(SPLIT_SHARES["val"] * n)
    split = np.empty(n, dtype=object)
    split[order[:n_train]] = "train"
    split[order[n_train:n_train + n_val]] = "val"
    split[order[n_train + n_val:]] = "test"
    return split


def build_dataset(seed=SEED):
    people, pairs = load_people(), load_pairs()
    node_of = {pid: i for i, pid in enumerate(people["id"])}
    pairs["a_idx"] = pairs["person_a"].map(node_of)
    pairs["b_idx"] = pairs["person_b"].map(node_of)

    people["split"] = split_people(len(people), seed)
    # A date belongs to the "newest" split of its two people: any date involving
    # a test person is a test date, so test people never appear in training
    rank = {"train": 0, "val": 1, "test": 2}
    names = ["train", "val", "test"]
    r = np.maximum(people["split"].map(rank).to_numpy()[pairs["a_idx"]],
                   people["split"].map(rank).to_numpy()[pairs["b_idx"]])
    pairs["split"] = [names[i] for i in r]

    def tensors(part):
        sub = pairs[pairs["split"] == part]
        return (torch.tensor(sub[["a_idx", "b_idx"]].to_numpy().T),
                torch.tensor(sub["match"].to_numpy(), dtype=torch.float))

    x = build_node_features(people)
    train_pairs, train_labels = tensors("train")
    val_pairs, val_labels = tensors("val")
    test_pairs, test_labels = tensors("test")
    return {
        "people": people,
        "pairs": pairs,
        "x": x,
        # At evaluation the model sees every training match; val/test people have none
        "edge_index": undirected(train_pairs[:, train_labels == 1]),
        "train": (train_pairs, train_labels),
        "val": (val_pairs, val_labels),
        "test": (test_pairs, test_labels),
    }


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


def train_step(model, optimizer, ds, use_graph=True):
    model.train()
    optimizer.zero_grad()
    pairs, labels = ds["train"]
    num_nodes = ds["x"].size(0)

    # Real dates give real non-matches, so no negative sampling is needed
    supervised = torch.rand(labels.size(0)) < SUPERVISION_RATIO
    if use_graph:
        message = pairs[:, (labels == 1) & ~supervised]
        cold = torch.rand(num_nodes) < COLD_START_RATIO
        message = message[:, ~(cold[message[0]] | cold[message[1]])]
        edge_index = undirected(message)
    else:
        edge_index = torch.empty(2, 0, dtype=torch.long)

    logits = model(ds["x"], edge_index, pairs[:, supervised])
    loss = F.binary_cross_entropy_with_logits(logits, labels[supervised])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def predict_probs(model, ds, part, use_graph=True):
    model.eval()
    pairs, labels = ds[part]
    edge_index = ds["edge_index"] if use_graph else torch.empty(2, 0, dtype=torch.long)
    return torch.sigmoid(model(ds["x"], edge_index, pairs)).numpy(), labels.numpy()


def best_threshold(model, ds, use_graph=True):
    """Cut-off that maximises F1 on validation. Only ~16% of dates match, so 0.5 is far too high."""
    probs, labels = predict_probs(model, ds, "val", use_graph)
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    return float(thresholds[np.argmax(f1[:-1])])


def evaluate(model, ds, part, use_graph=True, threshold=0.5):
    probs, labels = predict_probs(model, ds, part, use_graph)
    predicted = probs >= threshold
    return {
        "auc": roc_auc_score(labels, probs),
        "ap": average_precision_score(labels, probs),
        "precision": precision_score(labels, predicted, zero_division=0),
        "recall": recall_score(labels, predicted),
        "f1": f1_score(labels, predicted),
        "match_rate": labels.mean(),  # AP of a random guess
    }


def train(use_graph=True, verbose=True, seed=SEED):
    """Train on the speed dating data; returns everything needed to inspect the run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ds = build_dataset(seed)
    if verbose:
        counts = ds["people"]["split"].value_counts().to_dict()
        print(f"{ds['x'].size(0)} people ({counts}), {ds['x'].size(1)} features, "
              f"{len(ds['pairs'])} dates, {int(ds['pairs']['match'].sum())} matches")

    model = MatchSAGE(ds["x"].size(1))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)

    best_val_auc, best_state = 0.0, None
    history = []
    for epoch in range(1, EPOCHS + 1):
        loss = train_step(model, optimizer, ds, use_graph)
        val = evaluate(model, ds, "val", use_graph)
        history.append({"epoch": epoch, "loss": loss, "val_auc": val["auc"], "val_ap": val["ap"]})
        if val["auc"] > best_val_auc:
            best_val_auc = val["auc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and epoch % 25 == 0:
            print(f"epoch {epoch:3d}  loss {loss:.4f}  val auc {val['auc']:.4f}  val ap {val['ap']:.4f}")

    model.load_state_dict(best_state)
    return {
        "model": model,
        "ds": ds,
        "history": pd.DataFrame(history),
        "best_val_auc": best_val_auc,
        "threshold": best_threshold(model, ds, use_graph),
    }


def compare_graph(seeds=(1, 2, 3, 4, 5)):
    """Test AUC with and without message passing, averaged over seeds (the test set is small)."""
    for use_graph in (True, False):
        aucs, aps = [], []
        for seed in seeds:
            run = train(use_graph=use_graph, verbose=False, seed=seed)
            test = evaluate(run["model"], run["ds"], "test", use_graph)
            aucs.append(test["auc"])
            aps.append(test["ap"])
        label = "GraphSAGE    " if use_graph else "features only"
        print(f"{label}  test auc {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}   ap {np.mean(aps):.4f} +/- {np.std(aps):.4f}")


def main():
    run = train()
    model, ds, threshold = run["model"], run["ds"], run["threshold"]
    test = evaluate(model, ds, "test", threshold=threshold)
    print(f"\nbest val auc {run['best_val_auc']:.4f}, decision threshold {threshold:.3f} (max F1 on val)")
    print(f"test auc {test['auc']:.4f}  ap {test['ap']:.4f} (random guess: {test['match_rate']:.4f})")
    print(f"test precision {test['precision']:.4f}  recall {test['recall']:.4f}  f1 {test['f1']:.4f}")
    torch.save({"state_dict": model.state_dict(), "threshold": threshold}, "match_sage.pt")
    print("saved match_sage.pt")


if __name__ == "__main__":
    import sys

    if "--compare" in sys.argv:
        compare_graph()
    else:
        main()
