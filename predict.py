import argparse

import torch

from sage_model import MatchSAGE, build_node_features, undirected
from speed_dating import load_pairs, load_people


def load(model_path="match_sage.pt"):
    """Load the trained model plus the full graph: every person and every known match."""
    people, pairs = load_people(), load_pairs()
    node_of = {pid: i for i, pid in enumerate(people["id"])}
    matched = pairs[pairs["match"] == 1]
    edge_pairs = torch.tensor([matched["person_a"].map(node_of).to_list(),
                               matched["person_b"].map(node_of).to_list()])

    checkpoint = torch.load(model_path)
    x = build_node_features(people)
    model = MatchSAGE(x.size(1))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return {
        "model": model,
        "threshold": checkpoint["threshold"],
        "people": people,
        "pairs": pairs,
        "x": x,
        "edge_index": undirected(edge_pairs),
        "node_of": node_of,
    }


@torch.no_grad()
def predict_pair(ctx, id_a, id_b):
    """Probability that two participants (by id) match."""
    pair = torch.tensor([[ctx["node_of"][id_a]], [ctx["node_of"][id_b]]])
    return torch.sigmoid(ctx["model"](ctx["x"], ctx["edge_index"], pair)).item()


@torch.no_grad()
def recommend(ctx, person_id, top_k=10):
    """Rank opposite-gender people this participant never dated by match probability."""
    people, pairs = ctx["people"], ctx["pairs"]
    gender = people.loc[people["id"] == person_id, "gender"].item()
    dated = (set(pairs.loc[pairs["person_a"] == person_id, "person_b"])
             | set(pairs.loc[pairs["person_b"] == person_id, "person_a"]))
    candidates = people[(people["gender"] != gender) & ~people["id"].isin(dated)]

    node = ctx["node_of"][person_id]
    idx = [ctx["node_of"][c] for c in candidates["id"]]
    pair_index = torch.tensor([[node] * len(idx), idx])
    probs = torch.sigmoid(ctx["model"](ctx["x"], ctx["edge_index"], pair_index)).numpy()
    ranked = candidates.assign(p_match=probs).sort_values("p_match", ascending=False)
    return ranked.head(top_k).reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict matches with the trained GraphSAGE model.")
    parser.add_argument("person", type=int, help="participant id (1-552)")
    parser.add_argument("other", type=int, nargs="?", help="second participant's id; omit to get recommendations")
    parser.add_argument("--top", type=int, default=10, help="number of recommendations")
    args = parser.parse_args()

    ctx = load()
    if args.other is not None:
        p = predict_pair(ctx, args.person, args.other)
        verdict = "match" if p >= ctx["threshold"] else "no match"
        print(f"person {args.person} & person {args.other}: p(match) = {p:.3f} -> {verdict} "
              f"(threshold {ctx['threshold']:.3f})")
    else:
        recs = recommend(ctx, args.person, args.top)
        print(f"top {args.top} new matches for person {args.person}:")
        print(recs[["id", "p_match", "gender", "age", "goal", "field", "race"]].to_string(index=False))
