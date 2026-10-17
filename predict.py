import argparse

import pandas as pd
import torch

from sage_model import MatchSAGE, build_graph


def load(model_path="match_sage.pt", people_path="people.json", matches_path="matches.csv"):
    people = pd.read_json(people_path)
    matches = pd.read_csv(matches_path)
    data = build_graph(people, matches)
    model = MatchSAGE(data.num_node_features)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model, data, people, matches


@torch.no_grad()
def predict_pair(model, data, person_a, person_b):
    """Probability that two people (1-based ids) match."""
    pair = torch.tensor([[person_a - 1], [person_b - 1]])
    return torch.sigmoid(model(data.x, data.edge_index, pair)).item()


@torch.no_grad()
def recommend(model, data, matches, person, top_k=10):
    """Rank everyone this person hasn't matched with yet by match probability."""
    already = set(matches.loc[matches["source"] == person, "target"]) | set(
        matches.loc[matches["target"] == person, "source"]
    )
    candidates = [p for p in range(1, data.num_nodes + 1) if p != person and p not in already]

    pairs = torch.tensor([[person - 1] * len(candidates), [c - 1 for c in candidates]])
    probs = torch.sigmoid(model(data.x, data.edge_index, pairs))
    ranked = pd.DataFrame({"id": candidates, "p_match": probs.numpy()})
    return ranked.sort_values("p_match", ascending=False).head(top_k).reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict matches with the trained GraphSAGE model.")
    parser.add_argument("person", type=int, help="id of the person (1-200)")
    parser.add_argument("other", type=int, nargs="?", help="second person's id; omit to get recommendations")
    parser.add_argument("--top", type=int, default=10, help="number of recommendations")
    args = parser.parse_args()

    model, data, people, matches = load()

    if args.other is not None:
        p = predict_pair(model, data, args.person, args.other)
        print(f"person {args.person} & person {args.other}: p(match) = {p:.3f}")
    else:
        recs = recommend(model, data, matches, args.person, args.top)
        cols = ["id", "gender", "age", "relationship_goal", "likes"]
        print(f"top {args.top} new matches for person {args.person}:")
        print(recs.merge(people[cols], on="id").to_string(index=False))
