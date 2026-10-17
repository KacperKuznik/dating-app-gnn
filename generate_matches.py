import numpy as np
import pandas as pd

from generate_dataset import generate_people

SEED = 42
# Calibrated so each person ends up with roughly 10 matches
MATCH_THRESHOLD = 3.0
EDUCATION_RANK = {"high school": 0, "bachelor": 1, "master": 2, "phd": 3}


def compatibility(a, b):
    likes_a, likes_b = set(a["likes"]), set(b["likes"])
    dislikes_a, dislikes_b = set(a["dislikes"]), set(b["dislikes"])

    score = 1.5 * len(likes_a & likes_b)
    # Something one person loves and the other hates is a strong turn-off
    score -= 1.5 * (len(likes_a & dislikes_b) + len(likes_b & dislikes_a))
    score += 0.5 * len(dislikes_a & dislikes_b)
    score += 2.5 if a["relationship_goal"] == b["relationship_goal"] else -2.5
    score -= 0.3 * abs(a["age"] - b["age"])
    score -= 0.5 * abs(EDUCATION_RANK[a["education"]] - EDUCATION_RANK[b["education"]])
    return score


def generate_matches(people, threshold=MATCH_THRESHOLD, seed=SEED):
    """Return every pair (i, j), i < j, labelled 1 if they match, else 0.

    A pair matches when its compatibility plus some noise passes the
    threshold, so the label is learnable but not perfectly deterministic.
    """
    rng = np.random.default_rng(seed)
    records = people.to_dict("records")
    rows = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            score = compatibility(records[i], records[j]) + rng.normal(0, 1.0)
            rows.append({
                "source": records[i]["id"],
                "target": records[j]["id"],
                "score": round(score, 3),
                "match": int(score > threshold),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    people = generate_people()
    pairs = generate_matches(people)
    matches = pairs[pairs["match"] == 1]
    matches[["source", "target"]].to_csv("matches.csv", index=False)

    degree = pd.concat([matches["source"], matches["target"]]).value_counts()
    degree = degree.reindex(people["id"], fill_value=0)
    print(f"pairs: {len(pairs)}, matches: {len(matches)} ({len(matches) / len(pairs):.1%})")
    print(f"matches per person: mean {degree.mean():.1f}, min {degree.min()}, max {degree.max()}")
    print(f"people with no match: {(degree == 0).sum()}")
