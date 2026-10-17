import numpy as np
import pandas as pd

SEED = 42
N_PEOPLE = 200

INTERESTS = [
    "hiking", "gaming", "cooking", "travel", "reading", "movies", "music",
    "dancing", "gym", "running", "yoga", "photography", "art", "board games",
    "anime", "football", "basketball", "cycling", "swimming", "camping",
    "coffee", "wine", "craft beer", "clubbing", "concerts", "theatre",
    "pets", "dogs", "cats", "fashion", "tech", "politics", "astrology",
    "meditation", "volunteering", "skiing", "surfing", "podcasts",
    "smoking", "vegan food", "spicy food", "karaoke", "museums", "poetry",
]

GENDERS = ["male", "female", "non-binary"]
GENDER_PROBS = [0.47, 0.47, 0.06]

# Mean / std of height in cm per gender
HEIGHT_PARAMS = {"male": (180, 7), "female": (166, 6.5), "non-binary": (172, 8)}

RELATIONSHIP_GOALS = ["long term", "short term"]

EDUCATION_LEVELS = ["high school", "bachelor", "master", "phd"]


def education_for_age(rng, age):
    # Younger people can't realistically hold higher degrees yet
    if age < 21:
        probs = [1.0, 0.0, 0.0, 0.0]
    elif age < 24:
        probs = [0.45, 0.50, 0.05, 0.0]
    elif age < 28:
        probs = [0.25, 0.40, 0.32, 0.03]
    else:
        probs = [0.20, 0.35, 0.35, 0.10]
    return rng.choice(EDUCATION_LEVELS, p=probs)


def generate_people(n=N_PEOPLE, seed=SEED):
    rng = np.random.default_rng(seed)
    rows = []
    for person_id in range(1, n + 1):
        gender = rng.choice(GENDERS, p=GENDER_PROBS)
        age = int(np.clip(rng.normal(27, 6), 18, 50))
        mean, std = HEIGHT_PARAMS[gender]
        height = int(round(np.clip(rng.normal(mean, std), 150, 205)))

        # Likes and dislikes are drawn from the same pool without overlap
        n_likes = rng.integers(3, 8)
        n_dislikes = rng.integers(2, 6)
        picked = rng.choice(INTERESTS, size=n_likes + n_dislikes, replace=False)
        likes = sorted(picked[:n_likes].tolist())
        dislikes = sorted(picked[n_likes:].tolist())

        # Older users lean slightly more towards long-term relationships
        p_long = 0.45 + 0.015 * (age - 18)
        goal = RELATIONSHIP_GOALS[0] if rng.random() < min(p_long, 0.85) else RELATIONSHIP_GOALS[1]

        rows.append({
            "id": person_id,
            "gender": gender,
            "age": age,
            "height_cm": height,
            "education": education_for_age(rng, age),
            "relationship_goal": goal,
            "likes": likes,
            "dislikes": dislikes,
        })

    df = pd.DataFrame(rows)
    df["gender"] = df["gender"].astype("category")
    df["education"] = pd.Categorical(df["education"], categories=EDUCATION_LEVELS, ordered=True)
    df["relationship_goal"] = df["relationship_goal"].astype("category")
    return df


if __name__ == "__main__":
    people = generate_people()
    # JSON keeps likes/dislikes as real lists when loaded back with pd.read_json
    people.to_json("people.json", orient="records", indent=2)
    people.to_csv("people.csv", index=False)
    print(people.head(10).to_string())
    print()
    print(people.describe(include="all").T[["count", "unique", "top", "mean", "min", "max"]])
