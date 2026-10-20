"""Load the raw Kaggle file into Neo4j as a graph of participants and their dates.

Only answers from the sign-up survey (filled in *before* the event) go on the
Person nodes. Ratings given during or after a date would leak the outcome.
"""
import pandas as pd

import graph_db
from speed_dating import FIELDS, GENDERS, GOALS, HOBBIES, PREFERENCES, RACES, SELF_RATINGS

RAW_PATH = "dataset/Speed Dating Data.csv"

# Raw survey column -> readable property name on Person nodes
RENAME = {
    "iid": "id",
    "field_cd": "field",
    "date": "date_frequency",
    "go_out": "go_out_frequency",
    "imprace": "importance_same_race",
    "imprelig": "importance_same_religion",
    "exphappy": "expected_happiness",
    **dict(zip(["attr1_1", "sinc1_1", "intel1_1", "fun1_1", "amb1_1", "shar1_1"], PREFERENCES)),
    **dict(zip(["attr3_1", "sinc3_1", "intel3_1", "fun3_1", "amb3_1"], SELF_RATINGS)),
}
RAW_PROFILE_COLUMNS = ["iid", "wave", "gender", "age", "field_cd", "goal", "race", "date", "go_out",
                       "imprace", "imprelig", "exphappy"] + HOBBIES + [c for c in RENAME if c.endswith("_1")]


def build_profiles(raw):
    # The raw file has one row per person per date; survey answers repeat on each row
    people = raw[RAW_PROFILE_COLUMNS].groupby("iid", as_index=False).first().rename(columns=RENAME)
    people[HOBBIES] = people[HOBBIES].clip(0, 10)  # a few answers were outside the 1-10 scale
    # Waves 6-9 rated preferences 1-10, the rest split 100 points: turn both into shares
    people[PREFERENCES] = people[PREFERENCES].div(people[PREFERENCES].sum(axis=1), axis=0)
    people["gender"] = people["gender"].map(GENDERS)
    people["field"] = people["field"].map(FIELDS)
    people["goal"] = people["goal"].map(GOALS)
    people["race"] = people["race"].map(RACES)
    return people


def build_matches(raw):
    # Each date appears twice (once per person): keep one row per pair
    pairs = raw[raw["iid"] < raw["pid"]][["iid", "pid", "match"]]
    return pairs.rename(columns={"iid": "person_a", "pid": "person_b"}).astype(int)


if __name__ == "__main__":
    raw = pd.read_csv(RAW_PATH, encoding="latin-1").dropna(subset=["pid"])
    profiles, matches = build_profiles(raw), build_matches(raw)
    graph_db.write_graph(profiles, matches)
    print(f"Neo4j at {graph_db.URI}: {len(profiles)} people, {len(matches)} dates, "
          f"{matches['match'].sum()} matches")
