"""Speed Dating data: codebook constants and loaders for the two project CSVs.

profiles.csv: one row per participant (sign-up survey answers), keyed by `id`.
matches.csv:  one row per date, `person_a`, `person_b`, `wave`, `match` (1 = both said yes).
Both are built from the raw Kaggle file by prepare_data.py.
"""
import pandas as pd

PROFILES_PATH = "profiles.csv"
MATCHES_PATH = "matches.csv"

GENDERS = {0: "female", 1: "male"}
GOALS = {
    1: "fun night out",
    2: "meet new people",
    3: "get a date",
    4: "serious relationship",
    5: "to say I did it",
    6: "other",
}
FIELDS = {
    1: "law", 2: "math", 3: "social science / psychology", 4: "medical science / pharma / biotech",
    5: "engineering", 6: "english / writing / journalism", 7: "history / religion / philosophy",
    8: "business / econ / finance", 9: "education / academia", 10: "biology / chemistry / physics",
    11: "social work", 12: "undergrad / undecided", 13: "political science / intl affairs",
    14: "film", 15: "fine arts", 16: "languages", 17: "architecture", 18: "other",
}
RACES = {
    1: "black", 2: "white", 3: "latino", 4: "asian", 5: "native american", 6: "other",
}
# date_frequency / go_out_frequency are stored as these codes (1 = most often)
FREQUENCY = {
    1: "several times a week", 2: "twice a week", 3: "once a week", 4: "twice a month",
    5: "once a month", 6: "several times a year", 7: "almost never",
}
# How much they enjoy each activity, 1-10
HOBBIES = [
    "sports", "tvsports", "exercise", "dining", "museums", "art", "hiking", "gaming", "clubbing",
    "reading", "tv", "theater", "movies", "concerts", "music", "shopping", "yoga",
]
PREFERENCE_NAMES = ["attractive", "sincere", "intelligent", "fun", "ambitious", "shared interests"]
# What they look for in a partner, as shares of their total importance (sum to 1)
PREFERENCES = [f"pref_{name.replace(' ', '_')}" for name in PREFERENCE_NAMES]
# How they rate themselves, 1-10 (the survey has no "shared interests" self-rating)
SELF_RATINGS = [f"self_{name}" for name in PREFERENCE_NAMES[:5]]
OTHER_NUMERIC = [
    "age", "date_frequency", "go_out_frequency",
    "importance_same_race", "importance_same_religion", "expected_happiness",
]


def load_people(path=PROFILES_PATH):
    return pd.read_csv(path)


def load_pairs(path=MATCHES_PATH):
    return pd.read_csv(path)
