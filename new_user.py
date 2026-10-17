import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

from sage_model import build_node_features
from speed_dating import (
    FIELDS, FREQUENCY, GENDERS, GOALS, HOBBIES, PREFERENCE_NAMES, PREFERENCES, RACES, SELF_RATINGS,
)
from visualize import (
    GENDER_COLORS, INK, INK_2, MATCH_LINE, MUTED, SURFACE, draw_connections, match_graph, network_layout,
    no_axes,
)

NEW_USER_COLOR = INK
FREQUENCY_CODE = {label: code for code, label in FREQUENCY.items()}
SELF_RATING_NAMES = PREFERENCE_NAMES[:5]  # no "shared interests" self-rating in the survey
LIKE_LEVEL = 8  # a hobby rated this high (1-10) counts as a shared passion


def validate_profile(profile):
    choices = {
        "gender": list(GENDERS.values()), "goal": list(GOALS.values()), "field": list(FIELDS.values()),
        "race": list(RACES.values()),
        "date_frequency": list(FREQUENCY.values()), "go_out_frequency": list(FREQUENCY.values()),
    }
    for key, allowed in choices.items():
        if profile.get(key) not in allowed:
            raise ValueError(f"{key} must be one of {allowed}, got {profile.get(key)!r}")
    for key, names in (("hobbies", HOBBIES), ("looking_for", PREFERENCE_NAMES), ("self_rating", SELF_RATING_NAMES)):
        missing = set(names) - set(profile.get(key, {}))
        if missing:
            raise ValueError(f"{key} is missing: {sorted(missing)}")
    if sum(profile["looking_for"].values()) <= 0:
        raise ValueError("looking_for needs at least one non-zero importance")
    if not 18 <= profile["age"] <= 100:
        raise ValueError("age must be between 18 and 100")


def profile_to_row(profile, person_id):
    """Turn a friendly profile dict into a row shaped like speed_dating.load_people()."""
    validate_profile(profile)
    total = sum(profile["looking_for"].values())
    row = {
        "id": person_id, "wave": 0,
        "gender": profile["gender"], "goal": profile["goal"], "field": profile["field"], "race": profile["race"],
        "age": profile["age"],
        "date_frequency": FREQUENCY_CODE[profile["date_frequency"]],
        "go_out_frequency": FREQUENCY_CODE[profile["go_out_frequency"]],
        "importance_same_race": profile["importance_same_race"],
        "importance_same_religion": profile["importance_same_religion"],
        "expected_happiness": profile["expected_happiness"],
    }
    row.update({h: profile["hobbies"][h] for h in HOBBIES})
    row.update({col: profile["looking_for"][name] / total for col, name in zip(PREFERENCES, PREFERENCE_NAMES)})
    row.update({col: profile["self_rating"][name] for col, name in zip(SELF_RATINGS, SELF_RATING_NAMES)})
    return row


@torch.no_grad()
def find_matches(ctx, profile, top_k=10):
    """Score a brand-new user against every opposite-gender participant.

    The user is appended as an extra node with no edges: GraphSAGE builds their
    embedding from their own answers, which is what the cold-start training in
    sage_model.train_step prepares it for.
    Returns (ranked candidates, embeddings of all nodes incl. the user).
    """
    people = ctx["people"]
    user = profile_to_row(profile, person_id=int(people["id"].max()) + 1)
    everyone = pd.concat([people, pd.DataFrame([user])], ignore_index=True)

    x = build_node_features(everyone, reference=people)
    model = ctx["model"]
    model.eval()
    z = model.encoder(x, ctx["edge_index"])
    user_idx = len(people)
    # Opposite gender only (every date in the data was male-female), and skip the few
    # participants who left the survey blank: there is no real profile to recommend
    answered = people[HOBBIES].notna().any(axis=1).to_numpy()
    candidates = np.flatnonzero((people["gender"].to_numpy() != profile["gender"]) & answered)
    pairs = torch.tensor([[user_idx] * len(candidates), candidates.tolist()])
    probs = torch.sigmoid(model.decoder(z, pairs)).numpy()

    ranked = people.iloc[candidates].copy()
    ranked["p_match"] = probs
    # Same cut-off the model was evaluated with (max F1 on validation, saved in match_sage.pt)
    ranked["predicted"] = np.where(probs >= ctx["threshold"], "match", "no match")
    user_hobbies = np.array([user[h] for h in HOBBIES], dtype=float)
    hobby_matrix = ranked[HOBBIES].to_numpy(dtype=float)
    ranked["shared_passions"] = [
        [h for h, a, b in zip(HOBBIES, user_hobbies, row) if a >= LIKE_LEVEL and b >= LIKE_LEVEL]
        for row in hobby_matrix
    ]
    # Same idea as the dataset's int_corr: how similarly the two rate the 17 hobbies
    ranked["interest_corr"] = [
        np.corrcoef(user_hobbies, row)[0, 1] if np.std(row) > 0 else np.nan for row in hobby_matrix
    ]
    cols = ["id", "p_match", "predicted", "age", "goal", "field", "race", "interest_corr", "shared_passions"]
    ranked = ranked.sort_values("p_match", ascending=False).head(top_k).reset_index(drop=True)
    return ranked[cols].round({"p_match": 3, "interest_corr": 2}), z.numpy()


def _legend(ax, n_matches):
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=c,
               markeredgecolor=SURFACE, label=gender)
        for gender, c in GENDER_COLORS.items()
    ]
    handles.append(Line2D([], [], label="match", **MATCH_LINE))
    handles.append(Line2D([], [], color=NEW_USER_COLOR, linewidth=1.2, linestyle="--", label="your top match"))
    handles.append(Line2D([], [], marker="*", linestyle="", markersize=16,
                          markerfacecolor=NEW_USER_COLOR, markeredgecolor=SURFACE, label="you"))
    handles.append(Line2D([], [], marker="o", linestyle="", markersize=10, markerfacecolor="none",
                          markeredgecolor=NEW_USER_COLOR, markeredgewidth=2,
                          label=f"your top {n_matches} matches"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=9)


def plot_on_network(ax, ctx, matches):
    people = ctx["people"]
    graph = match_graph(people, ctx["pairs"])
    # Same layout as visualize.py, so the picture is familiar
    pos = network_layout(graph)
    fixed = list(graph.nodes)
    gender = people.set_index("id")["gender"].to_dict()

    user = "you"
    top = matches["id"].tolist()
    graph.add_edges_from((user, t) for t in top)
    pos = nx.spring_layout(graph, pos=pos, fixed=fixed, seed=42, k=0.08, iterations=100)

    others = [n for n in graph.nodes if n != user]
    existing = [(a, b) for a, b in graph.edges if user not in (a, b)]
    nx.draw_networkx_edges(graph, pos, edgelist=existing, ax=ax, edge_color=MATCH_LINE["color"],
                           width=MATCH_LINE["linewidth"], alpha=MATCH_LINE["alpha"])
    nx.draw_networkx_edges(graph, pos, edgelist=[(user, t) for t in top], ax=ax,
                           edge_color=NEW_USER_COLOR, width=1.2, style="--")
    nx.draw_networkx_nodes(graph, pos, nodelist=others, ax=ax, node_size=18, edgecolors=SURFACE,
                           linewidths=0.6, node_color=[GENDER_COLORS[gender[n]] for n in others])
    nx.draw_networkx_nodes(graph, pos, nodelist=top, ax=ax, node_size=90, edgecolors=NEW_USER_COLOR,
                           linewidths=2, node_color=[GENDER_COLORS[gender[n]] for n in top])
    ax.scatter(*pos[user], marker="*", s=550, color=NEW_USER_COLOR, edgecolors=SURFACE,
               linewidths=1.5, zorder=5)
    for t in top[:3]:
        ax.annotate(f"#{t}", pos[t], xytext=(6, 6), textcoords="offset points", color=INK_2, fontsize=9)
    no_axes(ax)
    ax.set_title("You on the real match network")
    ax.text(0, -0.13, "Dashed lines = your predicted top matches. Clusters = speed-dating events.",
            transform=ax.transAxes, color=MUTED, fontsize=9)
    _legend(ax, len(top))


def plot_on_embeddings(ax, ctx, z, matches):
    people = ctx["people"]
    xy = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(z)
    user = len(people)
    node_of = ctx["node_of"]
    top = [node_of[i] for i in matches["id"]]
    colors = people["gender"].map(GENDER_COLORS).to_numpy()
    pairs = ctx["pairs"]
    matched = pairs[pairs["match"] == 1]

    draw_connections(ax, xy, zip(matched["person_a"].map(node_of), matched["person_b"].map(node_of)),
                     colors=MATCH_LINE["color"], linewidths=0.5, alpha=0.2)
    draw_connections(ax, xy, [(user, t) for t in top], colors=NEW_USER_COLOR, linewidths=1.2,
                     linestyles="--", zorder=3)
    ax.scatter(xy[:user, 0], xy[:user, 1], c=colors, s=24, edgecolors=SURFACE, linewidths=0.8, zorder=2)
    ax.scatter(xy[top, 0], xy[top, 1], c=colors[top], s=90, edgecolors=NEW_USER_COLOR, linewidths=2, zorder=4)
    ax.scatter(*xy[user], marker="*", s=550, color=NEW_USER_COLOR, edgecolors=SURFACE,
               linewidths=1.5, zorder=5)
    no_axes(ax)
    ax.set_title("You in the learned embedding space (t-SNE)")
    ax.text(0, -0.13, "Close points = people the model sees as similar. Dashed lines = your top matches.",
            transform=ax.transAxes, color=MUTED, fontsize=9)
    _legend(ax, len(top))


def plot_new_user(ctx, matches, z):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    plot_on_network(axes[0], ctx, matches)
    plot_on_embeddings(axes[1], ctx, z, matches)
    best = matches.iloc[0]
    fig.suptitle(f"Your closest match: participant {best['id']} (p = {best['p_match']:.2f})",
                 x=0.012, ha="left", fontsize=16, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=3)
    return fig
