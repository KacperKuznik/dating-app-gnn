import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

from generate_dataset import EDUCATION_LEVELS, GENDERS, INTERESTS, RELATIONSHIP_GOALS
from generate_matches import compatibility
from sage_model import build_node_features
from visualize import AXIS, GOAL_COLORS, INK, INK_2, MUTED, SURFACE, no_axes

NEW_USER_COLOR = INK


def validate_profile(profile):
    required = {"gender", "age", "height_cm", "education", "relationship_goal", "likes", "dislikes"}
    missing = required - profile.keys()
    if missing:
        raise ValueError(f"profile is missing: {sorted(missing)}")
    choices = {"gender": GENDERS, "education": EDUCATION_LEVELS, "relationship_goal": RELATIONSHIP_GOALS}
    for key, allowed in choices.items():
        if profile[key] not in allowed:
            raise ValueError(f"{key} must be one of {allowed}, got {profile[key]!r}")
    unknown = (set(profile["likes"]) | set(profile["dislikes"])) - set(INTERESTS)
    if unknown:
        raise ValueError(f"unknown interests: {sorted(unknown)}. Pick from INTERESTS.")
    both = set(profile["likes"]) & set(profile["dislikes"])
    if both:
        raise ValueError(f"can't both like and dislike: {sorted(both)}")
    if not 18 <= profile["age"] <= 100:
        raise ValueError("age must be between 18 and 100")


@torch.no_grad()
def find_matches(model, data, people, profile, top_k=10):
    """Score a brand-new user against everyone in the dataset.

    The user is appended as an extra node with no edges: GraphSAGE builds their
    embedding from their own features, which is what the cold-start training
    in sage_model.train_step prepares it for.
    Returns (ranked candidates, embeddings of all nodes incl. the user).
    """
    validate_profile(profile)
    user = {"id": len(people) + 1, **profile}
    user["likes"], user["dislikes"] = sorted(profile["likes"]), sorted(profile["dislikes"])
    everyone = pd.concat([people, pd.DataFrame([user])], ignore_index=True)

    x = build_node_features(everyone, reference=people)
    user_idx = len(people)
    model.eval()
    z = model.encoder(x, data.edge_index)
    pairs = torch.tensor([[user_idx] * len(people), list(range(len(people)))])
    probs = torch.sigmoid(model.decoder(z, pairs)).numpy()

    ranked = people.copy()
    ranked["p_match"] = probs
    # Score from the rule that generated the training labels, as a sanity check
    ranked["rule_score"] = [compatibility(user, row) for row in people.to_dict("records")]
    ranked["shared_likes"] = [sorted(set(user["likes"]) & set(likes)) for likes in people["likes"]]
    ranked = ranked.sort_values("p_match", ascending=False).head(top_k).reset_index(drop=True)
    cols = ["id", "p_match", "rule_score", "gender", "age", "height_cm", "education",
            "relationship_goal", "shared_likes", "likes", "dislikes"]
    return ranked[cols], z.numpy()


def _legend(ax, n_matches):
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=c,
               markeredgecolor=SURFACE, label=goal)
        for goal, c in GOAL_COLORS.items()
    ]
    handles.append(Line2D([], [], marker="*", linestyle="", markersize=16,
                          markerfacecolor=NEW_USER_COLOR, markeredgecolor=SURFACE, label="you"))
    handles.append(Line2D([], [], marker="o", linestyle="", markersize=10, markerfacecolor="none",
                          markeredgecolor=NEW_USER_COLOR, markeredgewidth=2,
                          label=f"your top {n_matches} matches"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=9)


def plot_on_network(ax, people, data, matches):
    graph = nx.Graph()
    graph.add_nodes_from(range(data.num_nodes))
    graph.add_edges_from(data.edge_index.t().tolist())
    # Same layout as visualize.py, so the picture is familiar
    pos = nx.spring_layout(graph, seed=42, k=0.25)

    user = data.num_nodes
    top = [i - 1 for i in matches["id"]]
    graph.add_node(user)
    graph.add_edges_from((user, t) for t in top)
    pos = nx.spring_layout(graph, pos=pos, fixed=list(range(data.num_nodes)), seed=42, k=0.25)

    colors = people["relationship_goal"].map(GOAL_COLORS).to_numpy()
    existing = [(a, b) for a, b in graph.edges if user not in (a, b)]
    nx.draw_networkx_edges(graph, pos, edgelist=existing, ax=ax, edge_color=AXIS, width=0.4, alpha=0.5)
    nx.draw_networkx_edges(graph, pos, edgelist=[(user, t) for t in top], ax=ax,
                           edge_color=NEW_USER_COLOR, width=1.2, style="--")
    nx.draw_networkx_nodes(graph, pos, nodelist=list(range(data.num_nodes)), ax=ax,
                           node_color=colors, node_size=22, edgecolors=SURFACE, linewidths=0.8)
    nx.draw_networkx_nodes(graph, pos, nodelist=top, ax=ax, node_color=colors[top], node_size=90,
                           edgecolors=NEW_USER_COLOR, linewidths=2)
    ax.scatter(*pos[user], marker="*", s=550, color=NEW_USER_COLOR, edgecolors=SURFACE,
               linewidths=1.5, zorder=5)
    for t in top[:3]:
        ax.annotate(f"#{t + 1}", pos[t], xytext=(6, 6), textcoords="offset points", color=INK_2, fontsize=9)
    no_axes(ax)
    ax.set_title("You on the match network")
    ax.text(0, -0.1, "Dashed lines = your predicted top matches", transform=ax.transAxes,
            color=MUTED, fontsize=9)
    _legend(ax, len(top))


def plot_on_embeddings(ax, people, z, matches):
    xy = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(z)
    user = len(people)
    top = [i - 1 for i in matches["id"]]
    colors = people["relationship_goal"].map(GOAL_COLORS).to_numpy()

    ax.scatter(xy[:user, 0], xy[:user, 1], c=colors, s=36, edgecolors=SURFACE, linewidths=1.0)
    ax.scatter(xy[top, 0], xy[top, 1], c=colors[top], s=90, edgecolors=NEW_USER_COLOR, linewidths=2)
    ax.scatter(*xy[user], marker="*", s=550, color=NEW_USER_COLOR, edgecolors=SURFACE,
               linewidths=1.5, zorder=5)
    no_axes(ax)
    ax.set_title("You in the learned embedding space (t-SNE)")
    ax.text(0, -0.1, "Close points = people the model sees as similar", transform=ax.transAxes,
            color=MUTED, fontsize=9)
    _legend(ax, len(top))


def plot_new_user(people, data, matches, z):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    plot_on_network(axes[0], people, data, matches)
    plot_on_embeddings(axes[1], people, z, matches)
    best = matches.iloc[0]
    fig.suptitle(f"Your closest match: person {best['id']} (p = {best['p_match']:.2f})",
                 x=0.012, ha="left", fontsize=16, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=3)
    return fig
