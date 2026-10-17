import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, roc_curve

from sage_model import predict_probs, train

# Reference palette (dataviz skill): categorical slots 1-2 and light-mode chart chrome
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
GENDER_COLORS = {"female": ORANGE, "male": BLUE}
# How a real match is drawn on every plot
MATCH_LINE = {"color": MUTED, "linewidth": 0.7, "alpha": 0.45}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.frameon": False,
    "legend.labelcolor": INK_2,
})


def gender_legend(ax, loc="upper right"):
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=c,
               markeredgecolor=SURFACE, label=gender)
        for gender, c in GENDER_COLORS.items()
    ]
    handles.append(Line2D([], [], label="match", **MATCH_LINE))
    ax.legend(handles=handles, loc=loc)


def draw_connections(ax, xy, edges, **style):
    """A straight line between the 2D points of every (i, j) node-index pair."""
    segments = [(xy[i], xy[j]) for i, j in edges]
    ax.add_collection(LineCollection(segments, **{"zorder": 1, **style}))


def no_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def match_graph(people, pairs):
    """Every person with at least one match, connected to the people they matched with."""
    graph = nx.Graph()
    matched = pairs[pairs["match"] == 1]
    graph.add_edges_from(zip(matched["person_a"], matched["person_b"]))
    nx.set_node_attributes(graph, people.set_index("id")["gender"].to_dict(), "gender")
    return graph


def network_layout(graph):
    return nx.spring_layout(graph, seed=42, k=0.15, iterations=100)


def plot_network(ax, people, pairs):
    graph = match_graph(people, pairs)
    pos = network_layout(graph)
    nodes = list(graph.nodes)
    colors = [GENDER_COLORS[graph.nodes[n]["gender"]] for n in nodes]
    sizes = [10 + 8 * graph.degree(n) for n in nodes]

    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=MATCH_LINE["color"],
                           width=MATCH_LINE["linewidth"], alpha=MATCH_LINE["alpha"])
    nx.draw_networkx_nodes(graph, pos, nodelist=nodes, ax=ax, node_color=colors, node_size=sizes,
                           edgecolors=SURFACE, linewidths=0.8)
    no_axes(ax)
    n_alone = len(people) - graph.number_of_nodes()
    ax.set_title(f"Real match network: {graph.number_of_edges()} matches")
    ax.text(0, -0.02, f"Clusters = speed-dating events. Node size = number of matches. "
            f"{n_alone} people with no match not shown.", transform=ax.transAxes, color=MUTED, fontsize=9)
    gender_legend(ax)


def plot_embeddings(ax, model, ds):
    model.eval()
    with torch.no_grad():
        z = model.encoder(ds["x"], ds["edge_index"]).numpy()
    xy = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(z)
    colors = ds["people"]["gender"].map(GENDER_COLORS).to_numpy()
    matched = ds["pairs"][ds["pairs"]["match"] == 1]

    # Lighter than on the network: matched people are rarely close here, so lines cross a lot
    draw_connections(ax, xy, zip(matched["a_idx"], matched["b_idx"]), colors=MATCH_LINE["color"],
                     linewidths=0.5, alpha=0.2)
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=26, edgecolors=SURFACE, linewidths=0.8, zorder=2)
    no_axes(ax)
    ax.set_title("Learned GraphSAGE embeddings (t-SNE)")
    ax.text(0, -0.02, "Close points = people the model sees as similar. Lines = real matches.",
            transform=ax.transAxes, color=MUTED, fontsize=9)
    gender_legend(ax)


def plot_roc(ax, curves):
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--")
    ax.text(0.62, 0.55, "random guess", color=MUTED, fontsize=9, rotation=38)
    for (label, probs, labels), color in zip(curves, [BLUE, ORANGE]):
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{label} (AUC {auc:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC on test people (never seen in training)")
    ax.legend(loc="lower right")


def plot_probability_split(ax, probs, labels, threshold):
    # Shares within each class, so the 16% of matches aren't dwarfed by non-matches
    bins = np.linspace(0, 1, 11)
    width = (bins[1] - bins[0]) / 2
    neg, _ = np.histogram(probs[labels == 0], bins=bins)
    pos, _ = np.histogram(probs[labels == 1], bins=bins)
    ax.bar(bins[:-1] + width * 0.5, neg / neg.sum(), width=width, color=ORANGE, label="no match",
           edgecolor=SURFACE, linewidth=1)
    ax.bar(bins[:-1] + width * 1.5, pos / pos.sum(), width=width, color=BLUE, label="real match",
           edgecolor=SURFACE, linewidth=1)
    ax.axvline(threshold, color=INK_2, linewidth=1, linestyle=":")
    ax.annotate(f"decision threshold {threshold:.2f}", (threshold, 0.55), xytext=(0.3, 0.55),
                xycoords=ax.get_xaxis_transform(), textcoords=ax.get_xaxis_transform(),
                va="center", color=INK_2, fontsize=9,
                arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})
    ax.set_xlim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Predicted p(match)")
    ax.set_ylabel("Share of test dates in class")
    ax.set_title("Predicted probability by true outcome")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)


def plot_loss(ax, history):
    ax.plot(history["epoch"], history["loss"], color=BLUE, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss")
    ax.set_ylim(bottom=0)
    ax.set_title("Training loss")


def plot_val_auc(ax, history):
    best = history.loc[history["val_auc"].idxmax()]
    late = best["epoch"] > history["epoch"].max() / 2
    ax.plot(history["epoch"], history["val_auc"], color=BLUE, linewidth=2)
    ax.scatter([best["epoch"]], [best["val_auc"]], s=70, color=BLUE, edgecolors=SURFACE,
               linewidths=2, zorder=3)
    ax.annotate(f"best: {best['val_auc']:.3f} @ epoch {int(best['epoch'])}\n(checkpoint kept)",
                (best["epoch"], best["val_auc"]),
                # Put the label on whichever side of the peak has room
                xytext=(best["epoch"] + (-50 if late else 50), history["val_auc"].min() + 0.04),
                textcoords="data", ha="right" if late else "left", color=INK_2, fontsize=9,
                arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})
    ax.set_ylim(top=history["val_auc"].max() + 0.01)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUC")
    ax.set_title("Validation AUC")


def main():
    run = train(verbose=False)
    plain = train(use_graph=False, verbose=False)
    model, ds = run["model"], run["ds"]
    probs, labels = predict_probs(model, ds, "test")
    plain_probs, plain_labels = predict_probs(plain["model"], plain["ds"], "test", use_graph=False)

    fig, axes = plt.subplots(2, 3, figsize=(19, 11.5))
    plot_network(axes[0, 0], ds["people"], ds["pairs"])
    plot_embeddings(axes[0, 1], model, ds)
    plot_roc(axes[0, 2], [("GraphSAGE", probs, labels), ("features only", plain_probs, plain_labels)])
    plot_probability_split(axes[1, 0], probs, labels, run["threshold"])
    plot_loss(axes[1, 1], run["history"])
    plot_val_auc(axes[1, 2], run["history"])

    fig.suptitle("GraphSAGE match prediction on the Columbia Speed Dating data", x=0.012, ha="left",
                 fontsize=17, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=3, w_pad=3)
    fig.savefig("match_visualization.png", dpi=130)
    print("saved match_visualization.png")


if __name__ == "__main__":
    main()
