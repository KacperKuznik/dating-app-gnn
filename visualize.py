import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, roc_curve

from sage_model import train

# Reference palette (dataviz skill): categorical slots 1-2 and light-mode chart chrome
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
GOAL_COLORS = {"long term": BLUE, "short term": ORANGE}

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


def goal_legend(ax, loc="upper right"):
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=c,
               markeredgecolor=SURFACE, label=goal)
        for goal, c in GOAL_COLORS.items()
    ]
    ax.legend(handles=handles, title="Relationship goal", title_fontsize=9, loc=loc)


def no_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_network(ax, people, data):
    graph = nx.Graph()
    graph.add_nodes_from(range(data.num_nodes))
    graph.add_edges_from(data.edge_index.t().tolist())
    pos = nx.spring_layout(graph, seed=42, k=0.25)
    degree = np.array([graph.degree(n) for n in graph.nodes])
    colors = people["relationship_goal"].map(GOAL_COLORS).to_numpy()

    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=AXIS, width=0.4, alpha=0.6)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=colors, node_size=12 + 6 * degree,
                           edgecolors=SURFACE, linewidths=1.0)
    no_axes(ax)
    ax.set_title(f"Match network: {data.num_nodes} people, {graph.number_of_edges()} matches")
    ax.text(0, -0.02, "Node size = number of matches", transform=ax.transAxes, color=MUTED, fontsize=9)
    goal_legend(ax)


def plot_embeddings(ax, model, people, data):
    model.eval()
    with torch.no_grad():
        z = model.encoder(data.x, data.edge_index).numpy()
    xy = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(z)
    colors = people["relationship_goal"].map(GOAL_COLORS).to_numpy()

    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=40, edgecolors=SURFACE, linewidths=1.0)
    no_axes(ax)
    ax.set_title("Learned GraphSAGE embeddings (t-SNE)")
    ax.text(0, -0.02, "Close points = people the model sees as similar", transform=ax.transAxes,
            color=MUTED, fontsize=9)
    goal_legend(ax)


def test_predictions(model, test_data):
    model.eval()
    with torch.no_grad():
        logits = model(test_data.x, test_data.edge_index, test_data.edge_label_index)
    return torch.sigmoid(logits).numpy(), test_data.edge_label.numpy()


def plot_roc(ax, probs, labels):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--")
    ax.text(0.62, 0.55, "random guess", color=MUTED, fontsize=9, rotation=38)
    ax.plot(fpr, tpr, color=BLUE, linewidth=2)
    ax.fill_between(fpr, tpr, color=BLUE, alpha=0.08, linewidth=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC on test pairs (AUC = {auc:.3f})")


def plot_probability_split(ax, probs, labels):
    # Side-by-side bars per bin, so the two classes never blend into a third color
    bins = np.linspace(0, 1, 11)
    width = (bins[1] - bins[0]) / 2
    neg_counts, _ = np.histogram(probs[labels == 0], bins=bins)
    pos_counts, _ = np.histogram(probs[labels == 1], bins=bins)
    ax.bar(bins[:-1] + width * 0.5, neg_counts, width=width, color=ORANGE, label="not a match",
           edgecolor=SURFACE, linewidth=1)
    ax.bar(bins[:-1] + width * 1.5, pos_counts, width=width, color=BLUE, label="real match",
           edgecolor=SURFACE, linewidth=1)
    ax.axvline(0.5, color=INK_2, linewidth=1, linestyle=":")
    ax.text(0.51, 0.6, "threshold 0.5", transform=ax.get_xaxis_transform(), color=INK_2, fontsize=9)
    acc = ((probs > 0.5) == labels).mean()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Predicted p(match)")
    ax.set_ylabel("Test pairs")
    ax.set_title(f"Predicted probability by true label (accuracy = {acc:.2f})")
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
    ax.plot(history["epoch"], history["val_auc"], color=BLUE, linewidth=2)
    ax.scatter([best["epoch"]], [best["val_auc"]], s=70, color=BLUE, edgecolors=SURFACE,
               linewidths=2, zorder=3)
    ax.annotate(f"best: {best['val_auc']:.3f} @ epoch {int(best['epoch'])}\n(checkpoint kept)",
                (best["epoch"], best["val_auc"]), xytext=(110, best["val_auc"] - 0.002),
                textcoords="data", va="top", color=INK_2, fontsize=9,
                arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})
    ax.set_ylim(top=history["val_auc"].max() + 0.01)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUC")
    ax.set_title("Validation AUC")


def main():
    run = train(verbose=False)
    model, people, data, test_data = run["model"], run["people"], run["data"], run["test_data"]
    probs, labels = test_predictions(model, test_data)

    fig, axes = plt.subplots(2, 3, figsize=(19, 11.5))
    plot_network(axes[0, 0], people, data)
    plot_embeddings(axes[0, 1], model, people, data)
    plot_roc(axes[0, 2], probs, labels)
    plot_probability_split(axes[1, 0], probs, labels)
    plot_loss(axes[1, 1], run["history"])
    plot_val_auc(axes[1, 2], run["history"])

    fig.suptitle("GraphSAGE match prediction", x=0.012, ha="left", fontsize=17,
                 fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=3, w_pad=3)
    fig.savefig("match_visualization.png", dpi=130)
    print("saved match_visualization.png")


if __name__ == "__main__":
    main()
