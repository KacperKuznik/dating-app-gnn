"""Shared palette, matplotlib style and graph helpers for visualize.ipynb."""
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import font_manager
from matplotlib.lines import Line2D

# Reference palette (dataviz skill): categorical slots 1-2 and light-mode chart chrome
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
GENDER_COLORS = {"female": ORANGE, "male": BLUE}
# How a real match is drawn on every plot
MATCH_LINE = {"color": MUTED, "linewidth": 0.7, "alpha": 0.45}
# Date outcomes, when both kinds of date are drawn together
MATCH_COLOR, NO_MATCH_COLOR = "#2e9e5b", "#d64545"

# First of these that is installed; naming a missing font makes matplotlib warn on every text.
# DejaVu Sans ships with matplotlib, so there is always a match.
_INSTALLED = {f.name for f in font_manager.fontManager.ttflist}
FONT = next(f for f in ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"] if f in _INSTALLED)

plt.rcParams.update({
    "font.family": FONT,
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
