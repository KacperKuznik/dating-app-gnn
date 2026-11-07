"""Print-sized versions of the EDA.ipynb charts for the EDA report.

Same data and the same palette as the notebook, but drawn at the size they are printed at
(half or full text width of an A4 page), so the text stays readable. Saves vector PDFs.
Needs Neo4j running with the data loaded. Run from anywhere, e.g. from the repo root:
python eda/report/figures.py
"""
import os
import sys

# Project modules live in the repo root, two levels up from eda/report/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter

from plot_style import (
    GENDER_COLORS, INK_2, MATCH_COLOR, MATCH_LINE, MUTED, NO_MATCH_COLOR, SURFACE, match_graph, network_layout, no_axes,
)
from prepare_data import RAW_PATH
from speed_dating import GENDERS, PREFERENCE_NAMES, PREFERENCES, SELF_RATINGS, load_pairs, load_people

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
HALF, FULL = 3.3, 6.9  # inches: half and full text width
BAR, GAP = 0.36, 0.04
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.titlepad": 4, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.labelsize": 7.5,
                     "savefig.facecolor": "white", "figure.facecolor": "white", "axes.facecolor": "white"})


def capitalised(labels):
    return [label[0].upper() + label[1:] for label in labels]


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def hbars(ax, table, fmt="{:.0%}", percent=True, labels=None):
    """Female and male bar per row of `table`; only each gender's largest bar gets a value label."""
    y = np.arange(len(table))
    for sign, gender in ((-1, "female"), (1, "male")):
        values = table[gender].to_numpy(dtype=float)
        bars = ax.barh(y + sign * (BAR + GAP) / 2, values, height=BAR, color=GENDER_COLORS[gender],
                       label=(labels or {}).get(gender, gender))
        top = np.nanargmax(values)
        ax.bar_label(bars, labels=[fmt.format(v) if i == top else "" for i, v in enumerate(values)],
                     padding=2, color=INK_2, fontsize=6.5)
    ax.set_yticks(y, capitalised(list(table.index)))
    ax.invert_yaxis()
    ax.set_xlim(0, np.nanmax(table.to_numpy(dtype=float)) * 1.18)
    if percent:
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, labelcolor=INK_2)
    ax.legend(loc="lower right")


def histograms(values, bins, name, xlabel, median_fmt="{:.0f}", percent_x=False, height=1.95):
    fig, axes = plt.subplots(2, 1, figsize=(HALF, height), sharex=True, sharey=True)
    for ax, gender in zip(axes, ("female", "male")):
        v = values[gender].dropna()
        ax.hist(v, bins=bins, weights=np.full(len(v), 1 / len(v)), color=GENDER_COLORS[gender], rwidth=0.85)
        ax.axvline(v.median(), color=INK_2, linewidth=0.8)
        ax.set_title(f"{capitalised([gender])[0]} ({len(v)}), median {median_fmt.format(v.median())}",
                     fontsize=7.5, pad=2)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
    if percent_x:
        axes[1].xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[1].set_xlabel(xlabel)
    fig.tight_layout(h_pad=0.4)
    save(fig, name)


def main():
    os.makedirs(OUT, exist_ok=True)
    people, pairs = load_people(), load_pairs()
    raw = pd.read_csv(RAW_PATH, encoding="latin-1")
    # Raw-file columns not in the graph (partner ratings, decisions): same people and dates as the graph
    date_rows = raw.dropna(subset=["pid"])
    date_rows = date_rows[date_rows["iid"].isin(people["id"]) & date_rows["pid"].isin(people["id"])]
    date_rows = date_rows.assign(sex=date_rows["gender"].map(GENDERS))
    survey = date_rows.groupby("iid").first()

    # Match network
    graph = match_graph(people, pairs)
    pos = network_layout(graph)
    nodes = list(graph.nodes)
    fig, ax = plt.subplots(figsize=(HALF, 2.1))  # with the legend row below: about the height of the age figure
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=MATCH_LINE["color"], width=0.4, alpha=MATCH_LINE["alpha"])
    nx.draw_networkx_nodes(graph, pos, nodelist=nodes, ax=ax, linewidths=0.3, edgecolors=SURFACE,
                           node_color=[GENDER_COLORS[graph.nodes[n]["gender"]] for n in nodes],
                           node_size=[2 + 1.6 * graph.degree(n) for n in nodes])
    no_axes(ax)
    # Legend in one row below the network, so it covers no cluster (same entries as gender_legend)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=5, markerfacecolor=c, markeredgecolor=SURFACE,
                      label=gender) for gender, c in GENDER_COLORS.items()]
    handles.append(Line2D([], [], label="match", **{**MATCH_LINE, "linewidth": 1}))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0), ncol=3, borderaxespad=0.2)
    save(fig, "network")

    # Every date, one circle per event: women on the left half, men on the right, one line per date
    wave_of = people.set_index("id")["wave"]
    waves = sorted(people["wave"].unique())
    fig, axes = plt.subplots(4, 6, figsize=(HALF, 2.3))
    for ax, wave in zip(axes.flat, waves):
        here = people[people["wave"] == wave].sort_values(["gender", "id"])  # female first: left half
        dates = pairs[pairs["person_a"].map(wave_of) == wave]
        angles = np.pi / 2 + 2 * np.pi * np.arange(len(here)) / len(here)  # from the top, anticlockwise
        pos = {pid: (np.cos(a), np.sin(a)) for pid, a in zip(here["id"], angles)}
        for outcome, color, width, alpha in ((0, NO_MATCH_COLOR, 0.25, 0.35), (1, MATCH_COLOR, 0.6, 0.9)):
            for a, b in dates.loc[dates["match"] == outcome, ["person_a", "person_b"]].itertuples(index=False):
                ax.plot(*zip(pos[a], pos[b]), color=color, linewidth=width, alpha=alpha, solid_capstyle="round")
        ax.scatter(*zip(*pos.values()), s=1.5, color=[GENDER_COLORS[g] for g in here["gender"]], zorder=3,
                   linewidths=0)
        no_axes(ax)
        ax.set_aspect("equal")
        ax.set_title(f"{wave}: {dates['match'].mean():.0%}", fontsize=5.5, pad=1, color=INK_2, fontweight="normal",
                     loc="center")
    for ax in axes.flat[len(waves):]:
        ax.set_visible(False)
    fig.legend(handles=[Line2D([], [], color=MATCH_COLOR, linewidth=1.2, label="match"),
                        Line2D([], [], color=NO_MATCH_COLOR, linewidth=1.2, alpha=0.6, label="no match"),
                        Line2D([], [], marker="o", linestyle="", markersize=3.5, markerfacecolor=GENDER_COLORS["female"],
                               markeredgecolor=SURFACE, label="female"),
                        Line2D([], [], marker="o", linestyle="", markersize=3.5, markerfacecolor=GENDER_COLORS["male"],
                               markeredgecolor=SURFACE, label="male")],
               loc="lower right", bbox_to_anchor=(1, 0.01), ncol=2, fontsize=6, handlelength=1.2, columnspacing=0.8)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95, wspace=0.08, hspace=0.3)
    save(fig, "events")

    # Age
    histograms({g: people.loc[people["gender"] == g, "age"] for g in ("female", "male")},
               bins=np.arange(17.5, people["age"].max() + 1.5), name="age", xlabel="age (years)")

    # What they look for (stated) and what goes with a yes (revealed)
    looking_for = people.groupby("gender")[PREFERENCES].mean().T
    looking_for.index = PREFERENCE_NAMES
    fig, ax = plt.subplots(figsize=(HALF, 2.1))
    hbars(ax, looking_for)
    ax.set_xlabel("average share of the 100 points")
    save(fig, "looking_for")

    given = ["attr", "sinc", "intel", "fun", "amb", "shar"]
    drives_yes = pd.DataFrame({s: [d[c].corr(d["dec"]) for c in given] for s, d in date_rows.groupby("sex")},
                              index=PREFERENCE_NAMES)
    fig, ax = plt.subplots(figsize=(HALF, 2.1))
    hbars(ax, drives_yes, fmt="{:.2f}", percent=False)
    ax.set_xlabel("correlation of the rating given with saying yes")
    save(fig, "drives_yes")

    # Self-rating vs rating received from partners
    self_rating = people.groupby("gender")[SELF_RATINGS].mean().T
    self_rating.index = PREFERENCE_NAMES[:5]
    received = date_rows.groupby("iid")[["attr_o", "sinc_o", "intel_o", "fun_o", "amb_o"]].mean()
    received = received.groupby(survey["sex"]).mean().T
    received.index = PREFERENCE_NAMES[:5]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 1.65), sharey=True, sharex=True)
    y = np.arange(len(self_rating))
    for ax, gender in zip(axes, ("female", "male")):
        color = GENDER_COLORS[gender]
        ax.hlines(y, received[gender], self_rating[gender], color=MUTED, linewidth=1.5, zorder=2)
        ax.scatter(received[gender], y, s=26, color=color, edgecolors=SURFACE, linewidths=1, zorder=3)
        ax.scatter(self_rating[gender], y, s=26, facecolors="white", edgecolors=color, linewidths=1.3, zorder=3)
        ax.set_yticks(y, capitalised(self_rating.index))
        ax.set_xlim(5, 9)
        ax.set_xlabel("average rating (1-10)")
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, labelcolor=INK_2)
        ax.set_title({"female": "Women", "male": "Men"}[gender])
    axes[0].invert_yaxis()
    # One legend for both panels, in neutral grey: fill = who gave the rating (the panel says the gender)
    marker = dict(marker="o", linestyle="", markersize=5, markeredgewidth=1.2)
    fig.legend(handles=[Line2D([], [], color=MUTED, markerfacecolor=MUTED, label="rated by partners", **marker),
                        Line2D([], [], color=MUTED, markerfacecolor="white", label="rated by themselves", **marker)],
               loc="upper right", ncol=2, bbox_to_anchor=(1, 1.06))
    fig.tight_layout(w_pad=1.5)
    save(fig, "self_vs_partners")

    # How picky
    yes_rate = date_rows.groupby("iid")["dec"].mean()
    histograms({g: yes_rate[survey["sex"] == g] for g in ("female", "male")},
               bins=np.arange(-0.025, 1.05, 0.05), name="picky", xlabel="share of their dates they said yes to",
               median_fmt="{:.0%}", percent_x=True)
    print(f"saved figures to {OUT}")


if __name__ == "__main__":
    main()
