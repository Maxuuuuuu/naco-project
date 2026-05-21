
"""
Standalone plotting script for Yang Figure 1/2 reproduction results.

This file does NOT import or rerun yang_og_baseline.py.
It only uses the already-computed values from the completed run.

Run:
    python plot_yang_reproduction_from_saved_values.py
"""

import matplotlib.pyplot as plt
import numpy as np


RESULTS = {
    "Figure 1A": {
        "observed_strategy": [0.979, 0.042, 0.990],
        "expected_strategy": [1.000, 0.010, 1.000],
        "observed_frequencies": [0.177, 0.623, 0.150, 0.050],
        "expected_frequencies": [0.160, 0.610, 0.150, 0.080],
        "observed_cm": 0.8007,
        "expected_cm": 0.7700,
    },
    "Figure 1B": {
        "observed_strategy": [0.904, 0.042, 0.002],
        "expected_strategy": [1.000, 0.000, 0.000],
        "observed_frequencies": [0.002, 0.003, 0.580, 0.415],
        "expected_frequencies": [0.000, 0.000, 0.490, 0.510],
        "observed_cm": 0.0048,
        "expected_cm": 0.0000,
    },
    "Figure 2A": {
        "observed_strategy": [0.977, 0.009, 0.451],
        "expected_strategy": [1.000, 0.000, 0.270],
        "observed_frequencies": [0.082, 0.425, 0.337, 0.156],
        "expected_frequencies": [0.070, 0.500, 0.260, 0.170],
        "observed_cm": 0.5067,
        "expected_cm": 0.5700,
    },
    "Figure 2B": {
        "observed_strategy": [0.939, 0.024, 0.315],
        "expected_strategy": [0.940, 0.010, 0.100],
        "observed_frequencies": [0.063, 0.373, 0.375, 0.189],
        "expected_frequencies": [0.040, 0.280, 0.370, 0.310],
        "observed_cm": 0.4368,
        "expected_cm": 0.3100,
    },
}


def autolabel_bars(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_strategy_comparison():
    figures = list(RESULTS.keys())
    x = np.arange(len(figures))
    width = 0.11

    fig, ax = plt.subplots(figsize=(11, 6))

    labels = [
        "Observed p(F|L)", "Expected p(F|L)",
        "Observed p(L|F)", "Expected p(L|F)",
        "Observed p(L|B)", "Expected p(L|B)",
    ]
    offsets = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]

    values = []
    for fig_name in figures:
        obs = RESULTS[fig_name]["observed_strategy"]
        exp = RESULTS[fig_name]["expected_strategy"]
        values.append([obs[0], exp[0], obs[1], exp[1], obs[2], exp[2]])

    values = np.array(values)

    for i, label in enumerate(labels):
        bars = ax.bar(x + offsets[i] * width, values[:, i], width, label=label)
        autolabel_bars(ax, bars)

    ax.set_title("Yang reproduction: observed vs expected strategy values")
    ax.set_ylabel("Strategy probability")
    ax.set_xticks(x)
    ax.set_xticklabels(figures)
    ax.set_ylim(0, 1.15)
    ax.legend(ncol=2, fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("yang_strategy_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_frequency_comparison():
    figures = list(RESULTS.keys())
    x = np.arange(len(figures))
    width = 0.08

    fig, ax = plt.subplots(figsize=(12, 6))

    labels = [
        "Observed L", "Expected L",
        "Observed F", "Expected F",
        "Observed B", "Expected B",
        "Observed C", "Expected C",
    ]
    offsets = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]

    values = []
    for fig_name in figures:
        obs = RESULTS[fig_name]["observed_frequencies"]
        exp = RESULTS[fig_name]["expected_frequencies"]
        values.append([obs[0], exp[0], obs[1], exp[1], obs[2], exp[2], obs[3], exp[3]])

    values = np.array(values)

    for i, label in enumerate(labels):
        bars = ax.bar(x + offsets[i] * width, values[:, i], width, label=label)
        autolabel_bars(ax, bars)

    ax.set_title("Yang reproduction: observed vs expected positional frequencies")
    ax.set_ylabel("Position frequency")
    ax.set_xticks(x)
    ax.set_xticklabels(figures)
    ax.set_ylim(0, 0.75)
    ax.legend(ncol=4, fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("yang_frequency_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_cm_comparison():
    figures = list(RESULTS.keys())
    observed = [RESULTS[name]["observed_cm"] for name in figures]
    expected = [RESULTS[name]["expected_cm"] for name in figures]

    x = np.arange(len(figures))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))

    observed_bars = ax.bar(x - width / 2, observed, width, label="Observed CM")
    expected_bars = ax.bar(x + width / 2, expected, width, label="Expected CM")

    autolabel_bars(ax, observed_bars)
    autolabel_bars(ax, expected_bars)

    ax.set_title("Yang reproduction: observed vs expected collective mobility")
    ax.set_ylabel("CM = f_L + f_F")
    ax.set_xticks(x)
    ax.set_xticklabels(figures)
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("yang_cm_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    plot_strategy_comparison()
    plot_frequency_comparison()
    plot_cm_comparison()

    print("Saved plots:")
    print("  yang_strategy_comparison.png")
    print("  yang_frequency_comparison.png")
    print("  yang_cm_comparison.png")


if __name__ == "__main__":
    main()
