from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", required=True)
    parser.add_argument("--autoattack", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    epsilon = pd.read_csv(args.epsilon)
    autoattack = pd.read_csv(args.autoattack).iloc[-1]

    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), constrained_layout=True)
    axes[0].plot(
        epsilon["epsilon_pixels"],
        100 * epsilon["accuracy"],
        marker="o",
        linewidth=2,
        color="#4C78A8",
    )
    for x, value in zip(epsilon["epsilon_pixels"], 100 * epsilon["accuracy"], strict=True):
        axes[0].text(x, value + 2, f"{value:.2f}", ha="center", fontsize=8)
    axes[0].axvline(8, color="#E45756", linestyle="--", linewidth=1, label="training ε")
    axes[0].set_xticks(epsilon["epsilon_pixels"])
    axes[0].set_ylim(0, 85)
    axes[0].set_xlabel("Perturbation budget ε (/255)")
    axes[0].set_ylabel("PGD-20 test accuracy (%)")
    axes[0].set_title("Robustness sensitivity to ε")
    axes[0].legend(fontsize=8)

    value = 100 * float(autoattack["accuracy"])
    bar = axes[1].bar([0], [value], width=0.52, color="#59A14F")[0]
    axes[1].text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.2f}%", ha="center", fontsize=10)
    axes[1].set_xticks([0], ["AutoAttack\nstandard"])
    axes[1].set_ylim(0, 50)
    axes[1].set_ylabel("Robust accuracy (%)")
    axes[1].set_title("Third-party suite (fixed 1,000 samples)")
    axes[1].text(
        0.5,
        0.04,
        "torchattacks 3.5.1, L∞ ε=8/255",
        transform=axes[1].transAxes,
        ha="center",
        fontsize=8,
        color="#555555",
    )

    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
