from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def checkpoint_row(path: str, checkpoint: str) -> pd.Series:
    data = pd.read_csv(path)
    return data[data["checkpoint"] == checkpoint].set_index("attack")["accuracy"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", required=True)
    parser.add_argument("--rollback", required=True)
    parser.add_argument("--no-rollback", required=True)
    parser.add_argument("--optimized", required=True)
    parser.add_argument("--strong", required=True)
    parser.add_argument("--margin", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    fast = checkpoint_row(args.fast, "last.pt")
    rollback = checkpoint_row(args.rollback, "best_robust.pt")
    no_rollback = checkpoint_row(args.no_rollback, "best_robust.pt")
    optimized = checkpoint_row(args.optimized, "epoch09.pt")
    attacks = ["clean", "pgd10_8_255"]
    methods = ["Fast AT final", "Rollback + PGD-2", "Switch only", "Optimized"]
    values = 100 * np.array([[row[a] for a in attacks] for row in [fast, rollback, no_rollback, optimized]])

    strong = pd.read_csv(args.strong).set_index("attack")["accuracy"]
    margin = pd.read_csv(args.margin).set_index("attack")["accuracy"]
    strong_labels = ["PGD-10 R1", "PGD-20 R1", "PGD-50 R1", "PGD-20 R5", "CW-PGD-50"]
    strong_values = 100 * np.array([
        optimized["pgd10_8_255"],
        strong["pgd20_r1_8_255"],
        strong["pgd50_r1_8_255"],
        strong["pgd20_r5_8_255"],
        margin["cw_margin_pgd50_r1_8_255"],
    ])

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    x = np.arange(2)
    width = 0.19
    colors = ["#E45756", "#4C78A8", "#F58518", "#59A14F"]
    for i, (method, row, color) in enumerate(zip(methods, values, colors, strict=True)):
        bars = axes[0].bar(x + (i - 1.5) * width, row, width, label=method, color=color)
        for bar, value in zip(bars, row, strict=True):
            axes[0].text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}", ha="center", fontsize=7.5)
    axes[0].set_xticks(x, ["Clean", "PGD-10"])
    axes[0].set_ylim(0, 95)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_title("Method ablation")
    axes[0].legend(fontsize=8)

    bars = axes[1].bar(
        np.arange(5), strong_values, color=["#59A14F", "#76B7B2", "#4C78A8", "#B279A2", "#F28E2B"]
    )
    for bar, value in zip(bars, strong_values, strict=True):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.7, f"{value:.2f}", ha="center", fontsize=8)
    axes[1].set_xticks(np.arange(5), strong_labels, rotation=20)
    axes[1].set_ylim(38, 48)
    axes[1].set_ylabel("Robust test accuracy (%)")
    axes[1].set_title("Stronger white-box attacks")

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
