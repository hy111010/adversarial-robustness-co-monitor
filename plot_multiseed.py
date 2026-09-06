from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.table)
    metrics = ["clean_accuracy", "fgsm_accuracy", "pgd10_accuracy"]
    labels = ["Clean", "FGSM", "PGD-10"]
    colors = ["#4C78A8", "#F58518", "#59A14F"]

    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), constrained_layout=True)
    x = np.arange(len(data))
    width = 0.24
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors, strict=True)):
        values = 100 * data[metric].to_numpy()
        bars = axes[0].bar(x + (i - 1) * width, values, width, label=label, color=color)
        for bar, value in zip(bars, values, strict=True):
            axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.7, f"{value:.1f}", ha="center", fontsize=7.5)
    axes[0].set_xticks(x, [f"seed {seed}" for seed in data["seed"]])
    axes[0].set_ylim(0, 90)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_title("Results across random seeds")
    axes[0].legend(fontsize=8)

    means = 100 * data[metrics].mean().to_numpy()
    stds = 100 * data[metrics].std(ddof=1).to_numpy()
    bars = axes[1].bar(np.arange(3), means, yerr=stds, capsize=5, color=colors)
    for bar, mean, std in zip(bars, means, stds, strict=True):
        axes[1].text(bar.get_x() + bar.get_width() / 2, mean + std + 1, f"{mean:.2f}±{std:.2f}", ha="center", fontsize=8)
    axes[1].set_xticks(np.arange(3), labels)
    axes[1].set_ylim(0, 90)
    axes[1].set_ylabel("Test accuracy (%)")
    axes[1].set_title("Mean and sample standard deviation")

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
