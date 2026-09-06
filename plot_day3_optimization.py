from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_row(path: str, checkpoint: str) -> pd.Series:
    data = pd.read_csv(path)
    return data[data["checkpoint"] == checkpoint].set_index("attack")["accuracy"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--pgd2-refined", required=True)
    parser.add_argument("--pgd5-refined", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = [
        get_row(args.original, "best_robust.pt"),
        get_row(args.pgd2_refined, "best_balanced.pt"),
        get_row(args.pgd5_refined, "epoch09.pt"),
    ]
    methods = ["Original Day 3", "PGD-2 refinement", "PGD-5 refinement"]
    attacks = ["clean", "fgsm_8_255", "pgd10_8_255"]
    labels = ["Clean", "FGSM", "PGD-10"]
    values = 100 * np.array([[row[a] for a in attacks] for row in rows])

    x = np.arange(3)
    width = 0.24
    colors = ["#4C78A8", "#F58518", "#59A14F"]
    figure, axis = plt.subplots(figsize=(7.8, 4.7), constrained_layout=True)
    for i, (method, row, color) in enumerate(zip(methods, values, colors, strict=True)):
        bars = axis.bar(x + (i - 1) * width, row, width, label=method, color=color)
        for bar, value in zip(bars, row, strict=True):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 0.8, f"{value:.2f}", ha="center", fontsize=8)
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 90)
    axis.set_ylabel("Test accuracy (%)")
    axis.set_title("Day 3 refinement results")
    axis.grid(axis="y", alpha=0.22)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(fontsize=9)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
