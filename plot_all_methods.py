from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def checkpoint_values(path: str, checkpoint: str) -> pd.Series:
    data = pd.read_csv(path)
    return data[data["checkpoint"] == checkpoint].set_index("attack")["accuracy"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard", required=True)
    parser.add_argument("--fast", required=True)
    parser.add_argument("--recovery", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    standard = checkpoint_values(args.standard, "best.pt")
    early = checkpoint_values(args.fast, "best_robust.pt")
    final = checkpoint_values(args.fast, "last.pt")
    recovery = checkpoint_values(args.recovery, "best_robust.pt")
    attacks = ["clean", "fgsm_8_255", "pgd10_8_255"]
    labels = ["Clean", "FGSM", "PGD-10"]
    methods = ["Standard", "Fast AT (early)", "Fast AT (final)", "Rollback + PGD-2"]
    values = np.array([[row[a] for a in attacks] for row in [standard, early, final, recovery]]) * 100

    x = np.arange(len(labels))
    width = 0.19
    colors = ["#4C78A8", "#59A14F", "#E45756", "#B279A2"]
    figure, axis = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    for index, (method, row, color) in enumerate(zip(methods, values, colors, strict=True)):
        offset = (index - 1.5) * width
        bars = axis.bar(x + offset, row, width, label=method, color=color)
        for bar, value in zip(bars, row, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                max(value + 1, 1),
                f"{value:.1f}",
                ha="center",
                fontsize=7.5,
            )

    axis.set_xticks(x, labels)
    axis.set_ylabel("Test accuracy (%)")
    axis.set_ylim(0, 105)
    axis.set_title("Accuracy under clean inputs and white-box attacks")
    axis.grid(axis="y", alpha=0.22)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(fontsize=8.5, ncol=2)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
