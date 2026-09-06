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
    parser.add_argument("--standard", required=True)
    parser.add_argument("--fast", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    standard = pd.read_csv(args.standard).set_index("attack")["accuracy"]
    fast = pd.read_csv(args.fast)
    early = fast[fast["checkpoint"] == "best_robust.pt"].set_index("attack")["accuracy"]
    final = fast[fast["checkpoint"] == "last.pt"].set_index("attack")["accuracy"]
    attacks = ["clean", "fgsm_8_255", "pgd10_8_255"]
    labels = ["Clean", "FGSM", "PGD-10"]
    methods = ["Standard", "Fast AT (early stop)", "Fast AT (final)"]
    values = np.array([[standard[a] for a in attacks], [early[a] for a in attacks], [final[a] for a in attacks]]) * 100

    x = np.arange(len(labels))
    width = 0.24
    colors = ["#4C78A8", "#59A14F", "#E45756"]
    figure, axis = plt.subplots(figsize=(7.4, 4.5), constrained_layout=True)
    for index, (method, row, color) in enumerate(zip(methods, values, colors, strict=True)):
        bars = axis.bar(x + (index - 1) * width, row, width, label=method, color=color)
        for bar, value in zip(bars, row, strict=True):
            label_height = value + 1 if value >= 1 else 1
            axis.text(bar.get_x() + bar.get_width() / 2, label_height, f"{value:.1f}", ha="center", fontsize=8)

    axis.set_xticks(x, labels)
    axis.set_ylabel("Test accuracy (%)")
    axis.set_ylim(0, 105)
    axis.set_title("Standard training vs. fast adversarial training")
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
