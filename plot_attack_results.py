from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib.pyplot as plt
import pandas as pd


LABELS = {
    "clean": "Clean",
    "fgsm_8_255": "FGSM\n($\\epsilon$=8/255)",
    "pgd10_8_255": "PGD-10\n($\\epsilon$=8/255)",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    labels = [LABELS[value] for value in data["attack"]]
    values = 100 * data["accuracy"]
    colors = ["#4C78A8", "#F58518", "#E45756"]

    figure, axis = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)
    bars = axis.bar(labels, values, color=colors, width=0.62)
    axis.set_ylabel("Test accuracy (%)")
    axis.set_ylim(0, 100)
    axis.set_title("Standard ResNet-18 under white-box attacks")
    axis.grid(axis="y", alpha=0.22)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for bar, value in zip(bars, values, strict=True):
        offset = 1.2 if value >= 2 else 1.0
        axis.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:.2f}%", ha="center")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()

