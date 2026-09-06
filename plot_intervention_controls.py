from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot intervention accuracy/compute controls")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    labels = {
        "adaptive_cosine": "Adaptive cosine",
        "pgd_validation_monitor": "PGD monitor",
        "full_pgd2": "Full PGD-2",
        "unprotected_fgsm": "Unprotected FGSM",
    }
    colors = {
        "adaptive_cosine": "#2b78b8",
        "pgd_validation_monitor": "#e28e2c",
        "full_pgd2": "#4c9f70",
        "unprotected_fgsm": "#c94f4f",
    }
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for _, row in frame.iterrows():
        method = row["method"]
        axis.errorbar(
            row["training_seconds_mean"] / 60,
            100 * row["final_pgd10_mean"],
            xerr=row["training_seconds_std"] / 60,
            yerr=100 * row["final_pgd10_std"],
            fmt="o",
            markersize=8,
            capsize=4,
            color=colors[method],
            label=labels[method],
        )
    axis.set_xlabel("Training time (minutes, mean ± sample SD)")
    axis.set_ylabel("Validation PGD-10 accuracy (%, mean ± sample SD)")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220)
    plt.close(figure)
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
