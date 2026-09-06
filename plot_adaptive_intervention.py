from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot matched adaptive intervention results")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.input).sort_values("seed")
    x = np.arange(len(frame))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)

    axes[0].bar(
        x - width / 2,
        100 * frame["baseline_final_pgd10_acc"],
        width,
        label="Unprotected FGSM",
        color="#d95f5f",
    )
    axes[0].bar(
        x + width / 2,
        100 * frame["final_pgd10_acc"],
        width,
        label="Adaptive + PGD-2",
        color="#3978b5",
    )
    axes[0].set_ylabel("Validation PGD-10 accuracy (%)")
    axes[0].set_xlabel("Seed")
    axes[0].set_xticks(x, frame["seed"].astype(str))
    axes[0].set_ylim(0, 55)
    axes[0].legend(frameon=False)

    axes[1].bar(
        x - width / 2,
        100 * frame["baseline_final_clean_acc"],
        width,
        label="Unprotected FGSM",
        color="#d95f5f",
    )
    axes[1].bar(
        x + width / 2,
        100 * frame["final_clean_acc"],
        width,
        label="Adaptive + PGD-2",
        color="#3978b5",
    )
    axes[1].set_ylabel("Validation clean accuracy (%)")
    axes[1].set_xlabel("Seed")
    axes[1].set_xticks(x, frame["seed"].astype(str))
    axes[1].set_ylim(0, 100)
    axes[1].legend(frameon=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220)
    plt.close(figure)
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
