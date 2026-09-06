from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training metrics for one run")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics)
    output_path = Path(args.output) if args.output else metrics_path.with_name("training_curves.png")
    data = pd.read_csv(metrics_path)
    best_index = data["val_clean_acc"].idxmax()
    best = data.loc[best_index]

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes[0, 0].plot(data["epoch"], data["train_loss"], color="#2457A7")
    axes[0, 0].set(title="Training loss", xlabel="Epoch", ylabel="Cross-entropy")

    axes[0, 1].plot(data["epoch"], 100 * data["train_acc"], label="Train", color="#2457A7")
    axes[0, 1].plot(data["epoch"], 100 * data["val_clean_acc"], label="Validation", color="#D55E00")
    axes[0, 1].scatter([best["epoch"]], [100 * best["val_clean_acc"]], color="#009E73", zorder=3)
    axes[0, 1].annotate(
        f"best: {int(best['epoch'])}, {100 * best['val_clean_acc']:.2f}%",
        (best["epoch"], 100 * best["val_clean_acc"]),
        xytext=(-115, -24),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#444444"},
    )
    axes[0, 1].set(title="Clean accuracy", xlabel="Epoch", ylabel="Accuracy (%)", ylim=(0, 100))
    axes[0, 1].legend()

    axes[1, 0].plot(data["epoch"], data["lr"], color="#CC79A7")
    axes[1, 0].set(title="Cosine learning-rate schedule", xlabel="Epoch", ylabel="Learning rate")

    axes[1, 1].plot(data["epoch"], data["epoch_seconds"], color="#0072B2")
    axes[1, 1].axhline(data["epoch_seconds"].mean(), color="#666666", linestyle="--", linewidth=1)
    axes[1, 1].set(title="Training time per epoch", xlabel="Epoch", ylabel="Seconds")

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.suptitle("Standard ResNet-18 on CIFAR-10 (seed 17)", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    print(output_path)


if __name__ == "__main__":
    main()
