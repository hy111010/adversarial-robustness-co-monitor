from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot held-out detector ablation")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    frame = frame[frame["horizon_windows"] == 3].copy()
    order = [
        "adaptive_gradient_cosine_drop",
        "adaptive_gradient_sign_drop",
        "adaptive_cosine_accuracy_divergence",
        "adaptive_train_loss_drop",
        "adaptive_train_accuracy_rise",
    ]
    labels = ["Cosine drop", "Sign drop", "Cosine + acc.", "Loss drop", "Acc. rise"]
    frame = frame.set_index("feature").loc[order].reset_index()
    x = np.arange(len(frame))
    colors = ["#3978b5", "#66a3d2", "#8a6bb8", "#df9b45", "#cc5b5b"]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].bar(x, 100 * frame["event_recall"], color=colors)
    axes[0].set_ylabel("Event recall (%)")
    axes[0].set_ylim(0, 110)
    axes[1].bar(x, frame["false_alarms_per_run"], color=colors)
    axes[1].set_ylabel("False alarms per run")
    axes[2].bar(x, frame["auprc"], color=colors)
    axes[2].set_ylabel("AUPRC")
    axes[2].set_ylim(0, 1)
    for axis in axes:
        axis.set_xticks(x, labels, rotation=32, ha="right")
        axis.grid(axis="y", alpha=0.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220)
    plt.close(figure)
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
