from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.metrics)

    trigger_rows = data[data["collapse_triggered"] == 1]
    trigger_epoch = int(trigger_rows.iloc[0]["epoch"]) if not trigger_rows.empty else None
    best = data.loc[data["val_pgd10_acc"].idxmax()]

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes[0, 0].plot(data["epoch"], data["train_adv_loss"], color="#2457A7")
    axes[0, 0].set(title="Adversarial training loss", xlabel="Epoch", ylabel="Cross-entropy")

    axes[0, 1].plot(data["epoch"], 100 * data["val_clean_acc"], label="Clean", color="#4C78A8")
    axes[0, 1].plot(data["epoch"], 100 * data["val_fgsm_acc"], label="FGSM", color="#F58518")
    axes[0, 1].plot(data["epoch"], 100 * data["val_pgd10_acc"], label="PGD-10", color="#E45756")
    axes[0, 1].scatter([best["epoch"]], [100 * best["val_pgd10_acc"]], color="#009E73", zorder=4)
    axes[0, 1].annotate(
        f"best PGD: epoch {int(best['epoch'])}, {100 * best['val_pgd10_acc']:.2f}%",
        (best["epoch"], 100 * best["val_pgd10_acc"]),
        xytext=(-145, 18),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#444444"},
    )
    axes[0, 1].set(title="Validation accuracy", xlabel="Epoch", ylabel="Accuracy (%)", ylim=(0, 100))
    axes[0, 1].legend()

    axes[1, 0].plot(data["epoch"], data["lr_end"], color="#CC79A7")
    axes[1, 0].set(title="Cyclic learning rate", xlabel="Epoch", ylabel="Learning rate")

    colors = data["attack_mode"].map({"fgsm_rs": "#F58518", "pgd2": "#59A14F"})
    axes[1, 1].bar(data["epoch"], data["epoch_seconds"], color=colors)
    axes[1, 1].set(title="Training time per epoch", xlabel="Epoch", ylabel="Seconds")
    axes[1, 1].text(0.03, 0.94, "orange: FGSM-RS\ngreen: PGD-2", transform=axes[1, 1].transAxes, va="top")

    if trigger_epoch is not None:
        for axis in axes.flat:
            axis.axvline(trigger_epoch, color="#666666", linestyle="--", linewidth=1)
        axes[0, 1].annotate(
            "trigger: PGD-10 drop\nrollback to epoch 2",
            (trigger_epoch, 100 * trigger_rows.iloc[0]["val_pgd10_acc"]),
            xytext=(28, 22),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": "#444444"},
            fontsize=9,
        )

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.suptitle("Rollback + PGD-2 recovery dynamics (seed 17)", fontsize=14)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
