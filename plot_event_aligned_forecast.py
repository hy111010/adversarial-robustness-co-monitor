from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analyze_adaptive_forecast import add_adaptive_features
from analyze_collapse_forecast import first_collapse_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot traces aligned to the first collapse event")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.16632988750934596)
    parser.add_argument("--before", type=int, default=10)
    parser.add_argument("--after", type=int, default=4)
    args = parser.parse_args()
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    aligned_rows = []
    for path in args.inputs:
        frame = pd.read_csv(path).sort_values("global_step").reset_index(drop=True)
        frame = add_adaptive_features(frame, 5, 3)
        event = first_collapse_index(frame["label_pgd10_acc"].to_numpy(float), 0.20, 0.10)
        if event is None:
            continue
        match = re.search(r"seed(\d+)", path.parent.name)
        seed = match.group(1) if match else path.parent.name
        start = max(0, event - args.before)
        stop = min(len(frame), event + args.after + 1)
        view = frame.iloc[start:stop].copy()
        view["relative_window"] = view.index - event
        view["seed"] = seed
        aligned_rows.append(view)
        axes[0].plot(
            view["relative_window"],
            100 * view["label_pgd10_acc"],
            marker="o",
            markersize=3,
            linewidth=1.4,
            label=f"seed {seed}",
        )
        axes[1].plot(
            view["relative_window"],
            view["adaptive_gradient_cosine_drop"],
            marker="o",
            markersize=3,
            linewidth=1.4,
            label=f"seed {seed}",
        )
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1, label="collapse")
    axes[0].set_ylabel("PGD-10 label accuracy (%)")
    axes[0].set_xlabel("Windows relative to collapse")
    axes[0].set_ylim(0, 55)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].axhline(
        args.threshold, color="#b33a3a", linestyle=":", linewidth=1.5, label="frozen threshold"
    )
    axes[1].set_ylabel("Adaptive cosine-drop score")
    axes[1].set_xlabel("Windows relative to collapse")
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220)
    plt.close(figure)
    if aligned_rows:
        pd.concat(aligned_rows, ignore_index=True).to_csv(
            args.output.with_suffix(".csv"), index=False
        )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
