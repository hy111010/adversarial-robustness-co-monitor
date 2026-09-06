from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize compute-matched intervention controls")
    parser.add_argument("--adaptive", nargs="+", type=Path, required=True)
    parser.add_argument("--pgd-monitor", nargs="+", type=Path, required=True)
    parser.add_argument("--full-pgd2", nargs="+", type=Path, required=True)
    parser.add_argument("--unprotected", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path, required=True)
    return parser.parse_args()


def seed_of(path: Path) -> int:
    match = re.search(r"seed(\d+)", path.parent.name)
    if match is None:
        raise ValueError(f"Cannot infer seed from {path}")
    return int(match.group(1))


def load_group(paths: list[Path], method: str) -> list[dict]:
    rows = []
    for path in paths:
        frame = pd.read_csv(path)
        final = frame.iloc[-1]
        best = frame.loc[frame["val_pgd10_acc"].idxmax()]
        trigger_rows = (
            frame[frame["intervention_triggered"] == 1]
            if "intervention_triggered" in frame
            else frame.iloc[0:0]
        )
        if "total_training_seconds" in frame:
            training_seconds = float(frame["total_training_seconds"].sum())
        elif "estimated_core_epoch_seconds" in frame:
            training_seconds = float(frame["estimated_core_epoch_seconds"].sum())
        else:
            training_seconds = float(frame["epoch_seconds"].sum())
        rows.append(
            {
                "seed": seed_of(path),
                "method": method,
                "trigger_epoch": (
                    int(trigger_rows.iloc[0]["epoch"]) if len(trigger_rows) else None
                ),
                "best_epoch": int(best["epoch"]),
                "best_clean_acc": float(best["val_clean_acc"]),
                "best_pgd10_acc": float(best["val_pgd10_acc"]),
                "final_clean_acc": float(final["val_clean_acc"]),
                "final_pgd10_acc": float(final["val_pgd10_acc"]),
                "training_seconds": training_seconds,
                "path": str(path),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    rows = []
    rows += load_group(args.adaptive, "adaptive_cosine")
    rows += load_group(args.pgd_monitor, "pgd_validation_monitor")
    rows += load_group(args.full_pgd2, "full_pgd2")
    rows += load_group(args.unprotected, "unprotected_fgsm")
    frame = pd.DataFrame(rows).sort_values(["method", "seed"])
    seed_sets = frame.groupby("method")["seed"].apply(set)
    if len({tuple(sorted(value)) for value in seed_sets}) != 1:
        raise ValueError(f"Methods do not contain matched seeds: {seed_sets.to_dict()}")
    aggregate = (
        frame.groupby("method")
        .agg(
            seeds=("seed", "count"),
            final_clean_mean=("final_clean_acc", "mean"),
            final_clean_std=("final_clean_acc", "std"),
            final_pgd10_mean=("final_pgd10_acc", "mean"),
            final_pgd10_std=("final_pgd10_acc", "std"),
            training_seconds_mean=("training_seconds", "mean"),
            training_seconds_std=("training_seconds", "std"),
        )
        .reset_index()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    aggregate.to_csv(args.aggregate_output, index=False)
    print(aggregate.to_string(index=False))
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
