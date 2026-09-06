from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize matched adaptive intervention runs")
    parser.add_argument("--interventions", nargs="+", type=Path, required=True)
    parser.add_argument("--baselines", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path)
    return parser.parse_args()


def seed_from_frame(frame: pd.DataFrame, fallback: str) -> int:
    if "trigger_global_step" in frame and frame["trigger_global_step"].notna().any():
        return int(fallback.rsplit("seed", 1)[-1])
    return int(fallback.rsplit("seed", 1)[-1])


def main() -> None:
    args = parse_args()
    interventions = {}
    baselines = {}
    for path in args.interventions:
        frame = pd.read_csv(path)
        interventions[seed_from_frame(frame, path.parent.name)] = (path, frame)
    for path in args.baselines:
        frame = pd.read_csv(path)
        baselines[seed_from_frame(frame, path.parent.name)] = (path, frame)
    if set(interventions) != set(baselines):
        raise ValueError(
            f"Seed mismatch: interventions={sorted(interventions)}, baselines={sorted(baselines)}"
        )

    rows = []
    for seed in sorted(interventions):
        intervention_path, intervention = interventions[seed]
        baseline_path, baseline = baselines[seed]
        triggers = intervention[intervention["intervention_triggered"] == 1]
        trigger = triggers.iloc[0] if len(triggers) else None
        best_index = intervention["val_pgd10_acc"].idxmax()
        best = intervention.loc[best_index]
        final = intervention.iloc[-1]
        baseline_final = baseline.iloc[-1]
        rows.append(
            {
                "seed": seed,
                "triggered": int(trigger is not None),
                "trigger_epoch": int(trigger["epoch"]) if trigger is not None else None,
                "trigger_global_step": (
                    int(trigger["trigger_global_step"]) if trigger is not None else None
                ),
                "abandoned_fgsm_batches": (
                    int(trigger["abandoned_fgsm_batches"]) if trigger is not None else 0
                ),
                "best_epoch": int(best["epoch"]),
                "best_clean_acc": float(best["val_clean_acc"]),
                "best_pgd10_acc": float(best["val_pgd10_acc"]),
                "final_clean_acc": float(final["val_clean_acc"]),
                "final_pgd10_acc": float(final["val_pgd10_acc"]),
                "baseline_final_clean_acc": float(baseline_final["val_clean_acc"]),
                "baseline_final_pgd10_acc": float(baseline_final["val_pgd10_acc"]),
                "intervention_training_seconds": float(
                    intervention["total_training_seconds"].sum()
                ),
                "intervention_path": str(intervention_path),
                "baseline_path": str(baseline_path),
            }
        )
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    if args.aggregate_output is not None:
        numeric = (
            "best_clean_acc",
            "best_pgd10_acc",
            "final_clean_acc",
            "final_pgd10_acc",
            "baseline_final_clean_acc",
            "baseline_final_pgd10_acc",
            "intervention_training_seconds",
        )
        aggregates = []
        for column in numeric:
            aggregates.append(
                {
                    "metric": column,
                    "mean": float(output[column].mean()),
                    "sample_std": float(output[column].std(ddof=1)),
                    "seeds": len(output),
                }
            )
        args.aggregate_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(aggregates).to_csv(args.aggregate_output, index=False)
    print(output.to_string(index=False))
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
