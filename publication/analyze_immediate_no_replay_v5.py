from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "publication" / "immediate_no_replay_v5"
SEEDS = (17, 23, 42, 101, 202)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def metrics_path(folder: Path) -> Path:
    repaired = folder / "metrics_repaired.csv"
    return repaired if repaired.exists() else folder / "metrics.csv"


def first_collapse_step(seed: int) -> int:
    folder = ROOT / "outputs_publication" / f"dev_collapse_full_zero_eps8_lr03_seed{seed}"
    rows = read_csv(folder / "batch_metrics.csv")
    peak = -math.inf
    for row in rows:
        text = row.get("label_pgd10_acc", "") or row.get("observer_pgd10_acc", "")
        try:
            value = float(text)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        if peak > -math.inf and peak - value >= 0.20 and value <= 0.10:
            return int(float(row["global_step"]))
        peak = max(peak, value)
    raise RuntimeError(f"No registered CO event for seed {seed}")


def bootstrap(values: list[float], seed: int = 20260906) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(data, size=(100_000, len(data)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def method_folder(method: str, seed: int) -> Path:
    suffix = "dev" if seed in (17, 23, 42) else "holdout"
    if method == "immediate_replay":
        return ROOT / "outputs_publication" / f"adaptive_rollback_pgd2_{suffix}_seed{seed}"
    if method == "immediate_no_replay":
        return ROOT / "outputs_publication" / f"v5_immediate_no_replay_seed{seed}"
    if method == "next_epoch":
        return ROOT / "outputs_publication" / f"v2_adaptive_no_rollback_seed{seed}"
    raise ValueError(method)


def row_for(method: str, seed: int, collapse_step: int) -> dict[str, object]:
    folder = method_folder(method, seed)
    rows = read_csv(metrics_path(folder))
    triggers = [row for row in rows if row.get("intervention_triggered") == "1"]
    if len(triggers) != 1:
        raise RuntimeError(f"Expected one trigger row in {folder}; found {len(triggers)}")
    trigger = triggers[0]
    warning_step = int(float(trigger["trigger_global_step"]))
    if method == "immediate_no_replay" and trigger.get("pgd2_start_global_step"):
        pgd_start = int(float(trigger["pgd2_start_global_step"]))
    elif method == "immediate_replay":
        pgd_start = warning_step
    else:
        epoch_start = int(float(trigger["rolled_back_to_global_step"]))
        pgd_start = epoch_start + int(float(trigger["batches_processed"]))
    runtime_seconds = sum(float(row["total_training_seconds"]) for row in rows)
    return {
        "seed": seed,
        "method": method,
        "warning_step": warning_step,
        "unprotected_co_step": collapse_step,
        "warning_to_pgd2_start_updates": pgd_start - warning_step,
        "intervention_before_co": int(pgd_start < collapse_step),
        "final_clean_acc": float(rows[-1]["val_clean_acc"]),
        "final_pgd10_acc": float(rows[-1]["val_pgd10_acc"]),
        "runtime_minutes": runtime_seconds / 60.0,
        "folder": str(folder.relative_to(ROOT)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    methods = ("immediate_replay", "immediate_no_replay", "next_epoch")
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        collapse_step = first_collapse_step(seed)
        rows.extend(row_for(method, seed, collapse_step) for method in methods)

    with (OUT / "new_experiment_results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aggregate: dict[str, dict[str, float | int]] = {}
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        aggregate[method] = {
            "seeds": len(subset),
            "before_co_count": sum(int(row["intervention_before_co"]) for row in subset),
            "median_warning_to_pgd2_start_updates": statistics.median(
                int(row["warning_to_pgd2_start_updates"]) for row in subset
            ),
        }
        for metric in ("final_clean_acc", "final_pgd10_acc", "runtime_minutes"):
            values = [float(row[metric]) for row in subset]
            aggregate[method][f"{metric}_mean"] = statistics.mean(values)
            aggregate[method][f"{metric}_sample_std"] = statistics.stdev(values)

    comparisons: dict[str, dict[str, dict[str, float]]] = {}
    for other in ("immediate_replay", "next_epoch"):
        name = f"immediate_no_replay_minus_{other}"
        comparisons[name] = {}
        for metric in (
            "intervention_before_co",
            "warning_to_pgd2_start_updates",
            "final_clean_acc",
            "final_pgd10_acc",
            "runtime_minutes",
        ):
            differences = []
            for seed in SEEDS:
                left = next(row for row in rows if row["seed"] == seed and row["method"] == "immediate_no_replay")
                right = next(row for row in rows if row["seed"] == seed and row["method"] == other)
                differences.append(float(left[metric]) - float(right[metric]))
            low, high = bootstrap(differences)
            comparisons[name][metric] = {
                "mean_difference": float(np.mean(differences)),
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
            }

    payload = {
        "protocol": {
            "seeds": list(SEEDS),
            "threshold": 0.16632988750934596,
            "near_horizon_updates": 60,
            "co_drop": 0.20,
            "co_ceiling": 0.10,
            "bootstrap_resamples": 100_000,
        },
        "aggregate": aggregate,
        "paired_bootstrap": comparisons,
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
