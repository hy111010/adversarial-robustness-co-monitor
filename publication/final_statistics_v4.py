from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "publication" / "final_statistics_v4"
SEEDS = (17, 23, 42, 101, 202)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(row: dict[str, str], *keys: str) -> float:
    for key in keys:
        try:
            value = float(row.get(key, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return float("nan")


def mean_std(values: list[float]) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    return float(data.mean()), float(data.std(ddof=1))


def bootstrap_ci(differences: list[float], seed: int = 20260906) -> tuple[float, float]:
    data = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(100_000, len(data)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def matched_baselines() -> tuple[list[dict[str, object]], dict[str, dict[int, dict[str, float]]]]:
    component = read_csv(ROOT / "publication" / "supplemental_v2" / "component_ablation_runs.csv")
    methods: dict[str, dict[int, dict[str, float]]] = {}
    for row in component:
        methods.setdefault(row["method"], {})[int(row["seed"])] = {
            "clean": float(row["final_clean"]),
            "pgd10": float(row["final_pgd10"]),
            "minutes": float(row["training_minutes"]),
        }
    for method, prefix in (("FastAdv+", "v3_fastadvplus_seed"), ("GradAlign", "v3_gradalign_seed")):
        methods[method] = {}
        for seed in SEEDS:
            rows = read_csv(ROOT / "outputs_publication" / f"{prefix}{seed}" / "metrics.csv")
            seconds = sum(
                number(row, "core_training_seconds", "epoch_seconds", "total_training_seconds")
                for row in rows
            )
            methods[method][seed] = {
                "clean": number(rows[-1], "val_clean_acc"),
                "pgd10": number(rows[-1], "val_pgd10_acc"),
                "minutes": seconds / 60,
            }
    rows_out = []
    display_order = (
        "Adaptive, no rollback",
        "Adaptive + rollback",
        "Fixed switch epoch 13",
        "Fixed switch epoch 21",
        "Full PGD-2",
        "FastAdv+",
        "GradAlign",
    )
    for method in display_order:
        entries = methods[method]
        clean = [entries[seed]["clean"] for seed in SEEDS]
        pgd10 = [entries[seed]["pgd10"] for seed in SEEDS]
        minutes = [entries[seed]["minutes"] for seed in SEEDS]
        clean_mean, clean_std = mean_std(clean)
        pgd_mean, pgd_std = mean_std(pgd10)
        time_mean, time_std = mean_std(minutes)
        rows_out.append({
            "method": method,
            "seeds": 5,
            "clean_mean": clean_mean,
            "clean_sample_std": clean_std,
            "pgd10_mean": pgd_mean,
            "pgd10_sample_std": pgd_std,
            "training_minutes_mean": time_mean,
            "training_minutes_sample_std": time_std,
        })
    return rows_out, methods


def paired_intervals(methods: dict[str, dict[int, dict[str, float]]]) -> list[dict[str, object]]:
    comparisons = (
        ("rollback_minus_no_rollback", "Adaptive + rollback", "Adaptive, no rollback"),
        ("no_rollback_minus_fixed13", "Adaptive, no rollback", "Fixed switch epoch 13"),
        ("no_rollback_minus_fixed21", "Adaptive, no rollback", "Fixed switch epoch 21"),
        ("no_rollback_minus_full_pgd2", "Adaptive, no rollback", "Full PGD-2"),
        ("no_rollback_minus_fastadvplus", "Adaptive, no rollback", "FastAdv+"),
        ("no_rollback_minus_gradalign", "Adaptive, no rollback", "GradAlign"),
    )
    output = []
    for name, left, right in comparisons:
        for metric in ("clean", "pgd10", "minutes"):
            differences = [methods[left][seed][metric] - methods[right][seed][metric] for seed in SEEDS]
            low, high = bootstrap_ci(differences)
            output.append({
                "comparison": name,
                "metric": metric,
                "mean_difference": float(np.mean(differences)),
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
                "paired_seeds": 5,
            })
    return output


def detector_rows() -> list[dict[str, object]]:
    sources = (
        ("CIFAR-10", "PreActResNet-18", 7, 7, 7, 3, 0, 40),
        ("CIFAR-10", "ResNet-18", 5, 4, 2, 3, 0, 80),
        # The third planned collapse-prone CIFAR-100 trajectory did not meet
        # the registered CO definition. It is therefore a non-event and its
        # one alarm episode belongs in the false-alarm denominator together
        # with the two explicitly stable controls.
        ("CIFAR-100", "PreActResNet-18", 2, 2, 2, 3, 1, 40),
        ("CIFAR-10, eps=16/255", "PreActResNet-18", 3, 3, 3, 0, 0, 40),
    )
    output = []
    for dataset, architecture, events, detected, near, non_events, false_alarms, median_lead in sources:
        low, high = wilson(detected, events)
        output.append({
            "dataset": dataset,
            "architecture": architecture,
            "observed_events": events,
            "detected_before_event": detected,
            "detected_within_60_updates": near,
            "event_recall": detected / events,
            "wilson_95_low": low,
            "wilson_95_high": high,
            "non_event_runs": non_events,
            "false_alarm_episodes": false_alarms,
            "median_first_lead_updates": median_lead,
        })
    return output


def natural_summary() -> list[dict[str, object]]:
    observations = read_csv(ROOT / "publication" / "natural_co_timing_v4" / "natural_co_observations.csv")
    timing = read_csv(ROOT / "publication" / "natural_co_timing_v4" / "natural_co_intervention_timing.csv")
    output = []
    for recipe in ("standard", "highlr", "eps16"):
        subset = [row for row in observations if row["recipe"] == recipe]
        interventions = [row for row in timing if row["recipe"] == recipe]
        pgd = [float(row["final_pgd10_acc"]) for row in interventions]
        clean = [float(row["final_clean_acc"]) for row in interventions]
        output.append({
            "recipe": recipe,
            "runs": len(subset),
            "collapse_events": sum(int(row["collapse_event"]) for row in subset),
            "detected_events": sum(int(row["detected_before_collapse"]) for row in subset),
            "alarm_episodes": sum(int(row["alarm_episodes"]) for row in subset),
            "immediate_interventions": len(interventions),
            "intervention_before_collapse": sum(int(row["intervention_before_collapse"]) for row in interventions),
            "intervention_clean_mean": float(np.mean(clean)) if clean else float("nan"),
            "intervention_clean_sample_std": float(np.std(clean, ddof=1)) if len(clean) > 1 else float("nan"),
            "intervention_pgd10_mean": float(np.mean(pgd)) if pgd else float("nan"),
            "intervention_pgd10_sample_std": float(np.std(pgd, ddof=1)) if len(pgd) > 1 else float("nan"),
        })
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    baseline_rows, methods = matched_baselines()
    paired_rows = paired_intervals(methods)
    detector = detector_rows()
    natural = natural_summary()
    write_csv(OUT / "matched_baselines.csv", baseline_rows)
    write_csv(OUT / "paired_bootstrap_95ci.csv", paired_rows)
    write_csv(OUT / "detector_generalization.csv", detector)
    write_csv(OUT / "natural_co_summary.csv", natural)
    (OUT / "summary.json").write_text(json.dumps({
        "matched_baselines": baseline_rows,
        "paired_intervals": paired_rows,
        "detector_generalization": detector,
        "natural_co": natural,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"matched_baselines": baseline_rows, "natural_co": natural}, indent=2))


if __name__ == "__main__":
    main()
