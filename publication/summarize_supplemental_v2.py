from __future__ import annotations

import csv
import math
import os
import statistics
from collections import deque
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib")
)

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "publication" / "supplemental_v2"
THRESHOLD = 0.16632988750934596


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def completed_run(folder: Path, min_epoch: int = 30) -> bool:
    """Return true only for a run whose metrics contain the planned final epoch."""
    metrics_path = (
        folder / "metrics_repaired.csv"
        if (folder / "metrics_repaired.csv").exists()
        else folder / "metrics.csv"
    )
    if not metrics_path.exists():
        return False
    rows = read_csv(metrics_path)
    return bool(rows) and max(int(row["epoch"]) for row in rows) >= min_epoch


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else float("nan")


def trace_scores(path: Path) -> list[dict[str, float]]:
    rows = read_csv(path)
    history: deque[float] = deque(maxlen=5)
    output = []
    for row in rows:
        cosine = as_float(row, "free_gradient_cosine")
        baseline = statistics.median(history) if len(history) >= 3 else float("nan")
        score = baseline - cosine if math.isfinite(baseline) else float("nan")
        observer_label = as_float(row, "observer_pgd10_acc")
        dense_label = as_float(row, "label_pgd10_acc")
        pgd_label = observer_label if math.isfinite(observer_label) else dense_label
        output.append(
            {
                "epoch": float(row["epoch"]),
                "global_step": float(row["global_step"]),
                "score": score,
                "warning": float(math.isfinite(score) and score >= THRESHOLD),
                "pgd10_label": pgd_label,
            }
        )
        history.append(cosine)
    return output


def collapse_epoch(metrics: list[dict[str, str]]) -> int | None:
    peak = -1.0
    for row in metrics:
        accuracy = as_float(row, "val_pgd10_acc")
        if peak >= 0 and peak - accuracy >= 0.20 and accuracy <= 0.10:
            return int(row["epoch"])
        peak = max(peak, accuracy)
    return None


def analyze_detector_run(seed: int, folder: Path, trace_name: str) -> dict[str, object]:
    metrics = read_csv(folder / "metrics.csv")
    traces = trace_scores(folder / trace_name)
    event_epoch = None
    event_step = None
    labeled = [row for row in traces if math.isfinite(row["pgd10_label"])]
    label_peak = -1.0
    for row in labeled:
        accuracy = row["pgd10_label"]
        if label_peak >= 0 and label_peak - accuracy >= 0.20 and accuracy <= 0.10:
            event_epoch = int(row["epoch"])
            event_step = row["global_step"]
            break
        label_peak = max(label_peak, accuracy)
    if event_step is None:
        event_epoch = collapse_epoch(metrics)
    if event_epoch is not None and event_step is None:
        epoch_steps = [r["global_step"] for r in traces if int(r["epoch"]) == event_epoch]
        event_step = max(epoch_steps) if epoch_steps else None
    warning_steps = [r["global_step"] for r in traces if r["warning"]]
    warning_episodes: list[list[float]] = []
    for step in warning_steps:
        if not warning_episodes or step - warning_episodes[-1][-1] > 20:
            warning_episodes.append([step])
        else:
            warning_episodes[-1].append(step)
    if event_step is None:
        detected = False
        lead_updates = float("nan")
        false_alarms = len(warning_episodes)
    else:
        eligible = [s for s in warning_steps if event_step - 60 <= s < event_step]
        detected = bool(eligible)
        lead_updates = event_step - min(eligible) if eligible else float("nan")
        false_alarms = sum(
            max(episode) < event_step - 60 for episode in warning_episodes
        )
    final_pgd10 = as_float(metrics[-1], "val_pgd10_acc")
    return {
        "seed": seed,
        "collapse_epoch": event_epoch if event_epoch is not None else "",
        "collapse_step": event_step if event_step is not None else "",
        "detected_within_60_updates": int(detected),
        "lead_updates": lead_updates,
        "false_alarm_episodes": false_alarms,
        "total_warning_windows": len(warning_steps),
        "total_alarm_episodes": len(warning_episodes),
        "final_pgd10": final_pgd10,
        "terminal_below_10pct": int(final_pgd10 <= 0.10),
        "folder": str(folder.relative_to(ROOT)),
    }


def training_result(method: str, seed: int, folder: Path) -> dict[str, object]:
    # Early legacy recovery runs wrote short PGD-mode rows to metrics.csv;
    # metrics_repaired.csv restores the declared columns and is authoritative.
    metrics_path = (
        folder / "metrics_repaired.csv"
        if (folder / "metrics_repaired.csv").exists()
        else folder / "metrics.csv"
    )
    rows = read_csv(metrics_path)
    time_key = "total_training_seconds" if "total_training_seconds" in rows[0] else "epoch_seconds"
    seconds = sum(as_float(row, time_key) for row in rows)
    trigger_values = [row.get("trigger_epoch", "") for row in rows if row.get("trigger_epoch", "")]
    return {
        "method": method,
        "seed": seed,
        "final_clean": as_float(rows[-1], "val_clean_acc"),
        "final_pgd10": as_float(rows[-1], "val_pgd10_acc"),
        "training_minutes": seconds / 60.0,
        "trigger_epoch": trigger_values[-1] if trigger_values else "",
        "folder": str(folder.relative_to(ROOT)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    output = []
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        for metric in ("final_clean", "final_pgd10", "training_minutes"):
            values = [float(row[metric]) for row in subset]
            output.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": statistics.mean(values),
                    "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "seeds": len(values),
                }
            )
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    detector_runs = []
    for seed in (101, 202):
        folder = ROOT / "outputs_publication" / f"dev_collapse_full_zero_eps8_lr03_seed{seed}"
        if completed_run(folder):
            detector_runs.append(analyze_detector_run(seed, folder, "batch_metrics.csv"))
    for seed in (303, 404, 505, 606, 707):
        folder = ROOT / "outputs_publication" / f"v2_collapse_observe_seed{seed}"
        if completed_run(folder):
            detector_runs.append(analyze_detector_run(seed, folder, "detector_traces.csv"))
    write_csv(OUT / "expanded_heldout_detector.csv", detector_runs)

    stable_runs = []
    stable_sources = {
        17: (ROOT / "outputs_publication" / "fast_fgsm_rs_preact_seed17", "batch_metrics.csv"),
        23: (ROOT / "outputs_publication" / "v2_stable_fgsm_rs_observe_seed23", "detector_traces.csv"),
        42: (ROOT / "outputs_publication" / "v2_stable_fgsm_rs_observe_seed42", "detector_traces.csv"),
    }
    for seed, (folder, trace_name) in stable_sources.items():
        if completed_run(folder):
            stable_runs.append(analyze_detector_run(seed, folder, trace_name))
    write_csv(OUT / "stable_fgsm_rs_detector.csv", stable_runs)

    training_rows = []
    for seed in (17, 23, 42, 101, 202):
        suffix = "dev" if seed in (17, 23, 42) else "holdout"
        folder = ROOT / "outputs_publication" / f"adaptive_rollback_pgd2_{suffix}_seed{seed}"
        if completed_run(folder):
            training_rows.append(training_result("Adaptive + rollback", seed, folder))
        folder = ROOT / "outputs_publication" / f"v2_adaptive_no_rollback_seed{seed}"
        if completed_run(folder):
            training_rows.append(training_result("Adaptive, no rollback", seed, folder))
    for seed in (17, 23, 42, 101, 202):
        fixed_sources = (
            ("Fixed switch epoch 13", f"v2_fixed_early13_seed{seed}"),
            ("Fixed switch epoch 21", f"v2_fixed_late21_seed{seed}"),
            ("Full PGD-2", f"v2_full_pgd2_seed{seed}"),
        )
        for method, run_name in fixed_sources:
            folder = ROOT / "outputs_publication" / run_name
            if completed_run(folder):
                training_rows.append(training_result(method, seed, folder))
    write_csv(OUT / "component_ablation_runs.csv", training_rows)
    aggregate_rows = aggregate(training_rows)
    write_csv(OUT / "component_ablation_aggregate.csv", aggregate_rows)
    matched_seeds = {17, 101, 202}
    matched_rows = [
        row for row in training_rows if int(row["seed"]) in matched_seeds
    ]
    matched_aggregate_rows = aggregate(matched_rows)
    write_csv(OUT / "component_ablation_matched3.csv", matched_aggregate_rows)

    detected = sum(int(row["detected_within_60_updates"]) for row in detector_runs)
    collapsed = sum(row["collapse_epoch"] != "" for row in detector_runs)
    leads = [float(row["lead_updates"]) for row in detector_runs if math.isfinite(float(row["lead_updates"]))]
    false_alarms = sum(int(row["false_alarm_episodes"]) for row in detector_runs)
    terminal_collapses = sum(int(row["terminal_below_10pct"]) for row in detector_runs)
    stable_alarms = sum(int(row["total_alarm_episodes"]) for row in stable_runs)

    lookup = {
        (str(row["method"]), str(row["metric"])): row for row in aggregate_rows
    }
    method_order = list(dict.fromkeys(str(row["method"]) for row in training_rows))
    x = np.arange(len(method_order))
    robust_means = [100 * float(lookup[(m, "final_pgd10")]["mean"]) for m in method_order]
    robust_stds = [100 * float(lookup[(m, "final_pgd10")]["sample_std"]) for m in method_order]
    time_means = [float(lookup[(m, "training_minutes")]["mean"]) for m in method_order]
    time_stds = [float(lookup[(m, "training_minutes")]["sample_std"]) for m in method_order]
    label_map = {
        "Adaptive + rollback": "Adaptive\n+ rollback",
        "Adaptive, no rollback": "Adaptive\nno rollback",
        "Fixed switch epoch 13": "Fixed\nepoch 13",
        "Fixed switch epoch 21": "Fixed\nepoch 21",
        "Full PGD-2": "Full\nPGD-2",
    }
    labels = [label_map[m] for m in method_order]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].bar(x, robust_means, yerr=robust_stds, capsize=3, color="#3b7db8")
    axes[0].set_ylabel("Final validation PGD-10 accuracy (%)")
    axes[0].set_xticks(x, labels)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, time_means, yerr=time_stds, capsize=3, color="#df9b3f")
    axes[1].set_ylabel("Core training time (min)")
    axes[1].set_xticks(x, labels)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "component_ablation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# Supplemental v2 results",
        "",
        "## Expanded frozen-detector evaluation",
        "",
        f"- Collapse events observed: {collapsed}/{len(detector_runs)}",
        f"- Events warned within 60 updates: {detected}/{collapsed}",
        f"- Runs ending below 10% PGD-10: {terminal_collapses}/{len(detector_runs)}",
        f"- Median lead: {statistics.median(leads):.1f} updates" if leads else "- Median lead: n/a",
        f"- Early false-alarm episodes: {false_alarms} total ({false_alarms/len(detector_runs):.3f}/run)",
        f"- Stable FGSM-RS alarm episodes: {stable_alarms} across {len(stable_runs)} runs",
        "",
        "## Component ablation",
        "",
        "| Method | Seeds | Final clean | Final PGD-10 | Core time |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in method_order:
        clean = lookup[(method, "final_clean")]
        robust = lookup[(method, "final_pgd10")]
        timing = lookup[(method, "training_minutes")]
        lines.append(
            f"| {method} | {clean['seeds']} | "
            f"{100*float(clean['mean']):.2f} +/- {100*float(clean['sample_std']):.2f}% | "
            f"{100*float(robust['mean']):.2f} +/- {100*float(robust['sample_std']):.2f}% | "
            f"{float(timing['mean']):.2f} +/- {float(timing['sample_std']):.2f} min |"
        )
    rollback_by_seed = {
        int(row["seed"]): row
        for row in training_rows
        if row["method"] == "Adaptive + rollback"
    }
    no_rollback_by_seed = {
        int(row["seed"]): row
        for row in training_rows
        if row["method"] == "Adaptive, no rollback"
    }
    paired_seeds = sorted(rollback_by_seed.keys() & no_rollback_by_seed.keys())
    if paired_seeds:
        robust_deltas = [
            100
            * (
                float(rollback_by_seed[seed]["final_pgd10"])
                - float(no_rollback_by_seed[seed]["final_pgd10"])
            )
            for seed in paired_seeds
        ]
        time_deltas = [
            float(rollback_by_seed[seed]["training_minutes"])
            - float(no_rollback_by_seed[seed]["training_minutes"])
            for seed in paired_seeds
        ]
        rollback_wins = sum(delta > 0 for delta in robust_deltas)
        lines.extend(
            [
                "",
                "## Paired rollback effect",
                "",
                f"- Matched seeds: {len(paired_seeds)}",
                f"- Rollback minus no-rollback PGD-10: {statistics.mean(robust_deltas):+.2f} percentage points",
                f"- Seeds favoring rollback: {rollback_wins}/{len(paired_seeds)}",
                f"- Rollback minus no-rollback core time: {statistics.mean(time_deltas):+.2f} min",
            ]
        )
    lines.append("")
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
