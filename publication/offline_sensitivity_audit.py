from __future__ import annotations

import csv
import math
import statistics
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "publication" / "offline_sensitivity"
FROZEN_THRESHOLD = 0.16632988750934596
THRESHOLDS = (0.10, 0.12, 0.14, 0.16, FROZEN_THRESHOLD, 0.18, 0.20, 0.22, 0.24)
HORIZONS = (20, 40, 60, 80, 100)

HELDOUT = {
    101: ("dev_collapse_full_zero_eps8_lr03_seed101", "batch_metrics.csv"),
    202: ("dev_collapse_full_zero_eps8_lr03_seed202", "batch_metrics.csv"),
    303: ("v2_collapse_observe_seed303", "detector_traces.csv"),
    404: ("v2_collapse_observe_seed404", "detector_traces.csv"),
    505: ("v2_collapse_observe_seed505", "detector_traces.csv"),
    606: ("v2_collapse_observe_seed606", "detector_traces.csv"),
    707: ("v2_collapse_observe_seed707", "detector_traces.csv"),
}

NON_EVENT_CONTROLS = {
    17: ("fast_fgsm_rs_preact_seed17", "batch_metrics.csv"),
    23: ("v2_stable_fgsm_rs_observe_seed23", "detector_traces.csv"),
    42: ("v2_stable_fgsm_rs_observe_seed42", "detector_traces.csv"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def finite(row: dict[str, str], key: str) -> float:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def load_run(seed: int, folder_name: str, trace_name: str) -> dict[str, object]:
    folder = ROOT / "outputs_publication" / folder_name
    rows = read_csv(folder / trace_name)
    history: deque[float] = deque(maxlen=5)
    trace: list[dict[str, float]] = []
    for row in rows:
        cosine = finite(row, "free_gradient_cosine")
        baseline = statistics.median(history) if len(history) >= 3 else float("nan")
        score = baseline - cosine if math.isfinite(baseline) else float("nan")
        label = finite(row, "label_pgd10_acc")
        if not math.isfinite(label):
            label = finite(row, "observer_pgd10_acc")
        trace.append({
            "epoch": finite(row, "epoch"),
            "step": finite(row, "global_step"),
            "score": score,
            "label": label,
        })
        history.append(cosine)
    return {"seed": seed, "folder": folder, "trace": trace}


def event_step(run: dict[str, object], drop: float, ceiling: float) -> float | None:
    trace = run["trace"]
    peak = -1.0
    for row in trace:
        value = row["label"]
        if not math.isfinite(value):
            continue
        if peak >= 0 and peak - value >= drop and value <= ceiling:
            return row["step"]
        peak = max(peak, value)
    metrics = read_csv(run["folder"] / "metrics.csv")
    peak = -1.0
    event_epoch: int | None = None
    for row in metrics:
        value = finite(row, "val_pgd10_acc")
        if peak >= 0 and peak - value >= drop and value <= ceiling:
            event_epoch = int(float(row["epoch"]))
            break
        peak = max(peak, value)
    if event_epoch is None:
        return None
    steps = [row["step"] for row in trace if int(row["epoch"]) == event_epoch]
    return max(steps) if steps else None


def active_warning_steps(run: dict[str, object], threshold: float) -> list[float]:
    return [
        row["step"]
        for row in run["trace"]
        if math.isfinite(row["score"]) and row["score"] >= threshold
    ]


def alarm_starts(run: dict[str, object], threshold: float) -> list[float]:
    starts: list[float] = []
    active_before = False
    for row in run["trace"]:
        active = math.isfinite(row["score"]) and row["score"] >= threshold
        if active and not active_before:
            starts.append(row["step"])
        active_before = active
    return starts


def summarize(runs: list[dict[str, object]], threshold: float, drop: float, ceiling: float, horizon: int) -> dict[str, object]:
    events = detected = near = 0
    leads: list[float] = []
    non_events = false_episodes = non_events_with_alarm = 0
    for run in runs:
        event = event_step(run, drop, ceiling)
        active_steps = active_warning_steps(run, threshold)
        alarms = alarm_starts(run, threshold)
        if event is None:
            non_events += 1
            false_episodes += len(alarms)
            non_events_with_alarm += int(bool(alarms))
            continue
        events += 1
        prior = [step for step in active_steps if step < event]
        within = [step for step in prior if event - step <= horizon]
        detected += int(bool(prior))
        near += int(bool(within))
        if prior:
            leads.append(event - prior[0])
    return {
        "events": events,
        "detected_events": detected,
        "recall": detected / events if events else float("nan"),
        "near_events": near,
        "near_recall": near / events if events else float("nan"),
        "non_event_runs": non_events,
        "non_event_runs_with_alarm": non_events_with_alarm,
        "false_alarm_episodes": false_episodes,
        "median_first_lead_updates": statistics.median(leads) if leads else float("nan"),
    }


def write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def persistence_row(run: dict[str, object], seed: int) -> dict[str, object]:
    event = event_step(run, 0.20, 0.10)
    active_steps = active_warning_steps(run, FROZEN_THRESHOLD)
    first = min(step for step in active_steps if event is not None and step < event)
    segment = [row for row in run["trace"] if first <= row["step"] < event]
    active = [bool(math.isfinite(row["score"]) and row["score"] >= FROZEN_THRESHOLD) for row in segment]
    longest = current = 0
    for value in active:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return {
        "seed": seed,
        "first_warning_step": first,
        "registered_co_step": event,
        "warning_to_co_updates": event - first,
        "post_warning_windows": len(segment),
        "windows_above_threshold": sum(active),
        "fraction_above_threshold": sum(active) / len(active),
        "longest_consecutive_above_threshold_windows": longest,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    heldout = [load_run(seed, *spec) for seed, spec in HELDOUT.items()]
    controls = [load_run(seed, *spec) for seed, spec in NON_EVENT_CONTROLS.items()]
    all_primary = heldout + controls

    threshold_rows = []
    for threshold in THRESHOLDS:
        row = {"threshold": threshold}
        row.update(summarize(all_primary, threshold, 0.20, 0.10, 60))
        threshold_rows.append(row)
    write(OUT / "threshold_sensitivity.csv", threshold_rows)

    horizon_rows = []
    for horizon in HORIZONS:
        row = {"horizon_updates": horizon}
        row.update(summarize(all_primary, FROZEN_THRESHOLD, 0.20, 0.10, horizon))
        horizon_rows.append(row)
    write(OUT / "warning_horizon_sensitivity.csv", horizon_rows)

    definitions = (
        ("relaxed", 0.15, 0.15),
        ("original", 0.20, 0.10),
        ("same_drop_higher_ceiling", 0.20, 0.15),
        ("strict", 0.25, 0.05),
    )
    definition_rows = []
    for name, drop, ceiling in definitions:
        row = {"definition": name, "peak_drop": drop, "accuracy_ceiling": ceiling}
        row.update(summarize(all_primary, FROZEN_THRESHOLD, drop, ceiling, 60))
        definition_rows.append(row)
    write(OUT / "co_definition_sensitivity.csv", definition_rows)

    long_lead = []
    for seed in (23, 202):
        folder = f"dev_collapse_full_zero_eps8_lr03_seed{seed}"
        run = load_run(seed, folder, "batch_metrics.csv")
        long_lead.append(persistence_row(run, seed))
    write(OUT / "long_lead_signal_persistence.csv", long_lead)


if __name__ == "__main__":
    main()
