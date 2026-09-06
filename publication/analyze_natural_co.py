from __future__ import annotations

import csv
import json
import math
import statistics
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "publication" / "natural_co_timing_v4"
HIGH_LR_SEEDS = (101, 202, 303, 404, 505)
EPS16_SEEDS = (101, 202, 303)
STANDARD_SOURCES = {
    17: "fast_fgsm_rs_preact_seed17",
    23: "v2_stable_fgsm_rs_observe_seed23",
    42: "v2_stable_fgsm_rs_observe_seed42",
}
THRESHOLD = 0.16632988750934596


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def value(row: dict[str, str], *keys: str) -> float:
    for key in keys:
        text = row.get(key, "")
        try:
            number = float(text)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return float("nan")


def collapse_from_labels(traces: list[dict[str, str]]) -> tuple[int, int] | None:
    peak = -1.0
    for row in traces:
        accuracy = value(row, "observer_pgd10_acc", "label_pgd10_acc")
        if not math.isfinite(accuracy):
            continue
        if peak >= 0 and peak - accuracy >= 0.20 and accuracy <= 0.10:
            return int(float(row["epoch"])), int(float(row["global_step"]))
        peak = max(peak, accuracy)
    return None


def collapse_from_epochs(
    metrics: list[dict[str, str]], traces: list[dict[str, str]]
) -> tuple[int, int] | None:
    peak = -1.0
    for metric in metrics:
        accuracy = value(metric, "val_pgd10_acc")
        if peak >= 0 and peak - accuracy >= 0.20 and accuracy <= 0.10:
            epoch = int(float(metric["epoch"]))
            steps = [
                int(float(row["global_step"]))
                for row in traces
                if int(float(row["epoch"])) == epoch
            ]
            if steps:
                return epoch, max(steps)
        peak = max(peak, accuracy)
    return None


def scores(traces: list[dict[str, str]]) -> list[float]:
    history: deque[float] = deque(maxlen=5)
    result = []
    for row in traces:
        cosine = value(row, "free_gradient_cosine")
        baseline = statistics.median(history) if len(history) >= 3 else float("nan")
        result.append(baseline - cosine if math.isfinite(baseline) else float("nan"))
        history.append(cosine)
    return result


def analyze(recipe: str, seed: int) -> dict[str, object]:
    run_name = (
        STANDARD_SOURCES[seed]
        if recipe == "standard"
        else f"v4_natural_{recipe}_seed{seed}"
    )
    folder = ROOT / "outputs_publication" / run_name
    metrics = read_csv(folder / "metrics.csv")
    trace_path = folder / "detector_traces.csv"
    if not trace_path.exists():
        trace_path = folder / "batch_metrics.csv"
    traces = read_csv(trace_path)
    event = collapse_from_labels(traces) or collapse_from_epochs(metrics, traces)
    detector_scores = scores(traces)
    active = [math.isfinite(score) and score >= THRESHOLD for score in detector_scores]
    alarm_indices = [
        index
        for index, is_active in enumerate(active)
        if is_active and (index == 0 or not active[index - 1])
    ]
    alarm_steps = [int(float(traces[index]["global_step"])) for index in alarm_indices]
    event_step = event[1] if event else None
    prior = [step for step in alarm_steps if event_step is not None and step < event_step]
    return {
        "recipe": recipe,
        "seed": seed,
        "random_start": 1,
        "max_lr": 0.3 if recipe == "highlr" else 0.2,
        "epsilon_over_255": 16 if recipe == "eps16" else 8,
        "fgsm_step_size_over_255": 20 if recipe == "eps16" else 10,
        "collapse_event": int(event is not None),
        "collapse_epoch": event[0] if event else "",
        "collapse_step": event_step if event_step is not None else "",
        "detected_before_collapse": int(bool(prior)),
        "first_warning_step": prior[0] if prior else "",
        "warning_to_collapse_updates": event_step - prior[0] if prior else "",
        "alarm_episodes": len(alarm_steps),
        "final_clean_acc": value(metrics[-1], "val_clean_acc"),
        "final_pgd10_acc": value(metrics[-1], "val_pgd10_acc"),
        "folder": str(folder.relative_to(ROOT)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add_intervention_timing(observations: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for observation in observations:
        if not observation["collapse_event"] or not observation["detected_before_collapse"]:
            continue
        recipe = str(observation["recipe"])
        seed = int(observation["seed"])
        collapse_step = int(observation["collapse_step"])
        for timing in ("immediate", "deferred"):
            folder = ROOT / "outputs_publication" / f"v4_natural_{recipe}_{timing}_seed{seed}"
            metrics_path = folder / "metrics.csv"
            if not metrics_path.exists():
                continue
            rows = read_csv(metrics_path)
            trigger_rows = [row for row in rows if row.get("intervention_triggered") == "1"]
            if len(trigger_rows) != 1:
                continue
            trigger = trigger_rows[0]
            warning_step = int(float(trigger["trigger_global_step"]))
            if timing == "immediate":
                start_step = warning_step
            else:
                start_step = int(float(trigger["rolled_back_to_global_step"])) + int(
                    float(trigger["batches_processed"])
                )
            output.append(
                {
                    "recipe": recipe,
                    "seed": seed,
                    "timing": timing,
                    "warning_step": warning_step,
                    "unprotected_collapse_step": collapse_step,
                    "warning_to_collapse_updates": collapse_step - warning_step,
                    "pgd2_start_step": start_step,
                    "warning_to_pgd2_start_updates": start_step - warning_step,
                    "intervention_before_collapse": int(start_step < collapse_step),
                    "final_clean_acc": value(rows[-1], "val_clean_acc"),
                    "final_pgd10_acc": value(rows[-1], "val_pgd10_acc"),
                    "folder": str(folder.relative_to(ROOT)),
                }
            )
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    observations = [analyze("standard", seed) for seed in STANDARD_SOURCES]
    observations.extend(analyze("highlr", seed) for seed in HIGH_LR_SEEDS)
    observations.extend(
        analyze("eps16", seed)
        for seed in EPS16_SEEDS
        if (ROOT / "outputs_publication" / f"v4_natural_eps16_seed{seed}" / "metrics.csv").exists()
    )
    write_csv(OUT / "natural_co_observations.csv", observations)
    timing = add_intervention_timing(observations)
    if timing:
        write_csv(OUT / "natural_co_intervention_timing.csv", timing)
    summary = {
        recipe: {
            "runs": len([row for row in observations if row["recipe"] == recipe]),
            "collapse_events": sum(
                int(row["collapse_event"]) for row in observations if row["recipe"] == recipe
            ),
            "detected_events": sum(
                int(row["detected_before_collapse"])
                for row in observations
                if row["recipe"] == recipe
            ),
            "alarm_episodes": sum(
                int(row["alarm_episodes"]) for row in observations if row["recipe"] == recipe
            ),
        }
        for recipe in dict.fromkeys(str(row["recipe"]) for row in observations)
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
