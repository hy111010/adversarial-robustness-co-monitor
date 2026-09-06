from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_collapse_forecast import binary_metrics, first_collapse_index


CANDIDATES = {
    "raw_drop": 1,
    "mad_normalized_drop": 1,
    "online_percentile_persistent": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit/evaluate scale-normalized online CO detectors")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("fit", "eval"), required=True)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--collapse-drop", type=float, default=0.20)
    parser.add_argument("--collapse-ceiling", type=float, default=0.10)
    parser.add_argument("--horizon", type=int, default=3)
    return parser.parse_args()


def add_scores(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values("global_step").reset_index(drop=True).copy()
    cosine = frame["free_gradient_cosine"].astype(float)
    baseline = cosine.shift(1).rolling(5, min_periods=3).median()
    drop = baseline - cosine
    history_median = cosine.shift(1).rolling(20, min_periods=8).median()
    history_mad = (
        (cosine - history_median).abs().shift(1).rolling(20, min_periods=8).median()
    )
    frame["raw_drop"] = drop
    frame["mad_normalized_drop"] = drop / (1.4826 * history_mad + 1e-3)

    percentile = np.full(len(frame), np.nan, dtype=float)
    values = drop.to_numpy(float)
    for index, value in enumerate(values):
        prior = values[max(0, index - 40) : index]
        prior = prior[np.isfinite(prior)]
        if np.isfinite(value) and len(prior) >= 12:
            percentile[index] = (np.sum(prior < value) + 0.5 * np.sum(prior == value)) / len(prior)
    frame["online_percentile_persistent"] = percentile
    return frame


def load_runs(paths: list[Path], collapse_drop: float, collapse_ceiling: float) -> list[dict]:
    runs = []
    for path in paths:
        frame = add_scores(pd.read_csv(path))
        event_index = None
        label_column = next(
            (name for name in ("label_pgd10_acc", "observer_pgd10_acc") if name in frame),
            None,
        )
        if label_column is not None:
            peak = -np.inf
            for index, value in enumerate(frame[label_column].to_numpy(float)):
                if not np.isfinite(value):
                    continue
                if peak > -np.inf and value <= collapse_ceiling and peak - value >= collapse_drop:
                    event_index = index
                    break
                peak = max(peak, value)
        if event_index is None:
            metrics_path = path.parent / "metrics.csv"
            if metrics_path.exists():
                metrics = pd.read_csv(metrics_path).sort_values("epoch")
                metric_event = first_collapse_index(
                    metrics["val_pgd10_acc"].to_numpy(float), collapse_drop, collapse_ceiling
                )
                if metric_event is not None and "epoch" in frame:
                    event_epoch = int(metrics.iloc[metric_event]["epoch"])
                    epoch_indices = np.flatnonzero(frame["epoch"].to_numpy(int) == event_epoch)
                    if len(epoch_indices):
                        event_index = int(epoch_indices[-1])
        runs.append({"run": path.parent.name, "path": str(path), "frame": frame, "event_index": event_index})
    return runs


def alarm_crossings(scores: np.ndarray, threshold: float, persistence: int) -> np.ndarray:
    above = np.isfinite(scores) & (scores >= threshold)
    active = np.zeros(len(above), dtype=bool)
    if persistence == 1:
        active = above
    else:
        for index in range(persistence - 1, len(above)):
            active[index] = bool(above[index - persistence + 1 : index + 1].all())
    return np.flatnonzero(active & np.r_[True, ~active[:-1]])


def event_metrics(runs: list[dict], feature: str, threshold: float, horizon: int) -> dict:
    persistence = CANDIDATES[feature]
    events = detected = false_alarms = 0
    leads: list[int] = []
    details = []
    for run in runs:
        crossings = alarm_crossings(run["frame"][feature].to_numpy(float), threshold, persistence)
        event = run["event_index"]
        valid = np.array([], dtype=int)
        if event is None:
            false = len(crossings)
        else:
            events += 1
            valid = crossings[(crossings < event) & (crossings >= event - horizon)]
            false = int((crossings < event - horizon).sum())
            if len(valid):
                detected += 1
                leads.append(int(event - valid[0]))
        false_alarms += false
        details.append(
            {
                "run": run["run"],
                "event_index": event,
                "detected": int(len(valid) > 0),
                "lead_windows": int(event - valid[0]) if len(valid) else np.nan,
                "false_alarm_episodes": false,
                "alarm_episodes": len(crossings),
                "alarm_indices": ";".join(str(int(value)) for value in crossings),
            }
        )
    return {
        "events": events,
        "detected_events": detected,
        "event_recall": detected / events if events else np.nan,
        "false_alarms": false_alarms,
        "false_alarms_per_run": false_alarms / len(runs),
        "median_lead_windows": float(np.median(leads)) if leads else np.nan,
        "details": details,
    }


def pooled_rank_metrics(runs: list[dict], feature: str, horizon: int) -> tuple[float, float]:
    labels, scores = [], []
    for run in runs:
        event = run["event_index"]
        stop = len(run["frame"]) if event is None else event
        for index in range(stop):
            value = float(run["frame"].loc[index, feature])
            if not np.isfinite(value):
                continue
            labels.append(int(event is not None and 1 <= event - index <= horizon))
            scores.append(value)
    if not labels or sum(labels) == 0:
        return np.nan, np.nan
    return binary_metrics(np.asarray(labels, dtype=int), np.asarray(scores, dtype=float))


def fit_candidate(runs: list[dict], feature: str, horizon: int) -> dict:
    values = np.concatenate([run["frame"][feature].to_numpy(float) for run in runs])
    candidates = np.unique(values[np.isfinite(values)])
    best = None
    for threshold in candidates:
        metrics = event_metrics(runs, feature, float(threshold), horizon)
        lead = metrics["median_lead_windows"]
        rank = (
            metrics["detected_events"],
            -metrics["false_alarms"],
            -999.0 if not np.isfinite(lead) else lead,
            float(threshold),
        )
        if best is None or rank > best[0]:
            best = (rank, float(threshold), metrics)
    assert best is not None
    auroc, auprc = pooled_rank_metrics(runs, feature, horizon)
    return {"feature": feature, "threshold": best[1], "auroc": auroc, "auprc": auprc, **{k: v for k, v in best[2].items() if k != "details"}}


def main() -> None:
    args = parse_args()
    if args.mode == "eval" and args.frozen is None:
        raise ValueError("--mode eval requires --frozen")
    runs = load_runs(args.inputs, args.collapse_drop, args.collapse_ceiling)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "fit":
        fits = [fit_candidate(runs, feature, args.horizon) for feature in CANDIDATES]
        selected = max(
            fits,
            key=lambda row: (
                row["detected_events"],
                -row["false_alarms"],
                row["median_lead_windows"] if np.isfinite(row["median_lead_windows"]) else -999,
                row["auprc"],
            ),
        )
        payload = {
            "status": "frozen_after_development",
            "selection_rule": "event recall, false alarms, median lead, AUPRC",
            "horizon_windows": args.horizon,
            "selected_feature": selected["feature"],
            "threshold": selected["threshold"],
            "persistence": CANDIDATES[selected["feature"]],
            "candidate_results": fits,
        }
        (args.output_dir / "frozen_detector.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        payload = json.loads(args.frozen.read_text(encoding="utf-8"))
        feature = payload["selected_feature"]
        threshold = float(payload["threshold"])
        metrics = event_metrics(runs, feature, threshold, int(payload["horizon_windows"]))
        auroc, auprc = pooled_rank_metrics(runs, feature, int(payload["horizon_windows"]))
        summary = {"feature": feature, "threshold": threshold, "auroc": auroc, "auprc": auprc, **{k: v for k, v in metrics.items() if k != "details"}}
        pd.DataFrame([summary]).to_csv(args.output_dir / "summary.csv", index=False)
        pd.DataFrame(metrics["details"]).to_csv(args.output_dir / "per_run.csv", index=False)

    print(f"mode={args.mode} runs={len(runs)} output={args.output_dir}")


if __name__ == "__main__":
    main()
