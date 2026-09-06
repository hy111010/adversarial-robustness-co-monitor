from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_collapse_forecast import first_collapse_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen raw cosine-drop cross-domain evaluation")
    parser.add_argument("--collapse", nargs="*", type=Path, default=[])
    parser.add_argument("--stable", nargs="*", type=Path, default=[])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("fit", "eval"), required=True)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--collapse-drop", type=float, default=0.20)
    parser.add_argument("--collapse-ceiling", type=float, default=0.10)
    parser.add_argument("--near-horizon", type=int, default=3)
    return parser.parse_args()


def load_run(path: Path, drop: float, ceiling: float) -> dict:
    frame = pd.read_csv(path).sort_values("global_step").reset_index(drop=True)
    cosine = frame["free_gradient_cosine"].astype(float)
    baseline = cosine.shift(1).rolling(5, min_periods=3).median()
    frame["score"] = baseline - cosine
    event = None
    for label_name in ("label_pgd10_acc", "observer_pgd10_acc"):
        if label_name not in frame:
            continue
        peak = -np.inf
        for index, value in enumerate(frame[label_name].to_numpy(float)):
            if not np.isfinite(value):
                continue
            if peak > -np.inf and value <= ceiling and peak - value >= drop:
                event = index
                break
            peak = max(peak, value)
        if event is not None:
            break
    if event is None and (path.parent / "metrics.csv").exists():
        metrics = pd.read_csv(path.parent / "metrics.csv").sort_values("epoch")
        metric_event = first_collapse_index(metrics["val_pgd10_acc"].to_numpy(float), drop, ceiling)
        if metric_event is not None:
            event_epoch = int(metrics.iloc[metric_event]["epoch"])
            indices = np.flatnonzero(frame["epoch"].to_numpy(int) == event_epoch)
            if len(indices):
                event = int(indices[-1])
    return {"name": path.parent.name, "path": str(path), "frame": frame, "event": event}


def crossings(run: dict, threshold: float) -> np.ndarray:
    score = run["frame"]["score"].to_numpy(float)
    active = np.isfinite(score) & (score >= threshold)
    return np.flatnonzero(active & np.r_[True, ~active[:-1]])


def summarize(collapse_runs: list[dict], stable_runs: list[dict], threshold: float, near: int) -> tuple[dict, list[dict]]:
    detected = detected_near = 0
    leads = []
    rows = []
    for run in collapse_runs:
        alarms = crossings(run, threshold)
        event = run["event"]
        prior = alarms[alarms < event] if event is not None else np.array([], dtype=int)
        near_prior = prior[prior >= event - near] if event is not None else np.array([], dtype=int)
        if len(prior):
            detected += 1
            leads.append(int(event - prior[0]))
        if len(near_prior):
            detected_near += 1
        rows.append({
            "run": run["name"], "kind": "collapse", "event_index": event,
            "detected_before_event": int(len(prior) > 0),
            "detected_within_near_horizon": int(len(near_prior) > 0),
            "first_lead_windows": int(event - prior[0]) if len(prior) else np.nan,
            "alarm_episodes": len(alarms),
        })
    stable_false = 0
    for run in stable_runs:
        alarms = crossings(run, threshold)
        stable_false += len(alarms)
        rows.append({
            "run": run["name"], "kind": "stable", "event_index": np.nan,
            "detected_before_event": 0, "detected_within_near_horizon": 0,
            "first_lead_windows": np.nan, "alarm_episodes": len(alarms),
        })
    events = sum(run["event"] is not None for run in collapse_runs)
    summary = {
        "threshold": threshold,
        "planned_collapse_runs": len(collapse_runs),
        "observed_events": events,
        "detected_events": detected,
        "event_recall_any_lead": detected / events if events else np.nan,
        "detected_within_60_updates": detected_near,
        "recall_within_60_updates": detected_near / events if events else np.nan,
        "median_first_lead_windows": float(np.median(leads)) if leads else np.nan,
        "stable_runs": len(stable_runs),
        "stable_false_alarm_episodes": stable_false,
        "stable_false_alarms_per_run": stable_false / len(stable_runs) if stable_runs else np.nan,
    }
    return summary, rows


def main() -> None:
    args = parse_args()
    collapse_runs = [load_run(path, args.collapse_drop, args.collapse_ceiling) for path in args.collapse]
    stable_runs = [load_run(path, args.collapse_drop, args.collapse_ceiling) for path in args.stable]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "eval":
        if args.frozen:
            threshold = float(json.loads(args.frozen.read_text(encoding="utf-8"))["threshold"])
        elif args.threshold is not None:
            threshold = args.threshold
        else:
            raise ValueError("eval requires --threshold or --frozen")
    else:
        values = np.concatenate([run["frame"]["score"].to_numpy(float) for run in collapse_runs + stable_runs])
        candidates = np.unique(values[np.isfinite(values)])
        best = None
        for candidate in candidates:
            summary, _ = summarize(collapse_runs, stable_runs, float(candidate), args.near_horizon)
            # Prefer event coverage, then silence on stable runs, then warnings closer to collapse.
            lead = summary["median_first_lead_windows"]
            key = (
                summary["detected_events"],
                -summary["stable_false_alarm_episodes"],
                -lead if np.isfinite(lead) else -1e9,
                float(candidate),
            )
            if best is None or key > best[0]:
                best = (key, float(candidate))
        if best is None:
            raise ValueError("No finite detector scores")
        threshold = best[1]
        (args.output_dir / "frozen_detector.json").write_text(
            json.dumps({
                "status": "frozen_after_domain_development",
                "feature": "rolling-5 median cosine drop",
                "threshold": threshold,
                "selection": "event coverage, stable false alarms, proximity, threshold",
            }, indent=2), encoding="utf-8",
        )
    summary, rows = summarize(collapse_runs, stable_runs, threshold, args.near_horizon)
    pd.DataFrame([summary]).to_csv(args.output_dir / "summary.csv", index=False)
    pd.DataFrame(rows).to_csv(args.output_dir / "per_run.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
