from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_collapse_forecast import binary_metrics, first_collapse_index, fit_threshold


FEATURES = (
    "adaptive_gradient_cosine_drop",
    "adaptive_gradient_sign_drop",
    "adaptive_train_accuracy_rise",
    "adaptive_train_loss_drop",
    "adaptive_cosine_accuracy_divergence",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run-relative collapse forecast analysis (development v2)"
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("describe", "fit", "eval"), default="describe")
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--rolling-window", type=int, default=5)
    parser.add_argument("--minimum-history", type=int, default=3)
    parser.add_argument("--fit-horizon", type=int, default=1)
    parser.add_argument("--horizons", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--collapse-drop", type=float, default=0.20)
    parser.add_argument("--collapse-ceiling", type=float, default=0.10)
    return parser.parse_args()


def add_adaptive_features(
    frame: pd.DataFrame, rolling_window: int, minimum_history: int
) -> pd.DataFrame:
    frame = frame.copy()

    def prior_median(column: str) -> pd.Series:
        return (
            frame[column]
            .shift(1)
            .rolling(rolling_window, min_periods=minimum_history)
            .median()
        )

    cosine_baseline = prior_median("free_gradient_cosine")
    sign_baseline = prior_median("free_gradient_sign_agreement")
    accuracy_baseline = prior_median("train_adv_acc")
    loss_baseline = prior_median("train_adv_loss")

    frame["adaptive_gradient_cosine_drop"] = (
        cosine_baseline - frame["free_gradient_cosine"]
    )
    frame["adaptive_gradient_sign_drop"] = (
        sign_baseline - frame["free_gradient_sign_agreement"]
    )
    frame["adaptive_train_accuracy_rise"] = (
        frame["train_adv_acc"] - accuracy_baseline
    )
    frame["adaptive_train_loss_drop"] = loss_baseline - frame["train_adv_loss"]
    frame["adaptive_cosine_accuracy_divergence"] = (
        frame["adaptive_gradient_cosine_drop"]
        + frame["adaptive_train_accuracy_rise"]
    )
    return frame


def load_runs(args: argparse.Namespace) -> list[dict]:
    runs: list[dict] = []
    required = {
        "global_step",
        "label_pgd10_acc",
        "free_gradient_cosine",
        "free_gradient_sign_agreement",
        "train_adv_acc",
        "train_adv_loss",
    }
    for path in args.inputs:
        frame = pd.read_csv(path).sort_values("global_step").reset_index(drop=True)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frame = add_adaptive_features(frame, args.rolling_window, args.minimum_history)
        event_index = first_collapse_index(
            frame["label_pgd10_acc"].to_numpy(float),
            args.collapse_drop,
            args.collapse_ceiling,
        )
        runs.append(
            {"run": path.parent.name, "path": str(path), "frame": frame, "event_index": event_index}
        )
    return runs


def pooled_windows(
    runs: list[dict], feature: str, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    labels: list[int] = []
    scores: list[float] = []
    for run in runs:
        frame = run["frame"]
        event_index = run["event_index"]
        stop = len(frame) if event_index is None else event_index
        for index in range(stop):
            value = float(frame.loc[index, feature])
            if not np.isfinite(value):
                continue
            labels.append(int(event_index is not None and 1 <= event_index - index <= horizon))
            scores.append(value)
    return np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)


def event_metrics(
    runs: list[dict], feature: str, threshold: float, horizon: int
) -> dict[str, float | int]:
    events = 0
    detections = 0
    false_alarms = 0
    lead_times: list[int] = []
    for run in runs:
        scores = run["frame"][feature].to_numpy(float)
        above = np.isfinite(scores) & (scores >= threshold)
        crossings = np.flatnonzero(above & np.r_[True, ~above[:-1]])
        event_index = run["event_index"]
        if event_index is None:
            false_alarms += len(crossings)
            continue
        events += 1
        valid = crossings[(crossings < event_index) & (crossings >= event_index - horizon)]
        false_alarms += int((crossings < event_index - horizon).sum())
        if len(valid):
            detections += 1
            lead_times.append(int(event_index - valid[0]))
    return {
        "events": events,
        "detected_events": detections,
        "event_recall": detections / events if events else float("nan"),
        "false_alarms": false_alarms,
        "false_alarms_per_run": false_alarms / len(runs),
        "median_lead_windows": float(np.median(lead_times)) if lead_times else float("nan"),
    }


def main() -> None:
    args = parse_args()
    if args.mode == "eval" and args.thresholds is None:
        raise ValueError("--mode eval requires --thresholds")
    runs = load_runs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    event_rows = []
    for run in runs:
        event_index = run["event_index"]
        event_rows.append(
            {
                "run": run["run"],
                "input": run["path"],
                "windows": len(run["frame"]),
                "collapse_detected": event_index is not None,
                "collapse_window_index": event_index,
                "collapse_global_step": (
                    int(run["frame"].loc[event_index, "global_step"])
                    if event_index is not None
                    else None
                ),
            }
        )
    pd.DataFrame(event_rows).to_csv(args.output_dir / "events.csv", index=False)

    thresholds: dict[str, float] = {}
    if args.mode == "fit":
        for feature in FEATURES:
            labels, scores = pooled_windows(runs, feature, args.fit_horizon)
            thresholds[feature] = fit_threshold(labels, scores)
        threshold_payload = {
            "status": "frozen_after_development",
            "rolling_window": args.rolling_window,
            "minimum_history": args.minimum_history,
            "fit_horizon": args.fit_horizon,
            "thresholds": thresholds,
        }
        (args.output_dir / "thresholds.json").write_text(
            json.dumps(threshold_payload, indent=2), encoding="utf-8"
        )
    elif args.mode == "eval":
        payload = json.loads(args.thresholds.read_text(encoding="utf-8"))
        if int(payload["rolling_window"]) != args.rolling_window:
            raise ValueError("Rolling-window setting does not match frozen thresholds")
        if int(payload["minimum_history"]) != args.minimum_history:
            raise ValueError("Minimum-history setting does not match frozen thresholds")
        thresholds = {key: float(value) for key, value in payload["thresholds"].items()}

    metric_rows = []
    for feature in FEATURES:
        for horizon in args.horizons:
            labels, scores = pooled_windows(runs, feature, horizon)
            auroc, auprc = binary_metrics(labels, scores)
            row: dict[str, float | int | str] = {
                "feature": feature,
                "horizon_windows": horizon,
                "eligible_windows": len(labels),
                "positive_windows": int(labels.sum()),
                "prevalence": float(labels.mean()),
                "auroc": auroc,
                "auprc": auprc,
            }
            if feature in thresholds:
                row["risk_threshold"] = thresholds[feature]
                row.update(event_metrics(runs, feature, thresholds[feature], horizon))
            metric_rows.append(row)
    pd.DataFrame(metric_rows).to_csv(args.output_dir / "forecast_metrics.csv", index=False)
    print(f"runs={len(runs)} features={len(FEATURES)} mode={args.mode}")
    print(f"output={args.output_dir}")


if __name__ == "__main__":
    main()
