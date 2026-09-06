from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


RISK_DIRECTIONS = {
    "free_gradient_cosine": -1.0,
    "free_gradient_sign_agreement": -1.0,
    "free_prediction_disagreement": 1.0,
    "free_attack_loss_gain": -1.0,
    "free_boundary_fraction": 1.0,
    "control_gradient_participation_ratio": -1.0,
    "control_gradient_entropy": -1.0,
    "train_adv_loss": -1.0,
    "train_adv_acc": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-specified collapse-forecast analysis")
    parser.add_argument("inputs", nargs="+", type=Path, help="One batch_metrics.csv per run")
    parser.add_argument("--output-dir", type=Path, default=Path("publication/forecast_analysis"))
    parser.add_argument("--mode", choices=["describe", "fit", "eval"], default="describe")
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--collapse-drop", type=float, default=0.20)
    parser.add_argument("--collapse-ceiling", type=float, default=0.10)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3])
    return parser.parse_args()


def first_collapse_index(
    robust_accuracy: np.ndarray, minimum_drop: float, ceiling: float
) -> int | None:
    prior_peak = -np.inf
    for index, accuracy in enumerate(robust_accuracy):
        if index > 0 and accuracy <= ceiling and prior_peak - accuracy >= minimum_drop:
            return index
        prior_peak = max(prior_peak, float(accuracy))
    return None


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return float("nan"), float("nan")
    ascending_ranks = rankdata(scores, method="average")
    rank_sum = float(ascending_ranks[labels == 1].sum())
    auroc = (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    group_ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(scores) - 1]
    cumulative_positives = np.cumsum(sorted_labels)
    precisions = cumulative_positives[group_ends] / (group_ends + 1)
    group_starts = np.r_[0, group_ends[:-1] + 1]
    positives_per_group = np.add.reduceat(sorted_labels, group_starts)
    auprc = float((precisions * positives_per_group).sum() / positives)
    return auroc, auprc


def fit_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("Threshold fitting requires both positive and negative windows")
    candidates = np.unique(scores)
    best_threshold = float(candidates[0])
    best_youden = -np.inf
    for threshold in candidates:
        predictions = scores >= threshold
        tpr = float((predictions & (labels == 1)).sum() / positives)
        fpr = float((predictions & (labels == 0)).sum() / negatives)
        youden = tpr - fpr
        if youden > best_youden:
            best_youden = youden
            best_threshold = float(threshold)
    return best_threshold


def threshold_event_metrics(
    runs: list[dict], feature: str, threshold: float, horizon: int
) -> dict[str, float | int]:
    event_count = 0
    detected_events = 0
    false_alarms = 0
    lead_times: list[int] = []
    for run in runs:
        scores = run["frame"][feature].to_numpy(float) * RISK_DIRECTIONS[feature]
        crossings = np.flatnonzero((scores >= threshold) & np.r_[True, scores[:-1] < threshold])
        event_index = run["event_index"]
        if event_index is None:
            false_alarms += len(crossings)
            continue
        event_count += 1
        valid = crossings[(crossings < event_index) & (crossings >= event_index - horizon)]
        false_alarms += int((crossings < event_index - horizon).sum())
        if len(valid):
            detected_events += 1
            lead_times.append(int(event_index - valid[0]))
    return {
        "events": event_count,
        "detected_events": detected_events,
        "event_recall": detected_events / event_count if event_count else float("nan"),
        "false_alarms": false_alarms,
        "median_lead_windows": float(np.median(lead_times)) if lead_times else float("nan"),
    }


def load_runs(args: argparse.Namespace) -> list[dict]:
    runs = []
    for path in args.inputs:
        frame = pd.read_csv(path)
        required = {"global_step", "label_pgd10_acc"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frame = frame.sort_values("global_step").reset_index(drop=True)
        event_index = first_collapse_index(
            frame["label_pgd10_acc"].to_numpy(float),
            args.collapse_drop,
            args.collapse_ceiling,
        )
        runs.append({"run": path.parent.name, "path": str(path), "frame": frame, "event_index": event_index})
    return runs


def pooled_windows(runs: list[dict], feature: str, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    labels: list[int] = []
    scores: list[float] = []
    for run in runs:
        frame = run["frame"]
        event_index = run["event_index"]
        stop = len(frame) if event_index is None else event_index
        for index in range(stop):
            labels.append(
                int(event_index is not None and 1 <= event_index - index <= horizon)
            )
            scores.append(float(frame.loc[index, feature]) * RISK_DIRECTIONS[feature])
    return np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)


def main() -> None:
    args = parse_args()
    if args.mode == "eval" and args.thresholds is None:
        raise ValueError("--mode eval requires --thresholds from development runs")
    runs = load_runs(args)
    available_features = [
        feature
        for feature in RISK_DIRECTIONS
        if all(feature in run["frame"].columns for run in runs)
    ]
    if not available_features:
        raise ValueError("No pre-specified predictor columns were found")

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
    if args.mode == "eval":
        thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
    elif args.mode == "fit":
        fit_horizon = max(args.horizons)
        for feature in available_features:
            labels, scores = pooled_windows(runs, feature, fit_horizon)
            thresholds[feature] = fit_threshold(labels, scores)
        (args.output_dir / "thresholds.json").write_text(
            json.dumps(thresholds, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    metric_rows = []
    for feature in available_features:
        for horizon in args.horizons:
            labels, scores = pooled_windows(runs, feature, horizon)
            auroc, auprc = binary_metrics(labels, scores)
            row: dict[str, float | int | str] = {
                "feature": feature,
                "risk_direction": int(RISK_DIRECTIONS[feature]),
                "horizon_windows": horizon,
                "eligible_windows": len(labels),
                "positive_windows": int(labels.sum()),
                "prevalence": float(labels.mean()) if len(labels) else float("nan"),
                "auroc": auroc,
                "auprc": auprc,
            }
            if feature in thresholds:
                row["risk_threshold"] = thresholds[feature]
                row.update(threshold_event_metrics(runs, feature, thresholds[feature], horizon))
            metric_rows.append(row)
    pd.DataFrame(metric_rows).to_csv(args.output_dir / "forecast_metrics.csv", index=False)
    print(f"runs={len(runs)} features={len(available_features)} mode={args.mode}")
    print(f"output={args.output_dir}")


if __name__ == "__main__":
    main()
