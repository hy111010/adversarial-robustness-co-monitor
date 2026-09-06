from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_crossdomain_detector import load_run, summarize


ROOT = Path.cwd() if (Path.cwd() / "outputs_publication").exists() else Path(__file__).resolve().parents[1]
DEV_COLLAPSE = tuple(
    ROOT / "outputs_publication" / f"v3_cifar100_collapse_dev_seed{seed}" / "detector_traces.csv"
    for seed in (17, 23, 42)
)
DEV_STABLE = tuple(
    ROOT / "outputs_publication" / f"v3_cifar100_stable_dev_seed{seed}" / "detector_traces.csv"
    for seed in (17, 23, 42)
)
OUT = ROOT / "publication" / "cifar100_youden_audit.json"


def pooled_development_windows(
    collapse_runs: list[dict], stable_runs: list[dict]
) -> tuple[np.ndarray, np.ndarray]:
    labels: list[int] = []
    scores: list[float] = []
    collapse_ids = {id(run) for run in collapse_runs}
    for run in collapse_runs + stable_runs:
        frame = run["frame"]
        event = run["event"] if id(run) in collapse_ids else None
        stop = len(frame) if event is None else event
        for index in range(stop):
            score = float(frame.loc[index, "score"])
            if not np.isfinite(score):
                continue
            scores.append(score)
            labels.append(int(event is not None and event - index == 1))
    return np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)


def youden_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        raise RuntimeError("Youden calibration requires positive and negative windows")
    best_threshold = float("nan")
    best_j = -float("inf")
    for threshold in np.unique(scores):
        predicted = scores >= threshold
        tpr = float(np.sum(predicted & (labels == 1)) / positives)
        fpr = float(np.sum(predicted & (labels == 0)) / negatives)
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_threshold = float(threshold)
    return best_threshold, best_j


def main() -> None:
    collapse = [load_run(path, 0.20, 0.10) for path in DEV_COLLAPSE]
    stable = [load_run(path, 0.20, 0.10) for path in DEV_STABLE]
    labels, scores = pooled_development_windows(collapse, stable)
    threshold, best_j = youden_threshold(labels, scores)
    summary, _ = summarize(collapse, stable, threshold, near=3)
    payload = {
        "scope": "CIFAR-100 development traces only",
        "seeds": [17, 23, 42],
        "runs": 6,
        "positive_definition": "registered CO is the next 20-update window",
        "selection": "maximize Youden J over unique finite rolling-cosine-drop scores",
        "eligible_windows": int(len(labels)),
        "positive_windows": int(labels.sum()),
        "threshold": threshold,
        "youden_j": best_j,
        "development_event_summary": summary,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
