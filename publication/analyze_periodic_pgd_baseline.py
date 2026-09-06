"""Evaluate a fixed-cadence PGD-10 monitoring baseline on saved traces."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "publication" / "supplemental_v2" / "expanded_heldout_detector.csv"
OUTPUT = ROOT / "publication" / "supplemental_v2" / "periodic_pgd_monitor.csv"


def main() -> None:
    event_rows = pd.read_csv(EVENTS).set_index("seed")
    sources = {
        101: ("dev_collapse_full_zero_eps8_lr03_seed101", "batch_metrics.csv", "label_pgd10_acc"),
        202: ("dev_collapse_full_zero_eps8_lr03_seed202", "batch_metrics.csv", "label_pgd10_acc"),
        **{
            seed: (f"v2_collapse_observe_seed{seed}", "detector_traces.csv", "observer_pgd10_acc")
            for seed in (303, 404, 505, 606, 707)
        },
    }
    output = []
    for seed, (folder, filename, column) in sources.items():
        trace = pd.read_csv(ROOT / "outputs_publication" / folder / filename)
        # Fixed cadence: one PGD-10 validation batch after every five 20-update
        # trace windows. Warning-conditioned extra labels are deliberately ignored.
        scheduled = trace.iloc[4::5].copy()
        scheduled = scheduled[scheduled[column].map(math.isfinite)]
        peak = -1.0
        detection_step = None
        calls = 0
        event_step = float(event_rows.loc[seed, "collapse_step"])
        for _, row in scheduled.iterrows():
            calls += 1
            accuracy = float(row[column])
            if peak >= 0 and peak - accuracy >= 0.20 and accuracy <= 0.10:
                detection_step = float(row["global_step"])
                break
            peak = max(peak, accuracy)
        output.append(
            {
                "seed": seed,
                "event_step": int(event_step),
                "detected": int(detection_step is not None),
                "detection_step": "" if detection_step is None else int(detection_step),
                "delay_updates": "" if detection_step is None else int(detection_step - event_step),
                "pgd10_calls_until_detection_or_end": calls,
                "pgd10_calls_before_event": int((scheduled["global_step"] < event_step).sum()),
                "terminal_event": int(event_rows.loc[seed, "terminal_below_10pct"]),
            }
        )
    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


if __name__ == "__main__":
    main()
