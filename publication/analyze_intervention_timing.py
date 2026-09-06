from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "publication" / "intervention_timing_v3"
SEEDS = (17, 23, 42, 101, 202)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def first_collapse_step(seed: int) -> tuple[int, int]:
    folder = ROOT / "outputs_publication" / f"dev_collapse_full_zero_eps8_lr03_seed{seed}"
    traces = read_csv(folder / "batch_metrics.csv")
    peak = -1.0
    for row in traces:
        value_text = row.get("label_pgd10_acc", "") or row.get("observer_pgd10_acc", "")
        try:
            value = float(value_text)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        if peak >= 0 and peak - value >= 0.20 and value <= 0.10:
            return int(float(row["epoch"])), int(float(row["global_step"]))
        peak = max(peak, value)
    raise RuntimeError(f"No collapse event found for seed {seed}")


def metrics_path(folder: Path) -> Path:
    repaired = folder / "metrics_repaired.csv"
    return repaired if repaired.exists() else folder / "metrics.csv"


def intervention_row(folder: Path) -> tuple[list[dict[str, str]], dict[str, str]]:
    rows = read_csv(metrics_path(folder))
    triggered = [row for row in rows if row.get("intervention_triggered") == "1"]
    if len(triggered) != 1:
        raise RuntimeError(f"Expected one trigger row in {folder}, found {len(triggered)}")
    return rows, triggered[0]


def numeric(row: dict[str, str], key: str) -> float:
    return float(row[key])


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    event_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        collapse_epoch, collapse_step = first_collapse_step(seed)
        suffix = "dev" if seed in (17, 23, 42) else "holdout"
        methods = (
            (
                "immediate_epoch_replay",
                ROOT / "outputs_publication" / f"adaptive_rollback_pgd2_{suffix}_seed{seed}",
            ),
            (
                "deferred_next_epoch",
                ROOT / "outputs_publication" / f"v2_adaptive_no_rollback_seed{seed}",
            ),
        )
        for method, folder in methods:
            rows, trigger = intervention_row(folder)
            warning_step = int(float(trigger["trigger_global_step"]))
            if method == "immediate_epoch_replay":
                pgd2_start_step = warning_step
                post_warning_fgsm_updates = 0
            else:
                epoch_start = int(float(trigger["rolled_back_to_global_step"]))
                batches = int(float(trigger["batches_processed"]))
                pgd2_start_step = epoch_start + batches
                post_warning_fgsm_updates = pgd2_start_step - warning_step
            event_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "warning_epoch": int(float(trigger["epoch"])),
                    "warning_step": warning_step,
                    "collapse_epoch_unprotected": collapse_epoch,
                    "collapse_step_unprotected": collapse_step,
                    "warning_to_collapse_updates": collapse_step - warning_step,
                    "pgd2_start_step": pgd2_start_step,
                    "warning_to_pgd2_start_updates": pgd2_start_step - warning_step,
                    "post_warning_fgsm_updates": post_warning_fgsm_updates,
                    "collapse_before_pgd2_start": int(collapse_step <= pgd2_start_step),
                    "intervention_before_collapse": int(pgd2_start_step < collapse_step),
                    "final_clean_acc": numeric(rows[-1], "val_clean_acc"),
                    "final_pgd10_acc": numeric(rows[-1], "val_pgd10_acc"),
                    "source": str(folder.relative_to(ROOT)),
                }
            )
    write_csv(OUT / "event_timing.csv", event_rows)

    aggregate_rows: list[dict[str, object]] = []
    for method in ("immediate_epoch_replay", "deferred_next_epoch"):
        subset = [row for row in event_rows if row["method"] == method]
        aggregate_rows.append(
            {
                "method": method,
                "events": len(subset),
                "intervention_before_collapse_count": sum(
                    int(row["intervention_before_collapse"]) for row in subset
                ),
                "intervention_before_collapse_rate": statistics.mean(
                    int(row["intervention_before_collapse"]) for row in subset
                ),
                "median_warning_to_collapse_updates": statistics.median(
                    int(row["warning_to_collapse_updates"]) for row in subset
                ),
                "median_warning_to_pgd2_start_updates": statistics.median(
                    int(row["warning_to_pgd2_start_updates"]) for row in subset
                ),
                "mean_final_clean_acc": statistics.mean(
                    float(row["final_clean_acc"]) for row in subset
                ),
                "mean_final_pgd10_acc": statistics.mean(
                    float(row["final_pgd10_acc"]) for row in subset
                ),
            }
        )
    write_csv(OUT / "aggregate.csv", aggregate_rows)

    immediate = aggregate_rows[0]
    deferred = aggregate_rows[1]
    report = f"""# Intervention timing audit

The collapse time is measured on the seed-matched unprotected trajectory using the
registered PGD-10 drop rule. The warning time is the frozen detector crossing.
`immediate_epoch_replay` stops FGSM at the warning and starts PGD-2 immediately
after restoring the epoch-start state. `deferred_next_epoch` retains the current
epoch and starts PGD-2 at the next epoch boundary.

| Strategy | Events | PGD-2 before collapse | Median warning-to-collapse | Median warning-to-PGD-2 |
|---|---:|---:|---:|---:|
| Immediate epoch replay | {immediate['events']} | {immediate['intervention_before_collapse_count']}/{immediate['events']} | {immediate['median_warning_to_collapse_updates']:.0f} updates | {immediate['median_warning_to_pgd2_start_updates']:.0f} updates |
| Deferred next epoch | {deferred['events']} | {deferred['intervention_before_collapse_count']}/{deferred['events']} | {deferred['median_warning_to_collapse_updates']:.0f} updates | {deferred['median_warning_to_pgd2_start_updates']:.0f} updates |

This audit distinguishes statistical warning lead from actionable lead. Final clean
and PGD-10 accuracy remain in `event_timing.csv` for the matched outcome comparison.
"""
    (OUT / "README.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
