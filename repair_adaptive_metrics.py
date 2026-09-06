from __future__ import annotations

import argparse
import csv
from pathlib import Path


SHORT_FIELDS = (
    "epoch",
    "attack_mode",
    "train_adv_loss",
    "train_adv_acc",
    "epoch_seconds",
    "lr_start",
    "lr_end",
    "val_clean_acc",
    "val_pgd10_acc",
    "intervention_triggered",
    "trigger_epoch",
    "trigger_global_step",
    "rolled_back_to_global_step",
    "abandoned_fgsm_batches",
    "abandoned_fgsm_seconds",
    "total_training_seconds",
    "clean_eval_seconds",
    "pgd_monitor_seconds",
    "pgd_monitor_used_for_trigger",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair legacy adaptive-intervention CSV rows")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        raw_rows = list(reader)
    repaired = []
    for line_number, values in enumerate(raw_rows, start=2):
        if len(values) == len(header):
            repaired.append(dict(zip(header, values)))
        elif len(values) == len(SHORT_FIELDS):
            short = dict(zip(SHORT_FIELDS, values))
            repaired.append({field: short.get(field, "") for field in header})
        else:
            raise ValueError(
                f"Unexpected field count at line {line_number}: {len(values)}; "
                f"expected {len(header)} or {len(SHORT_FIELDS)}"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(repaired)
    print(f"rows={len(repaired)} output={args.output}")


if __name__ == "__main__":
    main()
