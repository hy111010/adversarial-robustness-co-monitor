from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "publication" / "IEEE_Adversarial_Robustness_Paper" / "figures"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    detector = read_csv(ROOT / "publication" / "final_statistics_v4" / "detector_generalization.csv")
    timing = read_csv(ROOT / "publication" / "intervention_timing_v3" / "event_timing.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.9, 3.1), constrained_layout=True)

    labels = ["C10\nPreAct", "C10\nResNet", "C100\nPreAct", "C10 $\\epsilon$16\nPreAct"]
    recall = np.array([float(row["event_recall"]) for row in detector])
    low = np.array([float(row["wilson_95_low"]) for row in detector])
    high = np.array([float(row["wilson_95_high"]) for row in detector])
    x = np.arange(len(labels))
    axes[0].bar(x, recall * 100, color="#3976AF", width=0.68)
    axes[0].errorbar(x, recall * 100, yerr=np.vstack(((recall-low)*100, (high-recall)*100)),
                     fmt="none", ecolor="#222222", capsize=3, linewidth=1)
    for i, row in enumerate(detector):
        axes[0].text(i, max(4, recall[i]*100-11),
                     f"{row['detected_before_event']}/{row['observed_events']}",
                     ha="center", color="white", fontsize=8, fontweight="bold")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 112)
    axes[0].set_ylabel("Event recall (%)")
    axes[0].set_title("(a) Transfer with 95% Wilson intervals", fontsize=9)
    axes[0].grid(axis="y", alpha=0.25)

    seeds = [17, 23, 42, 101, 202]
    immediate = {int(row["seed"]): row for row in timing if row["method"] == "immediate_epoch_replay"}
    deferred = {int(row["seed"]): row for row in timing if row["method"] == "deferred_next_epoch"}
    collapse_lead = np.array([int(immediate[seed]["warning_to_collapse_updates"]) for seed in seeds])
    deferred_start = np.array([int(deferred[seed]["warning_to_pgd2_start_updates"]) for seed in seeds])
    y = np.arange(len(seeds))
    axes[1].scatter(collapse_lead, y, marker="x", s=48, color="#C44E52", label="Unprotected CO")
    axes[1].scatter(np.zeros(len(seeds)), y, marker="o", s=34, color="#3976AF", label="Immediate-switch")
    axes[1].scatter(deferred_start, y, marker="s", s=34, color="#E39D35", label="Next epoch")
    for i in range(len(seeds)):
        axes[1].plot([0, max(collapse_lead[i], deferred_start[i])], [i, i], color="#AAAAAA", lw=0.7, zorder=0)
    axes[1].set_yticks(y, [str(seed) for seed in seeds])
    axes[1].set_xlabel("Updates after warning")
    axes[1].set_ylabel("Seed")
    axes[1].set_title("(b) Actionable intervention timing", fontsize=9)
    axes[1].legend(fontsize=7, loc="lower right")
    axes[1].grid(axis="x", alpha=0.25)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "transfer_and_timing.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
