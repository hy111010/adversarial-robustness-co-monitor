from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "publication" / "reduced_final_evaluation"
MODELS = {
    "Adaptive intervention": ROOT / "outputs_publication" / "adaptive_rollback_pgd2_holdout_seed101",
    "PGD-10 adversarial training": ROOT / "outputs_publication" / "pgd10_preact_seed17",
    "Fast FGSM-RS": ROOT / "outputs_publication" / "fast_fgsm_rs_preact_seed17",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    combined: list[dict[str, str]] = []
    for model, folder in MODELS.items():
        sources = (
            ("attack_results.csv", "full_test_10000"),
            ("strong_attack_results.csv", "full_test_10000"),
            ("official_autoattack_1000_results.csv", "fixed_test_subset_1000"),
        )
        for filename, scope in sources:
            path = folder / filename
            if not path.exists():
                raise FileNotFoundError(path)
            for row in read_rows(path):
                combined.append(
                    {
                        "model": model,
                        "attack": row["attack"],
                        "accuracy": row["accuracy"],
                        "evaluation_scope": scope,
                        "checkpoint": row["checkpoint"],
                    }
                )

    csv_path = OUT / "unified_attack_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=combined[0].keys())
        writer.writeheader()
        writer.writerows(combined)

    attacks = list(dict.fromkeys(row["attack"] for row in combined))
    lookup = {(row["model"], row["attack"]): float(row["accuracy"]) for row in combined}
    lines = [
        "# Unified final attack evaluation",
        "",
        "All models use CIFAR-10, PreActResNet-18 and L-inf epsilon=8/255. "
        "Clean/FGSM/PGD evaluations use all 10,000 test images; official AutoAttack "
        "uses the same fixed first 1,000 test images for every model.",
        "",
        "| Model | " + " | ".join(attacks) + " |",
        "|---|" + "---:|" * len(attacks),
    ]
    for model in MODELS:
        values = [f"{100 * lookup[(model, attack)]:.2f}%" for attack in attacks]
        lines.append("| " + model + " | " + " | ".join(values) + " |")
    lines.append("")
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
