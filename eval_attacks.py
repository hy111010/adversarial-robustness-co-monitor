from __future__ import annotations

import argparse
import time
from functools import partial
from pathlib import Path

from robust_exp.attacks import fgsm, pgd_linf
from robust_exp.checkpoints import load_model_checkpoint
from robust_exp.data import make_test_loader
from robust_exp.engine import evaluate
from robust_exp.utils import append_csv, choose_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate clean, FGSM, and PGD-10 accuracy")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--pgd-step-size", type=float, default=2 / 255)
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--max-batches", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint, model, architecture = load_model_checkpoint(checkpoint_path, device)
    loader = make_test_loader(args.data_root, args.batch_size, args.workers)
    attacks = {
        "clean": None,
        "fgsm_8_255": partial(fgsm, epsilon=args.epsilon),
        "pgd10_8_255": partial(
            pgd_linf,
            epsilon=args.epsilon,
            step_size=args.pgd_step_size,
            steps=args.pgd_steps,
            random_start=True,
        ),
    }
    output_path = checkpoint_path.parent / "attack_results.csv"
    for name, attack in attacks.items():
        start = time.perf_counter()
        accuracy = evaluate(model, loader, device, attack=attack, max_batches=args.max_batches)
        row = {
            "checkpoint": checkpoint_path.name,
            "architecture": architecture,
            "attack": name,
            "accuracy": accuracy,
            "epsilon": args.epsilon if attack else 0.0,
            "seconds": time.perf_counter() - start,
        }
        append_csv(output_path, row)
        print(f"{name}: accuracy={accuracy:.4f}, seconds={row['seconds']:.1f}")


if __name__ == "__main__":
    main()
