from __future__ import annotations

import argparse
import time
from functools import partial
from pathlib import Path

from robust_exp.attacks import pgd_linf, pgd_linf_restarts
from robust_exp.checkpoints import load_model_checkpoint
from robust_exp.data import make_test_loader
from robust_exp.engine import evaluate
from robust_exp.utils import append_csv, choose_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strong white-box robustness evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--step-size", type=float, default=2 / 255)
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
        "pgd20_r1_8_255": partial(
            pgd_linf,
            epsilon=args.epsilon,
            step_size=args.step_size,
            steps=20,
            random_start=True,
        ),
        "pgd50_r1_8_255": partial(
            pgd_linf,
            epsilon=args.epsilon,
            step_size=args.step_size,
            steps=50,
            random_start=True,
        ),
        "pgd20_r5_8_255": partial(
            pgd_linf_restarts,
            epsilon=args.epsilon,
            step_size=args.step_size,
            steps=20,
            restarts=5,
        ),
    }
    output_path = checkpoint_path.parent / "strong_attack_results.csv"
    for name, attack in attacks.items():
        start = time.perf_counter()
        accuracy = evaluate(model, loader, device, attack=attack, max_batches=args.max_batches)
        row = {
            "checkpoint": checkpoint_path.name,
            "architecture": architecture,
            "attack": name,
            "accuracy": accuracy,
            "epsilon": args.epsilon,
            "step_size": args.step_size,
            "seconds": time.perf_counter() - start,
        }
        append_csv(output_path, row)
        print(f"{name}: accuracy={accuracy:.4f}, seconds={row['seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()
