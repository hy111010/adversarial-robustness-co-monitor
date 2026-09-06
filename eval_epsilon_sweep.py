from __future__ import annotations

import argparse
import time
from functools import partial
from pathlib import Path

from robust_exp.attacks import pgd_linf
from robust_exp.checkpoints import load_model_checkpoint
from robust_exp.data import make_test_loader
from robust_exp.engine import evaluate
from robust_exp.utils import append_csv, choose_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="PGD-20 robustness across epsilon values")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = choose_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint, model, architecture = load_model_checkpoint(checkpoint_path, device)
    loader = make_test_loader(args.data_root, args.batch_size, args.workers)
    output_path = checkpoint_path.parent / "epsilon_sweep_results.csv"

    for epsilon_pixels in [2, 4, 8, 12, 16]:
        epsilon = epsilon_pixels / 255
        step_size = epsilon / 4
        attack = partial(
            pgd_linf,
            epsilon=epsilon,
            step_size=step_size,
            steps=args.steps,
            random_start=True,
        )
        start = time.perf_counter()
        accuracy = evaluate(model, loader, device, attack=attack, max_batches=args.max_batches)
        seconds = time.perf_counter() - start
        append_csv(
            output_path,
            {
            "checkpoint": checkpoint_path.name,
            "architecture": architecture,
                "attack": f"pgd{args.steps}",
                "epsilon_pixels": epsilon_pixels,
                "epsilon": epsilon,
                "step_size": step_size,
                "accuracy": accuracy,
                "seconds": seconds,
            },
        )
        print(f"epsilon={epsilon_pixels}/255: accuracy={accuracy:.4f}, seconds={seconds:.1f}", flush=True)


if __name__ == "__main__":
    main()
