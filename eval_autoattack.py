from __future__ import annotations

import argparse
import time
from pathlib import Path

import torchattacks

from robust_exp.checkpoints import load_model_checkpoint
from robust_exp.data import make_test_loader
from robust_exp.engine import evaluate
from robust_exp.utils import append_csv, choose_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Development-only evaluation with the third-party torchattacks AutoAttack"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--version", choices=["standard", "plus", "rand"], default="standard")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--output-name", default="third_party_autoattack_results.csv")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = choose_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint, model, architecture = load_model_checkpoint(checkpoint_path, device)
    model.eval()
    loader = make_test_loader(args.data_root, args.batch_size, args.workers)
    attack = torchattacks.AutoAttack(
        model,
        norm="Linf",
        eps=args.epsilon,
        version=args.version,
        n_classes=10,
        seed=args.seed,
        verbose=False,
    )
    start = time.perf_counter()
    def attack_adapter(_model, images, labels):
        return attack(images, labels)

    accuracy = evaluate(model, loader, device, attack=attack_adapter, max_batches=args.max_batches)
    seconds = time.perf_counter() - start
    evaluated_samples = None if args.max_batches is None else args.max_batches * args.batch_size
    append_csv(
        checkpoint_path.parent / args.output_name,
        {
            "checkpoint": checkpoint_path.name,
            "architecture": architecture,
            "attack": f"torchattacks_autoattack_{args.version}",
            "accuracy": accuracy,
            "epsilon": args.epsilon,
            "batch_size": args.batch_size,
            "max_batches": args.max_batches if args.max_batches is not None else "all",
            "evaluated_samples_upper_bound": evaluated_samples if evaluated_samples is not None else 10000,
            "seconds": seconds,
        },
    )
    print(f"AutoAttack-{args.version}: accuracy={accuracy:.4f}, seconds={seconds:.1f}", flush=True)


if __name__ == "__main__":
    main()
