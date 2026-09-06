from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
import time
from pathlib import Path

import torch
from autoattack import AutoAttack

from robust_exp.checkpoints import load_model_checkpoint
from robust_exp.data import make_test_loader
from robust_exp.utils import append_csv, choose_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official fra31 AutoAttack evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--loader-batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--version", choices=["standard", "plus", "rand"], default="standard")
    parser.add_argument("--max-samples", type=int, default=10_000)
    parser.add_argument("--output-name", default="official_autoattack_results.csv")
    parser.add_argument("--state-name", default="official_autoattack_state.json")
    parser.add_argument("--log-name", default="official_autoattack.log")
    return parser.parse_args()


def load_examples(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    loader = make_test_loader(args.data_root, args.loader_batch_size, args.workers)
    image_parts = []
    label_parts = []
    remaining = args.max_samples
    for images, labels in loader:
        if remaining <= 0:
            break
        take = min(remaining, labels.size(0))
        image_parts.append(images[:take])
        label_parts.append(labels[:take])
        remaining -= take
    if not image_parts:
        raise ValueError("--max-samples must be at least 1")
    return torch.cat(image_parts), torch.cat(label_parts)


def package_source() -> dict:
    distribution = importlib.metadata.distribution("autoattack")
    direct_url_path = Path(distribution.locate_file("autoattack-0.1.dist-info/direct_url.json"))
    direct_url = json.loads(direct_url_path.read_text(encoding="utf-8"))
    return {
        "version": distribution.version,
        "url": direct_url.get("url"),
        "sha256": direct_url.get("archive_info", {}).get("hashes", {}).get("sha256"),
    }


def read_state(path: Path) -> dict:
    """Read AutoAttack 0.1 state despite its lossy set JSON serialization."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.get("_attacks_to_run", [])
    if isinstance(expected, str):
        expected = ast.literal_eval(expected)
    completed = payload.get("_run_attacks", [])
    flags = payload.get("_robust_flags")
    return {
        "expected": set(expected),
        "completed": set(completed),
        "robust_accuracy": sum(bool(value) for value in flags) / len(flags),
        "clean_accuracy": float(payload.get("_clean_accuracy", float("nan"))),
    }


def main() -> None:
    args = parse_args()
    if args.max_samples > 10_000:
        raise ValueError("CIFAR-10 test evaluation cannot exceed 10,000 samples")
    seed_everything(args.seed)
    device = choose_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint, model, architecture = load_model_checkpoint(checkpoint_path, device)
    model.eval()
    images, labels = load_examples(args)
    state_path = checkpoint_path.parent / args.state_name
    log_path = checkpoint_path.parent / args.log_name
    source = package_source()

    start = time.perf_counter()
    existing = read_state(state_path) if state_path.exists() else None
    if existing is None or existing["completed"] != existing["expected"]:
        adversary = AutoAttack(
            model,
            norm="Linf",
            eps=args.epsilon,
            seed=args.seed,
            verbose=True,
            version=args.version,
            device=str(device),
            log_path=str(log_path),
        )
        adversary.run_standard_evaluation(
            images,
            labels,
            bs=args.batch_size,
            state_path=state_path,
        )
    seconds = time.perf_counter() - start
    state = read_state(state_path)
    completed = sorted(state["completed"])
    expected = sorted(state["expected"])
    row = {
        "checkpoint": checkpoint_path.name,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "architecture": architecture,
        "attack": f"official_autoattack_{args.version}",
        "accuracy": state["robust_accuracy"],
        "clean_accuracy": state["clean_accuracy"],
        "epsilon": args.epsilon,
        "samples": len(labels),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "completed_attacks": ";".join(completed),
        "expected_attacks": ";".join(expected),
        "complete": completed == expected,
        "seconds_this_invocation": seconds,
        "package_version": source["version"],
        "package_url": source["url"],
        "package_sha256": source["sha256"],
    }
    append_csv(checkpoint_path.parent / args.output_name, row)
    print(
        f"official AutoAttack-{args.version}: accuracy={state['robust_accuracy']:.4f} "
        f"samples={len(labels)} complete={row['complete']} seconds={seconds:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
