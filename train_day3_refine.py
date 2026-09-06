from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import torch
from torch import nn

from robust_exp.attacks import fgsm, pgd_linf
from robust_exp.data import make_loaders
from robust_exp.engine import evaluate, pgd_adversarial_train_one_epoch
from robust_exp.models import build_model
from robust_exp.utils import append_csv, choose_device, seed_everything, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-learning-rate PGD-2 refinement of Day 3")
    parser.add_argument("--checkpoint", default="outputs/rollback_pgd2_seed17/best_robust.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--run-name", default="day3_refined_seed17")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--train-step-size", type=float, default=4 / 255)
    parser.add_argument("--train-steps", type=int, default=2)
    parser.add_argument("--monitor-step-size", type=float, default=2 / 255)
    parser.add_argument("--monitor-steps", type=int, default=10)
    parser.add_argument("--monitor-batches", type=int, default=10)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    run_dir = Path(args.output_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "config.json",
        {**vars(args), "resolved_device": str(device), "torch": torch.__version__, "method": "day3_pgd2_refine"},
    )

    train_loader, val_loader = make_loaders(
        args.data_root,
        args.batch_size,
        args.workers,
        args.seed,
        args.val_size,
        args.limit_train,
        args.limit_val,
    )
    source = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = build_model().to(device)
    model.load_state_dict(source["model_state"])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.max_lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    # Reuse the descending half of the triangular scheduler: max_lr -> 0.
    refinement_steps = args.epochs * len(train_loader)
    total_steps = 2 * refinement_steps
    global_step = refinement_steps
    best_score = -1.0

    fgsm_attack = partial(fgsm, epsilon=args.epsilon)
    monitor_attack = partial(
        pgd_linf,
        epsilon=args.epsilon,
        step_size=args.monitor_step_size,
        steps=args.monitor_steps,
        random_start=True,
    )

    for epoch in range(1, args.epochs + 1):
        metrics, global_step = pgd_adversarial_train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            args.epsilon,
            args.train_step_size,
            args.train_steps,
            global_step,
            total_steps,
            args.max_lr,
        )
        val_clean = evaluate(model, val_loader, device)
        val_fgsm = evaluate(model, val_loader, device, attack=fgsm_attack, max_batches=args.monitor_batches)
        val_pgd = evaluate(model, val_loader, device, attack=monitor_attack, max_batches=args.monitor_batches)
        # Harmonic mean rewards balance and prevents one accuracy hiding a regression in the other.
        balanced_score = 2 * val_clean * val_pgd / max(val_clean + val_pgd, 1e-12)
        row = {
            "epoch": epoch,
            **metrics,
            "val_clean_acc": val_clean,
            "val_fgsm_acc": val_fgsm,
            "val_pgd10_acc": val_pgd,
            "balanced_score": balanced_score,
            "monitor_batches": args.monitor_batches,
        }
        append_csv(run_dir / "metrics.csv", row)
        checkpoint = {
            "epoch": epoch,
            "source_checkpoint": args.checkpoint,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_clean_acc": val_clean,
            "val_fgsm_acc": val_fgsm,
            "val_pgd10_acc": val_pgd,
            "balanced_score": balanced_score,
            "args": vars(args),
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if balanced_score > best_score:
            best_score = balanced_score
            torch.save(checkpoint, run_dir / "best_balanced.pt")
        print(
            f"epoch={epoch:02d} loss={metrics['train_adv_loss']:.4f} "
            f"clean={val_clean:.4f} fgsm={val_fgsm:.4f} pgd10={val_pgd:.4f} "
            f"score={balanced_score:.4f} seconds={metrics['epoch_seconds']:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
