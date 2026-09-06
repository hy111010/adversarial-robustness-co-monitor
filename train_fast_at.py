from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import torch
from torch import nn

from robust_exp.attacks import fgsm, pgd_linf
from robust_exp.data import make_loaders
from robust_exp.engine import evaluate, fast_adversarial_train_one_epoch
from robust_exp.models import build_model
from robust_exp.utils import append_csv, choose_device, seed_everything, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast adversarial training with random-start FGSM")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--run-name", default="fast_at_seed17")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-lr", type=float, default=0.2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--attack-step-size", type=float, default=10 / 255)
    parser.add_argument("--pgd-step-size", type=float, default=2 / 255)
    parser.add_argument("--pgd-steps", type=int, default=10)
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
        {**vars(args), "resolved_device": str(device), "torch": torch.__version__, "method": "FGSM-RS"},
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
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.0, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    total_steps = args.epochs * len(train_loader)
    global_step = 0
    best_pgd = -1.0

    fgsm_attack = partial(fgsm, epsilon=args.epsilon)
    pgd_attack = partial(
        pgd_linf,
        epsilon=args.epsilon,
        step_size=args.pgd_step_size,
        steps=args.pgd_steps,
        random_start=True,
    )

    for epoch in range(1, args.epochs + 1):
        metrics, global_step = fast_adversarial_train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            args.epsilon,
            args.attack_step_size,
            global_step,
            total_steps,
            args.max_lr,
        )
        val_clean = evaluate(model, val_loader, device)
        val_fgsm = evaluate(
            model, val_loader, device, attack=fgsm_attack, max_batches=args.monitor_batches
        )
        val_pgd = evaluate(
            model, val_loader, device, attack=pgd_attack, max_batches=args.monitor_batches
        )
        row = {
            "epoch": epoch,
            **metrics,
            "val_clean_acc": val_clean,
            "val_fgsm_acc": val_fgsm,
            "val_pgd10_acc": val_pgd,
            "monitor_batches": args.monitor_batches,
        }
        append_csv(run_dir / "metrics.csv", row)
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_clean_acc": val_clean,
            "val_fgsm_acc": val_fgsm,
            "val_pgd10_acc": val_pgd,
            "args": vars(args),
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if val_pgd > best_pgd:
            best_pgd = val_pgd
            torch.save(checkpoint, run_dir / "best_robust.pt")
        print(
            f"epoch={epoch:02d} loss={metrics['train_adv_loss']:.4f} "
            f"train_adv={metrics['train_adv_acc']:.4f} clean={val_clean:.4f} "
            f"fgsm={val_fgsm:.4f} pgd10={val_pgd:.4f} "
            f"seconds={metrics['epoch_seconds']:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
