from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from robust_exp.data import make_loaders
from robust_exp.engine import evaluate, train_one_epoch
from robust_exp.models import build_model
from robust_exp.utils import append_csv, choose_device, seed_everything, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a standard CIFAR-10 ResNet-18 baseline")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--run-name", default="standard_seed17")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
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
    write_json(run_dir / "config.json", {**vars(args), "resolved_device": str(device), "torch": torch.__version__})

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
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    best_val = -1.0

    for epoch in range(1, args.epochs + 1):
        metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_acc = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **metrics,
            "val_clean_acc": val_acc,
        }
        append_csv(run_dir / "metrics.csv", row)
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_clean_acc": val_acc,
            "args": vars(args),
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(checkpoint, run_dir / "best.pt")
        print(
            f"epoch={epoch:03d} loss={metrics['train_loss']:.4f} "
            f"train={metrics['train_acc']:.4f} val={val_acc:.4f} "
            f"seconds={metrics['epoch_seconds']:.1f}"
        )
        scheduler.step()


if __name__ == "__main__":
    main()

