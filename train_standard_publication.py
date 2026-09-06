from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from robust_exp.data import make_loaders
from robust_exp.engine import evaluate, train_one_epoch
from robust_exp.models import build_model
from robust_exp.utils import (
    append_csv,
    capture_rng_state,
    choose_device,
    restore_rng_state,
    seed_everything,
    write_json,
)
from train_pgd_at import epoch_lr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publication baseline: standard training")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="outputs_publication")
    parser.add_argument("--run-name", default="standard_preact_seed17")
    parser.add_argument(
        "--architecture", default="preact_resnet18", choices=["preact_resnet18", "resnet18"]
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=110)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    run_dir = Path(args.output_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    checkpoint_path = run_dir / "last.pt"
    if metrics_path.exists() and not args.resume:
        raise FileExistsError(
            f"{metrics_path} already exists; choose a new run name or pass --resume"
        )
    if args.resume and not checkpoint_path.exists():
        raise FileNotFoundError(f"Cannot resume because {checkpoint_path} does not exist")
    write_json(
        run_dir / "config.json",
        {**vars(args), "resolved_device": str(device), "torch": torch.__version__, "method": "standard"},
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
    model = build_model(architecture=args.architecture).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    start_epoch = 1
    best_clean = -1.0
    if args.resume:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_args = saved.get("args", {})
        for key in ("architecture", "seed", "batch_size"):
            if key in saved_args and saved_args[key] != getattr(args, key):
                raise ValueError(
                    f"Resume mismatch for {key}: checkpoint={saved_args[key]!r}, current={getattr(args, key)!r}"
                )
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        if "scaler_state" in saved:
            scaler.load_state_dict(saved["scaler_state"])
        start_epoch = int(saved["epoch"]) + 1
        best_clean = float(saved.get("best_clean", saved.get("val_clean_acc", -1.0)))
        restore_rng_state(saved.get("rng_state"), train_loader.generator)
        print(
            f"resumed={checkpoint_path} next_epoch={start_epoch:03d} best_clean={best_clean:.4f}",
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        lr = epoch_lr(args.lr, epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr
        metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_clean = evaluate(model, val_loader, device)
        row = {"epoch": epoch, **metrics, "val_clean_acc": val_clean}
        append_csv(metrics_path, row)
        checkpoint = {
            "epoch": epoch,
            "architecture": args.architecture,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "val_clean_acc": val_clean,
            "best_clean": max(best_clean, val_clean),
            "args": vars(args),
            "rng_state": capture_rng_state(train_loader.generator),
        }
        torch.save(checkpoint, checkpoint_path)
        if val_clean > best_clean:
            best_clean = val_clean
            torch.save(checkpoint, run_dir / "best_clean.pt")
        print(
            f"epoch={epoch:03d} lr={lr:.4g} loss={metrics['train_loss']:.4f} "
            f"clean={val_clean:.4f} seconds={metrics['epoch_seconds']:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
