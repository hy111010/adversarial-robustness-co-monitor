from __future__ import annotations

import argparse
import time
from functools import partial
from pathlib import Path

import torch
from torch import nn

from robust_exp.attacks import pgd_linf
from robust_exp.data import make_loaders
from robust_exp.diagnostics import cheap_collapse_signals
from robust_exp.engine import evaluate, fast_adversarial_train_one_epoch_traced
from robust_exp.models import build_model
from robust_exp.utils import (
    append_csv,
    capture_rng_state,
    choose_device,
    isolated_torch_rng,
    restore_rng_state,
    seed_everything,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publication FGSM-RS trace with PGD labels and cheap collapse diagnostics"
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar10")
    parser.add_argument("--output-root", default="outputs_publication")
    parser.add_argument("--run-name", default="fast_fgsm_rs_preact_seed17")
    parser.add_argument(
        "--architecture", default="preact_resnet18", choices=["preact_resnet18", "resnet18"]
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-lr", type=float, default=0.2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--attack-step-size", type=float, default=10 / 255)
    parser.add_argument(
        "--attack-init", choices=["random_uniform", "zero"], default="random_uniform"
    )
    parser.add_argument("--pgd-step-size", type=float, default=2 / 255)
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--pgd-monitor-batches", type=int, default=10)
    parser.add_argument("--diagnostic-batches", type=int, default=2)
    parser.add_argument("--trace-interval", type=int, default=20)
    parser.add_argument(
        "--batch-pgd-label-batches",
        type=int,
        default=1,
        help="PGD validation batches used only to label each trace point; 0 disables labels",
    )
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
    batch_metrics_path = run_dir / "batch_metrics.csv"
    checkpoint_path = run_dir / "last.pt"
    if metrics_path.exists() and not args.resume:
        raise FileExistsError(
            f"{metrics_path} already exists; choose a new run name or pass --resume"
        )
    if args.resume and not checkpoint_path.exists():
        raise FileNotFoundError(f"Cannot resume because {checkpoint_path} does not exist")

    write_json(
        run_dir / "config.json",
        {
            **vars(args),
            "resolved_device": str(device),
            "torch": torch.__version__,
            "method": (
                "FGSM-RS diagnostic trace"
                if args.attack_init == "random_uniform"
                else "FGSM zero-initialization diagnostic trace"
            ),
            "monitor_rng_isolated": True,
        },
    )
    train_loader, val_loader = make_loaders(
        args.data_root,
        args.batch_size,
        args.workers,
        args.seed,
        args.val_size,
        args.limit_train,
        args.limit_val,
        args.dataset,
    )
    num_classes = 100 if args.dataset == "cifar100" else 10
    model = build_model(num_classes=num_classes, architecture=args.architecture, dataset=args.dataset).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.0, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    total_steps = args.epochs * len(train_loader)
    global_step = 0
    start_epoch = 1
    best_pgd = -1.0

    if args.resume:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_args = saved.get("args", {})
        frozen = (
            "architecture",
            "seed",
            "epochs",
            "batch_size",
            "max_lr",
            "epsilon",
            "attack_step_size",
            "attack_init",
        )
        for key in frozen:
            if key in saved_args and saved_args[key] != getattr(args, key):
                raise ValueError(
                    f"Resume mismatch for {key}: checkpoint={saved_args[key]!r}, current={getattr(args, key)!r}"
                )
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        if "scaler_state" in saved:
            scaler.load_state_dict(saved["scaler_state"])
        global_step = int(saved["global_step"])
        start_epoch = int(saved["epoch"]) + 1
        best_pgd = float(saved.get("best_pgd", saved.get("val_pgd10_acc", -1.0)))
        restore_rng_state(saved.get("rng_state"), train_loader.generator)
        print(
            f"resumed={checkpoint_path} next_epoch={start_epoch:03d} best_pgd={best_pgd:.4f}",
            flush=True,
        )

    pgd_attack = partial(
        pgd_linf,
        epsilon=args.epsilon,
        step_size=args.pgd_step_size,
        steps=args.pgd_steps,
        random_start=True,
    )

    label_images = None
    label_labels = None
    if args.batch_pgd_label_batches > 0:
        image_parts = []
        label_parts = []
        with isolated_torch_rng(device, args.seed * 100_000 - 1):
            for batch_index, (images, labels) in enumerate(val_loader):
                if batch_index >= args.batch_pgd_label_batches:
                    break
                image_parts.append(images)
                label_parts.append(labels)
        if not image_parts:
            raise RuntimeError("No validation samples were available for batch-level PGD labels")
        label_images = torch.cat(image_parts).to(device, non_blocking=True)
        label_labels = torch.cat(label_parts).to(device, non_blocking=True)

    def label_trace_point(
        current_model: nn.Module, trace: dict[str, float | int]
    ) -> dict[str, float | int]:
        if args.batch_pgd_label_batches == 0:
            return {}
        assert label_images is not None and label_labels is not None
        label_seed = args.seed * 1_000_000 + int(trace["global_step"])
        with isolated_torch_rng(device, label_seed):
            label_start = time.perf_counter()
            current_model.eval()
            adversarial = pgd_attack(current_model, label_images, label_labels)
            with torch.no_grad():
                label_accuracy = float(
                    current_model(adversarial).argmax(1).eq(label_labels).float().mean().item()
                )
            label_seconds = time.perf_counter() - label_start
        return {
            "label_pgd10_acc": label_accuracy,
            "label_pgd10_seconds": label_seconds,
            "label_seed": label_seed,
            "label_batches": args.batch_pgd_label_batches,
        }

    for epoch in range(start_epoch, args.epochs + 1):
        metrics, global_step, batch_traces = fast_adversarial_train_one_epoch_traced(
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
            epoch,
            args.trace_interval,
            args.attack_init,
            label_trace_point,
        )
        for trace in batch_traces:
            append_csv(batch_metrics_path, trace)

        monitor_seed = args.seed * 100_000 + epoch
        with isolated_torch_rng(device, monitor_seed):
            clean_start = time.perf_counter()
            val_clean = evaluate(model, val_loader, device)
            clean_seconds = time.perf_counter() - clean_start

            diagnostic_start = time.perf_counter()
            diagnostics = cheap_collapse_signals(
                model,
                val_loader,
                device,
                args.epsilon,
                args.attack_step_size,
                max_batches=args.diagnostic_batches,
            )
            diagnostic_seconds = time.perf_counter() - diagnostic_start

            pgd_start = time.perf_counter()
            val_pgd = evaluate(
                model,
                val_loader,
                device,
                attack=pgd_attack,
                max_batches=args.pgd_monitor_batches,
            )
            pgd_seconds = time.perf_counter() - pgd_start

        row = {
            "epoch": epoch,
            **metrics,
            "val_clean_acc": val_clean,
            "val_pgd10_acc": val_pgd,
            **diagnostics,
            "clean_eval_seconds": clean_seconds,
            "diagnostic_seconds": diagnostic_seconds,
            "pgd_monitor_seconds": pgd_seconds,
            "diagnostic_batches": args.diagnostic_batches,
            "pgd_monitor_batches": args.pgd_monitor_batches,
            "monitor_seed": monitor_seed,
            "batch_pgd_label_seconds": sum(
                float(trace.get("label_pgd10_seconds", 0.0)) for trace in batch_traces
            ),
        }
        row["estimated_core_epoch_seconds"] = max(
            0.0, metrics["epoch_seconds"] - row["batch_pgd_label_seconds"]
        )
        append_csv(metrics_path, row)
        checkpoint = {
            "epoch": epoch,
            "architecture": args.architecture,
            "dataset": args.dataset,
            "num_classes": num_classes,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "global_step": global_step,
            "val_clean_acc": val_clean,
            "val_pgd10_acc": val_pgd,
            "diagnostics": diagnostics,
            "best_pgd": max(best_pgd, val_pgd),
            "args": vars(args),
            "rng_state": capture_rng_state(train_loader.generator),
        }
        torch.save(checkpoint, checkpoint_path)
        if val_pgd > best_pgd:
            best_pgd = val_pgd
            torch.save(checkpoint, run_dir / "best_robust.pt")
        print(
            f"epoch={epoch:03d} loss={metrics['train_adv_loss']:.4f} "
            f"clean={val_clean:.4f} pgd10={val_pgd:.4f} "
            f"diag_cos={diagnostics['diag_gradient_cosine']:.4f} "
            f"diag_disagree={diagnostics['diag_rs_prediction_disagreement']:.4f} "
            f"free_cos={metrics['free_gradient_cosine']:.4f} "
            f"train_s={metrics['epoch_seconds']:.1f} diag_s={diagnostic_seconds:.1f} "
            f"pgd_s={pgd_seconds:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
