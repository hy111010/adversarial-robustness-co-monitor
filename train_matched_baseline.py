from __future__ import annotations

import argparse
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from robust_exp.attacks import fgsm_random_start, pgd_linf
from robust_exp.data import make_loaders
from robust_exp.engine import evaluate
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
    parser = argparse.ArgumentParser(description="Matched FastAdv+ and GradAlign baselines")
    parser.add_argument("--method", choices=("fastadvplus", "gradalign"), required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar10")
    parser.add_argument("--output-root", default="outputs_publication")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--architecture", default="preact_resnet18", choices=("preact_resnet18", "resnet18"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-lr", type=float, default=0.3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--fgsm-step-size", type=float, default=10 / 255)
    parser.add_argument("--pgd-step-size", type=float, default=2 / 255)
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--pgd-monitor-batches", type=int, default=10)
    parser.add_argument("--fastadv-check-interval", type=int, default=20)
    parser.add_argument("--fastadv-pgd-block", type=int, default=20)
    parser.add_argument("--fastadv-drop", type=float, default=0.10)
    parser.add_argument("--gradalign-lambda", type=float, default=0.2)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def triangular_lr(global_step: int, total_steps: int, max_lr: float) -> float:
    progress = global_step / max(total_steps, 1)
    return max(0.0, max_lr * (2.0 * progress if progress <= 0.5 else 2.0 * (1.0 - progress)))


def gradalign_batch(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epsilon: float,
    regularization: float,
    use_amp: bool,
) -> tuple[float, float, float]:
    clean = images.detach().requires_grad_(True)
    with torch.amp.autocast(device_type=images.device.type, enabled=use_amp):
        clean_loss = F.cross_entropy(model(clean), labels)
    grad_clean = torch.autograd.grad(clean_loss, clean, only_inputs=True)[0].detach()

    random_point = (images.detach() + torch.empty_like(images).uniform_(-epsilon, epsilon)).clamp(0, 1)
    random_point.requires_grad_(True)
    # The second-order branch stays in FP32 for stable Hessian-vector backpropagation.
    with torch.amp.autocast(device_type=images.device.type, enabled=False):
        random_loss = F.cross_entropy(model(random_point.float()), labels)
        grad_random = torch.autograd.grad(
            random_loss, random_point, only_inputs=True, create_graph=True
        )[0]
        cosine = F.cosine_similarity(grad_clean.float().flatten(1), grad_random.flatten(1), dim=1)
        align_loss = regularization * (1.0 - cosine.mean())

    adversarial = (images.detach() + epsilon * grad_clean.sign()).clamp(0, 1)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type=images.device.type, enabled=use_amp):
        logits = model(adversarial)
        attack_loss = F.cross_entropy(logits, labels)
    total_loss = attack_loss + align_loss
    scaler.scale(total_loss).backward()
    scaler.step(optimizer)
    scaler.update()
    accuracy = logits.detach().argmax(1).eq(labels).float().mean().item()
    return float(attack_loss.detach()), float(align_loss.detach()), float(accuracy)


def standard_adversarial_batch(
    method: str,
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epsilon: float,
    fgsm_step_size: float,
    pgd_step_size: float,
    pgd_steps: int,
    use_amp: bool,
) -> tuple[float, float]:
    if method == "pgd":
        adversarial = pgd_linf(model, images, labels, epsilon, pgd_step_size, pgd_steps, True)
    else:
        adversarial = fgsm_random_start(model, images, labels, epsilon, fgsm_step_size)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type=images.device.type, enabled=use_amp):
        logits = model(adversarial)
        loss = F.cross_entropy(logits, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    accuracy = logits.detach().argmax(1).eq(labels).float().mean().item()
    return float(loss.detach()), float(accuracy)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    use_amp = device.type == "cuda"
    run_dir = Path(args.output_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    checkpoint_path = run_dir / "last.pt"
    if metrics_path.exists() and not args.resume:
        raise FileExistsError(f"{metrics_path} exists; pass --resume or choose a new name")
    if args.resume and not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    write_json(
        run_dir / "config.json",
        {
            **vars(args),
            "resolved_device": str(device),
            "torch": torch.__version__,
            "protocol": "matched CIFAR-10/PreActResNet-18/triangular-30-epoch baseline",
            "fastadvplus_definition": "R+FGSM with PGD-10 validation every s updates and a transient s-update PGD block after a 0.10 robust-accuracy drop",
            "gradalign_definition": "zero-start FGSM plus lambda*(1-cosine(input gradients at clean and random points)); second-order backprop through random branch",
        },
    )
    train_loader, val_loader = make_loaders(
        args.data_root, args.batch_size, args.workers, args.seed, args.val_size,
        args.limit_train, args.limit_val,
        args.dataset,
    )
    num_classes = 100 if args.dataset == "cifar100" else 10
    model = build_model(num_classes=num_classes, architecture=args.architecture, dataset=args.dataset).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0, momentum=args.momentum, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    total_steps = args.epochs * len(train_loader)
    global_step = 0
    start_epoch = 1
    best_pgd = -1.0
    best_detector_pgd = -1.0
    pgd_remaining = 0
    trigger_count = 0
    pgd_training_batches_total = 0

    if args.resume:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scaler.load_state_dict(saved["scaler_state"])
        global_step = int(saved["global_step"])
        start_epoch = int(saved["epoch"]) + 1
        best_pgd = float(saved["best_pgd"])
        best_detector_pgd = float(saved.get("best_detector_pgd", -1.0))
        pgd_remaining = int(saved.get("pgd_remaining", 0))
        trigger_count = int(saved.get("trigger_count", 0))
        pgd_training_batches_total = int(saved.get("pgd_training_batches_total", 0))
        restore_rng_state(saved.get("rng_state"), train_loader.generator)

    pgd_attack = partial(
        pgd_linf, epsilon=args.epsilon, step_size=args.pgd_step_size,
        steps=args.pgd_steps, random_start=True,
    )
    monitor_images = monitor_labels = None
    if args.method == "fastadvplus":
        with isolated_torch_rng(device, args.seed * 100_000 - 7):
            monitor_images, monitor_labels = next(iter(val_loader))
        monitor_images = monitor_images.to(device, non_blocking=True)
        monitor_labels = monitor_labels.to(device, non_blocking=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        training_loss = training_correct = training_total = 0.0
        align_loss_total = 0.0
        monitor_seconds = 0.0
        epoch_triggers = 0
        epoch_pgd_batches = 0
        first_lr = last_lr = 0.0

        for batch_index, (images, labels) in enumerate(train_loader, start=1):
            lr = triangular_lr(global_step, total_steps, args.max_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            if batch_index == 1:
                first_lr = lr
            last_lr = lr
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if args.method == "gradalign":
                loss, align_loss, accuracy = gradalign_batch(
                    model, images, labels, optimizer, scaler, args.epsilon,
                    args.gradalign_lambda, use_amp,
                )
                align_loss_total += align_loss * labels.size(0)
            else:
                batch_method = "pgd" if pgd_remaining > 0 else "fgsm"
                loss, accuracy = standard_adversarial_batch(
                    batch_method, model, images, labels, optimizer, scaler,
                    args.epsilon, args.fgsm_step_size, args.pgd_step_size,
                    args.pgd_steps, use_amp,
                )
                if pgd_remaining > 0:
                    pgd_remaining -= 1
                    epoch_pgd_batches += 1
                    pgd_training_batches_total += 1

            batch_size = labels.size(0)
            training_loss += loss * batch_size
            training_correct += accuracy * batch_size
            training_total += batch_size
            global_step += 1

            if args.method == "fastadvplus" and global_step % args.fastadv_check_interval == 0:
                assert monitor_images is not None and monitor_labels is not None
                monitor_start = time.perf_counter()
                with isolated_torch_rng(device, args.seed * 1_000_000 + global_step):
                    model.eval()
                    monitor_adv = pgd_attack(model, monitor_images, monitor_labels)
                    with torch.no_grad():
                        monitor_acc = model(monitor_adv).argmax(1).eq(monitor_labels).float().mean().item()
                    model.train()
                monitor_seconds += time.perf_counter() - monitor_start
                if best_detector_pgd >= 0 and best_detector_pgd - monitor_acc >= args.fastadv_drop:
                    if pgd_remaining == 0:
                        epoch_triggers += 1
                        trigger_count += 1
                    pgd_remaining = max(pgd_remaining, args.fastadv_pgd_block)
                best_detector_pgd = max(best_detector_pgd, monitor_acc)

        core_seconds = time.perf_counter() - epoch_start
        monitor_seed = args.seed * 100_000 + epoch
        with isolated_torch_rng(device, monitor_seed):
            clean_start = time.perf_counter()
            val_clean = evaluate(model, val_loader, device)
            clean_seconds = time.perf_counter() - clean_start
            pgd_start = time.perf_counter()
            val_pgd = evaluate(model, val_loader, device, attack=pgd_attack, max_batches=args.pgd_monitor_batches)
            pgd_seconds = time.perf_counter() - pgd_start

        row = {
            "epoch": epoch,
            "method": args.method,
            "train_adv_loss": training_loss / training_total,
            "train_adv_acc": training_correct / training_total,
            "gradalign_loss": align_loss_total / training_total if args.method == "gradalign" else 0.0,
            "val_clean_acc": val_clean,
            "val_pgd10_acc": val_pgd,
            "core_training_seconds": core_seconds,
            "online_monitor_seconds": monitor_seconds,
            "clean_eval_seconds": clean_seconds,
            "pgd_monitor_seconds": pgd_seconds,
            "lr_start": first_lr,
            "lr_end": last_lr,
            "fastadv_triggers_epoch": epoch_triggers,
            "fastadv_triggers_total": trigger_count,
            "pgd_training_batches_epoch": epoch_pgd_batches,
            "pgd_training_batches_total": pgd_training_batches_total,
            "pgd_remaining": pgd_remaining,
        }
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
            "best_pgd": max(best_pgd, val_pgd),
            "best_detector_pgd": best_detector_pgd,
            "pgd_remaining": pgd_remaining,
            "trigger_count": trigger_count,
            "pgd_training_batches_total": pgd_training_batches_total,
            "args": vars(args),
            "rng_state": capture_rng_state(train_loader.generator),
        }
        torch.save(checkpoint, checkpoint_path)
        if val_pgd > best_pgd:
            best_pgd = val_pgd
            torch.save(checkpoint, run_dir / "best_robust.pt")
        print(
            f"epoch={epoch:03d} method={args.method} clean={val_clean:.4f} "
            f"pgd10={val_pgd:.4f} core_s={core_seconds:.1f} online_s={monitor_seconds:.1f} "
            f"triggers={trigger_count} pgd_batches={pgd_training_batches_total}",
            flush=True,
        )


if __name__ == "__main__":
    main()
