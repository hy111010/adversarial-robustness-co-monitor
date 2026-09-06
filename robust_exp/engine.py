from __future__ import annotations

import time
from collections.abc import Callable

import torch
from torch import nn

from .attacks import fgsm_random_start, fgsm_trace, pgd_linf, trades_linf


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    start = time.perf_counter()
    use_amp = device.type == "cuda"

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * labels.size(0)
        correct += logits.argmax(1).eq(labels).sum().item()
        total += labels.size(0)

    return {
        "train_loss": total_loss / total,
        "train_acc": correct / total,
        "epoch_seconds": time.perf_counter() - start,
    }


def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    attack: Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    max_batches: int | None = None,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if attack is not None:
            images = attack(model, images, labels)
        with torch.no_grad():
            predictions = model(images).argmax(1)
        correct += predictions.eq(labels).sum().item()
        total += labels.size(0)
    return correct / total


def fast_adversarial_train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    epsilon: float,
    attack_step_size: float,
    global_step: int,
    total_steps: int,
    max_lr: float,
) -> tuple[dict[str, float], int]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    start = time.perf_counter()
    use_amp = device.type == "cuda"
    first_lr = None
    last_lr = None

    for images, labels in loader:
        progress = global_step / max(total_steps, 1)
        lr = max_lr * (2.0 * progress if progress <= 0.5 else 2.0 * (1.0 - progress))
        lr = max(lr, 0.0)
        for group in optimizer.param_groups:
            group["lr"] = lr
        first_lr = lr if first_lr is None else first_lr
        last_lr = lr

        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        adversarial = fgsm_random_start(model, images, labels, epsilon, attack_step_size)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(adversarial)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        correct += logits.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
        global_step += 1

    return (
        {
            "train_adv_loss": total_loss / total,
            "train_adv_acc": correct / total,
            "epoch_seconds": time.perf_counter() - start,
            "lr_start": float(first_lr or 0.0),
            "lr_end": float(last_lr or 0.0),
        },
        global_step,
    )


def fast_adversarial_train_one_epoch_traced(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    epsilon: float,
    attack_step_size: float,
    global_step: int,
    total_steps: int,
    max_lr: float,
    epoch: int,
    trace_interval: int = 20,
    attack_initialization: str = "random_uniform",
    trace_callback: Callable[
        [nn.Module, dict[str, float | int]], dict[str, float | int]
    ]
    | None = None,
    continue_with_pgd_after_stop: bool = False,
    recovery_step_size: float | None = None,
    recovery_steps: int | None = None,
) -> tuple[dict[str, float], int, list[dict[str, float | int]]]:
    """FGSM-RS training with endpoint signals that require no extra model passes."""
    if trace_interval < 1:
        raise ValueError("trace_interval must be at least 1")
    model.train()
    totals = {
        "train_adv_loss": 0.0,
        "train_adv_acc": 0.0,
        "free_gradient_cosine": 0.0,
        "free_gradient_sign_agreement": 0.0,
        "free_prediction_disagreement": 0.0,
        "free_attack_loss_gain": 0.0,
        "free_boundary_fraction": 0.0,
        "free_delta_linf": 0.0,
        "control_gradient_participation_ratio": 0.0,
        "control_gradient_entropy": 0.0,
    }
    window = {name: 0.0 for name in totals}
    total = 0
    diagnostic_total = 0
    window_total = 0
    traces: list[dict[str, float | int]] = []
    start_time = time.perf_counter()
    use_amp = device.type == "cuda"
    first_lr = None
    last_lr = None
    switched_to_pgd = False
    post_trigger_pgd_batches = 0

    for batch_index, (images, labels) in enumerate(loader, start=1):
        progress = global_step / max(total_steps, 1)
        lr = max_lr * (2.0 * progress if progress <= 0.5 else 2.0 * (1.0 - progress))
        lr = max(lr, 0.0)
        for group in optimizer.param_groups:
            group["lr"] = lr
        first_lr = lr if first_lr is None else first_lr
        last_lr = lr

        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if switched_to_pgd:
            if recovery_step_size is None or recovery_steps is None:
                raise ValueError("PGD continuation requires recovery parameters")
            adversarial = pgd_linf(
                model,
                images,
                labels,
                epsilon=epsilon,
                step_size=recovery_step_size,
                steps=recovery_steps,
                random_start=True,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(adversarial)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            with torch.no_grad():
                batch_correct = logits.detach().argmax(1).eq(labels).float().mean()
            batch_values = {
                "train_adv_loss": float(loss.detach().item()),
                "train_adv_acc": float(batch_correct.item()),
            }
            post_trigger_pgd_batches += 1
        else:
            adversarial, start_gradient, start_logits = fgsm_trace(
                model, images, labels, epsilon, attack_step_size, attack_initialization
            )
            adversarial.requires_grad_(True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(adversarial)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if adversarial.grad is None:
                raise RuntimeError("Endpoint input gradient was not retained")
            endpoint_gradient = adversarial.grad.detach().float()

            with torch.no_grad():
                flat_start = start_gradient.float().flatten(1)
                flat_endpoint = endpoint_gradient.flatten(1)
                gradient_cosine = nn.functional.cosine_similarity(
                    flat_start, flat_endpoint, dim=1
                ).mean()
                sign_agreement = (
                    start_gradient.sign()
                    .eq(endpoint_gradient.sign())
                    .float()
                    .flatten(1)
                    .mean()
                )
                prediction_disagreement = (
                    start_logits.argmax(1).ne(logits.detach().argmax(1)).float().mean()
                )
                endpoint_losses = nn.functional.cross_entropy(
                    logits.detach().float(), labels, reduction="none"
                )
                start_losses = nn.functional.cross_entropy(
                    start_logits.float(), labels, reduction="none"
                )
                attack_loss_gain = (endpoint_losses - start_losses).mean()
                delta = adversarial.detach() - images
                boundary_fraction = delta.abs().ge(epsilon - 1e-6).float().mean()
                delta_linf = delta.abs().flatten(1).max(dim=1).values.mean()
                batch_correct = logits.detach().argmax(1).eq(labels).float().mean()
                squared_gradient = flat_start.square()
                squared_sum = squared_gradient.sum(dim=1).clamp_min(1e-30)
                dimensions = flat_start.size(1)
                participation_ratio = (
                    squared_sum.square()
                    / squared_gradient.square().sum(dim=1).clamp_min(1e-30)
                    / dimensions
                ).mean()
                gradient_distribution = squared_gradient / squared_sum[:, None]
                gradient_entropy = (
                    -(gradient_distribution * gradient_distribution.clamp_min(1e-30).log()).sum(dim=1)
                    / torch.log(torch.tensor(float(dimensions), device=device))
                ).mean()

            batch_values = {
                "train_adv_loss": float(loss.detach().item()),
                "train_adv_acc": float(batch_correct.item()),
                "free_gradient_cosine": float(gradient_cosine.item()),
                "free_gradient_sign_agreement": float(sign_agreement.item()),
                "free_prediction_disagreement": float(prediction_disagreement.item()),
                "free_attack_loss_gain": float(attack_loss_gain.item()),
                "free_boundary_fraction": float(boundary_fraction.item()),
                "free_delta_linf": float(delta_linf.item()),
                "control_gradient_participation_ratio": float(participation_ratio.item()),
                "control_gradient_entropy": float(gradient_entropy.item()),
            }

        scaler.step(optimizer)
        scaler.update()
        batch_size = labels.size(0)
        for name, value in batch_values.items():
            totals[name] += value * batch_size
            if not switched_to_pgd:
                window[name] += value * batch_size
        total += batch_size
        if not switched_to_pgd:
            diagnostic_total += batch_size
            window_total += batch_size
        global_step += 1

        should_trace = not switched_to_pgd and (
            batch_index % trace_interval == 0 or batch_index == len(loader)
        )
        if should_trace:
            trace: dict[str, float | int] = {
                "epoch": epoch,
                "batch": batch_index,
                "global_step": global_step,
                "lr": lr,
                "window_samples": window_total,
                **{name: value / window_total for name, value in window.items()},
            }
            stop_epoch = False
            if trace_callback is not None:
                callback_values = trace_callback(model, trace)
                stop_epoch = bool(callback_values.pop("_stop_epoch", False))
                trace.update(callback_values)
                model.train()
            traces.append(trace)
            window = {name: 0.0 for name in totals}
            window_total = 0
            if stop_epoch:
                if continue_with_pgd_after_stop:
                    switched_to_pgd = True
                else:
                    break

    metrics = {
        name: value / (total if name in ("train_adv_loss", "train_adv_acc") else diagnostic_total)
        for name, value in totals.items()
    }
    metrics.update(
        {
            "epoch_seconds": time.perf_counter() - start_time,
            "lr_start": float(first_lr or 0.0),
            "lr_end": float(last_lr or 0.0),
            "batches_processed": batch_index,
            "epoch_completed": int(batch_index == len(loader)),
            "post_trigger_pgd_batches": post_trigger_pgd_batches,
        }
    )
    return metrics, global_step, traces


def pgd_adversarial_train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    epsilon: float,
    attack_step_size: float,
    attack_steps: int,
    global_step: int,
    total_steps: int,
    max_lr: float,
) -> tuple[dict[str, float], int]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    start = time.perf_counter()
    use_amp = device.type == "cuda"
    first_lr = None
    last_lr = None

    for images, labels in loader:
        progress = global_step / max(total_steps, 1)
        lr = max_lr * (2.0 * progress if progress <= 0.5 else 2.0 * (1.0 - progress))
        lr = max(lr, 0.0)
        for group in optimizer.param_groups:
            group["lr"] = lr
        first_lr = lr if first_lr is None else first_lr
        last_lr = lr

        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        adversarial = pgd_linf(
            model,
            images,
            labels,
            epsilon=epsilon,
            step_size=attack_step_size,
            steps=attack_steps,
            random_start=True,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(adversarial)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        correct += logits.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
        global_step += 1

    return (
        {
            "train_adv_loss": total_loss / total,
            "train_adv_acc": correct / total,
            "epoch_seconds": time.perf_counter() - start,
            "lr_start": float(first_lr or 0.0),
            "lr_end": float(last_lr or 0.0),
        },
        global_step,
    )


def pgd_adversarial_train_fixed_lr_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    epsilon: float,
    attack_step_size: float,
    attack_steps: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    start = time.perf_counter()
    use_amp = device.type == "cuda"
    lr = float(optimizer.param_groups[0]["lr"])

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        adversarial = pgd_linf(
            model,
            images,
            labels,
            epsilon=epsilon,
            step_size=attack_step_size,
            steps=attack_steps,
            random_start=True,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(adversarial)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * labels.size(0)
        correct += logits.argmax(1).eq(labels).sum().item()
        total += labels.size(0)

    return {
        "train_adv_loss": total_loss / total,
        "train_adv_acc": correct / total,
        "epoch_seconds": time.perf_counter() - start,
        "lr_start": lr,
        "lr_end": lr,
    }


def trades_train_fixed_lr_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    epsilon: float,
    attack_step_size: float,
    attack_steps: int,
    beta: float,
) -> dict[str, float]:
    """Train one epoch with the TRADES natural-plus-KL objective."""
    model.train()
    total_loss = 0.0
    total_natural_loss = 0.0
    total_robust_kl = 0.0
    correct = 0
    total = 0
    start = time.perf_counter()
    use_amp = device.type == "cuda"
    lr = float(optimizer.param_groups[0]["lr"])

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        adversarial = trades_linf(
            model,
            images,
            epsilon=epsilon,
            step_size=attack_step_size,
            steps=attack_steps,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            natural_logits = model(images)
            adversarial_logits = model(adversarial)
            natural_loss = nn.functional.cross_entropy(natural_logits, labels)
            robust_kl = nn.functional.kl_div(
                adversarial_logits.float().log_softmax(dim=1),
                natural_logits.float().softmax(dim=1),
                reduction="batchmean",
            )
            loss = natural_loss + beta * robust_kl
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_natural_loss += natural_loss.item() * batch_size
        total_robust_kl += robust_kl.item() * batch_size
        correct += natural_logits.argmax(1).eq(labels).sum().item()
        total += batch_size

    return {
        "train_trades_loss": total_loss / total,
        "train_natural_loss": total_natural_loss / total,
        "train_robust_kl": total_robust_kl / total,
        "train_clean_acc": correct / total,
        "epoch_seconds": time.perf_counter() - start,
        "lr_start": lr,
        "lr_end": lr,
    }
