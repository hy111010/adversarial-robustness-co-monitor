from __future__ import annotations

import argparse
import copy
import time
from collections import deque
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn

from robust_exp.attacks import pgd_linf
from robust_exp.data import make_loaders
from robust_exp.engine import (
    evaluate,
    fast_adversarial_train_one_epoch_traced,
    pgd_adversarial_train_one_epoch,
)
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


TRAIN_METRIC_FIELDS = (
    "train_adv_loss",
    "train_adv_acc",
    "free_gradient_cosine",
    "free_gradient_sign_agreement",
    "free_prediction_disagreement",
    "free_attack_loss_gain",
    "free_boundary_fraction",
    "free_delta_linf",
    "control_gradient_participation_ratio",
    "control_gradient_entropy",
    "epoch_seconds",
    "lr_start",
    "lr_end",
    "batches_processed",
    "epoch_completed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive cosine-drop warning with rollback and PGD-2 recovery"
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar10")
    parser.add_argument("--output-root", default="outputs_publication")
    parser.add_argument("--run-name", default="adaptive_rollback_pgd2_seed17")
    parser.add_argument(
        "--architecture", default="preact_resnet18", choices=("preact_resnet18", "resnet18")
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-lr", type=float, default=0.3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--fgsm-step-size", type=float, default=8 / 255)
    parser.add_argument("--fgsm-init", choices=("zero", "random_uniform"), default="zero")
    parser.add_argument("--recovery-step-size", type=float, default=4 / 255)
    parser.add_argument("--recovery-steps", type=int, default=2)
    parser.add_argument("--trace-interval", type=int, default=20)
    parser.add_argument("--detector-window", type=int, default=5)
    parser.add_argument("--detector-min-history", type=int, default=3)
    parser.add_argument("--detector-threshold", type=float, default=0.16632988750934596)
    parser.add_argument(
        "--detector",
        choices=("adaptive_cosine", "pgd_validation"),
        default="adaptive_cosine",
    )
    parser.add_argument("--pgd-detector-drop", type=float, default=0.10)
    parser.add_argument("--detector-pgd-batches", type=int, default=1)
    parser.add_argument("--start-in-rescue-mode", action="store_true")
    parser.add_argument(
        "--observe-only",
        action="store_true",
        help="record frozen-detector alarms without stopping or changing FGSM training",
    )
    parser.add_argument(
        "--disable-rollback",
        action="store_true",
        help="after an alarm, retain partial-epoch FGSM updates and begin PGD-2 next epoch",
    )
    parser.add_argument(
        "--immediate-switch-no-replay",
        action="store_true",
        help="retain warning-prefix FGSM updates and begin PGD-2 on the next mini-batch",
    )
    parser.add_argument(
        "--fixed-trigger-epoch",
        type=int,
        help="ignore the detector and begin PGD-2 at this epoch (timing ablation)",
    )
    parser.add_argument(
        "--observer-pgd-period",
        type=int,
        default=0,
        help="in observe-only mode, label every Nth trace window with fixed-batch PGD-10",
    )
    parser.add_argument(
        "--observer-pgd-followup-windows",
        type=int,
        default=3,
        help="densely label this many windows after each observe-only warning",
    )
    parser.add_argument("--pgd-step-size", type=float, default=2 / 255)
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--pgd-monitor-batches", type=int, default=10)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.observe_only and (
        args.disable_rollback or args.immediate_switch_no_replay or args.fixed_trigger_epoch is not None
    ):
        raise ValueError("--observe-only cannot be combined with intervention ablations")
    if args.disable_rollback and args.immediate_switch_no_replay:
        raise ValueError("choose either next-epoch switching or immediate no-replay switching")
    if args.fixed_trigger_epoch is not None and args.fixed_trigger_epoch < 1:
        raise ValueError("--fixed-trigger-epoch must be at least 1")
    if args.start_in_rescue_mode and args.fixed_trigger_epoch is not None:
        raise ValueError("choose either --start-in-rescue-mode or --fixed-trigger-epoch")
    if args.observer_pgd_period < 0 or args.observer_pgd_followup_windows < 0:
        raise ValueError("observer PGD labeling arguments must be non-negative")
    if args.observer_pgd_period > 0 and not args.observe_only:
        raise ValueError("--observer-pgd-period requires --observe-only")
    seed_everything(args.seed)
    device = choose_device(args.device)
    run_dir = Path(args.output_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    trace_path = run_dir / "detector_traces.csv"
    if metrics_path.exists():
        raise FileExistsError(f"{metrics_path} already exists; choose a new run name")

    write_json(
        run_dir / "config.json",
        {
            **vars(args),
            "resolved_device": str(device),
            "torch": torch.__version__,
            "method": (
                "frozen adaptive detector observation only"
                if args.observe_only
                else (
                    f"fixed epoch {args.fixed_trigger_epoch} switch + PGD-2"
                    if args.fixed_trigger_epoch is not None
                    else f"{args.detector} warning + "
                    f"{'immediate switch without replay' if args.immediate_switch_no_replay else ('no rollback' if args.disable_rollback else 'epoch-start rollback')} "
                    "+ PGD-2 recovery"
                )
            ),
            "detector_uses_validation_attack": args.detector == "pgd_validation",
            "detector_extra_forward_backward_passes": (
                args.pgd_steps * args.detector_pgd_batches
                if args.detector == "pgd_validation"
                else 0
            ),
            "threshold_status": "frozen from development seeds 17, 23, 42",
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
    best_pgd = -1.0
    rescue_mode = args.start_in_rescue_mode
    trigger_epoch: int | None = None
    trigger_global_step: int | None = None
    cosine_history: deque[float] = deque(maxlen=args.detector_window)
    best_detector_pgd = -1.0
    observer_followup_remaining = 0
    observer_trace_index = 0

    pgd_monitor = partial(
        pgd_linf,
        epsilon=args.epsilon,
        step_size=args.pgd_step_size,
        steps=args.pgd_steps,
        random_start=True,
    )
    detector_images = None
    detector_labels = None
    if args.detector == "pgd_validation" or args.observer_pgd_period > 0:
        image_parts = []
        label_parts = []
        for batch_index, (images, labels) in enumerate(val_loader):
            if batch_index >= args.detector_pgd_batches:
                break
            image_parts.append(images)
            label_parts.append(labels)
        if not image_parts:
            raise RuntimeError("No validation samples were available for the PGD detector")
        detector_images = torch.cat(image_parts).to(device, non_blocking=True)
        detector_labels = torch.cat(label_parts).to(device, non_blocking=True)

    for epoch in range(1, args.epochs + 1):
        fixed_switch_this_epoch = bool(
            not rescue_mode
            and args.fixed_trigger_epoch is not None
            and epoch >= args.fixed_trigger_epoch
        )
        if fixed_switch_this_epoch:
            rescue_mode = True
            trigger_epoch = epoch
            trigger_global_step = global_step
        attack_mode = "pgd2" if rescue_mode else f"fgsm_{args.fgsm_init}"
        should_snapshot = (
            not rescue_mode
            and not args.observe_only
            and not args.disable_rollback
            and not args.immediate_switch_no_replay
        )
        epoch_start_model = copy.deepcopy(model.state_dict()) if should_snapshot else None
        epoch_start_optimizer = copy.deepcopy(optimizer.state_dict()) if should_snapshot else None
        epoch_start_scaler = copy.deepcopy(scaler.state_dict()) if should_snapshot else None
        epoch_start_rng = capture_rng_state(train_loader.generator) if should_snapshot else None
        epoch_start_step = global_step
        triggered_this_epoch = fixed_switch_this_epoch
        abandoned_batches = 0
        abandoned_fgsm_seconds = 0.0

        if rescue_mode:
            metrics, global_step = pgd_adversarial_train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                scaler,
                args.epsilon,
                args.recovery_step_size,
                args.recovery_steps,
                global_step,
                total_steps,
                args.max_lr,
            )
        else:

            def detector_callback(
                current_model: nn.Module, trace: dict[str, float | int]
            ) -> dict[str, float | int]:
                nonlocal triggered_this_epoch, trigger_global_step, best_detector_pgd
                nonlocal observer_followup_remaining, observer_trace_index
                cosine = float(trace["free_gradient_cosine"])
                baseline = (
                    float(np.median(cosine_history))
                    if len(cosine_history) >= args.detector_min_history
                    else float("nan")
                )
                score = baseline - cosine if np.isfinite(baseline) else float("nan")
                detector_pgd_acc = float("nan")
                detector_seconds = 0.0
                # Fixed-schedule controls retain identical trace instrumentation
                # but must ignore the adaptive warning. Otherwise an early alarm
                # silently turns the control back into the proposed method.
                if args.fixed_trigger_epoch is not None:
                    triggered = False
                elif args.detector == "adaptive_cosine":
                    triggered = bool(
                        np.isfinite(score) and score >= args.detector_threshold
                    )
                else:
                    assert detector_images is not None and detector_labels is not None
                    detector_seed = args.seed * 1_000_000 + int(trace["global_step"])
                    detector_start = time.perf_counter()
                    with isolated_torch_rng(device, detector_seed):
                        model.eval()
                        adversarial = pgd_monitor(model, detector_images, detector_labels)
                        with torch.no_grad():
                            detector_pgd_acc = float(
                                model(adversarial)
                                .argmax(1)
                                .eq(detector_labels)
                                .float()
                                .mean()
                                .item()
                            )
                    model.train()
                    detector_seconds = time.perf_counter() - detector_start
                    triggered = bool(
                        best_detector_pgd >= 0
                        and best_detector_pgd - detector_pgd_acc >= args.pgd_detector_drop
                    )
                    best_detector_pgd = max(best_detector_pgd, detector_pgd_acc)
                cosine_history.append(cosine)
                if triggered:
                    triggered_this_epoch = True
                    if trigger_global_step is None:
                        trigger_global_step = int(trace["global_step"])
                observer_trace_index += 1
                observer_pgd_acc = float("nan")
                observer_pgd_seconds = 0.0
                scheduled_observer_label = bool(
                    args.observer_pgd_period > 0
                    and observer_trace_index % args.observer_pgd_period == 0
                )
                followup_observer_label = observer_followup_remaining > 0
                if triggered and args.observer_pgd_period > 0:
                    observer_followup_remaining = max(
                        observer_followup_remaining,
                        args.observer_pgd_followup_windows,
                    )
                if scheduled_observer_label or followup_observer_label or (
                    triggered and args.observer_pgd_period > 0
                ):
                    assert detector_images is not None and detector_labels is not None
                    observer_seed = args.seed * 10_000_000 + int(trace["global_step"])
                    observer_start = time.perf_counter()
                    with isolated_torch_rng(device, observer_seed):
                        current_model.eval()
                        adversarial = pgd_monitor(
                            current_model, detector_images, detector_labels
                        )
                        with torch.no_grad():
                            observer_pgd_acc = float(
                                current_model(adversarial)
                                .argmax(1)
                                .eq(detector_labels)
                                .float()
                                .mean()
                                .item()
                            )
                    current_model.train()
                    observer_pgd_seconds = time.perf_counter() - observer_start
                    if followup_observer_label and not triggered:
                        observer_followup_remaining -= 1
                return {
                    "detector_baseline": baseline,
                    "detector_score": score,
                    "detector_threshold": args.detector_threshold,
                    "detector_pgd10_acc": detector_pgd_acc,
                    "detector_seconds": detector_seconds,
                    "observer_pgd10_acc": observer_pgd_acc,
                    "observer_pgd10_seconds": observer_pgd_seconds,
                    "intervention_triggered": int(triggered),
                    "_stop_epoch": int(
                        triggered and not args.observe_only and not args.disable_rollback
                    ),
                }

            fgsm_metrics, partial_step, traces = fast_adversarial_train_one_epoch_traced(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                scaler,
                args.epsilon,
                args.fgsm_step_size,
                global_step,
                total_steps,
                args.max_lr,
                epoch,
                args.trace_interval,
                args.fgsm_init,
                detector_callback,
                continue_with_pgd_after_stop=args.immediate_switch_no_replay,
                recovery_step_size=args.recovery_step_size,
                recovery_steps=args.recovery_steps,
            )
            for trace in traces:
                append_csv(trace_path, trace)

            if triggered_this_epoch and not args.observe_only:
                abandoned_batches = int(fgsm_metrics["batches_processed"])
                abandoned_fgsm_seconds = float(fgsm_metrics["epoch_seconds"])
                rescue_mode = True
                trigger_epoch = epoch
                if args.disable_rollback:
                    abandoned_batches = 0
                    abandoned_fgsm_seconds = 0.0
                    attack_mode = "fgsm_warning_then_pgd2_next_epoch"
                    metrics = fgsm_metrics
                    global_step = partial_step
                elif args.immediate_switch_no_replay:
                    abandoned_batches = 0
                    abandoned_fgsm_seconds = 0.0
                    attack_mode = "fgsm_then_pgd2_same_epoch_no_replay"
                    metrics = fgsm_metrics
                    global_step = partial_step
                else:
                    assert epoch_start_model is not None
                    assert epoch_start_optimizer is not None
                    assert epoch_start_scaler is not None
                    model.load_state_dict(epoch_start_model)
                    optimizer.load_state_dict(epoch_start_optimizer)
                    scaler.load_state_dict(epoch_start_scaler)
                    restore_rng_state(epoch_start_rng, train_loader.generator)
                    global_step = epoch_start_step
                    attack_mode = "pgd2_after_warning"
                    metrics, global_step = pgd_adversarial_train_one_epoch(
                        model,
                        train_loader,
                        optimizer,
                        criterion,
                        device,
                        scaler,
                        args.epsilon,
                        args.recovery_step_size,
                        args.recovery_steps,
                        global_step,
                        total_steps,
                        args.max_lr,
                    )
            else:
                metrics = fgsm_metrics
                global_step = partial_step

        if triggered_this_epoch and trigger_epoch is None:
            trigger_epoch = epoch

        monitor_seed = args.seed * 100_000 + epoch
        with isolated_torch_rng(device, monitor_seed):
            clean_start = time.perf_counter()
            val_clean = evaluate(model, val_loader, device)
            clean_seconds = time.perf_counter() - clean_start
            pgd_start = time.perf_counter()
            val_pgd = evaluate(
                model,
                val_loader,
                device,
                attack=pgd_monitor,
                max_batches=args.pgd_monitor_batches,
            )
            pgd_seconds = time.perf_counter() - pgd_start

        normalized_metrics = {
            field: metrics.get(field, float("nan")) for field in TRAIN_METRIC_FIELDS
        }
        row = {
            "epoch": epoch,
            "attack_mode": attack_mode,
            **normalized_metrics,
            "val_clean_acc": val_clean,
            "val_pgd10_acc": val_pgd,
            "intervention_triggered": int(triggered_this_epoch),
            "trigger_epoch": trigger_epoch,
            "trigger_global_step": trigger_global_step,
            "rolled_back_to_global_step": (
                epoch_start_step
                if triggered_this_epoch
                and not args.disable_rollback
                and not args.immediate_switch_no_replay
                else ""
            ),
            "pgd2_start_global_step": (
                trigger_global_step
                if triggered_this_epoch
                and (args.immediate_switch_no_replay or not args.disable_rollback)
                else ""
            ),
            "post_trigger_pgd_batches": int(metrics.get("post_trigger_pgd_batches", 0)),
            "abandoned_fgsm_batches": abandoned_batches,
            "abandoned_fgsm_seconds": abandoned_fgsm_seconds,
            "total_training_seconds": float(metrics["epoch_seconds"]) + abandoned_fgsm_seconds,
            "clean_eval_seconds": clean_seconds,
            "pgd_monitor_seconds": pgd_seconds,
            "pgd_monitor_used_for_trigger": 0,
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
            "rescue_mode": rescue_mode,
            "trigger_epoch": trigger_epoch,
            "trigger_global_step": trigger_global_step,
            "best_pgd": max(best_pgd, val_pgd),
            "args": vars(args),
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if val_pgd > best_pgd:
            best_pgd = val_pgd
            torch.save(checkpoint, run_dir / "best_robust.pt")
        print(
            f"epoch={epoch:03d} mode={attack_mode} clean={val_clean:.4f} "
            f"pgd10={val_pgd:.4f} trigger={int(triggered_this_epoch)} "
            f"abandoned={abandoned_batches} "
            f"train_s={float(metrics['epoch_seconds']) + abandoned_fgsm_seconds:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
