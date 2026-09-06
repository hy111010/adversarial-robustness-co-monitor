from __future__ import annotations

import argparse
import copy
from functools import partial
from pathlib import Path

import torch
from torch import nn

from robust_exp.attacks import fgsm, pgd_linf
from robust_exp.data import make_loaders
from robust_exp.engine import evaluate, fast_adversarial_train_one_epoch, pgd_adversarial_train_one_epoch
from robust_exp.models import build_model
from robust_exp.utils import append_csv, choose_device, seed_everything, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PGD-monitored rollback with low-cost PGD-2 recovery")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--run-name", default="rollback_pgd2_seed17")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-lr", type=float, default=0.2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--fgsm-step-size", type=float, default=10 / 255)
    parser.add_argument("--recovery-step-size", type=float, default=4 / 255)
    parser.add_argument("--recovery-steps", type=int, default=2)
    parser.add_argument("--monitor-step-size", type=float, default=2 / 255)
    parser.add_argument("--monitor-steps", type=int, default=10)
    parser.add_argument("--monitor-batches", type=int, default=10)
    parser.add_argument("--drop-threshold", type=float, default=0.10)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--force-rescue-epoch", type=int, help="Development-only trigger for pipeline testing")
    parser.add_argument(
        "--disable-rollback",
        action="store_true",
        help="Ablation: switch to PGD-2 after detection without restoring the best checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    run_dir = Path(args.output_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "config.json",
        {**vars(args), "resolved_device": str(device), "torch": torch.__version__, "method": "rollback_pgd2"},
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
    best_epoch = 0
    best_model_state = None
    best_optimizer_state = None
    rescue_mode = False
    trigger_epoch = None

    fgsm_attack = partial(fgsm, epsilon=args.epsilon)
    monitor_attack = partial(
        pgd_linf,
        epsilon=args.epsilon,
        step_size=args.monitor_step_size,
        steps=args.monitor_steps,
        random_start=True,
    )

    for epoch in range(1, args.epochs + 1):
        attack_mode = "pgd2" if rescue_mode else "fgsm_rs"
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
            metrics, global_step = fast_adversarial_train_one_epoch(
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
            )

        val_clean = evaluate(model, val_loader, device)
        val_fgsm = evaluate(model, val_loader, device, attack=fgsm_attack, max_batches=args.monitor_batches)
        val_pgd = evaluate(model, val_loader, device, attack=monitor_attack, max_batches=args.monitor_batches)
        forced = args.force_rescue_epoch == epoch
        collapse = not rescue_mode and best_pgd >= 0 and (best_pgd - val_pgd >= args.drop_threshold or forced)
        rolled_back_to = ""

        if not collapse and val_pgd > best_pgd:
            best_pgd = val_pgd
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())

        if collapse:
            if best_model_state is None or best_optimizer_state is None:
                raise RuntimeError("Recovery triggered before a valid robust checkpoint was saved")
            if args.disable_rollback:
                rolled_back_to = "disabled"
            else:
                rolled_back_to = best_epoch
                model.load_state_dict(best_model_state)
                optimizer.load_state_dict(best_optimizer_state)
            rescue_mode = True
            trigger_epoch = epoch

        row = {
            "epoch": epoch,
            "attack_mode": attack_mode,
            **metrics,
            "val_clean_acc": val_clean,
            "val_fgsm_acc": val_fgsm,
            "val_pgd10_acc": val_pgd,
            "collapse_triggered": int(collapse),
            "rolled_back_to_epoch": rolled_back_to,
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
            "rescue_mode": rescue_mode,
            "trigger_epoch": trigger_epoch,
            "args": vars(args),
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if not collapse and val_pgd >= best_pgd:
            torch.save(checkpoint, run_dir / "best_robust.pt")

        if collapse and args.disable_rollback:
            event = " switch=PGD2(no rollback)"
        elif collapse:
            event = f" rollback={best_epoch}->PGD2"
        else:
            event = ""
        print(
            f"epoch={epoch:02d} mode={attack_mode} loss={metrics['train_adv_loss']:.4f} "
            f"clean={val_clean:.4f} fgsm={val_fgsm:.4f} pgd10={val_pgd:.4f}{event} "
            f"seconds={metrics['epoch_seconds']:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
