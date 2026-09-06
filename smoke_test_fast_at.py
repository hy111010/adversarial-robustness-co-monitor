from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from robust_exp.attacks import fgsm_random_start
from robust_exp.engine import fast_adversarial_train_one_epoch
from robust_exp.models import build_model
from robust_exp.utils import choose_device, seed_everything


def main() -> None:
    seed_everything(17)
    device = choose_device("auto")
    model = build_model().to(device)
    images = torch.rand(16, 3, 32, 32)
    labels = torch.randint(0, 10, (16,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=8)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0, momentum=0.9, weight_decay=5e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    parameter_before = next(model.parameters()).detach().clone()

    model.train()
    probe_images, probe_labels = images[:4].to(device), labels[:4].to(device)
    adversarial = fgsm_random_start(model, probe_images, probe_labels, 8 / 255, 10 / 255)
    max_delta = (adversarial - probe_images).abs().max().item()
    assert adversarial.min().item() >= 0.0 and adversarial.max().item() <= 1.0
    assert max_delta <= 8 / 255 + 1e-6

    metrics, steps = fast_adversarial_train_one_epoch(
        model,
        loader,
        optimizer,
        nn.CrossEntropyLoss(),
        device,
        scaler,
        8 / 255,
        10 / 255,
        global_step=0,
        total_steps=len(loader),
        max_lr=0.2,
    )
    assert steps == len(loader)
    assert torch.isfinite(torch.tensor(metrics["train_adv_loss"]))
    assert not torch.equal(parameter_before, next(model.parameters()).detach())
    print(f"max_linf={max_delta:.6f}")
    print(metrics)
    print("fast AT smoke test passed")


if __name__ == "__main__":
    main()
