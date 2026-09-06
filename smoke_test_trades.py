from __future__ import annotations

import torch

from robust_exp.attacks import trades_linf
from robust_exp.models import build_model
from robust_exp.utils import seed_everything


def main() -> None:
    seed_everything(17)
    model = build_model(architecture="preact_resnet18")
    model.train()
    images = torch.rand(2, 3, 32, 32)
    epsilon = 8 / 255
    adversarial = trades_linf(model, images, epsilon, 2 / 255, steps=2)
    maximum_delta = (adversarial - images).abs().max().item()
    assert model.training
    assert maximum_delta <= epsilon + 1e-6
    assert adversarial.min().item() >= 0.0
    assert adversarial.max().item() <= 1.0
    assert all(parameter.grad is None for parameter in model.parameters())
    print({"max_linf_delta": maximum_delta, "mode_restored": model.training})


if __name__ == "__main__":
    main()
