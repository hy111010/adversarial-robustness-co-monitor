from __future__ import annotations

import torch

from robust_exp.data import make_loaders
from robust_exp.diagnostics import cheap_collapse_signals
from robust_exp.models import build_model
from robust_exp.utils import choose_device, seed_everything


def main() -> None:
    seed_everything(17)
    device = choose_device("auto")
    _, loader = make_loaders("data", 32, 0, 17, val_size=64, limit_val=64)
    model = build_model(architecture="preact_resnet18").to(device)
    values = cheap_collapse_signals(model, loader, device, 8 / 255, 10 / 255, max_batches=1)
    assert all(torch.isfinite(torch.tensor(value)) for value in values.values())
    assert 0 <= values["diag_rs_prediction_disagreement"] <= 1
    assert -1 <= values["diag_gradient_cosine"] <= 1
    print(values)


if __name__ == "__main__":
    main()
