from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .models import build_model


def load_model_checkpoint(
    path: str | Path, device: torch.device
) -> tuple[dict[str, Any], nn.Module, str]:
    """Load a trusted local checkpoint and reconstruct its recorded architecture."""
    checkpoint_path = Path(path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(f"{checkpoint_path} is not a supported experiment checkpoint")
    saved_args = checkpoint.get("args", {})
    architecture = checkpoint.get("architecture", saved_args.get("architecture", "resnet18"))
    num_classes = int(checkpoint.get("num_classes", saved_args.get("num_classes", 10)))
    dataset = checkpoint.get("dataset", saved_args.get("dataset", "cifar10"))
    model = build_model(num_classes=num_classes, architecture=architecture, dataset=dataset).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint, model, architecture
