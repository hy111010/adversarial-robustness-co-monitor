from __future__ import annotations

import csv
import json
import os
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = list(row.keys())
    if exists:
        with path.open("r", newline="", encoding="utf-8-sig") as existing_handle:
            reader = csv.reader(existing_handle)
            try:
                fieldnames = next(reader)
            except StopIteration as error:
                raise ValueError(f"Existing CSV is empty and has no header: {path}") from error
        unexpected = set(row) - set(fieldnames)
        if unexpected:
            raise ValueError(
                f"CSV schema mismatch for {path}; unexpected fields: {sorted(unexpected)}"
            )
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def capture_rng_state(loader_generator: torch.Generator | None = None) -> dict[str, Any]:
    """Capture epoch-boundary randomness so an interrupted run can be resumed."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    if loader_generator is not None:
        state["loader_generator"] = loader_generator.get_state()
    return state


def restore_rng_state(
    state: dict[str, Any] | None, loader_generator: torch.Generator | None = None
) -> None:
    """Restore a state produced by :func:`capture_rng_state` when available."""
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])
    if loader_generator is not None and state.get("loader_generator") is not None:
        loader_generator.set_state(state["loader_generator"].cpu())


@contextmanager
def isolated_torch_rng(device: torch.device, seed: int):
    """Run stochastic monitoring reproducibly without advancing training RNG streams."""
    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        yield
