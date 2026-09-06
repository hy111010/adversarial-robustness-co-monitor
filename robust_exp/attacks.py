from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _input_gradient(model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    images = images.detach().requires_grad_(True)
    loss = F.cross_entropy(model(images), labels)
    return torch.autograd.grad(loss, images, only_inputs=True)[0]


def fgsm(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, epsilon: float) -> torch.Tensor:
    grad = _input_gradient(model, images, labels)
    return (images.detach() + epsilon * grad.sign()).clamp(0.0, 1.0)


def fgsm_random_start(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    step_size: float,
) -> torch.Tensor:
    """Random-start FGSM projected into the pixel-space L-infinity ball."""
    clean = images.detach()
    adversarial = (clean + torch.empty_like(clean).uniform_(-epsilon, epsilon)).clamp(0.0, 1.0)
    grad = _input_gradient(model, adversarial, labels)
    adversarial = adversarial.detach() + step_size * grad.sign()
    delta = (adversarial - clean).clamp(-epsilon, epsilon)
    return (clean + delta).clamp(0.0, 1.0).detach()


def fgsm_trace(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    step_size: float,
    initialization: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FGSM plus already-computed start gradients/logits for no-extra-pass diagnostics."""
    clean = images.detach()
    if initialization == "random_uniform":
        start = (clean + torch.empty_like(clean).uniform_(-epsilon, epsilon)).clamp(0.0, 1.0)
    elif initialization == "zero":
        start = clean.clone()
    else:
        raise ValueError(f"Unknown FGSM initialization: {initialization}")
    start = start.detach().requires_grad_(True)
    start_logits = model(start)
    start_loss = F.cross_entropy(start_logits, labels)
    start_gradient = torch.autograd.grad(start_loss, start, only_inputs=True)[0]
    candidate = start.detach() + step_size * start_gradient.detach().sign()
    delta = (candidate - clean).clamp(-epsilon, epsilon)
    adversarial = (clean + delta).clamp(0.0, 1.0).detach()
    return adversarial, start_gradient.detach(), start_logits.detach()


def fgsm_random_start_trace(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    step_size: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return fgsm_trace(model, images, labels, epsilon, step_size, "random_uniform")


def pgd_linf(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    step_size: float,
    steps: int,
    random_start: bool = True,
) -> torch.Tensor:
    clean = images.detach()
    if random_start:
        adversarial = clean + torch.empty_like(clean).uniform_(-epsilon, epsilon)
        adversarial = adversarial.clamp(0.0, 1.0)
    else:
        adversarial = clean.clone()

    for _ in range(steps):
        grad = _input_gradient(model, adversarial, labels)
        adversarial = adversarial.detach() + step_size * grad.sign()
        delta = (adversarial - clean).clamp(-epsilon, epsilon)
        adversarial = (clean + delta).clamp(0.0, 1.0)
    return adversarial.detach()


def pgd_linf_restarts(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    step_size: float,
    steps: int,
    restarts: int,
) -> torch.Tensor:
    """Return the highest-loss adversarial example per input across PGD restarts."""
    if restarts < 1:
        raise ValueError("restarts must be at least 1")
    best_adversarial = None
    best_loss = torch.full((images.size(0),), -torch.inf, device=images.device)
    for _ in range(restarts):
        candidate = pgd_linf(model, images, labels, epsilon, step_size, steps, random_start=True)
        with torch.no_grad():
            losses = F.cross_entropy(model(candidate), labels, reduction="none")
        replace = losses > best_loss
        best_loss = torch.where(replace, losses, best_loss)
        if best_adversarial is None:
            best_adversarial = candidate.clone()
        else:
            best_adversarial[replace] = candidate[replace]
    assert best_adversarial is not None
    return best_adversarial.detach()


def pgd_linf_margin(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    step_size: float,
    steps: int,
    random_start: bool = True,
) -> torch.Tensor:
    """L-infinity PGD maximizing the CW-style logit margin (max other - true)."""
    clean = images.detach()
    if random_start:
        adversarial = (clean + torch.empty_like(clean).uniform_(-epsilon, epsilon)).clamp(0.0, 1.0)
    else:
        adversarial = clean.clone()
    for _ in range(steps):
        adversarial = adversarial.detach().requires_grad_(True)
        logits = model(adversarial)
        true_logits = logits.gather(1, labels[:, None]).squeeze(1)
        other_logits = logits.masked_fill(
            F.one_hot(labels, num_classes=logits.size(1)).bool(), -torch.inf
        ).max(dim=1).values
        margin = other_logits - true_logits
        grad = torch.autograd.grad(margin.mean(), adversarial, only_inputs=True)[0]
        adversarial = adversarial.detach() + step_size * grad.sign()
        delta = (adversarial - clean).clamp(-epsilon, epsilon)
        adversarial = (clean + delta).clamp(0.0, 1.0)
    return adversarial.detach()


def trades_linf(
    model: nn.Module,
    images: torch.Tensor,
    epsilon: float,
    step_size: float,
    steps: int,
) -> torch.Tensor:
    """Generate the KL-maximizing L-infinity examples used by TRADES training."""
    clean = images.detach()
    was_training = model.training
    model.eval()
    with torch.no_grad():
        clean_probabilities = model(clean).float().softmax(dim=1)
    adversarial = (clean + 0.001 * torch.randn_like(clean)).clamp(0.0, 1.0)
    for _ in range(steps):
        adversarial = adversarial.detach().requires_grad_(True)
        adversarial_log_probabilities = model(adversarial).float().log_softmax(dim=1)
        divergence = F.kl_div(
            adversarial_log_probabilities, clean_probabilities, reduction="batchmean"
        )
        gradient = torch.autograd.grad(divergence, adversarial, only_inputs=True)[0]
        adversarial = adversarial.detach() + step_size * gradient.sign()
        delta = (adversarial - clean).clamp(-epsilon, epsilon)
        adversarial = (clean + delta).clamp(0.0, 1.0)
    model.train(was_training)
    return adversarial.detach()
