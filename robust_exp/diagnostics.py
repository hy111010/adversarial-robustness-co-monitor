from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _gradient(model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    variable = images.detach().requires_grad_(True)
    loss = F.cross_entropy(model(variable), labels)
    return torch.autograd.grad(loss, variable, only_inputs=True)[0].detach()


def _rs_fgsm_with_gradient(
    model: nn.Module,
    clean: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    step_size: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    start = (clean + torch.empty_like(clean).uniform_(-epsilon, epsilon)).clamp(0.0, 1.0)
    gradient = _gradient(model, start, labels)
    candidate = start + step_size * gradient.sign()
    delta = (candidate - clean).clamp(-epsilon, epsilon)
    return (clean + delta).clamp(0.0, 1.0).detach(), gradient


def cheap_collapse_signals(
    model: nn.Module,
    loader,
    device: torch.device,
    epsilon: float,
    step_size: float,
    max_batches: int = 2,
) -> dict[str, float]:
    """Two-start FGSM diagnostics intended as candidate PGD-free warning signals."""
    model.eval()
    totals = {
        "diag_rs_prediction_disagreement": 0.0,
        "diag_gradient_cosine": 0.0,
        "diag_gradient_sign_agreement": 0.0,
        "diag_clean_rs_jsd": 0.0,
        "diag_rs1_accuracy": 0.0,
        "diag_rs2_accuracy": 0.0,
    }
    count = 0
    for batch_index, (images, labels) in enumerate(loader):
        if batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        adv1, grad1 = _rs_fgsm_with_gradient(model, images, labels, epsilon, step_size)
        adv2, grad2 = _rs_fgsm_with_gradient(model, images, labels, epsilon, step_size)
        with torch.no_grad():
            clean_logits = model(images)
            logits1 = model(adv1)
            logits2 = model(adv2)
            pred1 = logits1.argmax(1)
            pred2 = logits2.argmax(1)
            clean_log_probs = clean_logits.float().log_softmax(dim=1)
            adv_log_probs = logits1.float().log_softmax(dim=1)
            p_clean = clean_log_probs.exp()
            p_adv = adv_log_probs.exp()
            mixture_log_probs = torch.logsumexp(
                torch.stack((clean_log_probs, adv_log_probs)), dim=0
            ) - torch.log(torch.tensor(2.0, device=images.device))
            jsd = 0.5 * (
                (p_clean * (clean_log_probs - mixture_log_probs)).sum(dim=1)
                + (p_adv * (adv_log_probs - mixture_log_probs)).sum(dim=1)
            )
            flat1 = grad1.flatten(1)
            flat2 = grad2.flatten(1)
            cosine = F.cosine_similarity(flat1, flat2, dim=1)
            sign_agreement = grad1.sign().eq(grad2.sign()).float().flatten(1).mean(dim=1)
        batch_size = labels.size(0)
        totals["diag_rs_prediction_disagreement"] += pred1.ne(pred2).float().sum().item()
        totals["diag_gradient_cosine"] += cosine.sum().item()
        totals["diag_gradient_sign_agreement"] += sign_agreement.sum().item()
        totals["diag_clean_rs_jsd"] += jsd.sum().item()
        totals["diag_rs1_accuracy"] += pred1.eq(labels).float().sum().item()
        totals["diag_rs2_accuracy"] += pred2.eq(labels).float().sum().item()
        count += batch_size
    if count == 0:
        raise RuntimeError("No validation samples were available for diagnostics")
    return {name: value / count for name, value in totals.items()}
