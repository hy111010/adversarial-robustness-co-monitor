from __future__ import annotations

import torch
import torch.nn.functional as F

from robust_exp.attacks import fgsm, pgd_linf
from robust_exp.models import build_model
from robust_exp.utils import choose_device, seed_everything


def main() -> None:
    seed_everything(17)
    device = choose_device("auto")
    model = build_model().to(device)
    model.eval()
    images = torch.rand(4, 3, 32, 32, device=device)
    labels = torch.randint(0, 10, (4,), device=device)

    logits = model(images)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    model.zero_grad(set_to_none=True)

    epsilon = 8 / 255
    fgsm_images = fgsm(model, images, labels, epsilon)
    pgd_images = pgd_linf(model, images, labels, epsilon, 2 / 255, steps=3)
    for name, adversarial in (("FGSM", fgsm_images), ("PGD", pgd_images)):
        max_delta = (adversarial - images).abs().max().item()
        assert adversarial.min().item() >= 0.0 and adversarial.max().item() <= 1.0
        assert max_delta <= epsilon + 1e-6
        print(f"{name}: max_linf={max_delta:.6f}")

    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print("smoke test passed")


if __name__ == "__main__":
    main()
