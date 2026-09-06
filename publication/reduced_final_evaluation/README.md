# Unified final attack evaluation

All models use CIFAR-10, PreActResNet-18 and L-inf epsilon=8/255. Clean/FGSM/PGD evaluations use all 10,000 test images; official AutoAttack uses the same fixed first 1,000 test images for every model.

| Model | clean | fgsm_8_255 | pgd10_8_255 | pgd20_r1_8_255 | pgd50_r1_8_255 | pgd20_r5_8_255 | official_autoattack_standard |
|---|---:|---:|---:|---:|---:|---:|---:|
| Adaptive intervention | 82.97% | 53.12% | 47.20% | 45.89% | 45.46% | 45.65% | 42.60% |
| PGD-10 adversarial training | 82.47% | 56.36% | 51.76% | 50.98% | 50.77% | 50.84% | 48.70% |
| Fast FGSM-RS | 82.48% | 54.15% | 47.77% | 46.49% | 46.24% | 46.36% | 43.00% |
