# Novelty audit: fast adversarial training collapse forecasting

Status: working audit, not a novelty claim. Last updated 2026-09-02.

## Candidate being tested

Use signals already available inside an FGSM-RS update—especially the alignment of
the input gradient at the random-start point with the endpoint input gradient
obtained during the ordinary parameter backward pass—to forecast catastrophic
overfitting. The intended distinction is operational: no additional model forward
or backward pass for the signal, batch-level warning, and a response evaluated
under an explicit compute budget.

The endpoint signal may still overlap unpublished or differently named prior work.
It cannot be described as novel until targeted full-text and citation-chain searches
are complete and the forecasting experiments succeed.

## Closest prior work and exclusion boundary

| Work | Relevant idea | What cannot be claimed | Required distinction/test |
|---|---|---|---|
| Wong et al., ICLR 2020, Fast is Better than Free | FGSM-RS and PGD-based early stopping | Fast FGSM training or PGD monitoring itself | Compare signal cost and warning lead time against PGD monitoring |
| Li et al., 2020, FastAdv+/FastAdvW | Detect with validation PGD every 20 batches; temporarily or permanently switch to PGD | Detect-and-switch logic in general | Hold the response rule fixed; replace only the detector and compare compute/false alarms |
| Andriushchenko & Flammarion, NeurIPS 2020, GradAlign | Input-gradient alignment diagnoses local nonlinearity and is used as a regularizer | Gradient alignment as a new mechanism | Emphasize reuse of gradients already present in the update; benchmark against GradAlign and ablate endpoints |
| Kim et al., AAAI 2021 | Boundary distortion/high curvature and use of perturbations along the adversarial direction | Curvature/boundary explanation | Treat endpoint consistency only as a forecasting variable unless causal evidence is added |
| ZeroGrad, 2023 | Small-gradient instability; nearly costless gradient modification | Costless gradient-statistic remedy | Compare to gradient magnitude/concentration controls; do not equate detection with prevention |
| Fast-BAT, ICML 2022 | Bi-level formulation and non-sign fast training | A generic improved single-step optimizer | Include as a strong fast-training reference if implementation is reproducible |
| ConvergeSmooth, ICCV 2023 | Loss-convergence outliers and adjacent-epoch smoothing | Loss jumps as a new warning signal | Include loss-only forecasting baseline at the same temporal resolution |
| Layer-Aware AWP, ICML 2024 | Early distortion in former layers and pseudo-robust shortcuts | Layer sensitivity or weight perturbation as new | If layer signals are added, compare directly and count their overhead |
| Adaptive Norm Selection, NeurIPS 2025 workshop | Gradient concentration controls adaptive norm selection | Gradient concentration as a new cause/signal | Include participation-ratio and entropy baselines using the same gradients |
| Label Information Elimination, ICCV 2025 | Transferable label information after CO; suppression-based mitigation | Label leakage/transfer as a new mechanism | Test whether the proposed signal precedes, rather than merely reflects, post-CO behavior |

## Primary sources

- Fast is Better than Free: https://arxiv.org/abs/2001.03994
- FastAdv+/FastAdvW: https://arxiv.org/abs/2006.03089
- GradAlign: https://proceedings.neurips.cc/paper/2020/hash/b8ce47761ed7b3b6f48b583350b7f9e4-Abstract.html
- Boundary distortion analysis: https://ojs.aaai.org/index.php/AAAI/article/view/16989
- ZeroGrad: https://doi.org/10.1016/j.iswa.2023.200258
- Fast-BAT: https://proceedings.mlr.press/v162/zhang22ak.html
- ConvergeSmooth: https://openaccess.thecvf.com/content/ICCV2023/html/Zhao_Fast_Adversarial_Training_with_Smooth_Convergence_ICCV_2023_paper.html
- Layer-Aware AWP: https://proceedings.mlr.press/v235/lin24v.html
- Adaptive Norm Selection: https://openreview.net/pdf?id=isvrCnlBZb
- Label Information Elimination: https://openaccess.thecvf.com/content/ICCV2025/html/Pan_Mitigating_Catastrophic_Overfitting_in_Fast_Adversarial_Training_via_Label_Information_ICCV_2025_paper.html

## Decision rule

Advance this direction only if the no-extra-pass signal forecasts a pre-registered
PGD-collapse event earlier than cheap loss/accuracy controls across development
seeds and materially reduces detector cost relative to FastAdv+'s PGD monitor.
Otherwise retain it as a negative analysis and choose a different contribution.
