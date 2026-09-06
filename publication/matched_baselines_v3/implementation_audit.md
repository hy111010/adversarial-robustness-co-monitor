# Matched baseline implementation audit

## Shared protocol

- CIFAR-10 with the existing deterministic 45,000/5,000 split.
- PreActResNet-18, 30 epochs, triangular learning-rate schedule peaking at 0.3.
- SGD, momentum 0.9, weight decay 5e-4, batch size 128, AMP on CUDA.
- Threat radius 8/255; the same validation PGD-10 and checkpoint rule are used for all methods.
- Formal results use seeds 17, 23, 42, 101, and 202.

## FastAdv+

The implementation follows Algorithm 1 and the hyperparameter paragraph in Bai Li et al.,
"Towards Understanding Fast Adversarial Training" (arXiv:2006.03089): R+FGSM is the
default update, validation robustness is checked every 20 optimizer updates, a drop of 0.10
activates a transient 20-update PGD block, and training then returns to R+FGSM. The local
implementation uses PGD-10 both for the detector and transient recovery so it is compatible
with the shared evaluator. Monitoring time and PGD training batches are recorded separately.

## GradAlign

The implementation follows Maksym Andriushchenko and Nicolas Flammarion,
"Understanding and Improving Fast Adversarial Training" (NeurIPS 2020) and the authors'
public implementation: zero-start FGSM training is augmented with
`lambda * (1 - cosine(g_clean, g_random))`, with lambda 0.2. Only the gradient at the random
point retains the second-order graph, matching the authors' lower-cost option. The second-order
branch is evaluated in FP32 for numerical stability; the adversarial classification loss uses AMP.

## Local validation

Both modes passed a CPU smoke run that exercised training, PGD validation, checkpoint writing,
and metric-schema writing. Full GPU runs are executed serially after the matched-control queue.
