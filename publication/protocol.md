# Publication-grade experimental protocol (frozen v1)

## Status

All existing Day 1–4 runs are pilot experiments. They remain useful for debugging and hypothesis generation but are not publication main results.

## Research question under validation

Can catastrophic overfitting in fast adversarial training be forecast using a substantially cheaper signal than iterative-PGD validation, early enough to enable a compute-budgeted response without sacrificing robustness?

This is a candidate question, not yet a novelty claim. It will be retained only after a broader literature audit and a multi-seed forecasting study.

## Primary protocol

- Dataset: CIFAR-10; secondary validation on CIFAR-100 or SVHN.
- Primary architecture: PreActResNet-18; second architecture required for final submission.
- Threat model: untargeted L-infinity, epsilon 8/255.
- Optimizer: SGD, momentum 0.9, weight decay 5e-4.
- Strong training reference: PGD-10, step size 2/255, 110 epochs, LR 0.1 with 10x drops at epochs 100 and 105.
- Model selection: fixed validation split, best validation PGD-10 checkpoint.
- Final evaluation: full 10,000-example official AutoAttack standard plus PGD-50 with 10 restarts.
- Repetitions: at least 3 seeds for development; 5 seeds for final central claims where feasible.
- Report: mean, sample standard deviation, training wall time, per-epoch cost, and attack configuration.

## Data-use policy

- Development and forecasting runs use a deterministic 45,000/5,000 split of the
  CIFAR-10 training set. The official test set is not used for tuning or checkpoint selection.
- After the method, thresholds, and epoch rule are frozen, final runs will use a
  49,000/1,000 training/monitor split for every method that requires online monitoring.
- Test attacks are run once per frozen final checkpoint. Development results and
  final results will be labeled separately in every table.

## Collapse-forecast study (pre-specified before formal traces)

- Resolution: non-overlapping windows of 20 optimizer updates.
- Ground-truth label: PGD-10 accuracy on a fixed held-out validation batch, with an
  independently derived deterministic attack seed at every window.
- Collapse event: a decline of at least 20 percentage points from the run's prior
  PGD peak and an absolute PGD-10 accuracy at or below 10% within one window.
- Candidate predictors: endpoint-gradient cosine/sign agreement, start-versus-end
  prediction disagreement, attack loss gain, boundary fraction, and perturbation norm.
- Cheap controls: adversarial training loss/accuracy, learning rate, gradient
  concentration, and adjacent-window loss change.
- Forecast horizons: 1, 2, and 3 windows before the first collapse event.
- Report: event recall, false alarms per run, median warning lead time, AUROC, AUPRC,
  detector wall time, and additional forward/backward-pass count.
- Thresholds are selected only on development seeds and then frozen for held-out seeds.

## Required baselines

1. Standard training.
2. PGD adversarial training.
3. TRADES.
4. Fast FGSM-RS.
5. FastAdv+ or FastAdvW.
6. GradAlign and/or a recent strong fast-AT method with public code.

## Novelty gate

The proposed method must satisfy all of the following before it is called a contribution:

1. It is distinguishable from FastAdv+, adaptive step-size training, adaptive norm selection, GradAlign, ConvergeSmooth, and recent label-information methods.
2. Its detection signal predicts collapse before a PGD robustness drop across many independent seeds.
3. It reduces monitoring/training compute under a fixed robustness target.
4. Ablations isolate each component.
5. The result generalizes beyond one dataset or one architecture.

## Stop conditions

- If the proposed signal does not reliably precede collapse, report it as a negative result and do not construct a method around it.
- If a claimed gain disappears under full AutoAttack or multiple seeds, remove the claim.
- Test-set results must never be used for checkpoint or hyperparameter selection.
