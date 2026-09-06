# Adaptive intervention: preliminary held-out results

## Scope

These are matched CIFAR-10 validation results for held-out seeds 101 and 202 on
PreActResNet-18. They are not final test-set or AutoAttack results. The detector
threshold was frozen before both runs. The intervention rolls back to the start of
the trigger epoch and uses PGD-2 for the restarted epoch and all remaining epochs.
The epoch-level PGD-10 evaluation is recorded for analysis and checkpoint selection
but is not used by the adaptive detector.

## Results

| Method | Clean accuracy | PGD-10 accuracy | Core training time |
|---|---:|---:|---:|
| Unprotected zero-init FGSM | 81.81 ± 1.34% | 0.16 ± 0.22% | 15.64 ± 0.23 min |
| Adaptive warning + rollback + PGD-2 | 83.26 ± 0.20% | 46.64 ± 0.44% | 18.64 ± 1.25 min |

Values are mean ± sample standard deviation over two seeds. Relative to the matched
unprotected runs, the intervention increases final validation PGD-10 accuracy by
46.48 percentage points and clean accuracy by 1.45 points. Core training time rises
by 19.1%. The early false alarm in seed 202 triggers PGD-2 at epoch 13, explaining
most of the between-seed compute difference; seed 101 triggers at epoch 21.

## Interpretation

The intervention prevents the terminal robustness collapse on both held-out runs.
This is encouraging but still preliminary: the remaining three intervention seeds,
the matched PGD-validation detector, full PGD-2, a second architecture, and strong
final attacks are required before making a publication claim.
