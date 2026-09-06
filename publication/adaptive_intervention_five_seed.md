# Adaptive intervention: five-seed development result

## Matched result

| Method | Final clean accuracy | Final PGD-10 accuracy | Core training time |
|---|---:|---:|---:|
| Unprotected zero-init FGSM | 83.27 ± 1.77% | 0.14 ± 0.19% | 15.53 ± 0.16 min |
| Adaptive warning + rollback + PGD-2 | 83.14 ± 0.62% | 46.83 ± 0.55% | 18.68 ± 0.70 min |

Values are mean ± sample standard deviation over seeds 17, 23, 42, 101, and 202
on the fixed CIFAR-10 validation protocol. The adaptive intervention improves final
PGD-10 accuracy by 46.69 percentage points while changing clean accuracy by -0.13
points. Its measured core-training overhead is 20.25% relative to the unprotected
FGSM traces. Validation evaluation time is excluded from both methods.

## Trigger behavior

| Seed | Trigger epoch | Trigger global step | Discarded FGSM batches before rollback |
|---:|---:|---:|---:|
| 17 | 17 | 5984 | 352 |
| 23 | 19 | 6476 | 140 |
| 42 | 19 | 6456 | 120 |
| 101 | 21 | 7100 | 60 |
| 202 | 13 | 4524 | 300 |

All five runs triggered once and finished with PGD-10 accuracy between 46.17% and
47.34%. Seed 202 contains the registered detector's early false alarm and therefore
switches to PGD-2 sooner; this increases compute but does not destroy clean or robust
accuracy.

## Claim boundary

This table establishes repeatable prevention of catastrophic overfitting under the
development validation attack. It does not yet establish superiority to a PGD-based
monitor or full PGD-2, and it is not a final test-set result. Those controls and the
registered second-architecture experiment are running in the chained queues.
