# Supplemental protocol v2 (frozen before execution)

## Questions

1. Does the frozen cosine-drop detector generalize beyond the two original
   held-out collapse events?
2. Is the detector useful when a standard random-start FGSM-RS run is already
   stable?
3. Which component matters: detection time, rollback, or PGD-2 recovery?

## Detector generalization

- The detector remains unchanged: 20-update windows, median of the preceding
  five windows with at least three observations, threshold 0.16632988750934596.
- No threshold or feature is refit.
- New collapse seeds: 303, 404, 505, 606, and 707.
- These are combined only at reporting time with the previously held-out seeds
  101 and 202, yielding seven held-out runs.
- Training remains unprotected zero-initialized FGSM for all 30 epochs.
- The first epoch satisfying a 20-point PGD-10 decline from the previous peak
  and absolute PGD-10 accuracy at or below 10% is the event. An alarm in the
  preceding 60 optimizer updates is a successful early warning.

## Stable FGSM-RS pass-through test

- Seeds: 17 (existing frozen trace), 23, and 42.
- Random-uniform initialization, FGSM step 10/255, epsilon 8/255, maximum
  learning rate 0.2, 30 epochs.
- The detector is observation-only. Any alarm in a non-collapsing run is a
  false alarm. This test determines whether the detector can wrap FGSM-RS
  without changing its training path when no failure is present.

## Matched component ablations

All use CIFAR-10, PreActResNet-18, 30 epochs, maximum learning rate 0.3,
epsilon 8/255, and PGD-2 recovery step size 4/255.

- Frozen detector + rollback + PGD-2: existing five seeds 17, 23, 42, 101, 202.
- Frozen detector + no rollback + PGD-2: the same five seeds. The warning epoch
  finishes under FGSM and PGD-2 begins at the next epoch.
- Fixed early switch at epoch 13: seeds 17, 101, 202.
- Fixed late switch at epoch 21: seeds 17, 101, 202.
- Full PGD-2 from epoch 1: seeds 17, 101, 202.

The two fixed schedules apply the identical PGD-2 response without using the
detector. Epoch 13 and epoch 21 are the lower and upper endpoints of trigger
epochs observed in the original five-seed study. The comparison therefore asks
whether adaptive timing improves the robustness-cost trade-off relative to a
single schedule chosen from the observed trigger range.

## Reporting rule

Rollback is retained as a contribution only if the matched no-rollback runs
show a consistent benefit. The method is described as a safety wrapper for
FGSM-RS only if the stable pass-through test produces no material false-alarm
cost. Test claims for a new primary variant require PGD-50, multi-restart PGD,
and official AutoAttack evaluation.
