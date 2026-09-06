# Adaptive collapse detector: development addendum

## Status and provenance

This detector is an exploratory revision proposed after inspecting development
seeds 17, 23, and 42. Results on those seeds are therefore development results,
not independent confirmation. All choices below are frozen before evaluating new
held-out seeds.

## Frozen v2 rule

- Trace resolution: one non-overlapping window per 20 optimizer updates.
- Recent baseline: median of the preceding five windows; at least three preceding
  windows are required. The current window is never included in its own baseline.
- Cosine drop: recent median endpoint-gradient cosine minus current cosine.
- Accuracy rise: current adversarial training accuracy minus its recent median.
- Primary score: cosine drop. Its frozen development threshold is 0.1663298875.
- Secondary ablation: cosine drop plus accuracy rise. It is retained to test whether
  the training-accuracy term adds value, but it is not the primary detector.
- The score uses statistics already recorded during ordinary FGSM training and
  requires no validation PGD call.
- Threshold fitting target: the single window immediately preceding the registered
  PGD collapse event. The threshold is fit on development seeds only and then frozen.
- Development seeds: 17, 23, and 42. Held-out seeds registered before execution:
  101 and 202.
- Held-out evaluation horizons: one, two, and three windows (20, 40, and 60 updates).
- Primary held-out metrics: event recall, false alarms per run, median warning lead,
  AUROC, and AUPRC. Training accuracy rise and cosine drop are retained as component
  baselines.

## Decision gate

The combined detector advances to intervention experiments only if it detects the
collapse on held-out seeds before the registered PGD event and does not increase
false alarms relative to the best component baseline. Otherwise it is rejected or
reported as a negative result.

## Registered architecture generalization

After completing the PreActResNet-18 intervention and compute controls, the frozen
detector is evaluated without retuning on torchvision ResNet-18 using seeds 101 and
202. Both the unprotected zero-initialized FGSM trace and the matched adaptive
rollback/PGD-2 intervention are run. A failure to transfer is reported as a negative
result rather than repaired by architecture-specific threshold selection.
