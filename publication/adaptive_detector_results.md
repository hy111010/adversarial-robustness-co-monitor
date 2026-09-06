# Adaptive detector: frozen held-out results

## Experimental status

The adaptive rule was designed on seeds 17, 23, and 42. Its rolling window,
minimum history, feature definition, and threshold were frozen before running
held-out seeds 101 and 202. The held-out runs use the same CIFAR-10 split policy,
PreActResNet-18 architecture, zero-initialized FGSM configuration, and registered
PGD-10 collapse definition.

## Primary result

At a three-window (60-update) horizon, the frozen adaptive cosine-drop detector
recalled both held-out collapse events. It produced one earlier alarm across the
two runs, or 0.5 false alarms per run. The median warning lead was 2.5 windows.
Its pooled held-out AUROC was 0.9975 and AUPRC was 0.7794.

## Component ablation

| Detector feature | Event recall | False alarms/run | Median lead (windows) | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|
| Adaptive cosine drop (primary) | 100% | 0.5 | 2.5 | 0.9975 | 0.7794 |
| Adaptive sign-agreement drop | 100% | 0.5 | 2.5 | 0.9983 | 0.8135 |
| Cosine drop + training-accuracy rise | 100% | 0.5 | 2.0 | 0.9970 | 0.7654 |
| Adaptive training-loss drop | 100% | 4.0 | 2.0 | 0.9402 | 0.1971 |
| Adaptive training-accuracy rise | 50% | 2.5 | 2.0 | 0.8729 | 0.1612 |

The sign-agreement variant is slightly stronger numerically on two held-out
events, but it was not the registered primary endpoint. It is reported as an
ablation and not substituted post hoc for the primary detector. The main evidence
is that both gradient-consistency features have substantially fewer false alarms
and higher AUPRC than the training-statistic controls.

## Limitations

Two held-out events are not sufficient for a final generalization claim. The
detector still requires validation on a second architecture or dataset, and its
intervention benefit must be compared with a PGD-validation monitor under matched
training and response rules.
