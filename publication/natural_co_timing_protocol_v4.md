# Natural-CO and actionable-lead protocol

This stage addresses whether the detector remains useful outside zero-start FGSM.
All runs use random-start FGSM, an 8/255 L-infinity budget, a 10/255 FGSM step,
PreActResNet-18, CIFAR-10, 30 epochs, and the same evaluator. Three completed runs
with the standard triangular schedule and peak LR 0.2 provide the stable
pass-through reference. Five held-out seeds (101, 202, 303, 404, and 505) use the
same schedule with peak LR 0.3, fixed before inspecting their outcomes. This is a
learning-rate stress condition with random starts rather than a zero-start attack
construction.

The detector threshold remains frozen at 0.16632988750934596. Natural CO is defined
by the registered PGD-10 rule: a drop of at least 20 percentage points to at most
10% accuracy. Every observed natural event is reported. Detector-guided intervention
is run only when the frozen detector warns before that event, paired by seed and
configuration. Immediate epoch replay is applied to at most the first three
detected natural events. The immediate-versus-next-epoch comparison uses the
already completed five-seed paired timing experiment.

For each event, the report records warning-to-CO updates, warning-to-PGD-2-start
updates, whether CO occurred before PGD-2 started, and whether intervention started
before collapse. Stable trajectories contribute false-alarm counts. This protocol
does not treat failure to produce a natural CO event as evidence of detector success.

The five epsilon-8 held-out runs produced no natural CO. A prespecified follow-up
therefore evaluates three held-out random-start runs at epsilon 16/255 with a
20/255 FGSM step and peak LR 0.2. This larger-budget setting is motivated by prior
FGSM-RS work showing that random starts are reliable mainly at smaller budgets.
It is reported as a threat-budget stress test and is not mixed with epsilon-8
robust-accuracy comparisons.
