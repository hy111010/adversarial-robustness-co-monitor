# Final experiment and claim audit

## Seed roles

| Role | Seeds | Evidence used in paper |
|---|---|---|
| CIFAR-10 detector development | 17, 23, 42 | Frozen threshold 0.1663298875 |
| CIFAR-10 held-out detector evaluation | 101, 202, 303, 404, 505, 606, 707 | Seven registered events; 7/7 within 60 updates |
| Intervention/component ablation | 17, 23, 42, 101, 202 | Immediate no-replay, immediate replay, next-epoch, fixed-switch, and full-PGD-2 controls |
| CIFAR-10 non-event controls | 17, 23, 42 | Random-start FGSM; zero alarm episodes |
| Baseline comparison | 17, 23, 42, 101, 202 | Controlled FastAdv+ and GradAlign reimplementations |
| ResNet-18 event evaluation | 303, 404, 505, 606, 707 | 4/5 detected, 2/5 within 60 updates |
| ResNet-18 non-event controls | 17, 23, 42 | Zero alarm episodes |
| CIFAR-100 development | 17, 23, 42 | Dataset-specific threshold 0.6332649834303036 |
| CIFAR-100 held-out event candidates | 101, 202, 303 | Two registered events; seed 303 is a non-event |

The 7/7 primary held-out statistic was checked against
`supplemental_v2/expanded_heldout_detector.csv`; it contains no development
seed. Development seeds do appear in the five-seed intervention ablation, as
declared.

## Non-event correction

`forecast_analysis/v3_cifar100_quick_heldout3/per_run.csv` records events for
held-out seeds 101 and 202 but no event for seed 303. Seed 303 nevertheless has
one alarm episode. Together with the two explicitly stable held-out runs, this
gives three CIFAR-100 non-event trajectories and one alarm episode. Table I and
`final_statistics_v4/detector_generalization.csv` were corrected accordingly.

## ResNet-18 Near-lead correction

The saved `v3_resnet18_heldout5/per_run.csv` records the two ResNet-18 events
within the frozen 60-update near horizon with first-warning leads of 3 and 2
trace windows, i.e., 60 and 40 optimizer updates. Their Near-lead median is
therefore 50 updates. The previous value 80 was the median first-warning lead
over all four detected events (60, 40, 100, and 480 updates), not the median
among Near events; Table I now reports 50.

## Monitor implementation

`robust_exp/attacks.py::fgsm_trace` computes the attack-start gradient needed
to construct the FGSM example. `robust_exp/engine.py::fast_adversarial_train_one_epoch_traced`
retains the endpoint input gradient from the normal parameter backward and
computes cosine similarity under `torch.no_grad()`. The monitor does not call
an auxiliary validation attack, extra forward evaluation, extra backward, or
additional `autograd.grad`. “Attack-free” therefore applies to the monitor,
not to the FGSM base-training update.

## Replay implementation and timing

`train_adaptive_intervention.py` creates one epoch-start in-memory deep copy of
model, optimizer, and AMP scaler state. It also captures Python, NumPy,
PyTorch CPU/CUDA, and loader-generator RNG state. The replay snapshot is not
written to disk. Snapshot creation and restoration occur outside the timed
training functions. Therefore 18.68 minutes is measured core training time,
not complete end-to-end time. No reliable peak-memory or snapshot-copy latency
measurement exists, so none is estimated in the paper.

## Immediate switching without replay

`publication/immediate_no_replay_v5/new_experiment_results.csv` contains all
15 method-by-seed rows for seeds 17, 23, 42, 101, and 202. The new response
keeps the warning-producing FGSM update, applies PGD-2 to the next mini-batch,
and remains on PGD-2. It intervenes before the matched unprotected CO in 5/5
cases, with zero intervening FGSM updates. Aggregate clean, PGD-10, and core
runtime are 83.06+/-0.65%, 47.05+/-0.74%, and 18.74+/-0.63 minutes.
Relative to epoch replay, its paired PGD-10 difference is +0.22 points with
95% CI [-0.22,+0.66]. Relative to next-epoch switching, it starts 157.6
updates earlier on average with CI [-252.0,-56.8]. The reported intervals use
100,000 paired bootstrap resamples and the frozen protocol.

## Epsilon-16/255 intervention identity

The 25.55%, 26.80%, and 27.11% PGD-10 values trace to
`v4_natural_eps16_immediate_seed101/202/303`. Their saved configurations name
the method `adaptive_cosine warning + epoch-start rollback + PGD-2 recovery`,
and their trigger rows record restored epoch-start steps and abandoned FGSM
batches. The manuscript therefore labels these values as the epoch-replay
variant; it does not attribute them to the primary no-replay response.

No immediate-switch/no-replay 16/255 result exists in the saved outputs. The
three `v4_natural_eps16_immediate_seed101/202/303` directories are configured
with epoch-start rollback and are therefore replay runs.

## Controlled baselines

Saved FastAdv+ configuration: one 128-image validation batch, PGD-10 with step
2/255 every 20 updates, a 0.10 drop from best observed probe accuracy as the
trigger, and a transient 20-update PGD-10 training block. The saved monitor
time is 3.1403 minutes on average; core plus online probe time is 30.4085
minutes on average.

Saved GradAlign configuration: lambda 0.2, uniform random perturbation in
[-epsilon, epsilon], cosine alignment between clean and random-point input
gradients, second-order graph on the random branch, batch size 128, SGD
momentum 0.9, weight decay 5e-4, and peak triangular learning rate 0.3.
Both baselines are controlled reimplementations, not claims of official code.

## Strong evaluation

The primary no-replay checkpoint was fixed as
`outputs_publication/v5_immediate_no_replay_seed101/best_robust.pt` before any
strong-attack result was observed. Its checkpoint timestamp predates all three
evaluation files, and no other seed or checkpoint was evaluated for
replacement. Full-test results are 82.95% clean, 46.93% PGD-10, 45.52%
PGD-50, and 45.66% PGD-20 with five restarts. AutoAttack-standard on the fixed
first 1,000 test images is complete at 42.30%.

Full-test PGD values were traced to
`reduced_final_evaluation/unified_attack_results.csv` and the corresponding
`strong_attack_results.csv` files. Immediate and next-epoch models use seed
101; Fast FGSM-RS and PGD-10 training references use seed 17. Each
`best_robust.pt` was selected by maximum recorded validation PGD-10 accuracy
within its run. The corresponding completed `official_autoattack_1000_results.csv`
records were also traced to those same four run directories and checkpoint
names. They use the first 1,000 examples in the unshuffled official CIFAR-10
test order and report 42.60% (immediate replay), 41.30% (next epoch), 43.00%
(Fast FGSM-RS), and 48.70% (PGD-10 training). The same protocol was newly run
on the seed-101 controlled FastAdv+ and GradAlign checkpoints, yielding 42.40%
and 44.40%, and on the pre-fixed primary no-replay checkpoint, yielding 42.30%.
All four AutoAttack-standard components are marked complete for each of the
seven records. These
values are reported as subset estimates, not as full-test AutoAttack. For each
run, the saved checkpoint predates both the strong-PGD and AutoAttack result
files and has not subsequently been modified. The result records identify
AutoAttack package version 0.1 and archive SHA-256
`87d423ee2794814fd003697ed1bec29b28416b6d048678a3f0983c27df5c2b1f`.

## Offline sensitivity

The Appendix tables come from `offline_sensitivity/*.csv`, generated by
`offline_sensitivity_audit.py` from saved traces. The analysis does not change
the frozen threshold, reporting horizon, event definition, or included cases.

## CIFAR-100 calibration reconciliation

The original `analyze_crossdomain_detector.py --mode fit` code generated the
CIFAR-100 threshold by lexicographically prioritizing detected development
events, fewer stable-run alarm episodes, shorter median lead, and then the
larger candidate threshold. The saved `frozen_detector.json` records this rule
and threshold 0.6332649834303036. A later window-level Youden-J audit happens
to reproduce the same numeric threshold from the six development traces, but
it was not the generating algorithm. That audit has one positive next-window
label among 2,877 eligible windows. Held-out traces were not used.

## Key-number reconciliation

The following manuscript values were reconciled to saved CSVs: 7/7, 4/5,
2/5 near, 2/2, 3/3, 5/5 versus 3/5, 46.83, 46.72, 47.09, 48.31, 18.68,
27.27, 45.71, 26.48, 42.60, 41.30, 43.00, 48.70, 42.40, 44.40, 47.05,
83.06, 18.74, 82.95, 46.93, 45.52, 45.66, and 42.30. The CIFAR-100 non-event
entry was the only material denominator correction found in this revision.
