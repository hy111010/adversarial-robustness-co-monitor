# Intervention timing audit

The collapse time is measured on the seed-matched unprotected trajectory using the
registered PGD-10 drop rule. The warning time is the frozen detector crossing.
`immediate_epoch_replay` stops FGSM at the warning and starts PGD-2 immediately
after restoring the epoch-start state. `deferred_next_epoch` retains the current
epoch and starts PGD-2 at the next epoch boundary.

| Strategy | Events | PGD-2 before collapse | Median warning-to-collapse | Median warning-to-PGD-2 |
|---|---:|---:|---:|---:|
| Immediate epoch replay | 5 | 5/5 | 60 updates | 0 updates |
| Deferred next epoch | 5 | 3/5 | 60 updates | 212 updates |

This audit distinguishes statistical warning lead from actionable lead. Final clean
and PGD-10 accuracy remain in `event_timing.csv` for the matched outcome comparison.
