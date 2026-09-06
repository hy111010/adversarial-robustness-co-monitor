# Supplemental v2 results

## Expanded frozen-detector evaluation

- Collapse events observed: 7/7
- Events warned within 60 updates: 7/7
- Runs ending below 10% PGD-10: 6/7
- Median lead: 40.0 updates
- Early false-alarm episodes: 1 total (0.143/run)
- Stable FGSM-RS alarm episodes: 0 across 3 runs

## Component ablation

| Method | Seeds | Final clean | Final PGD-10 | Core time |
|---|---:|---:|---:|---:|
| Adaptive + rollback | 5 | 83.14 +/- 0.62% | 46.83 +/- 0.55% | 18.68 +/- 0.70 min |
| Adaptive, no rollback | 5 | 83.09 +/- 0.57% | 46.72 +/- 1.05% | 18.14 +/- 0.58 min |
| Fixed switch epoch 13 | 5 | 83.32 +/- 0.70% | 46.41 +/- 0.50% | 19.37 +/- 0.21 min |
| Fixed switch epoch 21 | 5 | 82.35 +/- 1.12% | 45.75 +/- 0.94% | 17.85 +/- 0.21 min |
| Full PGD-2 | 5 | 83.57 +/- 1.02% | 46.81 +/- 0.90% | 21.60 +/- 0.13 min |

## Paired rollback effect

- Matched seeds: 5
- Rollback minus no-rollback PGD-10: +0.11 percentage points
- Seeds favoring rollback: 3/5
- Rollback minus no-rollback core time: +0.53 min
