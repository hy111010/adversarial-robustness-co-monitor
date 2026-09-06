# Publication experiment log

This ledger separates publication-protocol runs from the earlier Day 1-4 pilot work.
Validation metrics below are monitoring values, not final test-set results.

## Active

### PGD-AT / CIFAR-10 / PreActResNet-18 / seed 17

- Run directory: `outputs_publication/pgd10_preact_seed17`
- Started: 2026-09-02
- Status: running (110 planned epochs)
- Training attack: untargeted L-infinity PGD-10, epsilon 8/255, step size 2/255.
- Optimizer: SGD, momentum 0.9, weight decay 5e-4.
- Schedule: LR 0.1; divide by 10 at epochs 100 and 105.
- Split: 45,000 training / 5,000 fixed validation examples.
- Selection signal: PGD-10 on the first 10 fixed validation batches.
- Final evaluation still required: full CIFAR-10 test set, official AutoAttack standard,
  and PGD-50 with 10 restarts.

### Sequential queue

- After the active PGD-AT process exits, `publication/run_after_pgd.ps1` will start
  `fast_fgsm_rs_preact_seed17` for 30 epochs.
- The queue intentionally stops after this forecasting trace so its evidence can be
  checked before committing GPU time to further methods.

## Prepared

### FGSM-RS collapse-forecast trace

- Entry point: `train_fast_at_publication.py`
- Architecture and threat model match the PGD-AT baseline.
- Records two-start FGSM diagnostics, their wall-clock cost, and an independently
  seeded PGD-10 label at every epoch.
- Monitoring RNG is isolated from training RNG so diagnostics cannot alter the
  optimization trajectory merely by consuming random numbers.
- Changing diagnostic coverage from one to two validation batches produced 109/109
  bitwise-identical model-state tensors in a two-epoch isolation check.
- The training loop also records endpoint-gradient, prediction, loss-gain, and
  perturbation-boundary signals every 20 batches. These reuse the two model passes
  already required by FGSM-RS training; they add statistic computation but no new
  forward or backward pass.
- CPU smoke test passed on 2026-09-02.

### TRADES baseline

- Entry point: `train_trades.py`; beta 6, 10 KL-maximization steps.
- Uses the shared PreActResNet-18 backbone, split, optimizer, and learning-rate
  schedule for a controlled comparison rather than claiming an exact reproduction
  of the original WideResNet configuration.
- Projection, pixel bounds, model-mode restoration, and absence of leaked parameter
  gradients passed CPU checks on 2026-09-02.

### Standard-training baseline

- Entry point: `train_standard_publication.py`.
- Uses the same backbone, fixed split, optimizer, and piecewise schedule.
- Selects by validation clean accuracy; strong attacks are reserved for final evaluation.

### Final attack tooling

- `eval_official_autoattack.py` uses the official `fra31/auto-attack` implementation,
  not the similarly named wrapper in `torchattacks`.
- Installed package: `autoattack==0.1`, official master ZIP with SHA-256
  `87d423ee2794814fd003697ed1bec29b28416b6d048678a3f0983c27df5c2b1f`.
- The evaluator records package provenance, attack completion state, logs, and can
  resume an interrupted four-attack standard evaluation.
- All attack evaluators now reconstruct the architecture stored in each checkpoint;
  loading and forwarding the active PreActResNet-18 checkpoint passed on 2026-09-02.

## Non-result development runs

Directories whose names begin with `dev_` are implementation checks only. They must
not appear in result tables or be used to support scientific claims.
