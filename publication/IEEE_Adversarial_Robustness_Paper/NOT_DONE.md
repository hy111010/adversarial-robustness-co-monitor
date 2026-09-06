# High-cost experiments not performed in this revision

No new training was performed. This revision was restricted to the requested
strong evaluation of one pre-fixed checkpoint and audits of existing traces:

- ImageNet or another large-scale dataset.
- Architectures outside the current CIFAR/ResNet family.
- Large expansions of detector, intervention, or baseline seed counts.
- Retraining every FastAdv+ and GradAlign configuration or using new extensive
  hyperparameter searches.
- Full 10,000-image AutoAttack evaluation.
- Threshold retuning intended to turn the ResNet-18 4/5 result into 5/5.
- New training under alternative warning horizons or CO definitions; these are
  evaluated offline on saved traces only.
- Systematic snapshot-copy latency, peak GPU memory, disk-storage policy, or
  large-model engineering benchmarks.

The existing completed 1,000-image AutoAttack-standard records are included as
explicitly labeled subset estimates across seven pre-fixed checkpoints. They
do not replace the
unperformed full 10,000-image AutoAttack evaluation.
