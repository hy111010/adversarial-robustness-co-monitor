# Day 2 论文素材：Fast Adversarial Training

## 固定设置

- 方法：FGSM-RS（随机初始化的单步对抗训练）。
- 训练攻击：L∞，ε=8/255，单步步长 10/255。
- 优化：SGD，momentum 0.9，weight decay 5e-4。
- 学习率：15 epoch 三角循环学习率，0 → 0.2 → 0。
- 模型、数据划分和随机种子与 Standard Baseline 相同。
- 每个 epoch 在固定验证子集上监控 FGSM 与 PGD-10；测试集只用于训练后的正式评估。

## 正式测试结果

| Model checkpoint | Clean | FGSM | PGD-10 | Training time |
|---|---:|---:|---:|---:|
| Standard ResNet-18 | 94.72% | 38.58% | 0.06% | 30.79 min |
| Fast AT early-stop（epoch 2） | 37.57% | 23.45% | 18.55% | 1.03 min |
| Fast AT final（epoch 15） | 55.12% | 85.98% | 0.00% | 7.62 min |

注意：最终 Fast AT 的 85.98% FGSM Accuracy 不能解释为高鲁棒性。其 PGD-10 Accuracy 为 0%，说明模型过拟合单步攻击并产生梯度遮蔽。

## 关键观察

- 验证集 PGD-10 Accuracy 在 epoch 2 达到最高值 19.38%。
- epoch 3 开始明显下降，到 epoch 5 低于 1%，epoch 6 后持续为 0%。
- 与此同时，训练对抗准确率从 27.08% 持续升至 92.65%。
- 最终验证 FGSM Accuracy 为 84.92%，但 PGD-10 为 0%。
- 降低学习率没有使 PGD 鲁棒性恢复。

这些现象满足 catastrophic overfitting 的典型判据：单步攻击和训练目标持续改善，多步攻击鲁棒性却在很短时间内崩塌。

## 可用于正文的结果描述

FGSM-RS 对抗训练在早期获得了非零的多步攻击鲁棒性，其 epoch 2 权重在完整测试集上的 PGD-10 Accuracy 为 18.55%。然而，随着训练继续进行，验证集 PGD-10 Accuracy 从 epoch 3 开始快速下降，并在 epoch 6 后持续为零；与此同时，FGSM Accuracy 和训练对抗准确率继续上升。最终权重的 FGSM Accuracy 达到 85.98%，PGD-10 Accuracy 却为 0%，表明模型发生了明显的 catastrophic overfitting。该结果说明，仅使用单步攻击的高准确率不足以证明鲁棒性，训练期间必须使用多步攻击监控或稳定化机制。

## 图像

- `fig_fast_at_dynamics.png`：最关键的现象图，展示 PGD 崩塌和 FGSM 上升，建议放入正文。
- `fig_standard_vs_fast.png`：Standard、早停 Fast AT 和最终 Fast AT 的测试集对比。
- `training_metrics.csv`：15 epoch 完整日志。
- `attack_results.csv`：两个 Fast AT 检查点的正式测试结果。

推荐图题：

> Fig. X. Validation dynamics of FGSM-RS adversarial training. PGD-10 accuracy collapses to zero while FGSM accuracy continues to increase, indicating catastrophic overfitting.

> Fig. Y. Comparison of standard training, early-stopped Fast AT, and the final Fast AT checkpoint under clean inputs and white-box attacks.

