# Day 2：Fast Adversarial Training 与灾难性过拟合

## 实验目的

在与 Day 1 相同的 CIFAR-10、ResNet-18、数据划分和随机种子下，评估随机初始化 FGSM 对抗训练的效率与稳定性，并检查是否发生 catastrophic overfitting。

## 方法

Fast AT 使用 L∞ 约束下的 FGSM-RS 生成训练样本，ε=8/255，单步步长为 10/255。模型训练 15 epochs，学习率按三角周期从 0 升至 0.2 后降回 0。每个 epoch 使用固定验证数据监控 Clean、FGSM 和 PGD-10 Accuracy。

## 训练现象

- epoch 2：验证 PGD-10 Accuracy 达到最高值 19.38%。
- epoch 3：PGD-10 Accuracy 降至 4.37%，首次出现明显崩塌。
- epoch 5：PGD-10 Accuracy 降至 0.16%。
- epoch 6–15：PGD-10 Accuracy 持续为 0%。
- 最终训练对抗准确率为 92.65%，验证 FGSM Accuracy 为 84.92%。

训练目标与 FGSM 指标持续改善，而 PGD 指标归零，确认发生 catastrophic overfitting。

## 正式测试结果

| 检查点 | Clean | FGSM | PGD-10 | 训练时间 |
|---|---:|---:|---:|---:|
| Standard | 94.72% | 38.58% | 0.06% | 30.79 min |
| Fast AT epoch 2 | 37.57% | 23.45% | 18.55% | 1.03 min |
| Fast AT epoch 15 | 55.12% | 85.98% | 0.00% | 7.62 min |

## 阶段结论

Fast AT 在训练早期能够快速获得一定 PGD 鲁棒性，但当前设置下稳定窗口很短。简单早停可以保留部分鲁棒性，却得到较低的 Clean Accuracy 和有限的 PGD-10 Accuracy。最终模型的高 FGSM Accuracy 是误导性的，不能作为鲁棒性证据。

下一阶段应围绕“以很低额外成本检测并抑制 PGD 鲁棒性崩塌”设计轻量机制，并通过消融回答该机制是否优于单纯早停。
