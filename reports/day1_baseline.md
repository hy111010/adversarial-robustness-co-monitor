# Day 1：Standard 模型与攻击基线

## 实验设置

- 数据集：CIFAR-10，45,000 张训练图像、5,000 张固定验证图像、10,000 张测试图像。
- 模型：针对 32×32 输入修改首层并移除 max-pool 的 ResNet-18。
- 优化：SGD，初始学习率 0.1，momentum 0.9，weight decay 5e-4，余弦退火 100 epochs。
- 随机种子：17。
- 威胁模型：白盒 L∞，ε=8/255。
- FGSM：单步，步长 8/255。
- PGD-10：随机初始化，10 步，步长 2/255。

## 训练结果

- 最佳验证集 Clean Accuracy：95.16%（epoch 100）。
- 测试集 Clean Accuracy：94.72%。
- 总训练时间：30.79 分钟。
- 平均每 epoch 训练时间：18.48 秒（不含验证）。

## 攻击结果

| 模型 | Clean | FGSM（8/255） | PGD-10（8/255） |
|---|---:|---:|---:|
| Standard ResNet-18 | 94.72% | 38.58% | 0.06% |

相对于 Clean Accuracy：

- FGSM 下降 56.14 个百分点。
- PGD-10 下降 94.66 个百分点，模型在该攻击下几乎完全失效。

## 当前结论

正常训练能够得到较高的 CIFAR-10 分类准确率，但几乎不提供 PGD 鲁棒性。PGD-10 明显强于 FGSM，结果方向符合白盒攻击预期，可作为后续防御方法的对照基线。

这些结果只证明“模型脆弱”，还不能证明任何防御有效。下一阶段必须在相同数据划分、模型、ε 和评估攻击下训练 Fast Adversarial Training，并同时记录 Clean Accuracy、FGSM Accuracy、PGD-10 Accuracy 与训练成本。

## 结果文件

- `outputs/standard_seed17/metrics.csv`
- `outputs/standard_seed17/attack_results.csv`
- `outputs/standard_seed17/training_curves.png`
- `outputs/standard_seed17/best.pt`
