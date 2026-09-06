# Day 1 论文素材：Standard Baseline

本目录只包含正式 100-epoch 实验数据；`outputs/dev_baseline/` 中的短测试结果不得写入论文。

## 可直接放入论文的主结果表

| Model | Clean Accuracy | FGSM（ε=8/255） | PGD-10（ε=8/255） | Training Time |
|---|---:|---:|---:|---:|
| Standard ResNet-18 | 94.72% | 38.58% | 0.06% | 30.79 min |

推荐表题：

> Table I. Performance of the standard ResNet-18 on CIFAR-10 under white-box adversarial attacks.

表下注释：FGSM 与 PGD-10 均采用 L∞ 威胁模型，ε=8/255；PGD 步长为 2/255，并使用随机初始化。

## 可直接用于正文的数据

- 最佳验证集 Clean Accuracy：95.16%（epoch 100）。
- 测试集 Clean Accuracy：94.72%。
- FGSM Accuracy：38.58%，相对 Clean 下降 56.14 个百分点。
- PGD-10 Accuracy：0.06%，相对 Clean 下降 94.66 个百分点。
- 总训练时间：30.79 分钟。
- 平均每 epoch 纯训练时间：18.48 秒。
- 训练数据：45,000；固定验证数据：5,000；测试数据：10,000。
- 随机种子：17。

## 建议写进论文的结果描述

标准训练的 ResNet-18 在 CIFAR-10 测试集上取得了 94.72% 的正常准确率，表明模型具有良好的自然图像分类能力。然而，在相同的 L∞ 扰动预算 ε=8/255 下，其准确率在 FGSM 攻击下下降至 38.58%，在更强的 PGD-10 攻击下仅为 0.06%。这一结果说明，高正常准确率并不意味着模型具有对抗鲁棒性，并为后续对抗训练方法提供了明确的比较基线。

注意：上述段落是结果陈述，暂时不要写成“本文方法有效”；防御方法尚未完成。

## 图像

### `fig_attack_comparison.png`

优先用于论文实验结果部分。它直接展示 Standard 模型在 Clean、FGSM 和 PGD-10 下的准确率差异。

推荐图题：

> Fig. 1. Test accuracy of the standard ResNet-18 on clean CIFAR-10 images and under FGSM and PGD-10 attacks with ε=8/255.

### `fig_training_curves.png`

包含训练 loss、训练/验证准确率、学习率和每 epoch 耗时。正文篇幅不足时可只保留准确率曲线，完整四联图可放附录或实验过程说明。

推荐图题：

> Fig. 2. Training dynamics of the standard ResNet-18 on CIFAR-10 over 100 epochs.

## 文件说明

- `table_main_results.csv`：论文主表的 Day 1 行。
- `attack_results.csv`：正式测试集攻击原始输出。
- `training_metrics.csv`：100 个 epoch 的完整训练日志。
- `fig_attack_comparison.png`：攻击结果柱状图，300 dpi。
- `fig_training_curves.png`：训练过程四联图。
