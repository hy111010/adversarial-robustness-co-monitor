# Day 3 优化版论文素材

最终选用检查点：`outputs/day3_refined_pgd5_seed17/epoch09.pt`。

| 版本 | Clean | FGSM | PGD-10 | 累计训练时间 |
|---|---:|---:|---:|---:|
| 原 Day 3 | 78.85% | 49.30% | 44.39% | 10.18 min |
| PGD-2 精炼 | 81.56% | 50.96% | 44.67% | 16.64 min |
| PGD-5 精炼（最终） | **81.83%** | **52.08%** | **45.52%** | 28.77 min |

文件说明：

- `fig_day3_optimization.png`：三个版本的正式测试集对比。
- `table_optimization_results.csv`：可直接用于论文的结果表。
- `pgd2_refine_metrics.csv`：第一阶段精炼训练日志。
- `pgd5_refine_metrics.csv`：第二阶段精炼训练日志。
- `pgd2_attack_results.csv`、`pgd5_attack_results.csv`：完整测试集原始评估结果。

建议正文表述：低学习率 PGD-2 精炼首先改善模型收敛，随后使用更强的 PGD-5 进行短期强化。在相同 ε=8/255 约束下，最终模型的 Clean 和 PGD-10 Accuracy 相对原方法分别提高 2.98 和 1.13 个百分点。
