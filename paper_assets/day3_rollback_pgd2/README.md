# Day 3 论文素材：回滚 + PGD-2 恢复

## 可直接引用的核心数据

| 方法/检查点 | Clean | FGSM | PGD-10 | 训练时间 |
|---|---:|---:|---:|---:|
| Standard ResNet-18 | 94.72% | 38.58% | 0.06% | 30.79 min |
| Fast AT early-stop（epoch 2） | 37.57% | 23.45% | 18.55% | 1.03 min |
| Fast AT final（epoch 15） | 55.12% | 85.98% | 0.00% | 7.62 min |
| 回滚 + PGD-2（epoch 15） | 78.85% | 49.30% | 44.39% | 10.18 min |

## 关键现象

- epoch 3 检测到 PGD-10 从历史最佳 19.38% 降至 4.37%，触发回滚。
- 回滚到 epoch 2 后改用 PGD-2，epoch 4 的 PGD-10 恢复至 27.81%。
- 最终完整测试集 PGD-10 为 44.39%，原 Fast AT 最终模型为 0.00%。

## 文件说明

- `fig_recovery_dynamics.png`：崩溃、触发回滚和 PGD-2 恢复的完整训练动态，建议作为方法分析图。
- `fig_all_methods.png`：Standard、Fast AT 早停、Fast AT 最终模型和本文方案的攻击对比，建议作为主结果图。
- `table_main_results.csv`：可直接导入论文表格的核心结果。
- `training_metrics.csv`：15 epoch 的完整日志。
- `attack_results.csv`：最佳与最终检查点在完整测试集上的原始结果。

推荐图题：

> Fig. X. Validation dynamics of the rollback-and-recovery method. A sharp PGD-10 drop at epoch 3 triggers rollback to epoch 2 and a switch from FGSM-RS to PGD-2, after which multi-step robustness steadily recovers.

> Fig. Y. Test accuracy comparison under clean inputs, FGSM, and PGD-10. The rollback + PGD-2 method prevents the zero-robustness failure of the final Fast AT model.

## 论文表述边界

可以称为“本文设计的轻量恢复策略”或“课程项目中的改进方法”。现阶段不要写成“首次提出”，因为尚未完成系统性的全领域新颖性检索。正式结论还应由 Day 4 消融和多随机种子实验支撑。
