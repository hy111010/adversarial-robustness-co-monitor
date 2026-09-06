# Day 4 论文素材：强攻击与模块消融

- `fig_day4_validation.png`：左侧为回滚消融，右侧为强攻击结果。
- `table_strong_attacks.csv`：PGD-10/20/50 与多重启结果。
- `table_rollback_ablation.csv`：回滚和无回滚的正式测试对比。
- `strong_attack_results.csv`：强攻击原始输出。
- `no_rollback_metrics.csv`：无回滚消融的 15 轮日志。
- `no_rollback_attack_results.csv`：无回滚模型正式测试结果。

核心结论：PGD-50 和五重启 PGD-20 下准确率仍超过 44%，没有发现明显梯度遮蔽；禁用回滚后结果略好，说明回滚不是有效模块，崩溃检测后的攻击升级才是主要贡献。
