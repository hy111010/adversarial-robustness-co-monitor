# Day 4 多随机种子论文素材

核心结果：Clean 78.00% ± 1.11%，FGSM 49.91% ± 0.32%，PGD-10 45.29% ± 0.41%。三种子中仅 seed 17 触发攻击升级，说明方法会按训练状态分配计算量。

- `table_multiseed.csv`：三次正式实验及均值、标准差的基础数据。
- `fig_multiseed.png`：逐种子结果及误差棒。
- `seed*_metrics.csv`：训练日志。
- `seed*_attack_results.csv`：完整测试集评估结果。
