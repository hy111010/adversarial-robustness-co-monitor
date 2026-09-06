# Day 4 敏感性与 AutoAttack 论文素材

- `fig_sensitivity_autoattack.png`：左侧为完整测试集 PGD-20 的 ε 曲线；右侧为第三方 AutoAttack-1000 结果。
- `table_epsilon_sweep.csv`：不同 ε 的完整测试结果。
- `table_autoattack_1000.csv`：固定 1,000 样本的第三方套件结果。

引用时必须注明：PGD-20 曲线使用 10,000 张测试图片；AutoAttack 使用 `torchattacks 3.5.1` 的 standard 版本和固定前 1,000 张测试图片。两者样本量与攻击实现不同，不做严格的直接百分点比较。
