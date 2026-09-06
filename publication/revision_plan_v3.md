# 论文第三轮修订与补实验方案

## 总目标

本轮不以“增加实验数量”为唯一目标，而是建立一条可审计的证据链：检测信号能跨训练随机性、网络结构和数据集工作；与 FastAdv+、GradAlign 在同一实现和评估协议下比较；最终模型接受完整 10,000 张测试集的 AutoAttack；关键消融使用相同 seed，并报告差值的不确定性。

所有新增实验在运行前固定：数据划分、威胁半径、训练轮数、学习率日程、checkpoint 选择规则、攻击参数和 seed 列表。失败实验同样保留日志，不根据结果临时改口径。

## 1. 泛化性：从“绝对阈值”改为“尺度归一化的在线分数”

### 1.1 先做零训练成本的离线门控实验

现有 CIFAR-10/ResNet-18 两条 collapse 轨迹中，原始 cosine-drop 分数的 AUROC 约为 0.99，但从 PreActResNet-18 冻结的绝对阈值召回为 0/2，说明信号排序可迁移而数值尺度不可迁移。先利用已经保存的 detector traces 比较以下三个预先定义的在线版本：

1. 原始绝对 drop（现论文版本）；
2. rolling median/MAD 标准化分数；
3. rolling empirical-quantile 分数，并要求连续两个窗口确认。

只用原来的三个开发 seed 选择一种规则及单一阈值；随后一次性评估全部 CIFAR-10 held-out 轨迹、三条稳定 FGSM-RS 轨迹和两条现有 ResNet-18 轨迹。主指标为事件召回、每 run 误报数、提前量和 AUPRC。门槛设为：PreActResNet-18 事件召回不低于 6/7、稳定 run 无误报、ResNet-18 至少 2/2 命中且每 run 不超过一次误报。若门槛未达到，不进入昂贵的跨架构 intervention 训练。

### 1.2 跨网络结构

若归一化检测器通过离线门槛，再补 3 个 ResNet-18 collapse seed，使跨架构测试达到 5 个预先固定的 held-out seed；另跑 3 个稳定 FGSM-RS seed 检验误报。检测器参数不得在这些 run 上调节。只选择其中 3 个 collapse seed 做 detector-guided PGD-2 intervention，用来证明检测成功能够转化为最终鲁棒性，而不是对全部 5 个重复昂贵干预。

### 1.3 跨数据集

新增 SVHN + PreActResNet-18，因为图像尺寸不变、实现改动小，而且 GradAlign 官方实验也覆盖 SVHN。先用 2 个 pilot seed 确认 collapse-prone recipe；pilot 只用于确定训练配置，不进入最终结果。配置冻结后运行 3 个开发 seed 和 5 个 held-out seed，并增加 3 个稳定训练 seed 测误报。主文报告 detector 的事件级结果；3 个 held-out collapse seed 再做干预验证。若 SVHN 在固定预算内无法稳定产生至少 3 个 held-out collapse 事件，则不虚构泛化结论，改为报告“跨架构”并将跨数据集结果作为补充失败分析。

### 1.4 论文呈现

主表按 Dataset / Architecture / collapse events / recall / false alarms per run / median lead 分组。跨域结果只证明检测机制及归一化规则可迁移，不把单个共享阈值与整个训练算法的泛化混为一谈。

## 2. FastAdv+ 与 GradAlign 的严格数值 baseline

### 2.1 统一实现原则

在当前训练 harness 中增加两种 mode，保持模型、数据顺序、增强、优化器、混合精度、训练轮数、威胁半径与现有方法完全一致：

- GradAlign：依照官方实现，用干净点和随机扰动点的输入梯度余弦构造正则项；仅一个分支保留二阶反传，以匹配其节省计算的实现。
- FastAdv+：按论文定义周期性执行 PGD 鲁棒性检查，在检测到弱攻击过拟合后切换/恢复多步训练。监控频率、判定条件与动作在首次正式运行前冻结。

每个实现先通过单 batch 数值单元测试、一次短 smoke run、梯度与额外 forward/backward 次数审计。实现来源、commit、与官方代码的差异写入实验审计文件。

### 2.2 正式比较

在同一组 5 个 seed 上运行：不保护 FGSM、Fast FGSM-RS、GradAlign、FastAdv+、本文方法；PGD-2/PGD-10 作为高成本参考，不混入“快速方法排名”。所有 checkpoint 使用同一个预先规定的选择规则。统一报告：clean、PGD-10、PGD-50、AutoAttack、训练时间、峰值显存、额外攻击调用次数、collapse 率。

主要比较不是宣称本文鲁棒精度必然最高，而是比较“发现并处置不稳定性所需的在线成本”。数值表取代目前只有机制描述的 Table V；机制表移到附录或压缩为一段文字。

官方依据：GradAlign 使用作者公开实现与 NeurIPS 2020 论文；FastAdv+ 使用原论文算法定义。不得直接抄录不同网络、不同日程下的论文数字与本实验混排。

## 3. AutoAttack 扩展到完整 10,000 张

在训练与 checkpoint 选择全部冻结后才运行 AutoAttack，避免用最强评估结果反向选模型。对本文方法、Fast FGSM-RS、GradAlign、FastAdv+ 和 PGD-10 各选预先固定的代表 seed，使用相同官方 AutoAttack standard、同一批 10,000 张 CIFAR-10 测试图像、相同 batch size 和版本。脚本已有 `--max-samples 10000` 与 state 文件，按方法串行运行并允许断点恢复。

先用 1,000 张结果做运行完整性核对，不再把它作为主结果。正式表只放 full-test AutoAttack；1,000 张敏感性实验移入补充材料。每个 checkpoint 的哈希、AutoAttack 版本、攻击顺序、样本数和日志路径均记录。

## 4. 扩充 matched controls 并停止解读微小点差

当前 adaptive no-rollback 已有 5 个 seed，而 fixed epoch 13、fixed epoch 21、full PGD-2 各有 3 个有效 seed。补跑 seed 23 和 42，使四种 intervention/control 都在同一组 5 seeds 上比较，共新增 6 次训练。

报告每个 seed 的配对结果、均值与标准差、配对差值的 bootstrap 95% CI。对 0.x 个百分点且置信区间跨 0 的差异明确写成“未检测到可靠差异”，不再按均值大小宣称优劣。核心结论限定为：检测时机是否避免 terminal collapse、rollback 是否提供可重复增益、以及相应时间成本。

如果需要对小于 1 个百分点的差异做等价性结论，5 seeds 不足，应增加到至少 10 seeds 并预先规定等价界值；本轮更节省算力的做法是删除该强结论。

## 5. 压缩防御性文字

重写 Abstract、Introduction、Discussion 和 Conclusion：

- Abstract 只保留一次适用范围说明，先给问题、方法、核心数字与成本；
- Introduction 不提前连续列出“不是什么”，只清楚定义本文目标是 attack-free online monitoring and conditional mitigation；
- 将 Section V 中重复的“不优于 FGSM-RS、不证明跨架构、不是最优鲁棒训练”合并为 Discussion 末尾一个 Scope and Limitations 段落；
- 主文用新增数值 baseline 和泛化表主动回答问题，删去逐条回应审稿意见式句子；
- Conclusion 只总结已被表格支持的正面结论和一条未来工作，不重复全部限制。

## 6. 推荐执行顺序与停止规则

1. 离线归一化检测器回放（低成本，先决定跨架构是否值得继续）。
2. 补齐 5-seed matched controls（6 次训练）。
3. 实现并验证 FastAdv+、GradAlign；各先跑 1 seed，正确后再扩到 5 seeds。
4. 完成 ResNet-18 的 5-event/3-stable detector 泛化；通过后才跑 3 次 intervention。
5. 做 SVHN pilot；只有稳定产生可定义的事件才进入正式 3-dev/5-held-out 协议。
6. 所有 checkpoint 冻结后，串行跑五个模型的 full-test AutoAttack。
7. 汇总配对统计、改表格与正文、重新编译和逐页检查 PDF。

任何阶段若门槛失败，停止对应后续昂贵训练并如实收窄结论；不得换 seed、删失败 run 或在 held-out 数据上重新调阈值。

## 7. 预计成本

- 离线 detector 回放：分钟级，无训练。
- 补齐 matched controls：6 次 30-epoch 训练，按当前记录约 2 小时量级。
- FastAdv+/GradAlign：正式 10 次训练；GradAlign 因额外梯度计算最慢，约 4--6 小时量级。
- ResNet-18 泛化：6 次 detector-only/稳定训练加 3 次 intervention，约 3--4 小时量级。
- SVHN：pilot 与正式运行合计约 11--16 次，约 4--7 小时量级，取决于 collapse 是否稳定出现。
- 五个 full-test AutoAttack：现有 1,000 张约 11--13 分钟/模型，10,000 张预计合计约 8--12 GPU 小时，实际可因前序攻击淘汰样本而缩短。

最小可发表修订包为：归一化 detector 的跨架构结果、FastAdv+/GradAlign 五 seed 数值比较、5-seed matched controls、五模型 full-test AutoAttack、正文压缩。跨数据集 SVHN 是增强包，但若目标审稿意见明确要求“换数据集”，则应视为必做项。
