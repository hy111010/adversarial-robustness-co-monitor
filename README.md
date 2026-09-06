# 深度神经网络的对抗鲁棒性：实验与论文

本仓库保存论文的训练与评估代码、冻结实验协议、逐轮指标、检测器
trace、攻击结果、统计审计、论文图表和 IEEE LaTeX 源文件。实验覆盖
CIFAR-10/CIFAR-100、PreActResNet-18/ResNet-18、FGSM/PGD/AutoAttack、
灾难性过拟合预警以及条件式 PGD-2 恢复。

为使仓库能够在普通 GitHub 账户下稳定克隆，仓库不提交可重新下载的
数据集、Python 虚拟环境、临时工具和约 25.4GB 的模型权重。每个实验的
配置、CSV、日志和结果均保留；权重可按保存的配置重新训练得到。

论文工程位于 `publication/IEEE_Adversarial_Robustness_Paper/`，最终实验
审计见其中的 `AUDIT.md`，尚未执行的高成本项目见 `NOT_DONE.md`。

## 早期基线说明

本项目对应课程论文的第一阶段：建立可复现的 CIFAR-10 / ResNet-18 正常训练基线，并使用像素空间中的 FGSM 与 PGD-10 进行白盒评估。扰动约束统一为 `L∞, ε=8/255`。

## 1. 环境

PowerShell：

```powershell
E:\Scripts\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

验证 GPU 与代码链：

```powershell
.\.venv\Scripts\python.exe smoke_test.py
```

若多伦多大学官方源下载速度异常，可预先下载字节一致的镜像，并核对 torchvision 使用的官方 MD5：

```powershell
curl.exe -L -o data\cifar-10-python.tar.gz https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz
Get-FileHash -Algorithm MD5 data\cifar-10-python.tar.gz
# 必须得到 C58F30108F718F92721AF3B95E74349A
```

## 2. Day 1：正常模型与攻击基线

先用少量数据验证下载、训练、保存链路：

```powershell
.\.venv\Scripts\python.exe train.py --epochs 1 --limit-train 1024 --limit-val 512 --workers 0 --run-name dev_baseline
```

正式训练（默认 100 epochs，SGD、余弦退火、AMP）：

```powershell
.\.venv\Scripts\python.exe train.py --run-name standard_seed17
```

对白盒攻击进行最终评估：

```powershell
.\.venv\Scripts\python.exe eval_attacks.py --checkpoint outputs/standard_seed17/best.pt
```

输出目录会保存参数、逐 epoch 日志、最佳/最终权重和攻击结果。只有 `eval_attacks.py` 的测试集结果用于论文主表；开发阶段以固定验证集选择模型。

## 3. 实验口径

- 输入始终保持 `[0,1]`；归一化封装在模型内部，避免把 `8/255` 错用到标准化空间。
- FGSM：`ε=8/255`。
- PGD-10：`ε=8/255`，步长 `2/255`，随机初始化。
- 随机种子默认为 17；数据划分、训练与攻击随机性均被记录。
- 攻击时模型处于 `eval` 模式，但保留输入梯度；不会把模型参数梯度混入攻击过程。

下一阶段将在同一接口上加入 Fast Adversarial Training、每 epoch 鲁棒性追踪和训练耗时比较。
