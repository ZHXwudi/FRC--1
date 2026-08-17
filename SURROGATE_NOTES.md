# 同步控制 PyTorch 代理模型：方法与边界

## 1. 项目定位

本模块将候选人参与论文中的研究结构转成一个可复现的 AI 工程小项目。来源论文为：

> Guici Chen, Houxuan Zhang, Shiping Wen, Junhao Hu, Leimin Wang. Fixed/prescribed-time Synchronization of State-dependent Switching Neural Networks with stochastic disturbance and impulsive effects. *Neural Networks* 194 (2026) 108100. DOI: [10.1016/j.neunet.2025.108100](https://doi.org/10.1016/j.neunet.2025.108100).

候选人为第 2/5 作者。论文 CRediT 记录包括 Writing - original draft、Software、Investigation 和 Formal analysis。结构化记录见 [`data/candidate_paper_evidence.json`](data/candidate_paper_evidence.json)。

本仓库没有复制论文插图，也没有分发论文 PDF。

## 2. 保留了什么，改变了什么

保留的系统结构：

- 状态幅值决定连接矩阵的行级切换；
- 固定时滞状态进入动力学；
- Euler-Maruyama 随机扰动；
- 离散时刻的脉冲映射；
- 固定/预设时间控制所启发的非线性反馈调度。

主动改变或省略的内容：

- 论文示例矩阵经过缩放，只用于稳定的标准化二维基准；
- 没有复现 LMI 可行性条件、证明、收敛时间上界或论文数值图；
- 没有复现论文的完整比例-积分控制器和积分时滞项；
- 不使用 EAST、FRC 或其他装置数据；
- 不把该论文称为 PINN 论文。

因此，正确表述是“论文启发的 physics/control-informed 代理模型”，不是“用 PINN 复现了论文”。

## 3. 代理模型

输入为 11 维局部状态和外生条件：当前误差、时滞误差、距控制期限的时间、控制期限、脉冲增益、噪声尺度、两个切换模式和脉冲标志。网络为 `11-48-48-2` 的 SiLU 残差 MLP，输出一步状态增量。

训练损失为：

```text
L = L_supervised + 0.18 * L_dynamics
```

`L_supervised` 使用随机合成轨迹的一步状态增量，`L_dynamics` 使用解析确定性漂移与脉冲映射给出的期望增量。随机扩散没有被错误地写成确定性残差目标。

## 4. 防止时序泄漏

数据先按独立轨迹生成，再按 trajectory ID 做 70%/15%/15% 分组切分。同一条轨迹的相邻时间点不会跨训练、验证和测试集。训练域外集合另外扩大初始状态、噪声、控制期限和脉冲增益范围。

固定种子 2026 的仓库产物包含：

- 90 条域内轨迹：63/14/13 条用于训练、验证、测试；
- 16 条训练域外压力测试轨迹；
- 3,026 个可训练参数；
- PyTorch checkpoint 与 TorchScript 导出；
- 一步、递推、域外和 CPU 单样本延迟指标。

指标仅代表该合成基准。CPU eager 延迟是微基准，不是端到端服务延迟；本项目不宣称代理模型相对二维解析模拟器有加速优势。

## 5. 复现实验

```powershell
python scripts/train_surrogate.py
python -m pytest tests/test_surrogate.py
```

训练脚本会覆盖 `models/` 下的 checkpoint、TorchScript、指标 JSON 和示例滚动轨迹 CSV。固定随机种子用于可复现比较，但不同 PyTorch/BLAS 版本仍可能产生小幅浮点差异。

## 6. 向 FRC 场景迁移时必须补齐

这个代理模型不能直接用于 FRC。真正迁移前至少需要由装置和物理专家确认：

1. 状态、控制量、诊断通道、单位、采样时钟和缺失值口径；
2. 对应 FRC 形成、平移、压缩、维持或不稳定性任务的控制方程；
3. shot 级数据切分、工况覆盖和域外定义；
4. 一步误差之外的滚动稳定性、物理守恒、事件时序和不确定性指标；
5. 在线推理预算、故障降级、人工审批和禁止自动下发的安全边界。

项目的岗位价值不是“已经会做 FRC 控制”，而是证明候选人能把控制理论、PyTorch 建模、验证与智能体工具编排连成一条可审计链路。
