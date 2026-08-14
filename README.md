# FRC Equilibrium Reconstruction Lab

一个面向 AI 应用工程师面试展示的 Streamlit 科学可视化项目：使用合成的 FRC-like 轴对称磁场，演示稀疏磁探针下的物理约束平衡重建、传感器质量筛查、脉冲诊断和可复现实验复盘。

> 本项目只用于科普、工程原型与面试交流。它不是 Grad-Shafranov 平衡求解器，未经任何聚变装置数据验证，不得用于装置操作、实验参数下发或安全决策。

## 为什么做这个项目

项农（Nong Xiang）老师的公开论文显示，其研究与 EAST 托卡马克的平衡重建、电流模拟一致性、射频波-等离子体相互作用和集成建模软件紧密相关。其中，磁测量驱动的快速平衡重建与物理自洽约束，是 AI 应用工程可以做出有说服力原型的交叉点。

本项目不把 FRC 与托卡马克画等号，而是迁移一个通用方法论：

```text
稀疏诊断 -> 数据质量门禁 -> 物理约束重建 -> 基线对照
         -> 误差/不确定性 -> 实验复盘 -> 专家审阅
```

## 可交互内容

- **平衡重建**：对比合成参考场、磁通量约束反演与独立 IDW 插值。
- **物理验收**：评估矢量场 NRMSE、归一化 `div(B)` 残差和中平面场反转半径。
- **诊断质量**：注入漂移、尖峰或饱和故障，用稳健残差分数标记需复核探针。
- **脉冲复盘**：展示形成、压缩和衰减阶段的合成时序，并导出 CSV。
- **敏感性实验**：量化探针数量与测量噪声对两类方法的影响。
- **Agent 安全边界**：区分 Qwen/Dify 的编排与解释职责，以及 Python/物理工具的确定性计算职责。
- **研究映射**：将项农老师公开研究主线、项目技术证据和候选人经历放在同一张表里。

## 物理与算法

对轴对称系统，用一个极向磁通函数 `psi(r, z)` 同时表示径向与轴向磁场：

```text
Br = -(1/r) * d(psi)/dz
Bz =  (1/r) * d(psi)/dr
```

重建使用一组在 `r=0` 正则的平滑磁通基底，把所有探针的 `Br` 和 `Bz` 联合写成岭回归问题。反演只接收网格、探针测量和预先声明的几何先验，不读取合成参考场或真实系数。由于两个分量来自同一个 `psi`，重建场自然具有一致的磁通结构。IDW 基线则独立插值 `Br` 与 `Bz`，用来检验物理表示是否实际改善了结果。

合成参考场额外包含未纳入反演基底的整形项，因此反演不会获得人为的零误差。

## 本地运行

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

打开 `http://localhost:8501`。

## 测试

```powershell
python -m pytest
```

测试覆盖场反转、物理约束重建对 IDW 基线的精度优势、`div(B)` 残差和故障探针检出。

## 代码结构

```text
.
|-- streamlit_app.py          # 交互式科学仪表板
|-- src/frc_lab/physics.py    # 合成平衡、探针、重建与评估
|-- tests/test_physics.py     # 计算内核测试
|-- RESEARCH_NOTES.md         # 项农老师研究证据与边界
|-- requirements.txt
`-- .streamlit/config.toml
```

## 研究来源

- [Nong Xiang, ORCID 0000-0002-8663-0470](https://orcid.org/0000-0002-8663-0470)
- [Fast equilibrium reconstruction by deep learning on EAST tokamak](https://doi.org/10.1063/5.0152318)
- [Equilibrium reconstruction constrained by the consistency of current simulation on EAST](https://doi.org/10.1088/1741-4326/ad35d7)
- [Applying FAIR4RS principles to develop an integrated modeling environment for magnetic confinement fusion](https://doi.org/10.1038/s41597-023-02470-y)
- [Second-Harmonic Generation of Electron-Bernstein Waves in an Inhomogeneous Plasma](https://doi.org/10.1103/PhysRevLett.100.085002)
- [Interactions of plasma and guard limiter in front of lower hybrid wave antenna on EAST tokamak](https://doi.org/10.1088/1741-4326/ab082c)

详细的事实分层、研究主线和面试表达边界见 [RESEARCH_NOTES.md](RESEARCH_NOTES.md)。
