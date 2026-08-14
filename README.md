# 类 FRC 磁诊断与物理约束重建实验室

一个面向 AI 应用工程师面试展示的 Streamlit 科学可视化项目：使用合成的类 FRC 轴对称磁场，演示稀疏磁探针下的物理约束平衡重建、传感器质量筛查、脉冲诊断和可复现实验复盘。

> 本项目只用于科普、工程原型与面试交流。它不是 Grad-Shafranov 平衡求解器，未经任何聚变装置数据验证，不得用于装置操作、实验参数下发或安全决策。

## 为什么做这个项目

项农（Nong Xiang）老师的公开作者记录包含 EAST 托卡马克平衡重建、电流模拟一致性、射频波-等离子体相互作用和磁约束聚变集成建模软件。其中，磁测量驱动的快速平衡重建与可复现科研软件，是 AI 应用工程可以做出可验证原型的交叉点。

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
- **智能体安全边界**：区分 Qwen/Dify 的编排与解释职责，以及 Python/物理工具的确定性计算职责。
- **研究证据**：逐篇展示 DOI、期刊、作者位次、项目关系、不能宣称的内容和图片许可。
- **面试表达**：把公开研究问题、项目证据和候选人的储能智能体经历放在同一张表里。

## 数据与图片边界

- 重建、脉冲和敏感性图全部由固定随机种子的合成数据生成，不是 EAST 或 FRC 装置数据。
- RF 波与边界部件图是根据论文摘要或题名自行绘制的关系示意，不代表装置几何、尺寸或仿真结果。
- 页面只复用一张原论文图片：FyDev Figure 1。来源论文采用 CC BY 4.0，原始地址、署名和 SHA-256 均记录在 [`assets/SOURCES.md`](assets/SOURCES.md)。
- 六篇论文的结构化证据保存在 [`data/research_evidence.json`](data/research_evidence.json)，应用启动和测试时都会检查来源字段及图片哈希。

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

7 项测试覆盖场反转、物理约束重建对 IDW 基线的精度优势、`div(B)` 残差、故障探针检出，以及论文期刊、证据字段、图片许可与 SHA-256 完整性。

## 代码结构

```text
.
|-- streamlit_app.py          # 交互式科学仪表板
|-- src/frc_lab/physics.py    # 合成平衡、探针、重建与评估
|-- src/frc_lab/research.py   # 研究证据与图片来源校验
|-- tests/test_physics.py     # 计算内核测试
|-- tests/test_research_evidence.py
|-- data/research_evidence.json
|-- assets/
|   |-- fydev-workflow-figure1.png
|   `-- SOURCES.md
|-- RESEARCH_NOTES.md         # 项农老师研究证据与边界
|-- requirements.txt
`-- .streamlit/config.toml
```

## 研究来源

- [Nong Xiang, ORCID 0000-0002-8663-0470](https://orcid.org/0000-0002-8663-0470)
- [基于深度学习的 EAST 托卡马克快速平衡重建，AIP Advances 13(7)，2023](https://doi.org/10.1063/5.0152318)
- [EAST 上受电流模拟一致性约束的平衡重建](https://doi.org/10.1088/1741-4326/ad35d7)
- [低杂波参数不稳定性的粒子网格（PIC）模拟](https://doi.org/10.1063/5.0104505)
- [应用 FAIR4RS 原则开发磁约束聚变集成建模环境](https://doi.org/10.1038/s41597-023-02470-y)
- [非均匀等离子体中电子伯恩斯坦波的二次谐波生成](https://doi.org/10.1103/PhysRevLett.100.085002)
- [EAST 托卡马克低杂波天线前等离子体与保护限制器相互作用](https://doi.org/10.1088/1741-4326/ab082c)

详细的事实分层、研究主线和面试表达边界见 [RESEARCH_NOTES.md](RESEARCH_NOTES.md)。
