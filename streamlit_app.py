from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from frc_lab.physics import (  # noqa: E402
    cylindrical_divergence,
    generate_equilibrium,
    generate_shot,
    idw_reconstruct,
    make_grid,
    normalized_rmse,
    reconstruct_from_probes,
    reversal_radius,
    sample_probes,
)
from frc_lab.research import load_research_catalog, validate_research_catalog  # noqa: E402


RESEARCH_CATALOG = load_research_catalog(ROOT / "data" / "research_evidence.json")
CATALOG_ERRORS = validate_research_catalog(RESEARCH_CATALOG, ROOT)


st.set_page_config(
    page_title="类 FRC 磁诊断与物理约束重建实验室",
    page_icon="Ψ",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    :root { color-scheme: light; }
    html, body, [class*="css"] { letter-spacing: 0 !important; }
    .stApp { background: #f6f8fa; color: #17212b; }
    [data-testid="stSidebar"] { background: #eef2f3; border-right: 1px solid #d7dee2; }
    [data-testid="stHeader"] { background: rgba(246, 248, 250, 0.92); }
    [data-testid="stToolbar"], [data-testid="stElementToolbar"] { display: none !important; }
    .block-container { max-width: 1500px; padding-top: 1.4rem; padding-bottom: 3rem; }
    h1 { font-size: 1.75rem !important; font-weight: 650 !important; margin-bottom: .2rem !important; }
    h2 { font-size: 1.22rem !important; font-weight: 650 !important; margin-top: 1.35rem !important; }
    h3 { font-size: 1rem !important; font-weight: 650 !important; }
    p, li, label { color: #263642; }
    .app-kicker { color: #087f8c; font-size: .78rem; font-weight: 700; text-transform: uppercase; }
    .app-subtitle { color: #556570; max-width: 980px; margin-bottom: .65rem; }
    .boundary-note { border-left: 3px solid #d8782d; padding: .55rem .8rem; background: #fff7ed; color: #663c1c; }
    .method-note { border-left: 3px solid #087f8c; padding: .55rem .8rem; background: #ecfeff; color: #164e55; }
    .flow-node { border: 1px solid #cbd5da; border-radius: 6px; background: #ffffff; min-height: 118px; padding: .8rem; }
    .flow-node strong { display: block; color: #17212b; margin-bottom: .3rem; }
    .flow-node span { color: #596873; font-size: .86rem; }
    .flow-arrow { text-align: center; color: #087f8c; font-size: 1.2rem; padding-top: 2.4rem; }
    [data-testid="stMetric"] { background: #ffffff; border: 1px solid #dce2e5; border-radius: 6px; padding: .75rem .9rem; }
    [data-testid="stMetricLabel"] { color: #54636d; }
    [data-testid="stMetricValue"] { color: #17212b; font-size: 1.45rem; }
    [data-testid="stTabs"] button { letter-spacing: 0 !important; }
    [data-testid="stDataFrame"] { border: 1px solid #dce2e5; border-radius: 6px; }
    .research-row { padding: .65rem 0; border-bottom: 1px solid #dde3e6; }
    .research-row:last-child { border-bottom: 0; }
    .tag { display: inline-block; padding: .1rem .42rem; border-radius: 4px; background: #e6f6f7; color: #12616a; font-size: .76rem; margin-right: .3rem; }
    .evidence-note { border: 1px solid #cbd5da; border-left: 3px solid #087f8c; padding: .65rem .8rem; background: #ffffff; color: #34454f; }
    .figure-caption { color: #5c6972; font-size: .78rem; line-height: 1.5; margin-top: -.35rem; }
    .source-block { border-top: 1px solid #dde3e6; padding-top: .55rem; color: #52616b; font-size: .82rem; }
    .evidence-a { display: inline-block; padding: .08rem .38rem; border-radius: 4px; background: #e8f5ec; color: #236339; font-size: .74rem; font-weight: 700; }
    a { color: #096d78 !important; }
    @media (max-width: 700px) {
      .block-container { padding: 1rem .8rem 2rem; }
      h1 { font-size: 1.45rem !important; }
      .flow-node { min-height: auto; }
      .flow-arrow { padding-top: .25rem; transform: rotate(90deg); }
      [data-testid="stTabs"] button { padding-left: .32rem !important; padding-right: .32rem !important; }
      [data-testid="stTabs"] button p { font-size: .72rem !important; white-space: nowrap; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


PLOT_LAYOUT = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(
        family="Microsoft YaHei, Noto Sans CJK SC, Arial, sans-serif",
        color="#263642",
        size=12,
    ),
    margin=dict(l=58, r=28, t=58, b=52),
    hoverlabel=dict(bgcolor="#17212b", font_color="#ffffff"),
)

PLOT_CONFIG = {
    "displayModeBar": False,
    "displaylogo": False,
    "scrollZoom": False,
}

SHOT_COLUMN_LABELS = {
    "time_ms": "时间 [ms]",
    "coil_current_kA": "线圈电流 [kA]",
    "center_bz_T": "中心 Bz [T]",
    "edge_bz_T": "边界 Bz [T]",
    "density_1e19_m3": "密度代理量 [10¹⁹ m⁻³]",
    "energy_kJ": "储能代理量 [kJ]",
}


@st.cache_data(show_spinner=False)
def run_case(
    reversal_strength: float,
    elongation: float,
    probe_count: int,
    noise_percent: float,
    fault_mode: str,
    regularization: float,
):
    grid = make_grid()
    equilibrium = generate_equilibrium(
        grid,
        reversal_strength=reversal_strength,
        elongation=elongation,
    )
    probes = sample_probes(
        equilibrium,
        count=probe_count,
        noise_percent=noise_percent,
        fault_mode=fault_mode,
    )
    reconstruction = reconstruct_from_probes(
        grid,
        probes,
        elongation=elongation,
        regularization=regularization,
    )
    idw_br, idw_bz = idw_reconstruct(probes, grid)
    return grid, equilibrium, probes, reconstruction, idw_br, idw_bz


@st.cache_data(show_spinner=False)
def sensitivity_matrix(reversal_strength: float, elongation: float, regularization: float):
    probe_counts = [12, 18, 24, 32, 40]
    noise_levels = [0.0, 0.5, 1.0, 2.0, 4.0]
    grid = make_grid(nr=54, nz=76)
    equilibrium = generate_equilibrium(
        grid,
        reversal_strength=reversal_strength,
        elongation=elongation,
    )
    physics_errors = np.zeros((len(noise_levels), len(probe_counts)))
    idw_errors = np.zeros_like(physics_errors)
    for row, noise in enumerate(noise_levels):
        for col, count in enumerate(probe_counts):
            probes = sample_probes(
                equilibrium,
                count=count,
                noise_percent=noise,
                seed=100 + row * 10 + col,
            )
            reconstructed = reconstruct_from_probes(
                grid,
                probes,
                elongation=elongation,
                regularization=regularization,
            )
            idw_br, idw_bz = idw_reconstruct(probes, grid)
            physics_errors[row, col] = normalized_rmse(
                equilibrium.br,
                equilibrium.bz,
                reconstructed.br,
                reconstructed.bz,
            )
            idw_errors[row, col] = normalized_rmse(
                equilibrium.br,
                equilibrium.bz,
                idw_br,
                idw_bz,
            )
    return probe_counts, noise_levels, physics_errors, idw_errors


def field_comparison_figure(grid, equilibrium, probes, reconstruction, idw_bz):
    panels = [
        (equilibrium.bz, equilibrium.psi, "合成参考场"),
        (reconstruction.bz, reconstruction.psi, "磁通约束重建"),
        (idw_bz, None, "独立 IDW 基线"),
    ]
    limit = float(max(np.max(np.abs(panel[0])) for panel in panels))
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[panel[2] for panel in panels],
        horizontal_spacing=0.07,
    )
    for col, (bz, psi, _) in enumerate(panels, start=1):
        figure.add_trace(
            go.Heatmap(
                x=grid.z,
                y=grid.r,
                z=bz.T,
                zmin=-limit,
                zmax=limit,
                colorscale=[
                    [0.0, "#b4232b"],
                    [0.48, "#f7f7f4"],
                    [0.52, "#f7f7f4"],
                    [1.0, "#087f8c"],
                ],
                colorbar=dict(title="Bz [T]", x=1.015) if col == 3 else None,
                showscale=col == 3,
                showlegend=False,
                hovertemplate="轴向 z=%{x:.2f} m<br>半径 r=%{y:.2f} m<br>Bz=%{z:.3f} T<extra></extra>",
            ),
            row=1,
            col=col,
        )
        if psi is not None:
            figure.add_trace(
                go.Contour(
                    x=grid.z,
                    y=grid.r,
                    z=psi.T,
                    contours=dict(coloring="none", showlabels=False, size=0.045),
                    line=dict(color="#24333d", width=0.75),
                    showscale=False,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=col,
            )
        figure.add_trace(
            go.Scatter(
                x=probes.z,
                y=probes.r,
                mode="markers",
                marker=dict(
                    size=6,
                    color="#ffbf47",
                    line=dict(color="#17212b", width=0.7),
                    symbol="circle",
                ),
                name="磁探针",
                showlegend=False,
                hovertemplate="磁探针<br>轴向 z=%{x:.2f} m<br>半径 r=%{y:.2f} m<extra></extra>",
            ),
            row=1,
            col=col,
        )
        figure.update_xaxes(title_text="轴向位置 z [m]", row=1, col=col, showgrid=False)
        figure.update_yaxes(
            title_text="半径 r [m]" if col == 1 else None,
            row=1,
            col=col,
            showgrid=False,
            range=[0.0, grid.r[-1]],
        )
    figure.update_layout(
        **PLOT_LAYOUT,
        height=510,
    )
    figure.update_layout(margin=dict(l=58, r=28, t=56, b=52))
    figure.update_annotations(y=1.03)
    figure.add_annotation(
        x=0.5,
        y=-0.16,
        xref="paper",
        yref="paper",
        text="固定随机种子的合成类 FRC 场；非装置数据，非 EAST/FRC 实验重建结果",
        showarrow=False,
        font=dict(size=11, color="#687680"),
    )
    figure.update_layout(margin=dict(l=58, r=28, t=56, b=78))
    return figure


def midplane_figure(grid, equilibrium, reconstruction, idw_bz):
    mid = int(np.argmin(np.abs(grid.z)))
    figure = go.Figure()
    series = [
        (equilibrium.bz[mid], "参考场", "#17212b", "solid"),
        (reconstruction.bz[mid], "磁通约束重建", "#087f8c", "solid"),
        (idw_bz[mid], "IDW 基线", "#d8782d", "dash"),
    ]
    for values, label, color, dash in series:
        figure.add_trace(
            go.Scatter(
                x=grid.r,
                y=values,
                name=label,
                mode="lines",
                line=dict(color=color, width=2.2, dash=dash),
                hovertemplate=f"{label}<br>r=%{{x:.3f}} m<br>Bz=%{{y:.3f}} T<extra></extra>",
            )
        )
    figure.add_hline(y=0.0, line_color="#7c8991", line_width=1)
    figure.update_layout(
        **PLOT_LAYOUT,
        title="中平面轴向磁场反转",
        height=365,
        xaxis_title="半径 r [m]",
        yaxis_title="Bz [T]",
        legend=dict(orientation="h", y=1.13, x=0.0),
    )
    figure.update_xaxes(showgrid=True, gridcolor="#e7ebed")
    figure.update_yaxes(showgrid=True, gridcolor="#e7ebed")
    return figure


def residual_figure(probes, reconstruction):
    order = np.argsort(reconstruction.anomaly_score)[::-1]
    colors = ["#b4232b" if score >= 3.5 else "#087f8c" for score in reconstruction.anomaly_score[order]]
    labels = [f"P{index + 1:02d}" for index in order]
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=reconstruction.anomaly_score[order],
            marker_color=colors,
            customdata=np.column_stack(
                [probes.r[order], probes.z[order], reconstruction.residual[order]]
            ),
            hovertemplate=(
                "%{x}<br>异常分数=%{y:.2f}<br>r=%{customdata[0]:.2f} m"
                "<br>z=%{customdata[1]:.2f} m<br>残差=%{customdata[2]:.4f} T<extra></extra>"
            ),
        )
    )
    figure.add_hline(
        y=3.5,
        line_dash="dash",
        line_color="#b4232b",
        annotation_text="复核阈值",
        annotation_position="top right",
    )
    figure.update_layout(
        **PLOT_LAYOUT,
        title="磁探针残差筛查",
        height=365,
        xaxis_title="探针编号",
        yaxis_title="稳健异常分数",
        showlegend=False,
    )
    figure.update_yaxes(showgrid=True, gridcolor="#e7ebed")
    return figure


def shot_figure(shot: pd.DataFrame):
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=shot.time_ms,
            y=shot.center_bz_T,
            name="中心 Bz",
            line=dict(color="#b4232b", width=2.2),
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=shot.time_ms,
            y=shot.edge_bz_T,
            name="边界 Bz",
            line=dict(color="#087f8c", width=2.0),
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=shot.time_ms,
            y=shot.density_1e19_m3,
            name="密度代理量",
            line=dict(color="#d8782d", width=2.0, dash="dot"),
        ),
        secondary_y=True,
    )
    figure.add_vrect(x0=0.35, x1=0.72, fillcolor="#ffbf47", opacity=0.12, line_width=0)
    figure.add_vrect(x0=1.08, x1=1.62, fillcolor="#087f8c", opacity=0.08, line_width=0)
    figure.add_annotation(x=0.53, y=1.05, yref="paper", text="形成", showarrow=False)
    figure.add_annotation(x=1.35, y=1.05, yref="paper", text="压缩", showarrow=False)
    figure.update_layout(
        **PLOT_LAYOUT,
        title="合成脉冲诊断概览（非装置数据）",
        height=440,
        legend=dict(orientation="h", y=1.15, x=0.0),
    )
    figure.update_xaxes(title_text="时间 [ms]", showgrid=True, gridcolor="#e7ebed")
    figure.update_yaxes(title_text="磁场强度 [T]", secondary_y=False, showgrid=True, gridcolor="#e7ebed")
    figure.update_yaxes(title_text="密度代理量 [10¹⁹ m⁻³]", secondary_y=True, showgrid=False)
    return figure


def sensitivity_figure(probe_counts, noise_levels, physics_errors, idw_errors):
    improvement = (idw_errors - physics_errors) * 100.0
    text = [[f"{value:+.1f} 百分点" for value in row] for row in improvement]
    figure = go.Figure(
        go.Heatmap(
            x=probe_counts,
            y=noise_levels,
            z=improvement,
            text=text,
            texttemplate="%{text}",
            colorscale=[[0.0, "#f7d8d5"], [0.5, "#f7f7f4"], [1.0, "#087f8c"]],
            colorbar=dict(title="NRMSE 改善 [百分点]"),
            hovertemplate="探针数=%{x}<br>噪声=%{y:.1f}%<br>改善=%{z:.2f} 百分点<extra></extra>",
        )
    )
    figure.update_layout(
        **PLOT_LAYOUT,
        title="物理约束方法相对 IDW 的精度改善（合成基准）",
        height=430,
        xaxis_title="磁探针数量",
        yaxis_title="测量噪声 [% RMS]",
    )
    return figure


def _add_flow_node(figure, x0, x1, y0, y1, text, fill="#eef7f7", border="#7da9ae"):
    figure.add_shape(
        type="rect",
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        line=dict(color=border, width=1.2),
        fillcolor=fill,
    )
    figure.add_annotation(
        x=(x0 + x1) / 2,
        y=(y0 + y1) / 2,
        text=text,
        showarrow=False,
        align="center",
        font=dict(size=12, color="#263642"),
    )


def _add_flow_arrow(figure, x0, x1, y):
    figure.add_annotation(
        x=x1,
        y=y,
        ax=x0,
        ay=y,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.4,
        arrowcolor="#708089",
        text="",
    )


def equilibrium_research_flow_figure():
    figure = go.Figure()
    published = [
        "EAST<br>磁测量",
        "深度神经网络<br>自动调参",
        "EFIT 极向磁通<br>监督目标",
        "磁面 / LCFS / l<sub>i</sub><br>实时约束评估",
    ]
    prototype = [
        "合成类 FRC<br>稀疏磁探针",
        "岭回归<br>平滑磁通基底",
        "共享 ψ 导出<br>Br / Bz",
        "NRMSE / div(B)<br>IDW 基线对照",
    ]
    columns = [
        ((0.45, 5.05), published, "#edf7f8", "#6f9fa5", "论文方法"),
        ((5.75, 10.35), prototype, "#fff6ea", "#d19a62", "本项目方法"),
    ]
    node_y = [4.35, 3.15, 1.95, 0.75]
    for (x0, x1), labels, fill, border, heading in columns:
        center = (x0 + x1) / 2
        figure.add_annotation(
            x=center,
            y=5.48,
            text=heading,
            showarrow=False,
            font=dict(size=13, color=border),
        )
        for index, (y0, label) in enumerate(zip(node_y, labels)):
            _add_flow_node(figure, x0, x1, y0, y0 + 0.68, label, fill, border)
            if index < len(node_y) - 1:
                figure.add_annotation(
                    x=center,
                    y=node_y[index + 1] + 0.73,
                    ax=center,
                    ay=y0 - 0.05,
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowwidth=1.3,
                    arrowcolor="#708089",
                    text="",
                )
    figure.add_shape(type="line", x0=5.4, x1=5.4, y0=0.65, y1=5.52, line=dict(color="#dce2e5", width=1))
    figure.add_annotation(
        x=5.4,
        y=0.12,
        text="方法映射示意：本项目未复现 DNN、EFIT、电流演化或真实装置控制",
        showarrow=False,
        font=dict(size=11, color="#687680"),
    )
    figure.update_xaxes(visible=False, range=[0.0, 10.8], fixedrange=True)
    figure.update_yaxes(visible=False, range=[-0.05, 5.75], fixedrange=True)
    layout = {**PLOT_LAYOUT, "height": 500, "margin": dict(l=12, r=12, t=14, b=16)}
    figure.update_layout(**layout)
    return figure


def rf_research_schematic():
    figure = go.Figure()
    figure.add_annotation(x=2.75, y=4.85, text="低杂波参数衰变<br>（2022）", showarrow=False, font=dict(size=13, color="#263642"))
    figure.add_annotation(x=8.25, y=4.85, text="等离子体-部件关系<br>（2019）", showarrow=False, font=dict(size=13, color="#263642"))
    _add_flow_node(figure, 0.55, 4.95, 3.65, 4.30, "低杂波泵波", "#edf7f8", "#6f9fa5")
    _add_flow_node(figure, 0.55, 4.95, 2.30, 2.95, "低杂边带", "#f4f7f8", "#9eabb1")
    _add_flow_node(figure, 0.55, 4.95, 0.95, 1.60, "低频模：ISQM / ICQM", "#f4f7f8", "#9eabb1")
    figure.add_annotation(x=2.75, y=3.0, ax=2.75, ay=3.6, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowcolor="#708089", text="")
    figure.add_annotation(x=2.75, y=1.65, ax=2.75, ay=3.6, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowcolor="#708089", text="")

    _add_flow_node(figure, 6.05, 10.45, 3.65, 4.30, "低杂波天线", "#fff6ea", "#d19a62")
    _add_flow_node(figure, 6.05, 10.45, 2.30, 2.95, "天线前边界等离子体", "#f4f7f8", "#9eabb1")
    _add_flow_node(figure, 6.05, 10.45, 0.95, 1.60, "保护限制器", "#fff0ed", "#c98277")
    figure.add_annotation(x=8.25, y=3.0, ax=8.25, ay=3.6, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowcolor="#708089", text="")
    figure.add_annotation(x=8.25, y=1.65, ax=8.25, ay=2.25, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowcolor="#708089", text="")
    figure.add_shape(type="line", x0=5.5, x1=5.5, y0=0.85, y1=4.55, line=dict(color="#dce2e5", width=1))
    figure.add_annotation(
        x=5.5,
        y=0.22,
        text="基于论文摘要/题名绘制的关系示意；非 EAST 几何、尺寸、波形或仿真结果",
        showarrow=False,
        font=dict(size=11, color="#687680"),
    )
    figure.update_xaxes(visible=False, range=[0.0, 11.0], fixedrange=True)
    figure.update_yaxes(visible=False, range=[0.0, 5.15], fixedrange=True)
    layout = {**PLOT_LAYOUT, "height": 500, "margin": dict(l=12, r=12, t=12, b=16)}
    figure.update_layout(**layout)
    return figure


def research_evidence_table(catalog):
    return pd.DataFrame(
        [
            {
                "论文": paper["title_zh"],
                "期刊 / 年份": f'{paper["journal"]} {paper["volume_issue"]}, {paper["year"]}',
                "项农老师作者位次": paper["author_role"],
                "证据": f'{paper["evidence_level"]} · DOI',
                "与项目关系": paper["project_relation"],
                "DOI": f'https://doi.org/{paper["doi"]}',
            }
            for paper in catalog["papers"]
        ]
    )


def format_radius(value: float | None) -> str:
    return "未检测到" if value is None else f"{value:.3f} m"


with st.sidebar:
    st.markdown("### 合成脉冲参数")
    reversal_strength = st.slider("磁场反转强度", 0.75, 1.35, 1.00, 0.05)
    elongation = st.slider("等离子体延伸率", 1.10, 2.10, 1.55, 0.05)
    probe_count = st.slider("磁探针数量", 12, 42, 24, 2)
    noise_percent = st.slider("探针噪声 [% RMS]", 0.0, 5.0, 1.0, 0.25)
    fault_label = st.selectbox("注入的探针状态", ["无故障", "漂移", "尖峰", "饱和"])
    regularization_label = st.select_slider(
        "岭回归正则化系数",
        options=["1e-5", "1e-4", "1e-3", "1e-2", "1e-1"],
        value="1e-3",
    )
    st.divider()
    st.caption("仅使用合成数据，用于教学和面试演示；不包含装置数据或操作指导。")

fault_mode = {"无故障": "None", "漂移": "Drift", "尖峰": "Spike", "饱和": "Saturation"}[fault_label]
regularization = float(regularization_label)

grid, equilibrium, probes, reconstruction, idw_br, idw_bz = run_case(
    reversal_strength,
    elongation,
    probe_count,
    noise_percent,
    fault_mode,
    regularization,
)

physics_error = normalized_rmse(
    equilibrium.br,
    equilibrium.bz,
    reconstruction.br,
    reconstruction.bz,
)
idw_error = normalized_rmse(equilibrium.br, equilibrium.bz, idw_br, idw_bz)
physics_div = cylindrical_divergence(reconstruction.br, reconstruction.bz, grid)
idw_div = cylindrical_divergence(idw_br, idw_bz, grid)
b_scale = float(np.sqrt(np.mean(equilibrium.br**2 + equilibrium.bz**2)))
physics_div_norm = float(np.sqrt(np.mean(physics_div**2)) * 0.70 / b_scale)
idw_div_norm = float(np.sqrt(np.mean(idw_div**2)) * 0.70 / b_scale)
reference_radius = reversal_radius(equilibrium.bz, grid)
reconstructed_radius = reversal_radius(reconstruction.bz, grid)

st.markdown('<div class="app-kicker">聚变诊断 / 面试演示项目</div>', unsafe_allow_html=True)
st.title("类 FRC 磁诊断与物理约束重建实验室")
st.markdown(
    '<div class="app-subtitle">面向稀疏磁诊断、物理约束重建、传感器质量复核和'
    "专家在环实验分析的可复现 Python 原型。</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="boundary-note"><strong>研究边界：</strong>项农老师已验证的公开研究聚焦托卡马克'
    "理论与计算，特别是 EAST 平衡重建和集成建模。本演示仅将其重建方法迁移到合成的"
    "类 FRC 磁场，不声称 FRC 与托卡马克平衡等价。</div>",
    unsafe_allow_html=True,
)

if CATALOG_ERRORS:
    st.error("研究证据目录校验失败：" + "；".join(CATALOG_ERRORS))

tab_equilibrium, tab_shot, tab_sensitivity, tab_research, tab_agent, tab_interview = st.tabs(
    ["平衡重建", "脉冲诊断", "敏感性", "研究证据", "智能体架构", "面试表达"]
)

with tab_equilibrium:
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric(
        "磁通约束重建 NRMSE",
        f"{physics_error:.2%}",
        delta=f"-{(idw_error - physics_error):.1%} 相对 IDW",
        delta_color="inverse",
    )
    metric_b.metric(
        "归一化 div(B) 残差",
        f"{physics_div_norm:.3f}",
        delta=f"-{(idw_div_norm - physics_div_norm):.3f} 相对 IDW",
        delta_color="inverse",
    )
    radius_delta = None
    if reference_radius is not None and reconstructed_radius is not None:
        radius_delta = f"误差 {abs(reconstructed_radius - reference_radius) * 1000:.0f} mm"
    metric_c.metric(
        "场反转半径",
        format_radius(reconstructed_radius),
        delta=radius_delta,
        delta_color="off",
    )

    st.markdown("#### 稀疏磁探针驱动的轴向磁场重建")
    st.plotly_chart(
        field_comparison_figure(grid, equilibrium, probes, reconstruction, idw_bz),
        width="stretch",
        config=PLOT_CONFIG,
    )
    left, right = st.columns([1.45, 1.0])
    with left:
        st.plotly_chart(
            midplane_figure(grid, equilibrium, reconstruction, idw_bz),
            width="stretch",
            config=PLOT_CONFIG,
        )
    with right:
        st.plotly_chart(
            residual_figure(probes, reconstruction),
            width="stretch",
            config=PLOT_CONFIG,
        )

    st.markdown(
        '<div class="method-note"><strong>物理约束：</strong>重建磁场由同一轴对称极向磁通函数导出。'
        "Br 与 Bz 联合拟合，因此输出保留共享磁通结构，并强烈抑制数值 div(B) 偏差；"
        "IDW 基线则对两个分量独立插值。</div>",
        unsafe_allow_html=True,
    )

with tab_shot:
    shot = generate_shot(reversal_strength=reversal_strength)
    st.plotly_chart(shot_figure(shot), width="stretch", config=PLOT_CONFIG)
    event_a, event_b, event_c = st.columns(3)
    reversal_rows = shot.loc[shot.center_bz_T < 0.0, "time_ms"]
    event_a.metric("首次场反转", f"{reversal_rows.iloc[0]:.2f} ms" if len(reversal_rows) else "未检测到")
    event_b.metric("密度代理量峰值", f"{shot.density_1e19_m3.max():.2f} × 10¹⁹ m⁻³")
    event_c.metric("储能代理量峰值", f"{shot.energy_kJ.max():.2f} kJ")
    st.dataframe(
        shot.iloc[::12].round(4),
        width="stretch",
        hide_index=True,
        column_config=SHOT_COLUMN_LABELS,
    )
    st.download_button(
        "下载合成脉冲 CSV",
        shot.rename(columns=SHOT_COLUMN_LABELS).to_csv(index=False).encode("utf-8-sig"),
        file_name="synthetic_frc_shot.csv",
        mime="text/csv",
    )

with tab_sensitivity:
    probe_counts, noise_levels, physics_errors, idw_errors = sensitivity_matrix(
        reversal_strength,
        elongation,
        regularization,
    )
    chart_col, detail_col = st.columns([1.6, 1.0])
    with chart_col:
        st.plotly_chart(
            sensitivity_figure(
                probe_counts,
                noise_levels,
                physics_errors,
                idw_errors,
            ),
            width="stretch",
            config=PLOT_CONFIG,
        )
    with detail_col:
        st.subheader("评估矩阵")
        best_row, best_col = np.unravel_index(np.argmax(idw_errors - physics_errors), physics_errors.shape)
        st.metric("最大 NRMSE 改善", f"{(idw_errors - physics_errors)[best_row, best_col]:.1%}")
        st.metric("对应探针数量", str(probe_counts[best_col]))
        st.metric("对应噪声水平", f"{noise_levels[best_row]:.1f}%")
        st.caption(
            "正值表示共享磁通表示对合成参考场的重建精度高于独立插值。"
        )
        sensitivity_table = pd.DataFrame(
            physics_errors,
            index=[f"{value:.1f}%" for value in noise_levels],
            columns=[str(value) for value in probe_counts],
        )
        st.dataframe(sensitivity_table.style.format("{:.2%}"), width="stretch")


with tab_research:
    st.subheader("项农老师研究证据地图")
    st.markdown(
        '<div class="evidence-note"><span class="evidence-a">A 级证据</span> '
        "本页只使用 DOI/Crossref 元数据、论文摘要和 ORCID 公开记录。论文事实、项目映射与图像许可"
        "分别记录；合成结果和概念示意不作为 EAST 或 FRC 实验结论。</div>",
        unsafe_allow_html=True,
    )

    paper_by_id = {paper["id"]: paper for paper in RESEARCH_CATALOG["papers"]}
    evidence_a, evidence_b, evidence_c = st.columns(3)
    evidence_a.metric("已核验论文", f'{len(RESEARCH_CATALOG["papers"])} 篇')
    evidence_b.metric(
        "合规复用原论文图",
        f'{sum(paper["figure_reused"] for paper in RESEARCH_CATALOG["papers"])} 张',
    )
    evidence_c.metric("真实装置数据", "0 组", delta="本项目仅用合成数据", delta_color="off")

    st.markdown("#### 1. 平衡重建：直接方法关联")
    st.plotly_chart(
        equilibrium_research_flow_figure(),
        width="stretch",
        config=PLOT_CONFIG,
    )
    fast_paper = paper_by_id["east_fast_equilibrium"]
    consistency_paper = paper_by_id["east_current_consistency"]
    source_left, source_right = st.columns(2)
    with source_left:
        st.markdown(
            f"**已核验事实：快速平衡重建**  \n"
            f"{fast_paper['journal']} {fast_paper['volume_issue']}（{fast_paper['year']}）；"
            f"项农老师为{fast_paper['author_role']}。论文以 EAST 磁测量为输入、EFIT 极向磁通为监督目标，"
            "比较内部磁面、最后闭合磁面和归一化内感，并报告满足实时控制的时间约束。  \n"
            f"[DOI：{fast_paper['doi']}](https://doi.org/{fast_paper['doi']})"
        )
    with source_right:
        st.markdown(
            f"**已核验事实：电流模拟一致性**  \n"
            f"{consistency_paper['journal']} {consistency_paper['volume_issue']}（{consistency_paper['year']}）；"
            f"项农老师为{consistency_paper['author_role']}。论文针对 POINT/MSE 等间接测量的不确定性，"
            "引入电流模拟一致性约束，并讨论 RF 驱动电流、q 剖面与波沉积区的相互作用。  \n"
            f"[DOI：{consistency_paper['doi']}](https://doi.org/{consistency_paper['doi']})"
        )
    st.warning(
        "边界：本项目的共享磁通表示与 div(B) 检查不是论文中的电流模拟一致性约束；"
        "本项目也没有复现 DNN、EFIT、q 剖面或 RF 沉积模型。"
    )

    st.markdown("#### 2. RF 波-等离子体与边界：领域关联")
    st.plotly_chart(rf_research_schematic(), width="stretch", config=PLOT_CONFIG)
    rf_pic = paper_by_id["lh_parametric_instability"]
    rf_limiter = paper_by_id["lh_antenna_limiter"]
    rf_ebw = paper_by_id["ebw_second_harmonic"]
    st.markdown(
        f"- **低杂波参数不稳定性**：{rf_pic['journal']}（{rf_pic['year']}），"
        f"{rf_pic['author_role']}；二维全粒子 PIC 研究 EAST 参数下低杂泵波向低杂边带和低频模的参数衰变。"
        f" [DOI](https://doi.org/{rf_pic['doi']})\n"
        f"- **天线前等离子体-保护限制器互作用**：{rf_limiter['journal']}（{rf_limiter['year']}），"
        f"{rf_limiter['author_role']}。当前仅依据题名和 DOI 元数据陈述研究对象。"
        f" [DOI](https://doi.org/{rf_limiter['doi']})\n"
        f"- **非均匀等离子体中的电子伯恩斯坦波二次谐波**：{rf_ebw['journal']}（{rf_ebw['year']}），"
        f"{rf_ebw['author_role']}。该论文证明波-等离子体研究线索，不直接支撑本项目重建算法。"
        f" [DOI](https://doi.org/{rf_ebw['doi']})"
    )

    st.markdown("#### 3. 集成建模与科研软件：工程直接关联")
    fydev_paper = paper_by_id["fydev_fair4rs"]
    image_col, explain_col = st.columns([1.45, 1.0])
    with image_col:
        st.image(
            str(ROOT / "assets" / "fydev-workflow-figure1.png"),
            caption=(
                "原论文 Figure 1（未修改）：FyDev 的查找、获取、构建、使用/复用与 Python 调用流程。"
                "Liu X, Yu Z, Xiang N., Scientific Data 10 (2023), CC BY 4.0。"
            ),
            use_container_width=True,
        )
    with explain_col:
        st.markdown(
            f"**证据**：{fydev_paper['journal']} {fydev_paper['volume_issue']}（{fydev_paper['year']}），"
            f"项农老师为{fydev_paper['author_role']}。论文介绍面向磁约束聚变研究软件的发现、获取、"
            "构建、复用、唯一标识与 Python 调用。\n\n"
            "**项目映射**：结构化工具元数据、Python 确定性计算、版本与哈希、自动测试、运行清单、"
            "Dify/Qwen 编排和专家审核。\n\n"
            "**边界**：本项目没有接入或复现 FyDev/EAST 集成建模环境。\n\n"
            f"[查看原论文](https://doi.org/{fydev_paper['doi']}) · "
            "[查看 CC BY 4.0 许可](https://creativecommons.org/licenses/by/4.0/)"
        )

    st.markdown("#### 证据总表")
    evidence_table = research_evidence_table(RESEARCH_CATALOG)
    st.dataframe(
        evidence_table,
        width="stretch",
        hide_index=True,
        column_config={"DOI": st.column_config.LinkColumn("DOI", display_text="打开论文")},
    )
    st.download_button(
        "下载研究证据 JSON",
        (ROOT / "data" / "research_evidence.json").read_bytes(),
        file_name="xiang_nong_research_evidence.json",
        mime="application/json",
    )
    st.caption(
        "身份边界：ORCID 0000-0002-8663-0470 的公开记录支持姓名与中国科学院等离子体物理研究所机构关系；"
        "本项目不推断行政职务。图像来源、许可与 SHA-256 见 assets/SOURCES.md。"
    )


with tab_agent:
    st.subheader("确定性科学工具驱动的实验复盘智能体")
    flow_columns = st.columns([1, 0.18, 1, 0.18, 1, 0.18, 1])
    flow = [
        ("数据接入与版本", "放电编号、通道结构、单位、时间戳、原始文件哈希"),
        ("质量门禁", "缺失、漂移、饱和、时间对齐、异常分数"),
        ("Python 科学工具", "磁通重建、残差、物理约束、不确定性、基线对比"),
        ("专家审阅报告", "证据关联总结、未解问题、人工审批"),
    ]
    for index, (title, body) in enumerate(flow):
        with flow_columns[index * 2]:
            st.markdown(
                f'<div class="flow-node"><strong>{title}</strong><span>{body}</span></div>',
                unsafe_allow_html=True,
            )
        if index < len(flow) - 1:
            with flow_columns[index * 2 + 1]:
                st.markdown('<div class="flow-arrow">→</div>', unsafe_allow_html=True)

    st.markdown("#### LLM 职责边界")
    llm_col, tool_col = st.columns(2)
    with llm_col:
        st.markdown(
            "**Qwen/Dify 负责**\n\n"
            "- 意图识别与工作流编排\n"
            "- 带引用的版本化知识检索\n"
            "- 工具结果解释与报告初稿\n"
            "- 证据不足时升级给专家"
        )
    with tool_col:
        st.markdown(
            "**确定性工具负责**\n\n"
            "- 信号对齐、单位校验与数据质量\n"
            "- 磁场重建与物理约束\n"
            "- 优化、不确定性与验收指标\n"
            "- 安全规则、权限与不可篡改日志"
        )

    manifest = {
        "项目": "类 FRC 磁诊断与物理约束重建实验室",
        "适用范围": "合成数据教学与面试演示",
        "输入参数": {
            "磁场反转强度": reversal_strength,
            "等离子体延伸率": elongation,
            "磁探针数量": probe_count,
            "探针噪声百分比": noise_percent,
            "探针状态": fault_label,
            "岭回归正则化系数": regularization,
        },
        "计算结果": {
            "物理约束_NRMSE": physics_error,
            "IDW_NRMSE": idw_error,
            "物理约束_divB残差": physics_div_norm,
            "IDW_divB残差": idw_div_norm,
            "重建场反转半径_m": reconstructed_radius,
        },
        "必须人工复核": True,
    }
    st.download_button(
        "下载可复现性清单",
        json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="reconstruction_manifest.json",
        mime="application/json",
    )

with tab_interview:
    st.subheader("面试表达：用可验证证据说明你的价值")
    st.markdown(
        '<div class="evidence-note"><strong>定位：</strong>不把自己包装成聚变物理专家；证明自己能把'
        "物理专家的问题转成可测试的 Python 工具、受约束的智能体工作流和可追溯的研究界面。</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### 90 秒项目演示顺序")
    demo_columns = st.columns(4)
    demo_steps = [
        ("01 · 边界", "先说明全是合成类 FRC 数据，不冒充 EAST/FRC 实验结论。"),
        ("02 · 方法", "展示稀疏探针、共享磁通重建、IDW 基线与三项验收指标。"),
        ("03 · 工程", "展示故障注入、敏感性矩阵、证据 JSON、版本哈希和自动测试。"),
        ("04 · 智能体", "说明 Dify/Qwen 只编排与解释，Python 工具计算，专家最终审批。"),
    ]
    for column, (title, body) in zip(demo_columns, demo_steps):
        with column:
            st.markdown(
                f'<div class="flow-node"><strong>{title}</strong><span>{body}</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown("#### 你的经历如何转化为岗位价值")
    mapping = pd.DataFrame(
        [
            {
                "公开研究问题": "稀疏磁测量 -> 平衡/状态重建",
                "项目证据": "磁通基底拟合 + IDW 基线 + NRMSE/div(B)/反转半径",
                "你的经历": "时序数据治理、复杂系统建模、Python/NumPy/Plotly",
                "岗位价值": "把诊断问题拆成数据、先验、模型、指标和复核闭环",
            },
            {
                "公开研究问题": "模型、诊断与物理约束的一致性",
                "项目证据": "Br/Bz 共用磁通函数；显式声明它不等同于电流模拟约束",
                "你的经历": "MILP/动态规划、状态依赖切换、随机扰动与脉冲效应",
                "岗位价值": "把数值可行、物理可行和实验允许拆成确定性门禁",
            },
            {
                "公开研究问题": "集成建模与可复现科研软件",
                "项目证据": "固定随机种子、模块化内核、测试、CSV/JSON、来源与许可",
                "你的经历": "赛唯储能 Dify 多智能体、工具调用、测算校验与报告交付",
                "岗位价值": "把科研原型做成可追溯、可回放、专家可复核的 AI 应用",
            },
        ]
    )
    st.dataframe(mapping, width="stretch", hide_index=True, height=275)

    st.markdown("#### 可直接使用的项目陈述")
    st.markdown(
        """
        > 我核验了您在 EAST 快速平衡重建、电流模拟一致性约束和 FyDev 集成建模方面的公开论文。这个原型没有冒充真实 FRC 平衡求解器，而是用合成数据验证一个工程命题：稀疏磁诊断下，将 Br 和 Bz 纳入统一磁通表示，能否比独立插值更好地保留场反转与无散结构。页面里的 DOI、作者位次、图片许可和不能宣称的边界都做了结构化记录。

        > 我在浙江赛唯数字能源做储能 AI 智能体时，核心测算不交给大模型自由生成，而是交给 Python、规则、动态规划和 MILP；Dify 负责参数校验、工作流编排、工具调用和报告解释。我能带来的价值，是把物理专家定义的问题工程化成可测试工具、数据质量门禁和专家在环的智能体流程。

        > 如果进入团队，我会先和物理专家明确数据字典、诊断几何、标注口径、验收指标和失效模式，再用历史实验做离线回放、跨放电验证和误差分解。在通过专家审核和安全评审前，不把任何智能体输出用于装置参数下发或在线闭环控制。
        """
    )

    st.markdown("#### 不能说什么，以及应该怎么说")
    claims = pd.DataFrame(
        [
            {
                "避免表述": "我复现了项农老师的 FRC 平衡算法",
                "严谨表述": "公开论文主线是 EAST 托卡马克；我只迁移了稀疏诊断和物理约束的方法论。",
            },
            {
                "避免表述": "我的 div(B) 就是论文的电流模拟一致性约束",
                "严谨表述": "两者都强调一致性，但约束对象不同；本项目没有电流演化、q 剖面或 RF 沉积模型。",
            },
            {
                "避免表述": "这些图是 EAST/FRC 实验图和实验精度",
                "严谨表述": "重建图来自固定随机种子的合成数据；关系图是概念示意；只有 FyDev Figure 1 是经许可复用的原论文图。",
            },
            {
                "避免表述": "智能体可以直接控制装置",
                "严谨表述": "智能体用于检索、编排、解释和报告；数值由确定性工具给出，专家负责审批。",
            },
        ]
    )
    st.dataframe(claims, width="stretch", hide_index=True)

    st.markdown("#### 可向项农老师请教的问题")
    st.markdown(
        "1. 在 EAST 快速平衡重建中，从离线 EFIT 监督结果迁移到实时代理模型时，团队最关注的失效模式是磁面几何误差、内部参数误差，还是跨放电泛化？\n"
        "2. 电流模拟一致性约束中，如何权衡间接诊断不确定性与电流演化模型本身的偏差？\n"
        "3. 如果把平衡重建或实验复盘接入智能体，您更希望优先解决数据版本、诊断异常、适用域，还是推理延迟？\n"
        "4. FyDev 一类可复现科研环境与智能体工具编排结合时，接口标准、元数据、评测集和审批日志应如何排序？"
    )

    st.caption(
        "身份来源：ORCID 0000-0002-8663-0470 记录项农（Nong Xiang）就职于中国科学院等离子体物理研究所。"
        "研究陈述基于公开 DOI 元数据和摘要，不预设项农老师在贝塔聚变的具体职责。"
    )
