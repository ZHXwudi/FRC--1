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


st.set_page_config(
    page_title="FRC Equilibrium Reconstruction Lab",
    page_icon="Ψ",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { color-scheme: light; }
    html, body, [class*="css"] { letter-spacing: 0 !important; }
    .stApp { background: #f6f8fa; color: #17212b; }
    [data-testid="stSidebar"] { background: #eef2f3; border-right: 1px solid #d7dee2; }
    [data-testid="stHeader"] { background: rgba(246, 248, 250, 0.92); }
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
    a { color: #096d78 !important; }
    @media (max-width: 700px) {
      .block-container { padding: 1rem .8rem 2rem; }
      h1 { font-size: 1.45rem !important; }
      .flow-node { min-height: auto; }
      .flow-arrow { padding-top: .25rem; transform: rotate(90deg); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


PLOT_LAYOUT = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(family="Arial, sans-serif", color="#263642", size=12),
    margin=dict(l=58, r=28, t=58, b=52),
    hoverlabel=dict(bgcolor="#17212b", font_color="#ffffff"),
)


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
        equilibrium,
        probes,
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
                equilibrium,
                probes,
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
        (equilibrium.bz, equilibrium.psi, "Synthetic reference"),
        (reconstruction.bz, reconstruction.psi, "Flux-constrained fit"),
        (idw_bz, None, "Independent IDW baseline"),
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
                hovertemplate="z=%{x:.2f} m<br>r=%{y:.2f} m<br>Bz=%{z:.3f} T<extra></extra>",
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
                name="Magnetic probe",
                showlegend=False,
                hovertemplate="Probe<br>z=%{x:.2f} m<br>r=%{y:.2f} m<extra></extra>",
            ),
            row=1,
            col=col,
        )
        figure.update_xaxes(title_text="Axial z [m]", row=1, col=col, showgrid=False)
        figure.update_yaxes(
            title_text="Radius r [m]" if col == 1 else None,
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
    return figure


def midplane_figure(grid, equilibrium, reconstruction, idw_bz):
    mid = int(np.argmin(np.abs(grid.z)))
    figure = go.Figure()
    series = [
        (equilibrium.bz[mid], "Reference", "#17212b", "solid"),
        (reconstruction.bz[mid], "Flux-constrained", "#087f8c", "solid"),
        (idw_bz[mid], "IDW baseline", "#d8782d", "dash"),
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
        title="Midplane field reversal",
        height=365,
        xaxis_title="Radius r [m]",
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
                "%{x}<br>score=%{y:.2f}<br>r=%{customdata[0]:.2f} m"
                "<br>z=%{customdata[1]:.2f} m<br>residual=%{customdata[2]:.4f} T<extra></extra>"
            ),
        )
    )
    figure.add_hline(
        y=3.5,
        line_dash="dash",
        line_color="#b4232b",
        annotation_text="Review threshold",
        annotation_position="top right",
    )
    figure.update_layout(
        **PLOT_LAYOUT,
        title="Probe residual screening",
        height=365,
        xaxis_title="Probe ID",
        yaxis_title="Robust anomaly score",
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
            name="Center Bz",
            line=dict(color="#b4232b", width=2.2),
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=shot.time_ms,
            y=shot.edge_bz_T,
            name="Edge Bz",
            line=dict(color="#087f8c", width=2.0),
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=shot.time_ms,
            y=shot.density_1e19_m3,
            name="Density proxy",
            line=dict(color="#d8782d", width=2.0, dash="dot"),
        ),
        secondary_y=True,
    )
    figure.add_vrect(x0=0.35, x1=0.72, fillcolor="#ffbf47", opacity=0.12, line_width=0)
    figure.add_vrect(x0=1.08, x1=1.62, fillcolor="#087f8c", opacity=0.08, line_width=0)
    figure.add_annotation(x=0.53, y=1.05, yref="paper", text="Formation", showarrow=False)
    figure.add_annotation(x=1.35, y=1.05, yref="paper", text="Compression", showarrow=False)
    figure.update_layout(
        **PLOT_LAYOUT,
        title="Synthetic pulse overview",
        height=440,
        legend=dict(orientation="h", y=1.15, x=0.0),
    )
    figure.update_xaxes(title_text="Time [ms]", showgrid=True, gridcolor="#e7ebed")
    figure.update_yaxes(title_text="Magnetic field [T]", secondary_y=False, showgrid=True, gridcolor="#e7ebed")
    figure.update_yaxes(title_text="Density proxy [10¹⁹ m⁻³]", secondary_y=True, showgrid=False)
    return figure


def sensitivity_figure(probe_counts, noise_levels, physics_errors, idw_errors):
    improvement = (idw_errors - physics_errors) * 100.0
    text = [[f"{value:+.1f} pp" for value in row] for row in improvement]
    figure = go.Figure(
        go.Heatmap(
            x=probe_counts,
            y=noise_levels,
            z=improvement,
            text=text,
            texttemplate="%{text}",
            colorscale=[[0.0, "#f7d8d5"], [0.5, "#f7f7f4"], [1.0, "#087f8c"]],
            colorbar=dict(title="NRMSE gain [pp]"),
            hovertemplate="Probes=%{x}<br>Noise=%{y:.1f}%<br>Gain=%{z:.2f} pp<extra></extra>",
        )
    )
    figure.update_layout(
        **PLOT_LAYOUT,
        title="Physics constraint advantage over IDW",
        height=430,
        xaxis_title="Magnetic probe count",
        yaxis_title="Measurement noise [% RMS]",
    )
    return figure


def format_radius(value: float | None) -> str:
    return "Not detected" if value is None else f"{value:.3f} m"


with st.sidebar:
    st.markdown("### Synthetic shot controls")
    reversal_strength = st.slider("Field reversal strength", 0.75, 1.35, 1.00, 0.05)
    elongation = st.slider("Plasma elongation", 1.10, 2.10, 1.55, 0.05)
    probe_count = st.slider("Magnetic probe count", 12, 42, 24, 2)
    noise_percent = st.slider("Probe noise [% RMS]", 0.0, 5.0, 1.0, 0.25)
    fault_label = st.selectbox("Injected probe condition", ["无故障", "漂移", "尖峰", "饱和"])
    regularization_label = st.select_slider(
        "Ridge regularization",
        options=["1e-5", "1e-4", "1e-3", "1e-2", "1e-1"],
        value="1e-3",
    )
    st.divider()
    st.caption("Synthetic and educational only. No device data or operating guidance is used.")

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

st.markdown('<div class="app-kicker">Fusion diagnostics / interview demonstrator</div>', unsafe_allow_html=True)
st.title("FRC Equilibrium Reconstruction Lab")
st.markdown(
    '<div class="app-subtitle">A reproducible Python prototype for sparse magnetic diagnostics, '
    "physics-constrained reconstruction, sensor quality review, and expert-in-the-loop experiment analysis.</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="boundary-note"><strong>Research boundary:</strong> Nong Xiang\'s verified work is centered '
    "on tokamak theory and computation, especially EAST equilibrium reconstruction and integrated modeling. "
    "This demo transfers that reconstruction methodology to a synthetic FRC-like field; it does not claim "
    "that FRC and tokamak equilibria are equivalent.</div>",
    unsafe_allow_html=True,
)

tab_equilibrium, tab_shot, tab_sensitivity, tab_agent, tab_interview = st.tabs(
    ["平衡重建", "脉冲诊断", "敏感性", "Agent 架构", "研究与面试"]
)

with tab_equilibrium:
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric(
        "Flux-constrained NRMSE",
        f"{physics_error:.2%}",
        delta=f"{(idw_error - physics_error):.1%} vs IDW",
    )
    metric_b.metric(
        "Normalized div(B) residual",
        f"{physics_div_norm:.3f}",
        delta=f"{(idw_div_norm - physics_div_norm):.3f} vs IDW",
    )
    radius_delta = None
    if reference_radius is not None and reconstructed_radius is not None:
        radius_delta = f"{abs(reconstructed_radius - reference_radius) * 1000:.0f} mm error"
    metric_c.metric("Reversal radius", format_radius(reconstructed_radius), delta=radius_delta)

    st.markdown("#### Axial-field reconstruction from sparse magnetic probes")
    st.plotly_chart(
        field_comparison_figure(grid, equilibrium, probes, reconstruction, idw_bz),
        width="stretch",
        config={"displaylogo": False, "scrollZoom": False},
    )
    left, right = st.columns([1.45, 1.0])
    with left:
        st.plotly_chart(midplane_figure(grid, equilibrium, reconstruction, idw_bz), width="stretch")
    with right:
        st.plotly_chart(residual_figure(probes, reconstruction), width="stretch")

    st.markdown(
        '<div class="method-note"><strong>What is constrained:</strong> the reconstructed field is derived '
        "from one axisymmetric poloidal flux function. Br and Bz are fitted together, so the output preserves "
        "the shared flux structure and strongly suppresses numerical div(B) violations. The IDW baseline "
        "interpolates each component independently.</div>",
        unsafe_allow_html=True,
    )

with tab_shot:
    shot = generate_shot(reversal_strength=reversal_strength)
    st.plotly_chart(shot_figure(shot), width="stretch")
    event_a, event_b, event_c = st.columns(3)
    reversal_rows = shot.loc[shot.center_bz_T < 0.0, "time_ms"]
    event_a.metric("First reversal", f"{reversal_rows.iloc[0]:.2f} ms" if len(reversal_rows) else "Not detected")
    event_b.metric("Peak density proxy", f"{shot.density_1e19_m3.max():.2f} × 10¹⁹ m⁻³")
    event_c.metric("Peak stored-energy proxy", f"{shot.energy_kJ.max():.2f} kJ")
    st.dataframe(
        shot.iloc[::12].round(4),
        width="stretch",
        hide_index=True,
        column_config={
            "time_ms": "Time [ms]",
            "coil_current_kA": "Coil current [kA]",
            "center_bz_T": "Center Bz [T]",
            "edge_bz_T": "Edge Bz [T]",
            "density_1e19_m3": "Density proxy [10¹⁹ m⁻³]",
            "energy_kJ": "Energy proxy [kJ]",
        },
    )
    st.download_button(
        "Download synthetic shot CSV",
        shot.to_csv(index=False).encode("utf-8"),
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
        )
    with detail_col:
        st.subheader("Evaluation matrix")
        best_row, best_col = np.unravel_index(np.argmax(idw_errors - physics_errors), physics_errors.shape)
        st.metric("Largest NRMSE advantage", f"{(idw_errors - physics_errors)[best_row, best_col]:.1%}")
        st.metric("At probe count", str(probe_counts[best_col]))
        st.metric("At noise level", f"{noise_levels[best_row]:.1f}%")
        st.caption(
            "Positive cells mean the shared flux representation reconstructs the synthetic reference more accurately than independent interpolation."
        )
        sensitivity_table = pd.DataFrame(
            physics_errors,
            index=[f"{value:.1f}%" for value in noise_levels],
            columns=[str(value) for value in probe_counts],
        )
        st.dataframe(sensitivity_table.style.format("{:.2%}"), width="stretch")

with tab_agent:
    st.subheader("Experiment-review agent with deterministic scientific tools")
    flow_columns = st.columns([1, 0.18, 1, 0.18, 1, 0.18, 1])
    flow = [
        ("数据接入与版本", "Shot ID, channel schema, units, timestamps, raw-file hash"),
        ("质量门禁", "Missingness, drift, saturation, time alignment, anomaly score"),
        ("Python 科学工具", "Flux reconstruction, residuals, constraints, uncertainty, comparison"),
        ("专家审阅报告", "Evidence-linked summary, unresolved issues, human approval"),
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

    st.markdown("#### LLM responsibility boundary")
    llm_col, tool_col = st.columns(2)
    with llm_col:
        st.markdown(
            "**Qwen/Dify handles**\n\n"
            "- Intent recognition and workflow orchestration\n"
            "- Versioned knowledge retrieval with citations\n"
            "- Tool-result explanation and report drafting\n"
            "- Escalation when evidence is insufficient"
        )
    with tool_col:
        st.markdown(
            "**Deterministic tools handle**\n\n"
            "- Signal alignment, unit checks, and data quality\n"
            "- Magnetic reconstruction and physical constraints\n"
            "- Optimization, uncertainty, and acceptance metrics\n"
            "- Safety rules, permissions, and immutable logs"
        )

    manifest = {
        "project": "FRC Equilibrium Reconstruction Lab",
        "scope": "synthetic educational demonstrator",
        "inputs": {
            "reversal_strength": reversal_strength,
            "elongation": elongation,
            "probe_count": probe_count,
            "noise_percent": noise_percent,
            "fault_mode": fault_mode,
            "regularization": regularization,
        },
        "results": {
            "physics_nrmse": physics_error,
            "idw_nrmse": idw_error,
            "physics_divergence_residual": physics_div_norm,
            "idw_divergence_residual": idw_div_norm,
            "reconstructed_reversal_radius_m": reconstructed_radius,
        },
        "review_required": True,
    }
    st.download_button(
        "Download reproducibility manifest",
        json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="reconstruction_manifest.json",
        mime="application/json",
    )

with tab_interview:
    st.subheader("项农老师研究方向：已验证事实与项目连接")
    st.markdown(
        """
        <div class="research-row">
          <span class="tag">平衡重建</span><span class="tag">深度学习</span>
          <strong>EAST 实时平衡重建</strong><br>
          <span>以磁测量为输入、EFIT 磁通量为目标，用深度网络实现高空间分辨率和实时控制时间约束下的快速重建。</span><br>
          <a href="https://doi.org/10.1063/5.0152318" target="_blank">Physics of Plasmas, 2023</a>
        </div>
        <div class="research-row">
          <span class="tag">平衡自洽</span><span class="tag">电流模拟</span>
          <strong>EAST 平衡与电流模拟一致性约束</strong><br>
          <span>针对 POINT/MSE 等间接诊断的不确定性，引入电流模拟一致性，并关注 RF 波驱动电流、q 剖面与沉积位置的耦合。</span><br>
          <a href="https://doi.org/10.1088/1741-4326/ad35d7" target="_blank">Nuclear Fusion, 2024</a>
        </div>
        <div class="research-row">
          <span class="tag">RF 波</span><span class="tag">等离子体互作用</span>
          <strong>低杂波、电子伯恩斯坦波与边界等离子体</strong><br>
          <span>公开论文覆盖电子伯恩斯坦波二次谐波生成，以及 EAST 上低杂波天线前等离子体与保护限制器互作用。</span><br>
          <a href="https://doi.org/10.1103/PhysRevLett.100.085002" target="_blank">Physical Review Letters, 2008</a>
          &nbsp;·&nbsp;
          <a href="https://doi.org/10.1088/1741-4326/ab082c" target="_blank">Nuclear Fusion, 2019</a>
        </div>
        <div class="research-row">
          <span class="tag">集成建模</span><span class="tag">FAIR4RS</span>
          <strong>磁约束聚变科研软件的可复现工程</strong><br>
          <span>FyDev 面向 EAST 研究软件的构建、部署、调用和版本化，强调唯一标识、Python 模块化、包管理和可复现工作流。</span><br>
          <a href="https://doi.org/10.1038/s41597-023-02470-y" target="_blank">Scientific Data, 2023</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 你的经历如何对上这些问题")
    mapping = pd.DataFrame(
        [
            {
                "项农老师关注的方法": "稀疏磁测量 -> 平衡/状态重建",
                "项目中的对应证据": "磁通量基底拟合 + IDW 基线 + NRMSE/div(B) 验收",
                "你的可迁移经历": "时序数据治理、复杂系统建模、Python/NumPy/Plotly",
                "面试价值": "能将诊断问题拆成数据、先验、模型与评估闭环",
            },
            {
                "项农老师关注的方法": "电流模拟一致性与物理约束",
                "项目中的对应证据": "Br/Bz 共用单一磁通函数，显式检验 div(B) 残差",
                "你的可迁移经历": "MILP/动态规划、状态依赖切换、随机扰动与脉冲效应",
                "面试价值": "不让 AI 跳过约束，把可行性与安全性变成确定性门禁",
            },
            {
                "项农老师关注的方法": "集成建模与可复现科研软件",
                "项目中的对应证据": "固定随机种子、模块化内核、测试、CSV/清单导出、明确适用域",
                "你的可迁移经历": "Dify 多智能体、工具调用、数据校验、报告交付",
                "面试价值": "可把物理研究原型做成可追溯、可回放、专家可复核的 AI 应用",
            },
        ]
    )
    st.dataframe(mapping, width="stretch", hide_index=True, height=255)

    st.markdown("#### 面试中的项目说法")
    st.markdown(
        """
        > 我注意到您的研究不只是用模型拟合实验数据，而是把平衡重建、电流模拟一致性和实时约束放在同一个问题里。我做的这个原型没有冒充真实 FRC 平衡求解器，而是用合成数据验证一个工程命题：稀疏诊断下，将 Br 和 Bz 纳入统一磁通表示，能否比独立插值更好地保留场反转与无散结构。

        > 我在赛唯做储能 AI Agent 时，核心数值并不交给大模型自由生成，而是交给 Python、规则和优化器；Dify 负责数据校验、流程编排、工具调用和报告解释。我认为这与聚变科研 AI 应用的要求是一致的：先保证物理和数据链路可验证，再让 LLM 提升人机交互与知识复用效率。

        > 如果进入团队，我会先用真实数据字典、诊断位置和专家标注替换演示假设，按实验批次做离线回放和误差分解；在通过物理专家审核前，不将它用于装置参数下发或在线闭环控制。
        """
    )

    st.caption(
        "Identity source: ORCID 0000-0002-8663-0470 lists Nong Xiang at the Institute of Plasma Physics, Chinese Academy of Sciences. "
        "All paper links above resolve through DOI metadata."
    )
