from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


OBSERVED = "#0F8B8D"
MODEL = "#7C3AED"
ACCENT = "#0EA5E9"
INK = "#102A43"
MUTED = "#64748B"
GRID = "#DCE7EF"


def _finish(fig: go.Figure, *, height: int = 360, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=48, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, family="Arial, Malgun Gothic, sans-serif"),
        hoverlabel=dict(bgcolor="white"),
        showlegend=legend,
        legend=dict(orientation="h", y=1.12, x=0),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def sec_context_figure(history: pd.DataFrame, row: pd.Series) -> go.Figure:
    values = history["sec_kwh_m3"].astype(float)
    q10, q50, q90 = values.quantile([0.10, 0.50, 0.90])
    lower = float(values.min())
    upper = float(values.max())
    padding = max((upper - lower) * 0.08, 0.05)

    fig = go.Figure()
    fig.add_shape(
        type="rect",
        x0=q10,
        x1=q90,
        y0=-0.22,
        y1=0.22,
        fillcolor="#DCECF1",
        line_width=0,
        layer="below",
    )
    fig.add_vline(x=q50, line_dash="dot", line_color=MUTED, annotation_text="중앙값")
    fig.add_trace(
        go.Scatter(
            x=[row["sec_kwh_m3"]],
            y=[0.07],
            mode="markers+text",
            marker=dict(size=15, color=OBSERVED, symbol="circle"),
            text=[f"기록 {row['sec_kwh_m3']:.2f}"],
            textposition="top center",
            name="기록값",
            hovertemplate="기록 SEC %{x:.3f} kWh/m³<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[row["model_sec_kwh_m3"]],
            y=[-0.07],
            mode="markers+text",
            marker=dict(size=15, color=MODEL, symbol="diamond"),
            text=[f"모델 {row['model_sec_kwh_m3']:.2f}"],
            textposition="bottom center",
            name="모델 추정",
            hovertemplate="모델 SEC %{x:.3f} kWh/m³<extra></extra>",
        )
    )
    fig.update_xaxes(
        range=[lower - padding, upper + padding],
        title="비에너지소비량(SEC, kWh/m³)",
    )
    fig.update_yaxes(visible=False, range=[-0.42, 0.42])
    fig.update_layout(title="전체 기록 분포 안에서 본 선택 시점")
    return _finish(fig, height=270, legend=False)


def local_trend_figure(history: pd.DataFrame, timestamp: pd.Timestamp) -> go.Figure:
    start = timestamp - pd.Timedelta(hours=24)
    end = timestamp + pd.Timedelta(hours=24)
    window = history.loc[history["timestamp"].between(start, end)].copy()
    if len(window) < 2:
        window = history.tail(96).copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=window["timestamp"],
            y=window["sec_kwh_m3"],
            mode="lines+markers",
            line=dict(color=OBSERVED, width=2.5),
            marker=dict(size=5),
            name="기록 SEC",
            hovertemplate="%{x|%m-%d %H:%M}<br>%{y:.3f} kWh/m³<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=window["timestamp"],
            y=window["model_sec_kwh_m3"],
            mode="lines",
            line=dict(color=MODEL, width=2, dash="dot"),
            name="모델 추정 SEC",
            hovertemplate="%{x|%m-%d %H:%M}<br>%{y:.3f} kWh/m³<extra></extra>",
        )
    )
    fig.add_vline(x=timestamp.timestamp() * 1000, line_color=INK, line_dash="dash")
    fig.update_layout(title="선택 시점 전후 SEC 추이")
    fig.update_yaxes(title="kWh/m³")
    return _finish(fig, height=360)


def monthly_operations_figure(monthly: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.16,
        subplot_titles=("월평균 1단 인입압력", "월평균 SEC"),
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["month_label"],
            y=monthly["pressure_stage1_bar"],
            mode="lines+markers",
            line=dict(color=OBSERVED, width=3),
            name="기록 압력",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["month_label"],
            y=monthly["model_pressure_bar"],
            mode="lines+markers",
            line=dict(color=MODEL, width=2, dash="dot"),
            name="모델 압력",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=monthly["month_label"],
            y=monthly["sec_kwh_m3"],
            marker_color=OBSERVED,
            opacity=0.78,
            name="기록 SEC",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["month_label"],
            y=monthly["model_sec_kwh_m3"],
            mode="lines+markers",
            line=dict(color=MODEL, width=2),
            name="모델 SEC",
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title="bar", row=1, col=1)
    fig.update_yaxes(title="kWh/m³", row=2, col=1)
    return _finish(fig, height=610)


def monthly_quality_figure(monthly: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.18,
        subplot_titles=("1단 생산수 TDS", "2단 생산수 TDS"),
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["month_label"],
            y=monthly["tds_stage1_mg_l"],
            mode="lines+markers",
            line=dict(color=ACCENT, width=3),
            fill="tozeroy",
            fillcolor="rgba(14,165,233,0.10)",
            name="1단 TDS",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["month_label"],
            y=monthly["tds_stage2_mg_l"],
            mode="lines+markers",
            line=dict(color=OBSERVED, width=3),
            fill="tozeroy",
            fillcolor="rgba(15,139,141,0.10)",
            name="2단 TDS",
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title="mg/L", row=1, col=1)
    fig.update_yaxes(title="mg/L", row=2, col=1)
    return _finish(fig, height=610, legend=False)


def observation_density_figure(monthly: pd.DataFrame) -> go.Figure:
    colors = [OBSERVED if value >= 60 else "#F59E0B" for value in monthly["exact_hour_density_pct"]]
    fig = go.Figure(
        go.Bar(
            x=monthly["month_label"],
            y=monthly["exact_hour_density_pct"],
            marker_color=colors,
            text=monthly["samples"].map(lambda value: f"{int(value):,}건"),
            textposition="outside",
            hovertemplate="%{x}<br>정시 관측 밀도 %{y:.1f}%<br>%{text}<extra></extra>",
        )
    )
    fig.add_hline(y=100, line_color=MUTED, line_dash="dot")
    fig.update_layout(title="월별 정시 관측 밀도와 표본 수")
    fig.update_yaxes(title="정시 관측 밀도 (%)", range=[0, 112])
    return _finish(fig, height=350, legend=False)


def error_watch_figure(
    history: pd.DataFrame,
    *,
    pressure_cutoff: float,
    sec_cutoff: float,
) -> go.Figure:
    """Show model/record gaps over time with data-derived review cutoffs."""
    frame = history.sort_values("timestamp")
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.14,
        subplot_titles=("1단 압력 모델-기록 차이", "SEC 모델-기록 차이"),
    )
    fig.add_trace(
        go.Scatter(
            x=frame["timestamp"],
            y=frame["pressure_error_bar"].abs(),
            mode="lines",
            line=dict(color=MODEL, width=1.5),
            name="압력 절대 차이",
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.2f} bar<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frame["timestamp"],
            y=frame["sec_error_kwh_m3"].abs(),
            mode="lines",
            line=dict(color=OBSERVED, width=1.5),
            name="SEC 절대 차이",
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.3f} kWh/m³<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=pressure_cutoff,
        row=1,
        col=1,
        line_color="#F59E0B",
        line_dash="dot",
        annotation_text="95백분위",
        annotation_position="top left",
    )
    fig.add_hline(
        y=sec_cutoff,
        row=2,
        col=1,
        line_color="#F59E0B",
        line_dash="dot",
        annotation_text="95백분위",
        annotation_position="top left",
    )
    fig.update_yaxes(title="bar", row=1, col=1)
    fig.update_yaxes(title="kWh/m³", row=2, col=1)
    fig.update_layout(title="시간순 모델-기록 차이와 통계적 검토 기준")
    return _finish(fig, height=560, legend=False)


def feature_signal_figure(
    frame: pd.DataFrame,
    *,
    title: str,
    value_label: str,
    signed: bool = False,
) -> go.Figure:
    """Render compact, readable model signal bars for the model card."""
    data = frame.sort_values("value", ascending=True).copy()
    colors = (
        [MODEL if value >= 0 else OBSERVED for value in data["value"]]
        if signed
        else [ACCENT] * len(data)
    )
    fig = go.Figure(
        go.Bar(
            x=data["value"],
            y=data["feature"],
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>%{x:.4f}<extra></extra>",
        )
    )
    if signed:
        fig.add_vline(x=0, line_color=MUTED, line_width=1)
    fig.update_layout(title=title)
    fig.update_xaxes(title=value_label)
    return _finish(fig, height=max(260, 72 * len(data) + 100), legend=False)


def sensitivity_figure(
    pressure_model,
    sec_model,
    scenario: dict[str, float],
    ranges: pd.DataFrame,
) -> go.Figure:
    temperatures = np.linspace(
        ranges.loc["temperature_c", "min"],
        ranges.loc["temperature_c", "max"],
        80,
    )
    pressure_input = pd.DataFrame(
        {
            "수온": temperatures,
            "수소이온농도": np.repeat(scenario["ph"], len(temperatures)),
        }
    )
    pressures = np.asarray(pressure_model.predict(pressure_input), dtype=float)

    pressure_axis = np.linspace(
        ranges.loc["pressure_stage1_bar", "min"],
        ranges.loc["pressure_stage1_bar", "max"],
        80,
    )
    sec_input = pd.DataFrame(
        {
            "총인": np.repeat(scenario["total_p_mg_l"], len(pressure_axis)),
            "화학적산소요구량": np.repeat(scenario["cod_mg_l"], len(pressure_axis)),
            "총질소": np.repeat(scenario["total_n_mg_l"], len(pressure_axis)),
            "탁도": np.repeat(scenario["turbidity_ntu"], len(pressure_axis)),
            "1차 인입압력": pressure_axis,
        }
    )
    sec = np.asarray(sec_model.predict(sec_input), dtype=float)

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.12,
        subplot_titles=("수온 변화에 따른 압력 추정", "압력 변화에 따른 SEC 추정"),
    )
    fig.add_trace(
        go.Scatter(x=temperatures, y=pressures, line=dict(color=MODEL, width=3), name="압력"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=pressure_axis, y=sec, line=dict(color=OBSERVED, width=3), name="SEC"),
        row=1,
        col=2,
    )
    fig.update_xaxes(title="수온 (°C)", row=1, col=1)
    fig.update_yaxes(title="bar", row=1, col=1)
    fig.update_xaxes(title="1단 인입압력 (bar)", row=1, col=2)
    fig.update_yaxes(title="kWh/m³", row=1, col=2)
    return _finish(fig, height=380, legend=False)

