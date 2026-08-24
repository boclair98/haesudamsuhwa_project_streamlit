from __future__ import annotations

import pandas as pd
import streamlit as st

from desalination.analytics import (
    anomaly_watchlist,
    diagnostics,
    enrich_history,
    monthly_summary,
    percentile_rank,
    predict_scenario,
    support_ranges,
)
from desalination.charts import (
    error_watch_figure,
    local_trend_figure,
    monthly_operations_figure,
    monthly_quality_figure,
    observation_density_figure,
    sec_context_figure,
    sensitivity_figure,
)
from desalination.resources import ResourceError, load_history, load_models


st.set_page_config(
    page_title="RO Lens · 해수담수화 공정 분석",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #102a43;
        --muted: #61758a;
        --teal: #0f8b8d;
        --teal-soft: #dff4f2;
        --violet: #7c3aed;
        --sky: #0ea5e9;
        --line: #dce7ef;
        --panel: #ffffff;
        --canvas: #f4f8fb;
    }
    html, body, [class*="css"] {
        font-family: Inter, Pretendard, "Noto Sans KR", "Malgun Gothic", Arial, sans-serif;
    }
    .stApp { background: var(--canvas); color: var(--ink); }
    [data-testid="stHeader"] { height: 0; background: transparent; }
    [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
    .block-container { max-width: 1240px; padding-top: 1.6rem; padding-bottom: 4rem; }
    #MainMenu, footer { visibility: hidden; }
    h1, h2, h3 { color: var(--ink); letter-spacing: -0.025em; }
    p, label, .stCaption { color: var(--muted); }
    a { color: #087f8c !important; }
    .hero {
        position: relative;
        overflow: hidden;
        padding: 32px 34px;
        margin-bottom: 18px;
        border-radius: 24px;
        color: white;
        background:
            radial-gradient(circle at 88% 12%, rgba(56,189,248,.35), transparent 30%),
            linear-gradient(120deg, #082f49 0%, #0f5265 52%, #0f766e 100%);
        box-shadow: 0 18px 44px rgba(8,47,73,.15);
    }
    .hero:after {
        content: "";
        position: absolute;
        width: 290px; height: 290px; right: -90px; bottom: -190px;
        border: 42px solid rgba(255,255,255,.08); border-radius: 50%;
    }
    .hero-kicker { font-size: .78rem; font-weight: 800; letter-spacing: .14em; opacity: .82; }
    .hero h1 { margin: 8px 0 8px; color: white; font-size: clamp(2rem, 4vw, 3.3rem); }
    .hero p { max-width: 760px; margin: 0; color: rgba(255,255,255,.82); font-size: 1rem; line-height: 1.7; }
    .badges { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 20px; }
    .badge {
        display: inline-flex; align-items: center; gap: 6px;
        border: 1px solid rgba(255,255,255,.24); border-radius: 999px;
        padding: 7px 11px; color: white; background: rgba(255,255,255,.10);
        font-size: .78rem; font-weight: 700;
    }
    .trustbar {
        display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;
        margin: 2px 0 22px;
    }
    .trustitem {
        padding: 13px 15px; border: 1px solid var(--line); border-radius: 14px;
        background: rgba(255,255,255,.72); color: var(--muted); font-size: .79rem;
    }
    .trustitem strong { display: block; margin-top: 3px; color: var(--ink); font-size: .94rem; }
    div[role="radiogroup"] {
        display: flex; flex-wrap: wrap; gap: 6px; padding: 6px; margin-bottom: 16px;
        border: 1px solid var(--line); border-radius: 14px; background: white;
    }
    div[role="radiogroup"] label {
        padding: 4px 12px; border-radius: 10px;
    }
    [data-testid="stMetric"] {
        min-height: 126px; padding: 18px 18px 14px;
        border: 1px solid var(--line); border-radius: 16px; background: var(--panel);
        box-shadow: 0 7px 18px rgba(15,35,55,.035);
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--ink); letter-spacing: -.035em; }
    [data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
        border: 1px solid var(--line); border-radius: 18px; background: white;
        padding: 6px; overflow: hidden;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line) !important; border-radius: 18px !important;
        background: rgba(255,255,255,.84);
    }
    .section-head { margin: 28px 0 14px; }
    .section-kicker { color: var(--teal); font-size: .75rem; font-weight: 900; letter-spacing: .12em; }
    .section-head h2 { margin: 4px 0 4px; font-size: 1.55rem; }
    .section-head p { margin: 0; font-size: .9rem; }
    .flow-grid {
        display: grid; grid-template-columns: repeat(5, minmax(0,1fr)); gap: 10px;
        margin: 8px 0 4px;
    }
    .flow-step {
        position: relative; min-height: 112px; padding: 16px;
        border: 1px solid var(--line); border-radius: 16px; background: white;
    }
    .flow-step:after {
        content: "→"; position: absolute; right: -15px; top: 42px; z-index: 2;
        width: 20px; height: 20px; color: var(--teal); font-weight: 900;
    }
    .flow-step:last-child:after { display: none; }
    .flow-index { color: var(--teal); font-weight: 900; font-size: .72rem; letter-spacing: .08em; }
    .flow-name { margin-top: 6px; color: var(--ink); font-weight: 800; }
    .flow-value { margin-top: 8px; color: var(--muted); font-size: .8rem; line-height: 1.45; }
    .note-card {
        padding: 18px 20px; border-left: 4px solid var(--teal); border-radius: 4px 14px 14px 4px;
        background: var(--teal-soft); color: #155e62; line-height: 1.65; font-size: .9rem;
    }
    .model-card {
        height: 100%; padding: 18px; border: 1px solid var(--line); border-radius: 16px; background: white;
    }
    .model-card .eyebrow { color: var(--violet); font-weight: 900; font-size: .74rem; letter-spacing: .09em; }
    .model-card h3 { margin: 6px 0 10px; font-size: 1.08rem; }
    .model-card p { margin: 5px 0; font-size: .84rem; line-height: 1.55; }
    .footer-note { margin-top: 42px; padding-top: 16px; border-top: 1px solid var(--line); font-size: .78rem; color: var(--muted); }
    @media (max-width: 820px) {
        .block-container { padding: 1rem 1rem 3rem; }
        .hero { padding: 24px 22px; border-radius: 18px; }
        .trustbar { grid-template-columns: repeat(2, 1fr); }
        .flow-grid { grid-template-columns: 1fr; }
        .flow-step:after { content: "↓"; right: 50%; top: auto; bottom: -18px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


PLOT_CONFIG = {
    "displayModeBar": False,
    "displaylogo": False,
    "responsive": True,
}


@st.cache_data(show_spinner="정제 공정 기록을 준비하는 중입니다…")
def cached_history():
    return load_history()


@st.cache_resource(show_spinner="검증된 모델을 불러오는 중입니다…")
def cached_models():
    return load_models()


@st.cache_data(show_spinner="전체 기록의 모델 추정치를 계산하는 중입니다…")
def cached_enriched(frame, _pressure_model, _sec_model):
    return enrich_history(frame, _pressure_model, _sec_model)


def section(kicker: str, title: str, description: str = "") -> None:
    st.markdown(
        f"""
        <div class="section-head">
          <div class="section-kicker">{kicker}</div>
          <h2>{title}</h2>
          <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def delta_text(
    current: float,
    previous: float | None,
    unit: str,
    comparison_label: str | None = "전월",
) -> str | None:
    if previous is None:
        return None
    difference = current - previous
    sign = "+" if difference > 0 else ""
    label = comparison_label or "비교"
    return f"{label} 대비 {sign}{difference:.2f} {unit}".strip()


def plot(fig) -> None:
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)


try:
    history, profile = cached_history()
    models = cached_models()
    history = cached_enriched(history, models.pressure, models.sec)
except ResourceError as exc:
    st.error(f"대시보드를 시작하지 못했습니다: {exc}")
    st.stop()

model_diagnostics = diagnostics(history)
monthly = monthly_summary(history)
ranges = support_ranges(history)
start_time = history["timestamp"].min()
end_time = history["timestamp"].max()

st.markdown(
    f"""
    <div class="hero">
      <div class="hero-kicker">RO LENS · PROCESS EXPLORER</div>
      <h1>해수담수화 공정 분석 데모</h1>
      <p>2021년 천수만 원수 관측자료와 저장소의 RO 샘플 공정 기록을 바탕으로
      1단 인입압력과 비에너지소비량(SEC)을 탐색합니다. 실제 설비 제어와 연결되지 않습니다.</p>
      <div class="badges">
        <span class="badge">● 연구용 데모</span>
        <span class="badge">2021 기록 데이터</span>
        <span class="badge">실시간 설비 연동 아님</span>
        <span class="badge">모델 추정값 별도 표시</span>
      </div>
    </div>
    <div class="trustbar">
      <div class="trustitem">사용 가능한 공정 기록<strong>{len(history):,}건</strong></div>
      <div class="trustitem">기록 기간<strong>{start_time:%Y-%m-%d} — {end_time:%Y-%m-%d}</strong></div>
      <div class="trustitem">압력 모델 전체기록 MAE<strong>{model_diagnostics['pressure']['mae']:.2f} bar</strong></div>
      <div class="trustitem">연쇄 SEC 모델 전체기록 MAE<strong>{model_diagnostics['sec_chain']['mae']:.2f} kWh/m³</strong></div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.radio(
    "대시보드 메뉴",
    ["운전 스냅샷", "기간 성과", "예측 실험실", "운영 인사이트", "데이터·모델 카드"],
    horizontal=True,
    label_visibility="collapsed",
)


if page == "운전 스냅샷":
    section(
        "HISTORICAL REPLAY",
        "실제 존재하는 기록 시점만 탐색",
        "기록값과 모델 추정값을 같은 시점에서 비교합니다.",
    )
    dates = sorted(history["timestamp"].dt.date.unique(), reverse=True)
    filter_col1, filter_col2, filter_col3 = st.columns([1.2, 1, 1.8])
    with filter_col1:
        selected_date = st.selectbox(
            "기록 날짜",
            dates,
            format_func=lambda value: value.strftime("%Y년 %m월 %d일"),
        )
    available = history.loc[history["timestamp"].dt.date.eq(selected_date)]
    with filter_col2:
        selected_timestamp = st.selectbox(
            "기록 시각",
            available["timestamp"].tolist(),
            index=len(available) - 1,
            format_func=lambda value: value.strftime("%H:%M"),
        )
    row = history.loc[history["timestamp"].eq(selected_timestamp)].iloc[0]
    sec_percentile = percentile_rank(history["sec_kwh_m3"], row["sec_kwh_m3"])
    with filter_col3:
        st.markdown(
            f"""
            <div class="note-card">
              선택 기록의 SEC는 전체 기록의 <strong>{sec_percentile:.0f}백분위</strong>에 위치합니다.
              이는 운영 적합 판정이 아니라 과거 분포 안에서의 상대적 위치입니다.
            </div>
            """,
            unsafe_allow_html=True,
        )

    section("SNAPSHOT", "선택 시점 공정 기록과 모델 추정", "청록은 기록값, 보라는 모델 추정값입니다.")
    metric_cols = st.columns(5)
    metric_cols[0].metric(
        "모델 추정 1단 압력",
        f"{row['model_pressure_bar']:.2f} bar",
        f"기록 대비 {row['pressure_error_bar']:+.2f} bar",
        delta_color="off",
    )
    metric_cols[1].metric("기록 1단 압력", f"{row['pressure_stage1_bar']:.2f} bar")
    metric_cols[2].metric(
        "모델 추정 SEC",
        f"{row['model_sec_kwh_m3']:.2f} kWh/m³",
        f"기록 대비 {row['sec_error_kwh_m3']:+.2f} kWh/m³",
        delta_color="off",
    )
    metric_cols[3].metric("기록 SEC", f"{row['sec_kwh_m3']:.2f} kWh/m³")
    metric_cols[4].metric("최종 생산수 TDS", f"{row['tds_stage2_mg_l']:.2f} mg/L")

    section("PROCESS", "RO 공정 흐름", "구조 설명용 공정도이며 자동 제어 명령을 생성하지 않습니다.")
    st.markdown(
        f"""
        <div class="flow-grid">
          <div class="flow-step"><div class="flow-index">01 · FEED</div><div class="flow-name">원수 유입</div><div class="flow-value">수온 {row['temperature_c']:.1f}°C<br>pH {row['ph']:.2f}</div></div>
          <div class="flow-step"><div class="flow-index">02 · PRE</div><div class="flow-name">전처리</div><div class="flow-value">탁도 {row['turbidity_ntu']:.2f} NTU<br>COD {row['cod_mg_l']:.2f} mg/L</div></div>
          <div class="flow-step"><div class="flow-index">03 · RO 1</div><div class="flow-name">1단 RO</div><div class="flow-value">압력 {row['pressure_stage1_bar']:.2f} bar<br>TDS {row['tds_stage1_mg_l']:.2f} mg/L</div></div>
          <div class="flow-step"><div class="flow-index">04 · RO 2</div><div class="flow-name">2단 RO</div><div class="flow-value">압력 {row['pressure_stage2_bar']:.2f} bar<br>TDS {row['tds_stage2_mg_l']:.2f} mg/L</div></div>
          <div class="flow-step"><div class="flow-index">05 · PRODUCT</div><div class="flow-name">생산수</div><div class="flow-value">SEC {row['sec_kwh_m3']:.2f} kWh/m³<br>유량·회수율 미수록</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chart_col1, chart_col2 = st.columns([1, 1.55])
    with chart_col1:
        plot(sec_context_figure(history, row))
    with chart_col2:
        plot(local_trend_figure(history, selected_timestamp))

    section("FEED WATER", "선택 시점 원수 조건", "모델에 사용된 수질 입력을 단위와 함께 표시합니다.")
    water_cols = st.columns(4)
    water_cols[0].metric("탁도", f"{row['turbidity_ntu']:.2f} NTU")
    water_cols[1].metric("COD", f"{row['cod_mg_l']:.2f} mg/L")
    water_cols[2].metric("총질소", f"{row['total_n_mg_l']:.3f} mg/L")
    water_cols[3].metric("총인", f"{row['total_p_mg_l']:.3f} mg/L")


elif page == "기간 성과":
    section(
        "MONTHLY REVIEW",
        "완전한 월 단위 비교",
        "월별 표본 수와 정시 관측 밀도를 함께 보아 희소한 기간의 과대 해석을 막습니다.",
    )
    month_labels = monthly["month_label"].tolist()
    month_select_col1, month_select_col2 = st.columns(2)
    with month_select_col1:
        selected_month = st.selectbox("분석 월", month_labels, index=len(month_labels) - 1)
    selected_index = month_labels.index(selected_month)
    with month_select_col2:
        default_compare_index = max(0, selected_index - 1)
        compare_month = st.selectbox(
            "비교 월",
            month_labels,
            index=default_compare_index,
            help="분석 월과 다른 월을 골라 지표 차이를 확인합니다.",
        )
    current = monthly.iloc[selected_index]
    compare_index = month_labels.index(compare_month)
    comparison = None if compare_index == selected_index else monthly.iloc[compare_index]
    comparison_label = "비교 월" if comparison is not None else None

    metric_cols = st.columns(5)
    metric_cols[0].metric(
        "월평균 1단 압력",
        f"{current['pressure_stage1_bar']:.2f} bar",
        delta_text(
            current["pressure_stage1_bar"],
            None if comparison is None else comparison["pressure_stage1_bar"],
            "bar",
            comparison_label,
        ),
        delta_color="off",
    )
    metric_cols[1].metric(
        "월평균 SEC",
        f"{current['sec_kwh_m3']:.2f} kWh/m³",
        delta_text(
            current["sec_kwh_m3"],
            None if comparison is None else comparison["sec_kwh_m3"],
            "kWh/m³",
            comparison_label,
        ),
        delta_color="off",
    )
    metric_cols[2].metric(
        "월평균 최종 TDS",
        f"{current['tds_stage2_mg_l']:.2f} mg/L",
        delta_text(
            current["tds_stage2_mg_l"],
            None if comparison is None else comparison["tds_stage2_mg_l"],
            "mg/L",
            comparison_label,
        ),
        delta_color="off",
    )
    metric_cols[3].metric("표본 수", f"{int(current['samples']):,}건")
    metric_cols[4].metric("정시 관측 밀도", f"{current['exact_hour_density_pct']:.1f}%")

    st.caption(
        "월평균은 생산량 가중치가 없는 관측값의 단순 평균입니다. "
        "델타는 선택한 비교 월 기준이며, 정시 관측 밀도는 해당 월 전체 시간 수 대비 분·초가 00인 기록의 비율입니다."
    )

    chart_col1, chart_col2 = st.columns([1.25, 1])
    with chart_col1:
        plot(monthly_operations_figure(monthly))
    with chart_col2:
        plot(monthly_quality_figure(monthly))
    plot(observation_density_figure(monthly))

    section("TABLE", "월간 요약 데이터", "차트에 사용한 집계값을 내려받을 수 있습니다.")
    summary_view = monthly[
        [
            "month_label",
            "samples",
            "exact_hour_density_pct",
            "pressure_stage1_bar",
            "sec_kwh_m3",
            "tds_stage1_mg_l",
            "tds_stage2_mg_l",
        ]
    ].rename(
        columns={
            "month_label": "월",
            "samples": "표본 수",
            "exact_hour_density_pct": "정시 관측 밀도(%)",
            "pressure_stage1_bar": "1단 압력(bar)",
            "sec_kwh_m3": "SEC(kWh/m³)",
            "tds_stage1_mg_l": "1단 TDS(mg/L)",
            "tds_stage2_mg_l": "2단 TDS(mg/L)",
        }
    )
    st.dataframe(
        summary_view.style.format(
            {
                "정시 관측 밀도(%)": "{:.1f}",
                "1단 압력(bar)": "{:.2f}",
                "SEC(kWh/m³)": "{:.2f}",
                "1단 TDS(mg/L)": "{:.2f}",
                "2단 TDS(mg/L)": "{:.2f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "월간 요약 CSV 받기",
        summary_view.to_csv(index=False).encode("utf-8-sig"),
        file_name="ro_monthly_summary_2021.csv",
        mime="text/csv",
    )


elif page == "예측 실험실":
    section(
        "WHAT-IF LAB",
        "기록 범위 안에서 조건 비교",
        "수온·pH로 압력을 추정한 뒤 그 결과와 원수 수질을 SEC 모델에 연결합니다.",
    )
    st.info(
        "입력 범위는 저장소의 사용 가능한 공정 기록 범위로 제한됩니다. "
        "결과는 연구용 추정치이며 설비 설정값이나 운전 권고가 아닙니다."
    )

    def slider_for(column: str, label: str, step: float, fmt: str = "%.2f") -> float:
        spec = ranges.loc[column]
        value = st.slider(
            label,
            min_value=float(spec["min"]),
            max_value=float(spec["max"]),
            value=float(spec["median"]),
            step=step,
            format=fmt,
        )
        st.caption(
            f"기록 중앙 90%: {spec['p05']:.3f}–{spec['p95']:.3f} {spec['unit']}"
        )
        return value

    input_col1, input_col2, input_col3 = st.columns(3)
    with input_col1:
        temperature = slider_for("temperature_c", "수온 (°C)", 0.1, "%.1f")
        ph = slider_for("ph", "pH", 0.01)
    with input_col2:
        turbidity = slider_for("turbidity_ntu", "탁도 (NTU)", 0.1)
        cod = slider_for("cod_mg_l", "COD (mg/L)", 0.1)
    with input_col3:
        total_n = slider_for("total_n_mg_l", "총질소 (mg/L)", 0.01, "%.3f")
        total_p = slider_for("total_p_mg_l", "총인 (mg/L)", 0.001, "%.3f")

    scenario = {
        "temperature_c": temperature,
        "ph": ph,
        "turbidity_ntu": turbidity,
        "cod_mg_l": cod,
        "total_n_mg_l": total_n,
        "total_p_mg_l": total_p,
    }
    scenario_pressure, scenario_sec = predict_scenario(
        models.pressure,
        models.sec,
        **scenario,
    )
    baseline = {column: float(ranges.loc[column, "median"]) for column in scenario}
    baseline_pressure, baseline_sec = predict_scenario(
        models.pressure,
        models.sec,
        **baseline,
    )
    scenario_percentile = percentile_rank(history["sec_kwh_m3"], scenario_sec)

    section("RESULT", "기준 시나리오와 변경 시나리오", "기준은 각 입력 변수의 기록 중앙값입니다.")
    result_cols = st.columns(4)
    result_cols[0].metric("기준 압력 추정", f"{baseline_pressure:.2f} bar")
    result_cols[1].metric(
        "시나리오 압력 추정",
        f"{scenario_pressure:.2f} bar",
        f"기준 대비 {scenario_pressure - baseline_pressure:+.2f} bar",
        delta_color="off",
    )
    result_cols[2].metric("기준 SEC 추정", f"{baseline_sec:.2f} kWh/m³")
    result_cols[3].metric(
        "시나리오 SEC 추정",
        f"{scenario_sec:.2f} kWh/m³",
        f"기준 대비 {scenario_sec - baseline_sec:+.2f} kWh/m³",
        delta_color="off",
    )
    st.caption(
        f"시나리오 SEC 추정치는 과거 기록 SEC 분포의 약 {scenario_percentile:.0f}백분위 위치입니다. "
        "분포 위치는 안전성·적합성 판정이 아닙니다."
    )
    plot(sensitivity_figure(models.pressure, models.sec, scenario, ranges))

    section("MODEL CARD", "모델의 범위와 읽는 법", "성능은 독립 검증이 아닌 전체 기록 적합도입니다.")
    card_col1, card_col2, card_col3 = st.columns(3)
    with card_col1:
        st.markdown(
            f"""
            <div class="model-card"><div class="eyebrow">PRESSURE MODEL</div>
            <h3>선형회귀 · 2개 입력</h3>
            <p><strong>입력</strong> 수온, pH</p>
            <p><strong>전체기록 MAE</strong> {model_diagnostics['pressure']['mae']:.3f} bar</p>
            <p><strong>전체기록 R²</strong> {model_diagnostics['pressure']['r2']:.3f}</p></div>
            """,
            unsafe_allow_html=True,
        )
    with card_col2:
        st.markdown(
            f"""
            <div class="model-card"><div class="eyebrow">SEC MODEL · CHAINED</div>
            <h3>랜덤포레스트 · 5개 입력</h3>
            <p><strong>입력</strong> 총인, COD, 총질소, 탁도, 추정 압력</p>
            <p><strong>전체기록 MAE</strong> {model_diagnostics['sec_chain']['mae']:.3f} kWh/m³</p>
            <p><strong>전체기록 R²</strong> {model_diagnostics['sec_chain']['r2']:.3f}</p></div>
            """,
            unsafe_allow_html=True,
        )
    with card_col3:
        st.markdown(
            """
            <div class="model-card"><div class="eyebrow">KNOWN LIMITS</div>
            <h3>모델에 없는 핵심 변수</h3>
            <p>염분·전기전도도, 원수/생산수 유량, 회수율, 막 상태, 차압, 약품 주입량이 포함되지 않았습니다.</p>
            <p>자동 제어·보증 성능·안전 판정에 사용할 수 없습니다.</p></div>
            """,
            unsafe_allow_html=True,
        )


elif page == "운영 인사이트":
    section(
        "REVIEW QUEUE",
        "모델-기록 차이가 큰 구간부터 점검",
        "전체 기록의 오차 분포에서 상대적으로 큰 차이를 찾아 검토 순서를 정합니다.",
    )
    st.warning(
        "아래 후보는 각 오차의 95백분위 이상인 기록입니다. 통계적 검토 큐일 뿐 "
        "안전·품질 이상 판정이나 자동 운전 지시가 아닙니다."
    )
    watchlist, thresholds = anomaly_watchlist(history)
    insight_cols = st.columns(4)
    insight_cols[0].metric("검토 후보", f"{int(thresholds['watch_count']):,}건")
    insight_cols[1].metric("전체 중 비중", f"{thresholds['watch_share_pct']:.1f}%")
    insight_cols[2].metric(
        "압력 차이 기준",
        f"{thresholds['pressure_cutoff']:.2f} bar",
        "95백분위",
        delta_color="off",
    )
    insight_cols[3].metric(
        "SEC 차이 기준",
        f"{thresholds['sec_cutoff']:.3f} kWh/m³",
        "95백분위",
        delta_color="off",
    )

    plot(
        error_watch_figure(
            history,
            pressure_cutoff=float(thresholds["pressure_cutoff"]),
            sec_cutoff=float(thresholds["sec_cutoff"]),
        )
    )

    section(
        "WATCHLIST",
        "상위 검토 후보",
        "모델과 기록의 차이가 큰 순서입니다. 원자료·센서 상태·운영 로그와 함께 확인하세요.",
    )
    review_columns = [
        "timestamp",
        "pressure_stage1_bar",
        "model_pressure_bar",
        "pressure_error_bar",
        "sec_kwh_m3",
        "model_sec_kwh_m3",
        "sec_error_kwh_m3",
        "temperature_c",
        "turbidity_ntu",
        "watch_reason",
    ]
    review_view = watchlist[review_columns].head(40).copy()
    review_view["timestamp"] = review_view["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    review_view = review_view.rename(
        columns={
            "timestamp": "시각",
            "pressure_stage1_bar": "기록 압력(bar)",
            "model_pressure_bar": "모델 압력(bar)",
            "pressure_error_bar": "압력 차이(bar)",
            "sec_kwh_m3": "기록 SEC(kWh/m³)",
            "model_sec_kwh_m3": "모델 SEC(kWh/m³)",
            "sec_error_kwh_m3": "SEC 차이(kWh/m³)",
            "temperature_c": "수온(°C)",
            "turbidity_ntu": "탁도(NTU)",
            "watch_reason": "검토 사유",
        }
    )
    st.dataframe(
        review_view.style.format(
            {
                "기록 압력(bar)": "{:.2f}",
                "모델 압력(bar)": "{:.2f}",
                "압력 차이(bar)": "{:+.2f}",
                "기록 SEC(kWh/m³)": "{:.3f}",
                "모델 SEC(kWh/m³)": "{:.3f}",
                "SEC 차이(kWh/m³)": "{:+.3f}",
                "수온(°C)": "{:.1f}",
                "탁도(NTU)": "{:.2f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "검토 후보 CSV 받기",
        watchlist.to_csv(index=False).encode("utf-8-sig"),
        file_name="ro_model_record_watchlist_2021.csv",
        mime="text/csv",
    )
    st.caption(
        "기준값은 현재 표시된 2021년 기록 전체에서 다시 계산합니다. "
        "상위 40건 표에는 원인으로 단정하지 않고 차이가 큰 변수만 표시합니다."
    )


else:
    section(
        "TRANSPARENCY",
        "데이터와 모델의 한계를 먼저 확인",
        "공공 원수 관측자료와 출처가 문서화되지 않은 RO 샘플 공정 기록을 구분합니다.",
    )
    overview_cols = st.columns(5)
    overview_cols[0].metric("원본 행", f"{profile.source_rows:,}건")
    overview_cols[1].metric("사용 가능 행", f"{profile.usable_rows:,}건")
    overview_cols[2].metric("중복 제거", f"{profile.duplicate_rows_removed:,}건")
    overview_cols[3].metric("결측 제거", f"{profile.incomplete_rows:,}건")
    overview_cols[4].metric("정시 기록 비중", f"{profile.exact_hour_share:.1%}")

    source_col1, source_col2 = st.columns(2)
    with source_col1:
        st.markdown(
            """
            <div class="model-card"><div class="eyebrow">DOCUMENTED SOURCE</div>
            <h3>천수만 원수 관측자료</h3>
            <p>저장소 파일명 기준: 해양환경공단 해양수질자동측정망 천수만(2021).</p>
            <p>이 출처 표시는 수온·pH·탁도·COD·총질소·총인 원수 변수에만 적용합니다.</p></div>
            """,
            unsafe_allow_html=True,
        )
    with source_col2:
        st.markdown(
            """
            <div class="model-card"><div class="eyebrow">UNDOCUMENTED SAMPLE DATA</div>
            <h3>RO 공정 기록</h3>
            <p>압력·생산수 TDS·SEC의 측정 설비, 보정 방법, 생산량 가중치가 저장소에 문서화되어 있지 않습니다.</p>
            <p>공개 배포에서는 실제 플랜트 실적이 아닌 샘플/가공 공정 데이터로 취급합니다.</p></div>
            """,
            unsafe_allow_html=True,
        )

    section("DATA QUALITY", "0값과 표본 밀도", "0이 실제 측정값인지 센서 결측 대체값인지 원자료만으로 확정할 수 없습니다.")
    zero_cols = st.columns(4)
    zero_cols[0].metric("탁도 0값", f"{profile.zero_counts['turbidity_ntu']:,}건")
    zero_cols[1].metric("COD 0값", f"{profile.zero_counts['cod_mg_l']:,}건")
    zero_cols[2].metric("총질소 0값", f"{profile.zero_counts['total_n_mg_l']:,}건")
    zero_cols[3].metric("총인 0값", f"{profile.zero_counts['total_p_mg_l']:,}건")
    plot(observation_density_figure(monthly))

    section("RANGES", "사용 기록의 변수 범위", "모델 학습 범위가 문서화되지 않아 앱은 사용 기록 범위를 대리 경계로 사용합니다.")
    range_view = ranges.reset_index()[["label", "unit", "min", "p05", "median", "p95", "max"]].rename(
        columns={
            "label": "변수",
            "unit": "단위",
            "min": "최솟값",
            "p05": "5백분위",
            "median": "중앙값",
            "p95": "95백분위",
            "max": "최댓값",
        }
    )
    st.dataframe(
        range_view.style.format(
            {"최솟값": "{:.3f}", "5백분위": "{:.3f}", "중앙값": "{:.3f}", "95백분위": "{:.3f}", "최댓값": "{:.3f}"}
        ),
        width="stretch",
        hide_index=True,
    )

    section("LIMITATIONS", "사용 전에 알아야 할 점")
    st.warning(
        "모델을 재현할 학습 파이프라인과 독립 시간순 검증 결과가 저장소에 없습니다. "
        "표시된 MAE·R²는 저장 모델을 전체 공정 기록에 다시 적용한 적합도이므로 일반화 성능보다 낙관적일 수 있습니다."
    )
    st.markdown(
        """
        - 이 앱은 실제 센서, PLC, SCADA, 제어 밸브와 연결되지 않습니다.
        - 생산수 유량과 원수 유량이 없어 생산량·회수율을 계산하지 않습니다.
        - 규제 준수, 식수 적합성, 막 세정 또는 운전 모드 전환을 판정하지 않습니다.
        - Joblib 모델은 신뢰된 저장소 아티팩트를 재패키징하고 SHA-256으로 무결성을 확인한 뒤 로드합니다.
        """
    )
    cleaned_export = history.drop(
        columns=["model_pressure_bar", "model_sec_kwh_m3", "pressure_error_bar", "sec_error_kwh_m3"]
    )
    st.download_button(
        "정제 공정 기록 CSV 받기",
        cleaned_export.to_csv(index=False).encode("utf-8-sig"),
        file_name="ro_history_clean_2021.csv",
        mime="text/csv",
    )

st.markdown(
    f"""
    <div class="footer-note">
      RO Lens 연구용 데모 · 데이터 기간 {start_time:%Y-%m-%d}–{end_time:%Y-%m-%d} ·
      모델 패키지 {models.manifest.get('bundle_version', 'unknown')} · 실제 설비 제어용 아님
    </div>
    """,
    unsafe_allow_html=True,
)
