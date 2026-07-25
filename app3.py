import datetime

import pandas as pd
import streamlit as st
import plotly.express as px
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go

from desalination.analytics import evaluate_models
from desalination.domain import (
    classify_energy,
    production_progress,
    quality_achievement,
)
from desalination.resources import (
    PROJECT_ROOT,
    ResourceError,
    load_all_data,
    load_prediction_models,
)

# 페이지 기본 설정
st.set_page_config(layout="wide", page_title="해수 담수화 streamlit", page_icon="🎈")

# --- 데이터 로딩 (앱 실행 시 한 번만 실행되도록 캐싱) ---
# 데이터 파일이 크거나 로딩이 오래 걸리는 경우, st.cache_data를 사용하면 앱 성능이 향상됩니다.
@st.cache_data(show_spinner="공정 데이터를 불러오는 중입니다...")
def load_data():
    try:
        return load_all_data()
    except ResourceError as exc:
        st.error(str(exc))
        return None, None, None, None, None

seawater, ro, df_quality, df_ro_monthly, df_seawater_quality = load_data()

# --- 모델 로딩 (앱 실행 시 한 번만 실행되도록 캐싱) ---
@st.cache_resource
def load_models():
    try:
        return load_prediction_models()
    except ResourceError as exc:
        st.error(str(exc))
        return None

# 데이터나 모델 로딩에 실패하면 앱 실행 중지
models = load_models()
if seawater is None or models is None:
    st.stop()

pressure_model, elec_model = models

st.header("해수담수화 플랜트 A")

with st.sidebar:
    st.header("데이터·모델 상태")
    st.metric("수질 관측 데이터", f"{len(seawater):,}건")
    st.caption(
        f"{seawater['관측일자'].min():%Y-%m-%d %H:%M} ~ "
        f"{seawater['관측일자'].max():%Y-%m-%d %H:%M}"
    )
    required_columns = ["수온", "수소이온농도", "총인", "화학적산소요구량", "총질소", "탁도"]
    missing_values = int(seawater[required_columns].isna().sum().sum())
    st.metric("핵심 변수 결측치", f"{missing_values:,}건")

    with st.expander("모델 진단 지표"):
        try:
            diagnostics = evaluate_models(pressure_model, elec_model, ro)
            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric(
                "압력 MAE", f"{diagnostics['pressure']['mae']:.3f} bar"
            )
            metric_col2.metric(
                "전력 MAE", f"{diagnostics['energy']['mae']:.3f} kWh/m³"
            )
            st.write(
                f"압력 R²: `{diagnostics['pressure']['r2']:.3f}` · "
                f"전력 R²: `{diagnostics['energy']['r2']:.3f}`"
            )
            st.caption(
                "저장소의 과거 공정 데이터에 대한 적합도입니다. "
                "별도 검증 데이터 성능으로 해석하면 안 됩니다."
            )
        except (ValueError, KeyError) as exc:
            st.warning(f"모델 진단을 계산하지 못했습니다: {exc}")

tab1, tab2, tab3 = st.tabs(['실시간 대시보드', '생산관리', '수질분석'])

# =================================================================================================
# 탭 1: 실시간 대시보드
# =================================================================================================
with tab1:
    st.write('### 실시간 대시보드')
    
    # 시스템 시각을 과거 데이터 연도로 억지 변환하지 않고 실제 데이터 범위를 사용합니다.
    min_time = seawater["관측일자"].min()
    initial_time = seawater["관측일자"].max()
    
    ## ----- 날짜/시간 입력 cols 구성 -----
    st.markdown("")
    col100, col101, col102, col103 = st.columns([0.1, 0.3, 0.1, 0.3])
    with col100:
        st.info('일시')
    with col101:
        input_date = st.date_input(
            label='일시',
            value=initial_time.date(),
            min_value=min_time.date(),
            max_value=initial_time.date(),
            label_visibility="collapsed",
        )
    with col102:
        st.info('시간')
    with col103:
        input_time = st.time_input(label='시간', value=initial_time.time(), step=3600, label_visibility="collapsed")
    
    # 입력받은 날짜/시간 합쳐서 datetime타입으로 변환
    date_time_str = f"{input_date.strftime('%Y-%m-%d')} {input_time.strftime('%H:00:00')}"
    date_time = pd.to_datetime(date_time_str)
    before_one_hour = date_time - datetime.timedelta(hours=1)
    
    st.divider()

    # 날짜에 해당되는 수질 데이터(입력값) 추출
    selected_seawater = seawater.loc[seawater['관측일자'] == date_time].head(1)
    input_p = selected_seawater[['수온', '수소이온농도']]
    input_e = selected_seawater[['총인', '화학적산소요구량', '총질소', '탁도']].copy()

    # =================================================================
    # 중요: 데이터가 있는지 확인하는 로직 추가 (ValueError 방지)
    # =================================================================
    if input_p.empty or input_e.empty:
        st.error(f"**{date_time.strftime('%Y-%m-%d %H시')}**에 해당하는 수질 데이터가 없습니다. 다른 시간을 선택해주세요.")
    else:
        # ----- 예측값 표시 -----
        st.markdown("##### 예측값 :blue[(자동 적용중)]")
        
        col100, col101, col102, col103 = st.columns([0.1, 0.2, 0.1, 0.2])
        
        # 예측된 1차 인입압력
        predicted_pressure_value = float(pressure_model.predict(input_p)[0])
        
        # 예측된 전력량
        input_e['1차 인입압력'] = predicted_pressure_value
        predicted_energy_value = float(elec_model.predict(input_e)[0])
        energy_status = classify_energy(predicted_energy_value)

        with col100:
            st.success('1차 인입압력  : ')
        with col101:
            st.success(f"{predicted_pressure_value:.3f} bar")

        with col102:
            st.success('사용 전력량    : ')
        with col103:
            if energy_status.level == "normal":
                st.success(f"{predicted_energy_value:.3f} kWh/m³")
            elif energy_status.level == "warning":
                st.warning(f"{predicted_energy_value:.3f} kWh/m³")
            else:
                st.error(f"{predicted_energy_value:.3f} kWh/m³")

        # ----- 운전현황 및 게이지 차트 표시 -----
        col200, col201 = st.columns([0.6, 0.4])
        with col200:
            st.markdown("##### 운전현황")
            st.image(str(PROJECT_ROOT / energy_status.image_name), caption=f"{energy_status.label} 운영")
            if energy_status.level == "normal":
                st.success(energy_status.message)
            elif energy_status.level == "warning":
                st.warning(energy_status.message)
            else:
                st.error(energy_status.message)
        
        with col201:
            st.markdown("##### 예측 전력량 (kwh/m³)")
            gauge_value = round(predicted_energy_value, 2)
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=gauge_value,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [2, 4]},
                    'steps': [
                        {'range': [2.5, 3.5], 'color': "#b0d779"}, # 정상
                        {'range': [3.5, 3.7], 'color': "#f4e291"}, # 주의
                        {'range': [3.7, 4.0], 'color': "#d77981"}  # 경고
                    ],
                    'bar': {'color': "black"},
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': gauge_value
                    }
                }
            ))
            fig.update_layout(height=250, margin={'t':0, 'b':0, 'l':0, 'r':0})
            st.plotly_chart(fig, use_container_width=True)


        st.divider()
        
        # ----- 상세 정보 (Metric) -----
        st.markdown("##### RO공정 실시간 정보")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        # 현재 데이터 가져오기
        tem_pressure1 = ro.loc[ro['일시'] == date_time, '1차 인입압력']
        tem_pressure2 = ro.loc[ro['일시'] == date_time, '2차 인입압력']
        tem_tds = ro.loc[ro['일시'] == date_time, '2차 생산수 TDS']
        tem_power = ro.loc[ro['일시'] == date_time, '전체 전력량']

        # 1시간 전 데이터 가져오기
        tem_pressure1_prev = ro.loc[ro['일시'] == before_one_hour, '1차 인입압력']
        tem_pressure2_prev = ro.loc[ro['일시'] == before_one_hour, '2차 인입압력']
        tem_tds_prev = ro.loc[ro['일시'] == before_one_hour, '2차 생산수 TDS']
        tem_power_prev = ro.loc[ro['일시'] == before_one_hour, '전체 전력량']

        # Metric 카드 표시 (데이터가 있을 때만 델타 계산)
        p1_val = float(tem_pressure1.iloc[0]) if not tem_pressure1.empty else "N/A"
        p1_delta = round(float(tem_pressure1.iloc[0] - tem_pressure1_prev.iloc[0]), 2) if not tem_pressure1.empty and not tem_pressure1_prev.empty else None
        col_m1.metric(label="1차 인입압력 (bar)", value=p1_val, delta=p1_delta)

        p2_val = float(tem_pressure2.iloc[0]) if not tem_pressure2.empty else "N/A"
        p2_delta = round(float(tem_pressure2.iloc[0] - tem_pressure2_prev.iloc[0]), 2) if not tem_pressure2.empty and not tem_pressure2_prev.empty else None
        col_m2.metric(label="2차 인입압력 (bar)", value=p2_val, delta=p2_delta)

        tds_val = float(tem_tds.iloc[0]) if not tem_tds.empty else "N/A"
        tds_delta = round(float(tem_tds.iloc[0] - tem_tds_prev.iloc[0]), 2) if not tem_tds.empty and not tem_tds_prev.empty else None
        col_m3.metric(label="최종 생산수 TDS (mg/L)", value=tds_val, delta=tds_delta)

        power_val = float(tem_power.iloc[0]) if not tem_power.empty else "N/A"
        power_delta = round(float(tem_power.iloc[0] - tem_power_prev.iloc[0]), 2) if not tem_power.empty and not tem_power_prev.empty else None
        col_m4.metric(label="사용 전력량 (kWh/m³)", value=power_val, delta=power_delta)
        
        st.divider()

        # ----- 담수 생산률 및 수질 달성률 -----
        col_pie, col_achieve = st.columns([0.4, 0.6])
        with col_pie:
            st.markdown("##### 담수 생산률 (%)")
            prod_percent = production_progress(date_time.hour, date_time.minute)
            prod = pd.DataFrame({'names':['생산률', ' '], 'values':[prod_percent, 100-prod_percent]})
            
            fig = px.pie(prod, values='values', names='names', hole=0.7, color_discrete_sequence=['#79b0d7', '#E0E0E0'])
            fig.update_traces(hoverinfo='label+percent+name', textinfo='none')
            fig.update(layout_showlegend=False)
            fig.update_layout(
                annotations=[dict(text=f"{prod_percent:.2f}%", x=0.5, y=0.5, font=dict(size=30, color='black'), showarrow=False)],
                height=250, margin={'t':20, 'b':20, 'l':20, 'r':20}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_achieve:
            st.markdown("##### 수질 달성률")
            selected_data = df_quality[df_quality['관측일자'] == date_time]
            if not selected_data.empty:
                row = selected_data.iloc[0]
                achievement = {
                    "탁도 달성률": quality_achievement(row["탁도"], row["↓탁도"], row["기준 탁도"]),
                    "COD 달성률": quality_achievement(
                        row["화학적산소요구량"],
                        row["↓화학적산소요구량"],
                        row["기준 화학적산소요구량"],
                    ),
                    "총질소 달성률": quality_achievement(
                        row["총질소"], row["↓총질소"], row["기준 총질소"]
                    ),
                    "총인 달성률": quality_achievement(
                        row["총인"], row["↓총인"], row["기준 총인"]
                    ),
                }

                st.markdown("##") # 공백 추가
                c1, c2, c3, c4 = st.columns(4)
                for column, (label, value) in zip((c1, c2, c3, c4), achievement.items()):
                    column.metric(label, f"{value:.1%}")
            else:
                st.info("해당 시간의 수질 달성률 데이터가 없습니다.")

# =================================================================================================
# 탭 2: 생산관리
# =================================================================================================
with tab2:
    st.write('### 생산관리')
    
    # 데이터가 비어있지 않은지 먼저 확인
    if df_ro_monthly is not None:
        ro_monthly_clean = df_ro_monthly.dropna(axis=0).copy()

        # 사용자로부터 날짜 입력 받기
        min_date = ro_monthly_clean['관측일자'].min().date()
        max_date = ro_monthly_clean['관측일자'].max().date()
        default_date = max_date # 기본값을 최신 날짜로 설정
        
        selected_date = st.date_input("기준 날짜 선택", value=default_date, min_value=min_date, max_value=max_date, key="tab2_date")
        selected_date = pd.to_datetime(selected_date)

        # 선택한 날짜까지 필터링
        filtered_data = ro_monthly_clean[
            ro_monthly_clean['관측일자'].dt.date <= selected_date.date()
        ].copy()
        
        # '관측월' 컬럼 생성
        filtered_data['관측월'] = filtered_data['관측일자'].dt.to_period('M').astype(str)

        # 월별로 데이터 집계
        monthly_data = filtered_data.groupby('관측월').mean(numeric_only=True).reset_index()

        st.divider()
        # --- Metric 카드 ---
        col101, col102, col103 = st.columns(3)
        
        selected_month_str = selected_date.strftime('%Y-%m')
        before_one_month_str = (selected_date - relativedelta(months=1)).strftime('%Y-%m')
        
        # 현재 선택 월 데이터
        press_series = monthly_data.loc[monthly_data['관측월'] == selected_month_str, '1차 인입압력']
        tds_series = monthly_data.loc[monthly_data['관측월'] == selected_month_str, '2차 생산수 TDS']
        power_series = monthly_data.loc[monthly_data['관측월'] == selected_month_str, '전체 전력량']

        # 한달 전 데이터
        press_1_series = monthly_data.loc[monthly_data['관측월'] == before_one_month_str, '1차 인입압력']
        tds_1_series = monthly_data.loc[monthly_data['관측월'] == before_one_month_str, '2차 생산수 TDS']
        power_1_series = monthly_data.loc[monthly_data['관측월'] == before_one_month_str, '전체 전력량']

        # Metric 카드 표시 (데이터 유무 확인)
        press_val = float(press_series.iloc[0]) if not press_series.empty else "N/A"
        press_delta = round(float(press_series.iloc[0] - press_1_series.iloc[0]), 2) if not press_series.empty and not press_1_series.empty else None
        col101.metric(label="월평균 1차 인입압력 (bar)", value=press_val, delta=press_delta)

        tds_val = float(tds_series.iloc[0]) if not tds_series.empty else "N/A"
        tds_delta = round(float(tds_series.iloc[0] - tds_1_series.iloc[0]), 2) if not tds_series.empty and not tds_1_series.empty else None
        col102.metric(label="월평균 2차 생산수TDS (mg/L)", value=tds_val, delta=tds_delta)

        power_val = float(power_series.iloc[0]) if not power_series.empty else "N/A"
        power_delta = round(float(power_series.iloc[0] - power_1_series.iloc[0]), 2) if not power_series.empty and not power_1_series.empty else None
        col103.metric(label="월평균 전력량 (kWh/m³)", value=power_val, delta=power_delta)
        
        st.divider()

        # --- 인입압력, TDS, 전력량 그래프 ---
        col201, col202 = st.columns(2)
        with col201:
            fig_p = px.bar(monthly_data, x="관측월", y=["1차 인입압력", "2차 인입압력"], color_discrete_sequence=px.colors.qualitative.Pastel, title="월별 평균 인입압력")
            fig_p.update_traces(texttemplate='%{y:.2f}', textposition='outside')
            fig_p.update_layout(yaxis_title="인입압력(bar)")
            st.plotly_chart(fig_p, use_container_width=True)
        
        with col202:
            fig_tds = px.line(monthly_data, x="관측월", y=["1차 생산수 TDS", "2차 생산수 TDS"], color_discrete_sequence=px.colors.qualitative.Pastel, title="월별 1,2차 생산수 TDS", markers=True)
            fig_tds.update_layout(yaxis_title="TDS (mg/L)")
            fig_tds.update_traces(mode="lines+markers+text", texttemplate='%{y:.2f}', textposition="top center")
            st.plotly_chart(fig_tds, use_container_width=True)
        
        fig_elec = px.bar(monthly_data, x="관측월", y='전체 전력량', color_discrete_sequence=px.colors.qualitative.Pastel, title="월별 평균 전력량")
        emean = monthly_data['전체 전력량'].mean()
        fig_elec.update_traces(texttemplate='%{y:.2f}', textposition='outside')
        fig_elec.update_layout(yaxis_title="전력량(kWh/m³)")
        fig_elec.add_hline(y=emean, line_width=2, line_dash="dash", line_color="black", annotation_text=f"평균 {emean:.2f}", annotation_position="bottom right")
        st.plotly_chart(fig_elec, use_container_width=True)

# =================================================================================================
# 탭 3: 수질 분석
# =================================================================================================
with tab3:
    st.write('### 수질 분석')

    # 월 선택에 따른 수온 및 전력량 변화
    st.markdown("##### 월별 수온 및 전력량 추이")
    col_radio, col_chart1, col_chart2 = st.columns([0.2, 0.4, 0.4])
    with col_radio:
        selected_month = st.radio('월 선택', range(1, 13), format_func=lambda x: f"{x}월", index=datetime.datetime.now().month - 1)
    
    # df_ro_monthly 데이터프레임의 '관측일자'에서 월을 추출하여 '관측월' 컬럼 추가
    ro_for_quality = df_ro_monthly.copy()
    ro_for_quality['관측월'] = ro_for_quality['관측일자'].dt.month
    month_data = ro_for_quality[ro_for_quality['관측월'] == selected_month]

    with col_chart1:
        fig = px.line(month_data, x='관측일자', y='수온', title=f'{selected_month}월 수온 추이', markers=True)
        fig.update_layout(xaxis_tickformat='%m-%d')
        st.plotly_chart(fig, use_container_width=True)
    with col_chart2:
        fig_power = px.line(month_data, x='관측일자', y='전체 전력량', title=f'{selected_month}월 전체 전력량', markers=True)
        fig_power.update_layout(xaxis_tickformat='%m-%d')
        st.plotly_chart(fig_power, use_container_width=True)
        
    st.divider()
    
    # 월별 평균 수질 데이터 시각화
    st.markdown("##### 월별 평균 원수 수질")
    if df_seawater_quality is not None:
        seawater_quality_clean = df_seawater_quality.dropna(axis=0).copy()
        seawater_quality_clean['관측월'] = seawater_quality_clean['관측일자'].dt.to_period('M').astype(str)
        monthly_seawater_data = seawater_quality_clean.groupby('관측월').mean(numeric_only=True).reset_index()

        col202, col203 = st.columns(2)
        with col202:
            fig = px.bar(monthly_seawater_data, x="관측월", y="유입된 탁도(NTU)", title="월별 평균 탁도")
            fig.add_hline(y=1, line_dash="solid", line_color="red", annotation_text="기준", annotation_position="bottom right")
            st.plotly_chart(fig, use_container_width=True)
        with col203:
            fig = px.bar(monthly_seawater_data, x="관측월", y="유입된 화학적산소요구량(mg/L)", title="월별 평균 화학적산소요구량")
            fig.add_hline(y=1, line_dash="solid", line_color="red", annotation_text="기준", annotation_position="bottom right")
            st.plotly_chart(fig, use_container_width=True)

        col204, col205 = st.columns(2)
        with col204:
            fig = px.bar(monthly_seawater_data, x="관측월", y="유입된 총인(mg/L)", title="월별 평균 총인")
            fig.add_hline(y=0.01, line_dash="solid", line_color="red", annotation_text="기준", annotation_position="bottom right")
            st.plotly_chart(fig, use_container_width=True)
        with col205:
            fig = px.bar(monthly_seawater_data, x="관측월", y="유입된 총질소(mg/L)", title="월별 평균 총질소")
            fig.add_hline(y=0.2, line_dash="solid", line_color="red", annotation_text="기준", annotation_position="bottom right")
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- 시뮬레이션을 통한 예측 ---
    st.markdown("##### 예측 시뮬레이션")

    # 1. 1차 인입압력 예측
    st.info("원수 수질에 따른 **1차 인입압력** 예측")
    col206, col207 = st.columns(2)
    with col206:
        input_temperature = st.slider("수온을 입력하세요:", min_value=0.0, max_value=31.0, value=15.0, step=0.1)
    with col207:
        input_concentration = st.slider("수소이온농도를 입력하세요:", min_value=7.0, max_value=9.0, value=8.0, step=0.1)
    
    # 2D 배열 형태로 모델에 입력
    input_data_pressure = pd.DataFrame(
        [[input_temperature, input_concentration]],
        columns=["수온", "수소이온농도"],
    )
    simulated_pressure = float(pressure_model.predict(input_data_pressure)[0])
    st.success(f"예측된 1차 인입압력: **{simulated_pressure:.3f} bar**")

    st.markdown("---")

    # 2. 전체 전력량 예측
    st.info("원수 수질 및 1차 인입압력에 따른 **전체 전력량** 예측")
    col208, col209 = st.columns(2)
    col210, col211 = st.columns(2)

    with col208:
        # 이전에 예측된 인입압력 값을 기본값으로 사용
        input_pressure = st.slider(
            "1차 인입압력을 입력하세요: ",
            min_value=30.0,
            max_value=70.0,
            value=min(max(simulated_pressure, 30.0), 70.0),
            step=0.1,
        )
    with col209:
        input_tin = st.slider("총인(mg/L)을 입력하세요:", min_value=0.0, max_value=0.1, value=0.02, step=0.001, format="%.3f")
    with col210:
        input_cod = st.slider("화학적산소요구량(mg/L)을 입력하세요:", min_value=0.0, max_value=3.0, value=1.5, step=0.1)
    with col211:
        input_tn = st.slider("총질소(mg/L)을 입력하세요:", min_value=0.0, max_value=0.5, value=0.1, step=0.01)
    
    # 탁도 슬라이더 추가
    input_turbidity = st.slider("탁도(NTU)를 입력하세요:", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

    # 모델 입력 순서: ['총인', '화학적산소요구량', '총질소', '탁도', '1차 인입압력']
    input_data_elec = pd.DataFrame(
        [[input_tin, input_cod, input_tn, input_turbidity, input_pressure]],
        columns=["총인", "화학적산소요구량", "총질소", "탁도", "1차 인입압력"],
    )
    simulated_energy = float(elec_model.predict(input_data_elec)[0])
    simulated_status = classify_energy(simulated_energy)
    st.success(f"예측된 전체 전력량: **{simulated_energy:.3f} kWh/m³**")
    if simulated_status.level == "normal":
        st.info(f"운영 판정: **{simulated_status.label}** · {simulated_status.message}")
    elif simulated_status.level == "warning":
        st.warning(f"운영 판정: **{simulated_status.label}** · {simulated_status.message}")
    else:
        st.error(f"운영 판정: **{simulated_status.label}** · {simulated_status.message}")



