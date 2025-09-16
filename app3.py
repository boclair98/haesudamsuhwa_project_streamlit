import pandas as pd
import numpy as np
import datetime
import streamlit as st
import folium
import joblib
import requests
import calendar
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import altair as alt
from PIL import Image
from urllib.parse import quote
from urllib.request import urlopen
from geopy.geocoders import Nominatim
from haversine import haversine
from dateutil.relativedelta import relativedelta
from time import sleep

# 페이지 기본 설정
st.set_page_config(layout="wide", page_title="해수 담수화 streamlit", page_icon="🎈")

# --- 데이터 로딩 (앱 실행 시 한 번만 실행되도록 캐싱) ---
# 데이터 파일이 크거나 로딩이 오래 걸리는 경우, st.cache_data를 사용하면 앱 성능이 향상됩니다.
@st.cache_data
def load_data():
    """
    앱에 필요한 모든 CSV 파일을 로드합니다.
    """
    try:
        seawater_df = pd.read_csv('해양환경공단_해양수질자동측정망_천수만(2021).csv', encoding='cp949')
        ro_df = pd.read_csv('RO공정데이터_0621.csv', encoding='cp949')
        water_quality_df = pd.read_csv('수질만데이터.csv', encoding='cp949')
        ro_monthly_df = pd.read_csv('RO공정데이터.csv', encoding='cp949')
        seawater_quality_df = pd.read_csv('해수수질데이터.csv', encoding='cp949')
        
        # 날짜 타입 변환
        seawater_df['관측일자'] = pd.to_datetime(seawater_df['관측일자'])
        ro_df['일시'] = pd.to_datetime(ro_df['일시'])
        water_quality_df['관측일자'] = pd.to_datetime(water_quality_df['관측일자'])
        ro_monthly_df['관측일자'] = pd.to_datetime(ro_monthly_df['관측일자'])
        seawater_quality_df['관측일자'] = pd.to_datetime(seawater_quality_df['관측일자'])

        return seawater_df, ro_df, water_quality_df, ro_monthly_df, seawater_quality_df
    except FileNotFoundError as e:
        st.error(f"오류: '{e.filename}' 파일을 찾을 수 없습니다. 코드 파일과 동일한 위치에 파일이 있는지 확인해주세요.")
        return None, None, None, None, None

seawater, ro, df_quality, df_ro_monthly, df_seawater_quality = load_data()

# --- 모델 로딩 (앱 실행 시 한 번만 실행되도록 캐싱) ---
@st.cache_resource
def load_models():
    """
    학습된 모델을 로드합니다. scikit-learn 버전 호환성 문제에 유의하세요.
    """
    try:
        pressure_model = joblib.load('LR_pressure.pkl')
        elec_model = joblib.load('RF_elec.pkl')
        return pressure_model, elec_model
    except Exception as e:
        st.error(f"모델을 로드하는 중 오류가 발생했습니다: {e}")
        st.error("❗**해결 방법**: 로컬 환경에서 `scikit-learn`을 `pip install --upgrade scikit-learn`으로 업데이트한 후, 모델 파일을 다시 학습시켜서 새로 저장하고 GitHub에 업로드해야 합니다.")
        return None, None

# 데이터나 모델 로딩에 실패하면 앱 실행 중지
if seawater is None or load_models() is None:
    st.stop()

pressure_model, elec_model = load_models()


st.header("해수담수화 플랜트 A")

tab1, tab2, tab3 = st.tabs(['실시간 대시보드', '생산관리', '수질분석'])

# =================================================================================================
# 탭 1: 실시간 대시보드
# =================================================================================================
with tab1:
    st.write('### 실시간 대시보드')
    
    # 현재 시간 기준으로 날짜/시간 초기값 설정 (데이터가 2021년 기준이므로 현재 시간에서 2년을 뺌)
    # 실제 운영 시에는 now()를 그대로 사용
    now = datetime.datetime.now()
    # 데이터가 2021년이므로, 현재 날짜와 맞추기 위해 연도 차이를 계산
    # 이 부분은 데이터의 연도에 맞게 유동적으로 조절해야 합니다.
    year_diff = now.year - 2021 
    
    initial_time = now - relativedelta(years=year_diff)
    before_one_hour_initial = initial_time - datetime.timedelta(hours=1)
    
    ## ----- 날짜/시간 입력 cols 구성 -----
    st.markdown("")
    col100, col101, col102, col103 = st.columns([0.1, 0.3, 0.1, 0.3])
    with col100:
        st.info('일시')
    with col101:
        input_date = st.date_input(label='일시', value=initial_time.date(), label_visibility="collapsed")
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
    input_p = seawater.loc[seawater['관측일자'] == date_time, ['수온', '수소이온농도']]
    input_e = seawater.loc[seawater['관측일자'] == date_time, ['총인', '화학적산소요구량', '총질소', '탁도']]

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
        y_pred1 = pressure_model.predict(input_p)
        
        # 예측된 전력량
        input_e['1차 인입압력'] = y_pred1
        y_pred2 = elec_model.predict(input_e)

        with col100:
            st.success('1차 인입압력  : ')
        with col101:
            st.success(f"{round(float(y_pred1), 3)} bar")

        with col102:
            st.success('사용 전력량    : ')
        with col103:
            if y_pred2 >= 2.5 and y_pred2 < 3.5:
                st.success(f"{round(float(y_pred2), 3)} kwh/m³")
            elif y_pred2 >= 3.5 and y_pred2 <= 3.7:
                st.warning(f"{round(float(y_pred2), 3)} kwh/m³")
            elif y_pred2 > 3.7:
                st.error(f"{round(float(y_pred2), 3)} kwh/m³")

        # ----- 운전현황 및 게이지 차트 표시 -----
        col200, col201 = st.columns([0.6, 0.4])
        with col200:
            st.markdown("##### 운전현황")
            if y_pred2 < 3.5:
                st.image('대시보드 구성도_정상_w.png', caption='정상 운영')
            elif y_pred2 <= 3.7:
                st.image('대시보드 구성도_주의_w.png', caption='주의 단계')
                st.warning("주의 단계 진입 : partial two pass로 전환 운영합니다.")
            else:
                st.image('대시보드 구성도_이상_w.png', caption='경고 단계')
                st.error("경고 단계 진입 : split partial two pass로 전환 운영합니다.")
        
        with col201:
            st.markdown("##### 예측 전력량 (kwh/m³)")
            gauge_value = round(float(y_pred2), 2)
            
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
        tds_delta = round(float(tem_tds.iloc[0] - tds_tds_prev.iloc[0]), 2) if not tem_tds.empty and not tem_tds_prev.empty else None
        col_m3.metric(label="최종 생산수 TDS (mg/L)", value=tds_val, delta=tds_delta)

        power_val = float(tem_power.iloc[0]) if not tem_power.empty else "N/A"
        power_delta = round(float(tem_power.iloc[0] - tem_power_prev.iloc[0]), 2) if not tem_power.empty and not tem_power_prev.empty else None
        col_m4.metric(label="사용 전력량 (kWh/m³)", value=power_val, delta=power_delta)
        
        st.divider()

        # ----- 담수 생산률 및 수질 달성률 -----
        col_pie, col_achieve = st.columns([0.4, 0.6])
        with col_pie:
            st.markdown("##### 담수 생산률 (%)")
            time_min = (date_time.hour * 60) + date_time.minute
            amount = 83.33 * time_min
            prod_percent = amount / 120000 * 100
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
                # 계산 로직은 제공된 원본 코드를 따름
                inflow_turbidity = selected_data['탁도'].values[0]; processing_turbidity = selected_data['↓탁도'].values[0]; standard_turbidity = selected_data['기준 탁도'].values[0]
                inflow_turbidity_standard_turbidity = inflow_turbidity if inflow_turbidity-standard_turbidity <= 1 else inflow_turbidity-standard_turbidity
                processed_ratio = (inflow_turbidity-processing_turbidity) / (inflow_turbidity_standard_turbidity) if (inflow_turbidity-processing_turbidity) != 0 else 1
                
                inflow_CO = selected_data['화학적산소요구량'].values[0]; processing_CO = selected_data['↓화학적산소요구량'].values[0]; standard_CO = selected_data['기준 화학적산소요구량'].values[0]
                inflow_CO_standard_CO = inflow_CO if inflow_CO-standard_CO <= 1 else inflow_CO-standard_CO
                processed_ratio1 = (inflow_CO-processing_CO) / (inflow_CO_standard_CO) if (inflow_CO-processing_CO) != 0 else 1

                inflow_N = selected_data['총질소'].values[0]; processing_N = selected_data['↓총질소'].values[0]; standard_N = selected_data['기준 총질소'].values[0]
                inflow_N_standard_N = inflow_N if inflow_N-standard_N <= 0.2 else inflow_N-standard_N
                processed_ratio2 = (inflow_N-processing_N) / (inflow_N_standard_N) if (inflow_N-processing_N) != 0 else 1

                inflow_P = selected_data['총인'].values[0]; processing_P = selected_data['↓총인'].values[0]; standard_P = selected_data['기준 총인'].values[0]
                inflow_P_standard_P = inflow_P if inflow_P-standard_P <= 0.01 else inflow_P-standard_P
                processed_ratio3 = (inflow_P-processing_P) / (inflow_P_standard_P) if (inflow_P-processing_P) != 0 else 1
                
                st.markdown("##") # 공백 추가
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("탁도 달성률", f"{processed_ratio:.2%}")
                c2.metric("COD 달성률", f"{processed_ratio1:.2%}")
                c3.metric("총질소 달성률", f"{processed_ratio2:.2%}")
                c4.metric("총인 달성률", f"{processed_ratio3:.2%}")
            else:
                st.info("해당 시간의 수질 달성률 데이터가 없습니다.")

# =================================================================================================
# 탭 2: 생산관리
# =================================================================================================
with tab2:
    st.write('### 생산관리')
    
    # 데이터가 비어있지 않은지 먼저 확인
    if df_ro_monthly is not None:
        df_ro_monthly.dropna(axis=0, inplace=True)
        
        # 사용자로부터 날짜 입력 받기
        min_date = df_ro_monthly['관측일자'].min().date()
        max_date = df_ro_monthly['관측일자'].max().date()
        default_date = max_date # 기본값을 최신 날짜로 설정
        
        selected_date = st.date_input("기준 날짜 선택", value=default_date, min_value=min_date, max_value=max_date, key="tab2_date")
        selected_date = pd.to_datetime(selected_date)

        # 선택한 날짜까지 필터링
        filtered_data = df_ro_monthly[df_ro_monthly['관측일자'].dt.date <= selected_date.date()].copy()
        
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
    df_ro_monthly['관측월'] = df_ro_monthly['관측일자'].dt.month
    month_data = df_ro_monthly[df_ro_monthly['관측월'] == selected_month]

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
        df_seawater_quality.dropna(axis=0, inplace=True)
        df_seawater_quality['관측월'] = df_seawater_quality['관측일자'].dt.to_period('M').astype(str)
        monthly_seawater_data = df_seawater_quality.groupby('관측월').mean(numeric_only=True).reset_index()

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
    input_data_pressure = [[input_temperature, input_concentration]]
    predicted_pressure = pressure_model.predict(input_data_pressure)
    st.success(f"예측된 1차 인입압력: **{predicted_pressure[0]:.3f} bar**")

    st.markdown("---")

    # 2. 전체 전력량 예측
    st.info("원수 수질 및 1차 인입압력에 따른 **전체 전력량** 예측")
    col208, col209 = st.columns(2)
    col210, col211 = st.columns(2)

    with col208:
        # 이전에 예측된 인입압력 값을 기본값으로 사용
        input_pressure = st.slider("1차 인입압력을 입력하세요: ", min_value=30.0, max_value=70.0, value=float(predicted_pressure[0]), step=0.1)
    with col209:
        input_tin = st.slider("총인(mg/L)을 입력하세요:", min_value=0.0, max_value=0.1, value=0.02, step=0.001, format="%.3f")
    with col210:
        input_cod = st.slider("화학적산소요구량(mg/L)을 입력하세요:", min_value=0.0, max_value=3.0, value=1.5, step=0.1)
    with col211:
        input_tn = st.slider("총질소(mg/L)을 입력하세요:", min_value=0.0, max_value=0.5, value=0.1, step=0.01)
    
    # 탁도 슬라이더 추가
    input_turbidity = st.slider("탁도(NTU)를 입력하세요:", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

    # 모델 입력 순서: ['총인', '화학적산소요구량', '총질소', '탁도', '1차 인입압력']
    input_data_elec = [[input_tin, input_cod, input_tn, input_turbidity, input_pressure]]
    predicted_electricity = elec_model.predict(input_data_elec)
    st.success(f"예측된 전체 전력량: **{predicted_electricity[0]:.3f} kWh/m³**")
