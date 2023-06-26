import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
from geopy.geocoders import Nominatim
from folium import plugins
from keras.models import load_model
from haversine import haversine
from urllib.parse import quote
import streamlit as st
import folium
import branca
from geopy.geocoders import Nominatim
import ssl
from urllib.request import urlopen
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import plotly.express as px
import joblib
import requests
from streamlit_folium import folium_static
from dateutil.relativedelta import relativedelta
from streamlit_extras.metric_cards import style_metric_cards
import plotly.graph_objects as go
from PIL import Image
st.set_page_config(page_title='해수 담수화 RO 플랜트',layout='wide')
tab1,tab2,tab3 = st.tabs(['실시간 대시보드','생산관리','수질분석'])
with tab1:
    st.title("해수담수화 플랜트 A")


# 데이터 불러오기
    seawater = pd.read_csv('해양환경공단_해양수질자동측정망_천수만(2021).csv', encoding='cp949') # 수질 데이터
    ro = pd.read_csv('RO공정데이터_0621.csv', encoding='cp949') # RO공정 데이터
    
# 관측일자 object 타입 -> datetime 타입으로 변환
    seawater['관측일자'] = pd.to_datetime(seawater['관측일자'])
    ro['일시'] = pd.to_datetime(ro['일시'])
    
# 현재 시간
    now = datetime.datetime.now()
    before_two_year = now - relativedelta(years=2)
    before_one_month = now - relativedelta(years=2, months=1)
    before_one_hour = now - datetime.timedelta(hours=1)
    before_one_hour = before_one_hour - relativedelta(years=2)
    before_one_hour = before_one_hour.strftime('%Y-%m-%d %H:00:00')
    before_one_hour = pd.to_datetime(before_one_hour)

    st.header("RO공정 대시보드")
    
    ## ----- 날짜/시간 입력 cols 구성 -----
    st.markdown("")
    
    col100, col101, col102, col103 = st.columns([0.1, 0.3, 0.1, 0.3])
    with col100:
        st.info('일시')
    with col101:
        input_date = st.date_input(label='일시', value=before_two_year, label_visibility="collapsed")
    with col102:
        st.info('시간')
    with col103:
        input_time = st.time_input(label='시간', value=before_two_year, step=60, label_visibility="collapsed")
    
    # 입력받은 날짜/시간 합쳐서 datetime타입으로 변환
    date = input_date.strftime('%Y-%m-%d')
    time = input_time.strftime('%H:00:00')
    date_time = date + ' ' + time
    date_time = pd.to_datetime(date_time)
    
   
    st.divider() # 분리줄(가로줄)
    
    
    # 날짜에 해당되는 수질 데이터(입력값) 추출
    input_p = seawater.loc[seawater['관측일자'] == date_time, ['수온', '수소이온농도']]
    input_e = seawater.loc[seawater['관측일자'] == date_time, ['총인', '화학적산소요구량', '총질소', '탁도']]
    
    # 예측 모델 불러오기
    pressure_model = joblib.load('LR_pressure.pkl') # '1차 인입압력' 예측 모델
    elec_model = joblib.load('RF_elec.pkl') # '전체 전력량' 예측 모델
    
    ## ----- 예측값 표시 -----
    st.markdown("")
    st.markdown("##### 예측값 :blue[(자동 적용중)]")
    
    col100, col101, col102, col103 = st.columns([0.1, 0.2, 0.1, 0.2])
    with col100:
        st.success('1차 인입압력  : ')
        
    with col101:
        # 예측된 1차 인입압력
        y_pred1 = pressure_model.predict(input_p)
        st.success(round(float(y_pred1), 3))    
    
    with col102:
        st.success('사용 전력량   : ')
        
    with col103:
        # 예측된 전력량
        input_e['1차 인입압력'] = y_pred1
        y_pred2 = elec_model.predict(input_e)
        st.success(round(float(y_pred2), 3))
    

    ## ----- 적용중인 1차 인입압력, 1차 인입압력에 따른 사용 전력량 표시 (+ 1시간 전 대비 값의 변화 표시) -----
    col200, col201 = st.columns([0.6, 0.3])
 

    with col200:
        st.markdown("")
        st.markdown("##### 운전현황")   
        image = Image.open('대시보드 공정 구성도_w(운전현황X).png')
        st.image(image)

    with col201:
        st.markdown("")
        st.markdown("##### 사용 전력량 (kwh/m3)")   
    
        # 전력량 게이지 차트
        elec = ro.loc[ro['일시'] == date_time, '전체 전력량']
        
        
        fig = go.Figure(go.Indicator(
            domain={'x': [0, .5], 'y': [0, .7]},
            value=0,
            mode="gauge",
            gauge={'axis': {'range': [1, 4.8]},
                   'steps': [
                       {'range': [0, 1.8], 'color': "#d77981"},
                       {'range': [1.8, 2.3], 'color': "#f4e291"},
                       {'range': [2.3, 3.5], 'color': "#b0d779"},
                       {'range': [3.5, 4], 'color': "#f4e291"},
                       {'range': [4, 4.8], 'color': "#d77981"}],
                   'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': .8, 'value': round(float(y_pred2), 3)}}))

        fig.update_layout(annotations=[dict(text=round(float(y_pred2), 3), 
                                            x=0.18, 
                                            y=0.2, 
                                            font=dict(size=40, color='black'), 
                                            showarrow=False)])
        fig.add_annotation(text='(kwh/m3)', 
                           x=0.185, 
                           y=0.1, 
                           font=dict(size=20, color='black'), 
                           showarrow=False)
        
        st.plotly_chart(fig)
    
    
    
    # 실시간 정보
    st.markdown(" ")
    st.markdown("##### 실시간 정보")  

    col200, col201, col202 = st.columns([0.25, 0.25, 0.5])
    with col200:
        st.markdown("##### :green[RO공정]")  
        st.markdown("#")
        st.markdown("##")
        tem = ro.loc[ro['일시'] == date_time, '1차 인입압력'] # 현재 날짜와 일치하는 1차 인입압력
        tem_1 = ro.loc[ro['일시'] == before_one_hour, '1차 인입압력'] # 현재 날짜 기준 한시간 전의 1차 인입압력

        col200.metric(label="1차 인입압력", value=tem, delta=round(float(tem.values - tem_1.values),2))


        st.markdown("#")


        tem = ro.loc[ro['일시'] == date_time, '2차 인입압력'] # 현재 날짜와 일치하는 2차 인입압력
        tem_1 = ro.loc[ro['일시'] == before_one_hour, '2차 인입압력'] # 현재 날짜 기준 한시간 전의 2차 인입압력

        col200.metric(label="2차 인입압력", value=tem, delta=round(float(tem.values - tem_1.values),2))

    with col201:
        st.markdown("#") 
        st.markdown("#") 
        st.markdown("#")

 

        tem = ro.loc[ro['일시'] == date_time, '2차 생산수 TDS'] # 현재 날짜와 일치하는 최종 생산수 TDS
        tem_1 = ro.loc[ro['일시'] == before_one_hour, '2차 생산수 TDS'] # 현재 날짜 기준 한시간 전의 최종 생산수 TDS

        col201.metric(label="최종 생산수 TDS", value=tem, delta=round(float(tem.values - tem_1.values),2))
        
        st.markdown("#")
            
            
        tem = ro.loc[ro['일시'] == date_time, '전체 전력량'] # 현재 날짜와 일치하는 전체 전력량
        tem_1 = ro.loc[ro['일시'] == before_one_hour, '전체 전력량'] # 현재 날짜 기준 한시간 전의 전체 전력량
        
        col201.metric(label="사용 전력량", value=tem, delta=round(float(tem.values - tem_1.values),2))
        
    
    with col202:
        st.markdown("##### 담수 생산률 (%)")
        
        # 담수 생산률
        time = (date_time.hour * 60) + date_time.minute
        amount = 83.33 * time
        prod = pd.DataFrame({'names':['생산률', ' '], 'values':[amount/120000*100, 100-(amount/120000*100)]})
        
        fig = px.pie(prod, 
                     values='values', 
                     names='names', 
                     title = ' ', 
                     hole = 0.7, 
                     color_discrete_sequence = ['#79b0d7', 'rgba(211, 211, 211, 1.0)'])
        fig.update_traces(hoverinfo='label+percent+name', textinfo='none')
        fig.update(layout_showlegend=False)
        fig.update_layout(annotations=[dict(text=str(round(amount/120000*100, 2))+"%", 
                                            x=0.5, 
                                            y=0.5, 
                                            font=dict(size=40, color='black'), 
                                            showarrow=False)],
                         title_x=0.42)
        
        st.plotly_chart(fig)    
    
    
    
    
    st.markdown("##### :blue[수질]")    
    # 수질 달성률
    df = pd.read_csv('수질만데이터.csv', encoding='cp949')
    
    df['관측일자'] = pd.to_datetime(df['관측일자'])

    # 선택한 관측일자에 해당하는 데이터 필터링
    selected_data = df[df['관측일자'] == date_time]

 

    # 유입 탁도, 처리중 탁도, 기준 탁도 값 가져오기
    inflow_turbidity = selected_data['탁도'].values[0]
    processing_turbidity = selected_data['↓탁도'].values[0]
    standard_turbidity = selected_data['기준 탁도'].values[0]
    #달성률 = ((5 - 4) / 4) * 100 = (1 / 4) * 100 = 25%

 


    if inflow_turbidity-standard_turbidity <= 1:
        inflow_turbidity_standard_turbidity = inflow_turbidity
    else:
        inflow_turbidity_standard_turbidity = inflow_turbidity-standard_turbidity

    processed_ratio = (inflow_turbidity-processing_turbidity) / (inflow_turbidity_standard_turbidity)
    if inflow_turbidity-processing_turbidity ==0:
        processed_ratio = 1
    reducing_ratio = 1-processed_ratio

 


    # 유입 화학적산소요구량, 처리중 화학적산소요구량, 기준 화학적산소요구량 값 가져오기
    inflow_CO = selected_data['화학적산소요구량'].values[0]
    processing_CO = selected_data['↓화학적산소요구량'].values[0]
    standard_CO = selected_data['기준 화학적산소요구량'].values[0]
    #달성률 = ((5 - 4) / 4) * 100 = (1 / 4) * 100 = 25%
    #1/0.37
    if inflow_CO-standard_CO <= 1:
        inflow_CO_standard_CO = inflow_CO
    else:
        inflow_CO_standard_CO = inflow_CO-standard_CO
    processed_ratio1 = (inflow_CO-processing_CO) / (inflow_CO_standard_CO)
    if inflow_CO-processing_CO ==0:
        processed_ratio1 = 1
    reducing_ratio1 = 1-processed_ratio1

 

    ###총질소
    inflow_N = selected_data['총질소'].values[0]
    processing_N = selected_data['↓총질소'].values[0]
    standard_N = selected_data['기준 총질소'].values[0]
    #달성률 = ((5 - 4) / 4) * 100 = (1 / 4) * 100 = 25%
    #1/0.37
    if inflow_N-standard_N <= 0.2:
        inflow_N_standard_N = inflow_N
    else:
        inflow_N_standard_N = inflow_N-standard_N
        
    processed_ratio2 = (inflow_N-processing_N) / (inflow_N_standard_N)
    if inflow_N-processing_N ==0:
        processed_ratio2 = 1
    reducing_ratio2 = 1-processed_ratio2

 

    ###총인
    inflow_P = selected_data['총인'].values[0]
    processing_P = selected_data['↓총인'].values[0]
    standard_P = selected_data['기준 총인'].values[0]
    #달성률 = ((5 - 4) / 4) * 100 = (1 / 4) * 100 = 25%
    #1/0.37
    if inflow_P-standard_P <= 0.01:
        inflow_P_standard_P = inflow_P
    else:
        inflow_P_standard_P = inflow_P-standard_P
    processed_ratio3 = (inflow_P-processing_P) / (inflow_P_standard_P)
    if inflow_N-processing_N ==0:
        processed_ratio3 = 1
    reducing_ratio3 = 1-processed_ratio3

 
    # Card content - Value

 

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("탁도 달성률", f"{processed_ratio:.2%}")
    col2.metric("COD 달성률", f"{processed_ratio1:.2%}")
    col3.metric("총질소 달성률", f"{processed_ratio2:.2%}")
    col4.metric("총인 달성률", f"{processed_ratio3:.2%}")
    style_metric_cards(box_shadow=False)

    
    
    
    
    # 수질 달성률
    
    
with tab2:
    st.write('### 생산관리')
    data = pd.read_csv('RO공정데이터.csv', encoding='cp949')
    data.dropna(axis=0, inplace=True)
    from streamlit_extras.colored_header import colored_header
    colored_header(label="해수담수화 플랜트 데이터 분석", description="월별 1차인입압력,  2차 생산수TDS,  전력량 평균",color_name="blue-90")


    # 사용자로부터 날짜 입력 받기
    min_date = pd.to_datetime(data['관측일자']).min().date()
    max_date = pd.to_datetime(data['관측일자']).max().date()
    default_date = min_date + (max_date - min_date) // 2
    
    selected_date = st.date_input("날짜 선택", value=default_date, min_value=min_date, max_value=max_date)
    selected_date = pd.to_datetime(selected_date)

    # 선택한 날짜까지 필터링
    filtered_data = data[pd.to_datetime(data['관측일자']).dt.date <= selected_date]

    # 관측일자를 연도-월 형식으로 변환 (문자열로 변환)
    filtered_data['관측일자'] = pd.to_datetime(filtered_data['관측일자']).dt.to_period('M').astype(str)

    # 월별로 데이터 집계
    monthly_data = filtered_data.groupby('관측일자').mean().reset_index()

    #월별 집계 데이터에 누적전력량 column 추가
    #monthly_data['누적전력량'] = monthly_data['전체 전력량'].cumsum()
    
    # col001,col002 = st.columns(2)
    # with col001:
        

    #metric 카드 작성
    col101, col102, col103 = st.columns(3)
    with col101:
    
            before_one_month = selected_date - relativedelta(months=1)
            press = monthly_data.loc[monthly_data['관측일자'] == selected_date.strftime('%Y-%m'), '1차 인입압력'] # 현재 날짜(월)와 일치하는 1차 인압압력
            press_1 = monthly_data.loc[monthly_data['관측일자'] == before_one_month.strftime('%Y-%m'), '1차 인입압력'] # 현재 날짜 기준 한달 전의 1차 인압압력

            col101.metric(label="1차 인압압력 (bar)", value=round(press, 2), delta=round(float(press.values - press_1.values),2))

    with col102:
    
            before_one_month = selected_date - relativedelta(months=1)
            tds = monthly_data.loc[monthly_data['관측일자'] == selected_date.strftime('%Y-%m'), '2차 생산수 TDS'] # 현재 날짜(월)와 일치하는 2차 생산수 TDS
            tds_1 = monthly_data.loc[monthly_data['관측일자'] == before_one_month.strftime('%Y-%m'), '2차 생산수 TDS'] # 현재 날짜 기준 한달 전 2차 생산수 TDS

            col102.metric(label="2차 생산수TDS (mg/L)", value=round( tds,2), delta=round(float(tds.values - tds_1.values),2))
        

        
        
    with col103:
    
            before_one_month = selected_date - relativedelta(months=1)
            powersum = monthly_data.loc[monthly_data['관측일자'] == selected_date.strftime('%Y-%m'), '전체 전력량'] # 현재 날짜(월)까지의 전체 전력량
            powersum_1 = monthly_data.loc[monthly_data['관측일자'] == before_one_month.strftime('%Y-%m'), '전체 전력량'] # 현재 날짜 기준 한달전까지 전체전력량

            col103.metric(label="월평균전력량 (kWh/m3)", value= round(powersum,2), delta=round(float(powersum.values -  powersum_1.values),2))

        



    style_metric_cards(box_shadow=False)



    
    #인입압력, TDS, 전력량 그래프

    col201, col202= st.columns(2)

    with col201:
        #인입압력
        fig_p = px.bar(monthly_data, x="관측일자", y=["1차 인입압력", "2차 인입압력"], color_discrete_sequence=px.colors.qualitative.Pastel, title="월별 평균 인입압력")

        # 그래프 출력
        fig_p.update_traces(texttemplate='%{y:.2f}', textposition='outside')  # 소수점 두 자리로 표시 및 막대 바깥에 텍스트 표시
        fig_p.update_layout(yaxis_title="인입압력(bar)")  # y축 레이블 설정
        st.plotly_chart(fig_p)
    
    
    
    with col202:
        # TDS
        fig_tds = px.line(monthly_data, x="관측일자", y=["1차 생산수 TDS", "2차 생산수 TDS"], color_discrete_sequence=px.colors.qualitative.Pastel, title="월별 1,2차 생산수 TDS")

        # Update the layout and axis labels
        fig_tds.update_layout(yaxis_title="TDS")  # Set y-axis label
        fig_tds.update_traces(mode="lines+markers+text",texttemplate='%{y:.2f}', textposition= "top center" )  # Add markers to the lines for data points

        # Display the line graph
        st.plotly_chart(fig_tds)


    
     #전력량
    fig_elec = px.bar(monthly_data, x="관측일자", y=['전체 전력량'], color_discrete_sequence=px.colors.qualitative.Pastel, title="월별 평균 전력량")

    # 그래프 출력
    emean = monthly_data['전체 전력량'].mean()
    fig_elec.update_traces(texttemplate='%{y:.2f}', textposition='outside')  # 소수점 두 자리로 표시 및 막대 바깥에 텍스트 표시
    fig_elec.update_layout(yaxis_title="전력량(kWh/m3)")  # y축 레이블 설정
    fig_elec.add_hline(y= emean, line_width=1, line_dash="dash", line_color="black", annotation_text="평균", annotation_position="bottom right") # 기준선 (평균)추가

    st.plotly_chart(fig_elec, use_container_width=True)


with tab3:
    st.write('### 수질 분석')
    def style_metric_cards(
        background_color: str = "#FFF",
        border_size_px: int = 1,
        border_color: str = "#CCC",
        border_radius_px: int = 5,
        border_left_color: str = "#9AD8E1",  # Update the border_left_color to black
        box_shadow: bool = True,
    ):
        box_shadow_str = (
            "box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15) !important;"
            if box_shadow
            else "box-shadow: none !important;"
        )
        st.markdown(
            f"""
            <style>
                div[data-testid="metric-container"] {{
                    background-color: {background_color};
                    border: {border_size_px}px solid {border_color};
                    padding: 5% 5% 5% 10%;
                    border-radius: {border_radius_px}px;
                    border-left: 0.5rem solid {border_left_color} !important;
                    {box_shadow_str}
                }}
            </style>
            """,
            unsafe_allow_html=True,
        )

    def preprocessing(df):
        x = df[['수온', '수소이온농도']]
        y = df['1차 인입압력']
        return x, y

    def preprocessing1(df1):
        x = df1[['총인', '화학적산소요구량', '총질소', '탁도', '1차 인입압력']]
        y = df1['전체 전력량']
        return x, y
    def draw_circle(value):
        radius = int(value * 20)
        circle = f'<svg width="40" height="40"><circle cx="20" cy="20" r="{radius}" fill="#1f77b4" /></svg>'
        return circle

    background_color = """
    <style>
    body {
        background-color: black;
    }
    </style>
    """
    st.markdown(background_color, unsafe_allow_html=True)

    st.header('해수 담수화 프로젝트')
    st.subheader('스트림릿 시각화')
    df = pd.read_csv('RO공정데이터.csv', encoding='cp949')
    df1 = pd.read_csv('RO공정데이터.csv', encoding='cp949')
    col200, col201,col199 = st.columns([0.2, 0.4,0.4])
    with col200:
            selected_month = st.radio('월 선택', range(1, 13), format_func=lambda x: calendar.month_name[x])
    with col201:
            df['관측일자'] = pd.to_datetime(df['관측일자'])
            df['관측월'] = df['관측일자'].dt.month
            month_data = df[df['관측월'] == selected_month]
            month_data = month_data[['관측일자', '수온']]
            fig = px.line(month_data, x='관측일자', y='수온', title='월별 수온 추이')
            fig.update_layout(xaxis_tickformat='%Y-%m-%d')
            st.plotly_chart(fig)
    with col199:
            df['관측일자'] = pd.to_datetime(df['관측일자'])
            df['관측월'] = df['관측일자'].dt.month
            month_data = df[df['관측월'] == selected_month]
            month_data = month_data[['관측일자', '수온']]
            df_selected_month = df[df['관측월'] == selected_month]
            fig_power = px.line(df_selected_month, x='관측일자', y='전체 전력량', title='월별 전체 전력량')
            fig_power.update_layout(xaxis_tickformat='%Y-%m-%d')
            st.plotly_chart(fig_power)

    df.drop(['관측일자', '2차 인입압력', '1차 생산수 TDS', '2차 생산수 TDS', '전체 전력량', '총인', '화학적산소요구량', '총질소', '탁도'], axis=1, inplace=True)

    df1.drop(['관측일자', '2차 인입압력', '1차 생산수 TDS', '2차 생산수 TDS'], axis=1, inplace=True)
    new_x, new_y = preprocessing(df)
    model_m = joblib.load('LR_pressure.pkl')
    def predict_pressure(input_data):
        predicted_pressure = model_m.predict(input_data)
        return predicted_pressure
    col206, col207 = st.columns([0.5, 0.5])
    with col206:
                input_temperature = st.slider("수온을 입력하세요:", min_value=0.0, max_value=31.0, value=5.0,step=0.1)
    with col207:
                input_concentration = st.slider("수소이온농도를 입력하세요:",min_value=0.0, max_value=11.0, value=5.0,step=0.1)
    input_data = [[input_temperature, input_concentration]]
    predicted_pressure = predict_pressure(input_data)
    st.subheader("1차 인입압력량 예측 결과")
    st.success(f"예측된 1차 인입압력: {predicted_pressure}")

    new_xx, new_yy = preprocessing1(df1)
    model_k = joblib.load('RF_elec.pkl')

    def predict_electricity(input_data):
        predicted_electricity = model_k.predict(input_data)
        return predicted_electricity


    col208, col209,col210,col211,col212 = st.columns([0.2, 0.2,0.2,0.2,0.2])
    with col208:
        input_pressure = st.slider("1차 인입압력을 입력하세요: ", min_value=0.0, max_value=61.0, value=5.0, step=0.1,      format="%.1f", key="pressure_slider")
    with col209:
        input_turbidity = st.slider("탁도를 입력하세요: ", min_value=0.0, max_value=5.0, value=2.5,step=0.1)
    with col210:
        input_nitrogen = st.slider("총 질소를 입력하세요: ", min_value=0.0, max_value=5.0, value=2.5,step=0.1)
    with col211:
        input_total_inorganic_nitrogen = st.slider("총인을 입력하세요: ",  min_value=0.0, max_value=5.0, value=2.5,step=0.1)
    with col212:
        input_chemical_oxygen_demand = st.slider("화학적산소요구량을 입력하세요: ",  min_value=0.0, max_value=5.0,    value=2.5,step=0.1)

    input_data1 = [[input_pressure, input_turbidity, input_nitrogen, input_total_inorganic_nitrogen,    input_chemical_oxygen_demand]]
    predicted_electricity = predict_electricity(input_data1)
    col1, col2, col3,col4 = st.columns(4) 
    col1.metric("탁도", f"{input_turbidity-1:.2f}"+'NTU', f"{(input_turbidity-1)-1:.2f}NTU ")
    col2.metric("총 질소", f"{input_nitrogen -0.2:.2f}mg/L", f"{(input_nitrogen -0.2-0.2):.2f}mg/L")
    col3.metric("총인", f"{input_total_inorganic_nitrogen-0.01:.2f}mg/L", f"{input_total_inorganic_nitrogen-0.01-0.01:.2f}mg/L")
    col4.metric("화학전산소요구량", f"{input_chemical_oxygen_demand-1:.2f}mg/L", f"{(input_chemical_oxygen_demand-1)-1:.2f}mg/L")
    style_metric_cards()

    st.subheader("전체 전력량 예측 결과")
    st.success(f"예측된 전체 전력량: {predicted_electricity}")
    col220, col221 = st.columns([0.3, 0.7])
    with col220:
        st.subheader("수질 비율")
        fig = px.pie(values=[input_pressure, input_turbidity, input_nitrogen,        input_total_inorganic_nitrogen,input_chemical_oxygen_demand], names=['1차 인입압력','탁도','총 질소','총인','화학적산소요구량'])
        fig.update_layout(
            showlegend=True,  
            legend_title="데이터"  
        )
        fig.update_traces(hole=.3)
        st.plotly_chart(fig)
    with col221:
        col1, col2, col3, col4, col5 = st.columns(5) 
        pie_labels = ['1차 인입압력', '탁도', '총 질소', '총인', '화학적산소요구량']
        pie_values = [input_pressure, input_turbidity, input_nitrogen, input_total_inorganic_nitrogen, input_chemical_oxygen_demand]
        pie_percentages = [f"{(val / sum(pie_values)) * 100:.2f}%" for val in pie_values]
        col1.metric(label="총 인입압력 비율", value=pie_percentages[0])
        col2.metric(label="탁도 비율", value=pie_percentages[1])
        col3.metric(label="총 질소 비율", value=pie_percentages[2])
        col4.metric(label="총인 비율", value=pie_percentages[3])
        col5.metric(label="화학적산소요구량 비율", value=pie_percentages[4])
    data = pd.read_csv('해수수질데이터.csv', encoding='cp949')
    data.dropna(axis=0, inplace=True)
    min_date = pd.to_datetime(data['관측일자']).min().date()
    max_date = pd.to_datetime(data['관측일자']).max().date()
    default_date = min_date + (max_date - min_date) // 2
    selected_date = st.date_input("날짜 선택", value=default_date, min_value=min_date, max_value=max_date, key="unique_key")
    col202, col203 = st.columns([0.5, 0.5])
    with col202:
            selected_date = pd.to_datetime(selected_date)
            filtered_data = data[pd.to_datetime(data['관측일자']).dt.date <= selected_date]
            filtered_data['관측일자'] = pd.to_datetime(filtered_data['관측일자']).dt.to_period('M').astype(str)
            monthly_data = filtered_data.groupby('관측일자').mean().reset_index()
            fig = px.bar(monthly_data, x="관측일자", y=["유입된 탁도(NTU)"], color_discrete_sequence=px.colors.qualitative.Pastel, title="월별 평균 탁도")
            fig.add_hline(y=1, line_dash="solid", line_color="black", annotation_text="기준", annotation_position="bottom right")
            fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')  # 소수점 두 자리로 표시 및 막대 바깥에 텍스트 표시
            fig.update_layout(yaxis_title="탁도")  # y축 레이블 설정
            st.plotly_chart(fig)
    with col203:
            selected_date = pd.to_datetime(selected_date)
            filtered_data = data[pd.to_datetime(data['관측일자']).dt.date <= selected_date]
            filtered_data['관측일자'] = pd.to_datetime(filtered_data['관측일자']).dt.to_period('M').astype(str)
            monthly_data = filtered_data.groupby('관측일자').mean().reset_index()
            fig = px.bar(monthly_data, x="관측일자", y=[ "유입된 화학적산소요구량(mg/L)"], color_discrete_sequence=px.colors.qualitative.Pastel, title="월별 평균 화학적산소요구량")
            fig.add_hline(y=1, line_dash="solid", line_color="black", annotation_text="기준", annotation_position="bottom right")
            fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')  # 소수점 두 자리로 표시 및 막대 바깥에 텍스트 표시
            fig.update_layout(yaxis_title="화학적산소요구량")  # y축 레이블 설정
            st.plotly_chart(fig)
    col204, col205 = st.columns([0.5, 0.5])
    with col204:
            selected_date = pd.to_datetime(selected_date)
            filtered_data = data[pd.to_datetime(data['관측일자']).dt.date <= selected_date]
            filtered_data['관측일자'] = pd.to_datetime(filtered_data['관측일자']).dt.to_period('M').astype(str)
            monthly_data = filtered_data.groupby('관측일자').mean().reset_index()

            fig = px.bar(monthly_data, x="관측일자", y=["유입된 총인(mg/L)"],     color_discrete_sequence=px.colors.qualitative.Pastel, title="월별 평균 총인")
            fig.add_hline(y=0.01, line_dash="solid", line_color="black", annotation_text="기준", annotation_position="bottom right")
            fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')  # 소수점 두 자리로 표시 및 막대 바깥에 텍스트 표시
            fig.update_layout(yaxis_title="총인")  # y축 레이블 설정
            st.plotly_chart(fig)
    with col205:
            selected_date = pd.to_datetime(selected_date)
            filtered_data = data[pd.to_datetime(data['관측일자']).dt.date <= selected_date]
            filtered_data['관측일자'] = pd.to_datetime(filtered_data['관측일자']).dt.to_period('M').astype(str)
            monthly_data = filtered_data.groupby('관측일자').mean().reset_index()
            fig = px.bar(monthly_data, x="관측일자", y=[ "유입된 총질소(mg/L)"], color_discrete_sequence=px.colors.qualitative.Pastel, title="월별 평균 총질소")
            fig.add_hline(y=0.2, line_dash="solid", line_color="black", annotation_text="기준", annotation_position="bottom right")
        
            fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')  # 소수점 두 자리로 표시 및 막대 바깥에 텍스트 표시
            fig.update_layout(yaxis_title="총질소")  # y축 레이블 설정
            st.plotly_chart(fig)
    hospital_data = {
        '위치': ['한국1', '한국2', '사우디3', '사우디4', '사우디5'],
        '위도': [35.1, 34.9, 26.3, 28.6, 25.9],
        '경도': [129.1, 129.2, 56.1, 51.3, 55.2],
        '수온': [25.5, 26.0, 25.8, 25.9, 26.2],
        'pH': [7.2, 7.5, 7.3, 7.1, 7.4],
        '염분': [35, 38, 40, 42, 39],
        '산도': [6.8, 7.1, 6.9, 6.7, 7.0]
}

    hospital_list = pd.DataFrame(hospital_data)

    def main():
        m = folium.Map(location=[35.15, 129.10], zoom_start=2)
        for idx, row in hospital_list.iterrows():
            html = """<!DOCTYPE html>
            <html>
                <table style="height: 156px; width: 330px;"> <tbody> <tr>
                <td style="background-color: #2A799C;">
                <div style="color: #ffffff;text-align:center;">위치</div></td>
                <td style="width: 230px;background-color: #C5DCE7;">{}</td>""".format(row['위치'])+"""</tr>
                <tr><td style="background-color: #2A799C;">
                <div style="color: #ffffff;text-align:center;">수온</div></td>
                <td style="width: 230px;background-color: #C5DCE7;">{}</td>""".format(row['수온'])+"""</tr>
                <tr><td style="background-color: #2A799C;">
                <div style="color: #ffffff;text-align:center;">pH</div></td>
                <td style="width: 230px;background-color: #C5DCE7;">{}</td>""".format(row['pH'])+"""</tr>
                </tbody> </table> </html> """

            iframe = branca.element.IFrame(html=html, width=350, height=150)
            popup_text = folium.Popup(iframe, parse_html=True)
            icon = folium.Icon(color="blue")
            folium.Marker(location=[row['위도'], row['경도']],
                          popup=popup_text, tooltip=row['위치'], icon=icon).add_to(m)
        folium_static(m)

    if __name__ == "__main__":
        main()
    water = pd.read_csv('인천수질데이터.csv', encoding='cp949')
    water1 = pd.read_csv('수질서비스.csv', encoding='cp949')
    user_input = st.text_input("지역명을 입력하세요.")
    filtered_data_water = water[water['loc_nm'].str.contains(user_input)]
    filtered_data_water1 = water1[water1['시설주소'].str.contains(user_input)]

    if not filtered_data_water.empty:
        st.write("수질 데이터:")
        st.write(filtered_data_water[['loc_nm', 'temp', 'ph', 'do_', 't_n', 't_p', 'cod']])  
        geolocator = Nominatim(user_agent="my_app")
        location = geolocator.geocode(user_input)
        if location:
            latitude = location.latitude
            longitude = location.longitude
            st.write("입력한 지역의 경도: ", longitude)
            st.write("입력한 지역의 위도: ", latitude)
            st.map(data=[{"latitude": latitude, "longitude": longitude, "tooltip": user_input}])
        else:
            st.write("입력한 지역의 좌표를 가져올 수 없습니다.")
    elif not filtered_data_water1.empty:
        st.write("수질 데이터:")
        st.write(filtered_data_water1[['시설주소', 'pH', '탁도']])
        geolocator = Nominatim(user_agent="my_app")
        location = geolocator.geocode(user_input)
        if location:
            latitude = location.latitude
            longitude = location.longitude
            st.write("입력한 지역의 경도: ", longitude)
            st.write("입력한 지역의 위도: ", latitude)
            st.map(data=[{"latitude": latitude, "longitude": longitude, "tooltip": user_input}])
        else:
            st.write("입력한 지역의 좌표를 가져올 수 없습니다.")
    else:
    
        geolocator = Nominatim(user_agent="my_app")
        location = geolocator.geocode(user_input)
        if location:
            latitude = location.latitude
            longitude = location.longitude
            st.write("입력한 지역의 경도: ", longitude)
            st.write("입력한 지역의 위도: ", latitude)

            st.map(data=[{"latitude": latitude, "longitude": longitude, "tooltip": user_input}])
        else:
            st.write("해당 지역의 좌표와 수질 데이터를 찾을 수 없습니다.")
            st.write("입력한 지역: ", user_input)

