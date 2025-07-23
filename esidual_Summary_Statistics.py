# --------------------------------------------------
# [4] 시각화한 내용을 Steamlit에 배포하세요.
# 위에서 생성한 sunspots_for_prophet.csv를 다운로드 받아, 루트/data 아래에 넣어주세요.
# --------------------------------------------------
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# 페이지 설정
st.set_page_config(page_title="🌞 Sunspot Forecast", layout="wide")
st.title("🌞 Prophet Forecast with Preprocessed Sunspot Data")

# ----------------------------------
# [1] 데이터 불러오기
# ----------------------------------
df = pd.read_csv("data/sunspots_for_prophet.csv")
df['ds'] = pd.to_datetime(df['ds'])

st.subheader("📄 불러온 데이터 미리보기")
st.dataframe(df.head())

# ----------------------------------
# [2] Prophet 모델 정의 및 학습
# ----------------------------------
model = Prophet()
model.add_seasonality(name='sunspot_cycle', period=11, fourier_order=5)
model.fit(df)

# ----------------------------------
# [3] 예측 수행
# ----------------------------------
future = model.make_future_dataframe(periods=30, freq='Y')
forecast = model.predict(future)

# ----------------------------------
# [4] 기본 시각화
# ----------------------------------
st.subheader("📈 Prophet Forecast Plot")
fig1 = model.plot(forecast)
st.pyplot(fig1)

st.subheader("📊 Forecast Components")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)

# ----------------------------------
# [5] 컨스텀 시각화: 실제값 vs 예측값 + 신력관
# ----------------------------------
st.subheader("📉 Custom Plot: Actual vs Predicted with Prediction Intervals")

fig3, ax = plt.subplots(figsize=(14, 6))
ax.plot(df["ds"], df["y"], label="Actual", color='blue',marker='o')
ax.plot(forecast["ds"], forecast["yhat"], llabel='Predicted', color='red',linestyle='--')
ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],
                 color='red', alpha=0.1, label="Prediction Interval")
ax.set_title("Sunspots: Actual vs. Predicted with Prediction Intervals")
ax.set_xlabel("Year")
ax.set_ylabel("Sunspot Count")
ax.legend()
st.pyplot(fig3)

# ----------------------------------
# [6] 잔차 반응 시각화
# ----------------------------------
st.subheader("📉 Residual Analysis (예측 오차 반응)")

merged = pd.merge(df, forecast[['ds', 'yhat']], on='ds', how='left')
merged['residual'] = merged['y'] - merged['yhat']

fig4, ax2 = plt.subplots(figsize=(14, 4))
ax2.plot(merged["ds"], merged["residual"], label="Residuals", color='purple',marker='o')
ax2.axhline(0, color='black', linestyle='--', linewidth=1)
ax2.set_title("Residual Analysis (Actual - Predicted)")
ax2.set_xlabel("Year")
ax2.set_ylabel("Residual")
ax2.legend()
ax2.grid(True)
st.pyplot(fig4)

# ----------------------------------
# [7] 잔차 통계 요약 출력
# ----------------------------------
st.subheader("📌 Residual Summary Statistics")
st.write(merged["residual"].describe())
