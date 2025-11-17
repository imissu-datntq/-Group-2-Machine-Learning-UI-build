# app_weather_ui.py
import streamlit as st
from datetime import datetime
import altair as alt


from weather_backend import (
    get_actual_and_forecast_for_ui,
    get_actual_hourly_for_date,
    get_forecast_hourly_for_date,
    DF_DAILY,
    HORIZON,
)

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Hanoi Temperature Forecast",
    page_icon="🌤️",
    layout="wide"
)

# ================== GLOBAL CSS ==================
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://raw.githubusercontent.com/DanhBitoo/-Group-2-Machine-Learning-Project/refs/heads/main/UI/assets/hanoi1.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .overlay {
        background: rgba(0, 0, 0, 0.45);
        border-radius: 24px;
        padding: 24px;
        margin-bottom: 20px;
    }
    .main-title {
        font-size: 32px;
        font-weight: 700;
        color: #ffffff;
        text-shadow: 0 2px 4px rgba(0,0,0,0.6);
        margin-bottom: 8px;
    }
    .sub-text {
        font-size: 14px;
        color: #f0f0f0;
        opacity: 0.9;
    }
    .temp-card-main {
        padding: 16px;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.4);
        background: linear-gradient(
            135deg,
            rgba(255,255,255,0.15),
            rgba(255,255,255,0.03)
        );
        display: flex;
        justify-content: space-between;
        align-items: center;
        backdrop-filter: blur(8px);
    }
    .temp-card-forecast {
        padding: 12px;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.4);
        background: linear-gradient(
            135deg,
            rgba(255,255,255,0.12),
            rgba(255,255,255,0.02)
        );
        text-align: center;
        min-height: 130px;
        backdrop-filter: blur(6px);
        color: #ffffff;
    }
    .temp-main-value {
        font-size: 34px;
        font-weight: 800;
        color: #ffffff;
        text-shadow: 0 2px 6px rgba(0,0,0,0.6);
    }
    .temp-main-label {
        font-size: 14px;
        opacity: 0.9;
        color: #f0f0f0;
    }
    .temp-date {
        font-size: 18px;
        font-weight: 600;
        color: #ffffff;
    }
    .temp-subdate {
        font-size: 12px;
        opacity: 0.8;
        color: #f0f0f0;
    }
    .temp-forecast-value {
        font-size: 24px;
        font-weight: 700;
        margin-top: 4px;
        margin-bottom: 4px;
    }
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ================== HEADER ==================
st.markdown(
    """
    <div class="overlay">
        <div class="main-title">🌤️ Hanoi Temperature Forecast</div>
        <div class="sub-text">
            • <b>Actual</b> temperature + <b>hourly actual</b> for selected date<br>
            • <b>5-day forecast</b> + <b>hourly prediction</b> when clicking on each card
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ================== SESSION STATE ==================
if "selected_forecast_date" not in st.session_state:
    st.session_state["selected_forecast_date"] = None

# ================== CHỌN NGÀY GỐC ==================
from weather_backend import get_origin_date_range_for_ui

min_date, max_date = get_origin_date_range_for_ui()

selected_date = st.date_input(
    "Select base date (within data range):",
    value=max_date,          # ✅ default date = last valid date
    min_value=min_date,
    max_value=max_date
)
st.markdown('</div>', unsafe_allow_html=True)

origin_ts = datetime.combine(selected_date, datetime.min.time())

# ================== DAILY + HOURLY THỰC ==================
try:
    actual_temp, fc_df, _ = get_actual_and_forecast_for_ui(origin_ts, horizon=HORIZON)
except Exception as e:
    st.error(f"Error during daily forecast: {e}")
    st.stop()

fc_df = fc_df.copy().sort_index()

# ----- Card nhiệt độ thực -----
st.markdown('<div class="overlay">', unsafe_allow_html=True)
st.markdown("### 📌 Actual temperature for selected date", unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="temp-card-main">
        <div>
            <div class="temp-date">
                {origin_ts.strftime("%d/%m/%Y")}
            </div>
            <div class="temp-main-label">
                Actual temperature (daily)
            </div>
        </div>
        <div class="temp-main-value">
            {actual_temp:.1f}°C
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

# ----- Biểu đồ hourly thực -----
st.markdown('<div class="overlay">', unsafe_allow_html=True)
st.markdown("### ⏱️ Hourly temperature (actual)", unsafe_allow_html=True)

try:
    df_hourly_actual = get_actual_hourly_for_date(origin_ts)
    df_hourly_actual = df_hourly_actual.copy()
    df_hourly_actual["hour"] = df_hourly_actual.index.hour.astype(int)

    # Dùng Altair để control trục
    chart_actual = (
        alt.Chart(df_hourly_actual.reset_index())
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "hour:Q",
                scale=alt.Scale(domain=[0, 23]),
                axis=alt.Axis(
                    title="Hour (0–23)",
                    tickMinStep=1
                ),
            ),
            y=alt.Y(
                "temp:Q",
                axis=alt.Axis(title="Temperature (°C)")
            ),
            tooltip=[
                alt.Tooltip("datetime:T", title="Time"),
                alt.Tooltip("hour:Q", title="Hour"),
                alt.Tooltip("temp:Q", title="Temperature (°C)", format=".1f"),
            ],
        )
        .properties(height=260)
    )

    st.altair_chart(chart_actual, use_container_width=True)
except Exception as e:
    st.warning(f"Could not retrieve actual hourly data: {e}")


st.markdown('</div>', unsafe_allow_html=True)

# ================== 5 NGÀY DỰ BÁO (DAILY) ==================
st.markdown('<div class="overlay">', unsafe_allow_html=True)
st.markdown("### 🔮 5-day forecast (daily)", unsafe_allow_html=True)

cols = st.columns(len(fc_df))

for i, (d, row) in enumerate(fc_df.iterrows()):
    with cols[i]:
        date_short = d.strftime("%d/%m")
        date_full = d.strftime("%d/%m/%Y")
        temp_pred = row["temp_pred"]

        st.markdown(
            f"""
            <div class="temp-card-forecast">
                <div class="temp-date">{date_short}</div>
                <div class="temp-subdate">{date_full}</div>
                <div class="temp-forecast-value">{temp_pred:.1f}°C</div>
                <div class="temp-main-label">Forecast (daily)</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button("View hourly", key=f"btn_{i}"):
            st.session_state["selected_forecast_date"] = d

st.markdown('</div>', unsafe_allow_html=True)

# ================== HOURLY DỰ BÁO CHO NGÀY ĐƯỢC CHỌN ==================
selected_forecast_date = st.session_state.get("selected_forecast_date", None)

if selected_forecast_date is not None:
    st.markdown('<div class="overlay">', unsafe_allow_html=True)
    st.markdown(
        f"### 🔎 Hourly forecast for {selected_forecast_date.strftime('%d/%m/%Y')}",
        unsafe_allow_html=True
    )

try:
    df_hourly_fc = get_forecast_hourly_for_date(
        selected_forecast_date
    )  # hoặc hàm bạn đang dùng

    df_plot = df_hourly_fc.copy()
    df_plot["hour"] = df_plot.index.hour.astype(int)

    chart_fc = (
        alt.Chart(df_plot.reset_index())
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "hour:Q",
                scale=alt.Scale(domain=[0, 23]),
                axis=alt.Axis(
                    title="Hour (0–23)",
                    tickMinStep=1
                ),
            ),
            y=alt.Y(
                "temp_pred:Q",
                axis=alt.Axis(title="Forecast Temperature (°C)")
            ),
            tooltip=[
                alt.Tooltip("datetime:T", title="Time"),
                alt.Tooltip("hour:Q", title="Hour"),
                alt.Tooltip("temp_pred:Q", title="Temperature (°C)", format=".1f"),
            ],
        )
        .properties(height=260)
    )

    st.altair_chart(chart_fc, use_container_width=True)
except Exception as e:
    st.error(f"Error creating hourly forecast: {e}")


    st.markdown('</div>', unsafe_allow_html=True)
