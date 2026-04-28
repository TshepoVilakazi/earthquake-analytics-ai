import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# ================================
# FIX PYTHON IMPORT PATH
# ================================
ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Earthquake Analytics Dashboard",
    page_icon="🌍",
    layout="wide",
)


# ================================
# DATA PATHS
# ================================
LOCAL_DATA_DIR = ROOT_DIR / "data"
SAMPLE_DATA_DIR = ROOT_DIR / "sample_data"

# Locally use full data/ folder.
# On Streamlit Cloud use sample_data/ folder.
DATA_DIR = LOCAL_DATA_DIR if LOCAL_DATA_DIR.exists() else SAMPLE_DATA_DIR

SILVER_FILE = DATA_DIR / "silver" / "earthquake_silver.csv"
GOLD_DAILY_FILE = DATA_DIR / "gold" / "earthquake_daily_summary.csv"
GOLD_MAGNITUDE_FILE = DATA_DIR / "gold" / "earthquake_magnitude_summary.csv"

PREDICTIONS_FILE = DATA_DIR / "gold" / "earthquake_predictions.csv"
METRICS_FILE = DATA_DIR / "gold" / "earthquake_model_metrics.txt"
FORECAST_FILE = DATA_DIR / "gold" / "earthquake_forecast.csv"
FEATURE_IMPORTANCE_FILE = DATA_DIR / "gold" / "feature_importance.csv"


# ================================
# LOAD DATA
# ================================
@st.cache_data(show_spinner=False)
def load_data():
    missing_files = []

    required_files = [
        SILVER_FILE,
        GOLD_DAILY_FILE,
        GOLD_MAGNITUDE_FILE,
    ]

    for file in required_files:
        if not file.exists():
            missing_files.append(str(file))

    if missing_files:
        st.error("Required data files are missing.")
        st.write("Missing files:")
        for file in missing_files:
            st.write(file)
        st.stop()

    silver = pd.read_csv(SILVER_FILE, low_memory=False)
    daily = pd.read_csv(GOLD_DAILY_FILE, low_memory=False)
    magnitude = pd.read_csv(GOLD_MAGNITUDE_FILE, low_memory=False)

    silver["event_time"] = pd.to_datetime(silver["event_time"], errors="coerce")

    if "event_date" in daily.columns:
        daily["event_date"] = pd.to_datetime(daily["event_date"], errors="coerce")

    predictions = None
    if PREDICTIONS_FILE.exists():
        predictions = pd.read_csv(PREDICTIONS_FILE, low_memory=False)
        predictions["event_date"] = pd.to_datetime(
            predictions["event_date"],
            errors="coerce",
        )

    forecast = None
    if FORECAST_FILE.exists():
        forecast = pd.read_csv(FORECAST_FILE, low_memory=False)
        forecast["forecast_date"] = pd.to_datetime(
            forecast["forecast_date"],
            errors="coerce",
        )

    feature_importance = None
    if FEATURE_IMPORTANCE_FILE.exists():
        feature_importance = pd.read_csv(FEATURE_IMPORTANCE_FILE, low_memory=False)

    return silver, daily, magnitude, predictions, forecast, feature_importance


silver_df, daily_df, magnitude_df, predictions_df, forecast_df, feature_importance_df = load_data()


# ================================
# TITLE
# ================================
st.title("🌍 Earthquake Analytics Dashboard")
st.caption(
    "USGS Earthquake Data | Bronze → Silver → Gold → ML Predictions → Forecast → Explainability → AI Assistant"
)

if DATA_DIR == SAMPLE_DATA_DIR:
    st.info(
        "This deployed version is using sample data. "
        "Run the pipeline locally to generate the full dataset."
    )


# ================================
# SIDEBAR FILTERS
# ================================
st.sidebar.header("Filters")

min_date = silver_df["event_time"].min().date()
max_date = silver_df["event_time"].max().date()

date_range = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

min_mag = float(silver_df["magnitude"].min())
max_mag = float(silver_df["magnitude"].max())

magnitude_range = st.sidebar.slider(
    "Magnitude range",
    min_value=min_mag,
    max_value=max_mag,
    value=(min_mag, max_mag),
)

selected_status = st.sidebar.multiselect(
    "Status",
    options=sorted(silver_df["status"].dropna().unique()),
    default=sorted(silver_df["status"].dropna().unique()),
)

selected_type = st.sidebar.multiselect(
    "Event Type",
    options=sorted(silver_df["event_type"].dropna().unique()),
    default=sorted(silver_df["event_type"].dropna().unique()),
)


# ================================
# APPLY FILTERS
# ================================
filtered_df = silver_df.copy()

if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = filtered_df[
        (filtered_df["event_time"].dt.date >= start_date)
        & (filtered_df["event_time"].dt.date <= end_date)
    ]

filtered_df = filtered_df[
    (filtered_df["magnitude"] >= magnitude_range[0])
    & (filtered_df["magnitude"] <= magnitude_range[1])
]

filtered_df = filtered_df[
    filtered_df["status"].isin(selected_status)
    & filtered_df["event_type"].isin(selected_type)
]


# ================================
# KPI CARDS
# ================================
total_events = len(filtered_df)
avg_magnitude = filtered_df["magnitude"].mean()
max_magnitude = filtered_df["magnitude"].max()
avg_depth = filtered_df["depth_km"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Events", f"{total_events:,}")
col2.metric(
    "Average Magnitude",
    round(avg_magnitude, 2) if pd.notna(avg_magnitude) else 0,
)
col3.metric(
    "Max Magnitude",
    round(max_magnitude, 2) if pd.notna(max_magnitude) else 0,
)
col4.metric(
    "Average Depth KM",
    round(avg_depth, 2) if pd.notna(avg_depth) else 0,
)

st.divider()


# ================================
# TABS
# ================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "Overview",
        "Map",
        "Magnitude Analysis",
        "Raw Data",
        "ML Predictions",
        "Forecast",
        "Feature Importance",
        "AI Assistant",
    ]
)


# ================================
# TAB 1: OVERVIEW
# ================================
with tab1:
    st.subheader("Earthquakes Over Time")

    events_over_time = (
        filtered_df.dropna(subset=["event_time"])
        .assign(event_date=lambda df: df["event_time"].dt.date)
        .groupby("event_date")
        .size()
        .reset_index(name="total_events")
    )

    st.line_chart(
        events_over_time,
        x="event_date",
        y="total_events",
    )

    st.subheader("Top 10 Places by Number of Events")

    top_places = (
        filtered_df["place"]
        .fillna("Unknown")
        .value_counts()
        .head(10)
        .reset_index()
    )

    top_places.columns = ["place", "total_events"]

    st.bar_chart(
        top_places,
        x="place",
        y="total_events",
    )


# ================================
# TAB 2: MAP
# ================================
with tab2:
    st.subheader("Earthquake Locations")

    map_df = filtered_df.dropna(subset=["latitude", "longitude"]).copy()

    st.map(
        map_df,
        latitude="latitude",
        longitude="longitude",
    )

    st.caption(f"Showing {len(map_df):,} earthquake locations.")


# ================================
# TAB 3: MAGNITUDE ANALYSIS
# ================================
with tab3:
    st.subheader("Magnitude Band Distribution")

    if "magnitude_band" in filtered_df.columns:
        mag_band = (
            filtered_df["magnitude_band"]
            .fillna("Unknown")
            .value_counts()
            .reset_index()
        )

        mag_band.columns = ["magnitude_band", "total_events"]

        st.bar_chart(
            mag_band,
            x="magnitude_band",
            y="total_events",
        )
    else:
        st.info("Magnitude band column not found.")

    st.subheader("Average Depth by Magnitude Band")

    if "magnitude_band" in filtered_df.columns:
        depth_by_mag = (
            filtered_df.groupby("magnitude_band", dropna=False)["depth_km"]
            .mean()
            .reset_index()
        )

        st.bar_chart(
            depth_by_mag,
            x="magnitude_band",
            y="depth_km",
        )


# ================================
# TAB 4: RAW DATA
# ================================
with tab4:
    st.subheader("Silver Layer Data")

    st.dataframe(
        filtered_df,
        width="stretch",
    )

    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name="filtered_earthquake_data.csv",
        mime="text/csv",
    )


# ================================
# TAB 5: ML PREDICTIONS
# ================================
with tab5:
    st.subheader("🤖 Earthquake Activity Prediction")

    if predictions_df is None:
        st.warning("Prediction file not found. Run: python -m src.model")
    else:
        high_risk_days = predictions_df[
            predictions_df["risk_label"] == "High Activity"
        ]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Prediction Rows", f"{len(predictions_df):,}")
        col2.metric("High Activity Days", f"{len(high_risk_days):,}")
        col3.metric(
            "Normal Days",
            f"{len(predictions_df) - len(high_risk_days):,}",
        )

        max_events_day = predictions_df.loc[predictions_df["total_events"].idxmax()]
        col4.metric("Max Events in a Day", int(max_events_day["total_events"]))

        st.subheader("Daily Activity and Prediction")

        st.line_chart(
            predictions_df,
            x="event_date",
            y=["total_events", "predicted_high_activity"],
        )

        st.subheader("High Activity Days")

        st.dataframe(
            high_risk_days.sort_values("event_date", ascending=False),
            width="stretch",
        )

        st.subheader("All Prediction Data")

        st.dataframe(
            predictions_df.sort_values("event_date", ascending=False),
            width="stretch",
        )

        csv_predictions = predictions_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download ML predictions as CSV",
            data=csv_predictions,
            file_name="earthquake_predictions.csv",
            mime="text/csv",
        )

        if METRICS_FILE.exists():
            st.subheader("Model Metrics")
            st.text(METRICS_FILE.read_text(encoding="utf-8"))
        else:
            st.info("Model metrics file not found.")


# ================================
# TAB 6: FORECAST
# ================================
with tab6:
    st.subheader("🔮 Next 7 Days Earthquake Activity Forecast")

    if forecast_df is None:
        st.warning("Forecast file not found. Run: python -m src.forecast")
    else:
        high_forecast = forecast_df[forecast_df["risk_label"] == "High Risk"]
        medium_forecast = forecast_df[forecast_df["risk_label"] == "Medium Risk"]
        low_forecast = forecast_df[forecast_df["risk_label"] == "Low Risk"]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Forecast Days", f"{len(forecast_df):,}")
        col2.metric("High Risk Days", f"{len(high_forecast):,}")
        col3.metric("Medium Risk Days", f"{len(medium_forecast):,}")
        col4.metric("Low Risk Days", f"{len(low_forecast):,}")

        st.subheader("Forecast Risk Score")

        st.line_chart(
            forecast_df,
            x="forecast_date",
            y="risk_score",
        )

        st.subheader("Forecasted Event Volume")

        st.bar_chart(
            forecast_df,
            x="forecast_date",
            y="forecast_total_events",
        )

        st.subheader("Forecast Table")

        forecast_display = forecast_df.copy()
        forecast_display["forecast_date"] = forecast_display["forecast_date"].dt.date

        st.dataframe(
            forecast_display.sort_values("forecast_date"),
            width="stretch",
        )

        csv_forecast = forecast_display.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download forecast as CSV",
            data=csv_forecast,
            file_name="earthquake_forecast.csv",
            mime="text/csv",
        )


# ================================
# TAB 7: FEATURE IMPORTANCE
# ================================
with tab7:
    st.subheader("📊 Model Feature Importance")

    if feature_importance_df is None:
        st.warning(
            "Feature importance file not found. Run: python -m src.feature_importance"
        )
    else:
        feature_importance_df = feature_importance_df.sort_values(
            "importance",
            ascending=False,
        )

        top_feature = feature_importance_df.iloc[0]

        col1, col2 = st.columns(2)

        col1.metric("Top Feature", top_feature["feature"])
        col2.metric("Top Importance Score", round(top_feature["importance"], 4))

        st.subheader("Feature Importance Ranking")

        st.bar_chart(
            feature_importance_df,
            x="feature",
            y="importance",
        )

        st.subheader("Feature Importance Table")

        st.dataframe(
            feature_importance_df,
            width="stretch",
        )

        csv_feature_importance = feature_importance_df.to_csv(index=False).encode(
            "utf-8"
        )

        st.download_button(
            label="Download feature importance as CSV",
            data=csv_feature_importance,
            file_name="feature_importance.csv",
            mime="text/csv",
        )


# ================================
# TAB 8: AI ASSISTANT
# ================================
with tab8:
    st.subheader("🤖 Ask Your Earthquake Data")

    st.markdown(
        """
        <style>
        .chat-header {
            padding: 14px 18px;
            border-radius: 14px;
            background-color: #111827;
            border: 1px solid #374151;
            margin-bottom: 16px;
        }
        .chat-header h4 {
            margin: 0;
            color: #f9fafb;
        }
        .chat-header p {
            margin: 6px 0 0 0;
            color: #d1d5db;
            font-size: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="chat-header">
            <h4>Earthquake AI Assistant</h4>
            <p>Ask questions or request graphs: forecast risk, feature importance, events over time, average magnitude, or activity distribution.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "Hey 👋 I’m your offline earthquake analytics assistant. "
                    "Ask me questions or ask me to draw graphs."
                ),
            }
        ]

    col1, col2 = st.columns([1, 5])

    with col1:
        clear_chat = st.button("🧹 Clear chat")

    if clear_chat:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": "Chat cleared ✅ What would you like to ask next?",
            }
        ]
        st.rerun()

    def render_chart(chart_type: str):
        if chart_type == "forecast_risk":
            if forecast_df is not None:
                st.line_chart(forecast_df, x="forecast_date", y="risk_score")
            else:
                st.warning("Forecast data not found. Run: python -m src.forecast")

        elif chart_type == "forecast_events":
            if forecast_df is not None:
                st.bar_chart(
                    forecast_df,
                    x="forecast_date",
                    y="forecast_total_events",
                )
            else:
                st.warning("Forecast data not found. Run: python -m src.forecast")

        elif chart_type == "feature_importance":
            if feature_importance_df is not None:
                chart_df = feature_importance_df.sort_values(
                    "importance",
                    ascending=False,
                )
                st.bar_chart(chart_df, x="feature", y="importance")
            else:
                st.warning(
                    "Feature importance data not found. Run: python -m src.feature_importance"
                )

        elif chart_type == "activity_distribution":
            if predictions_df is not None:
                chart_df = predictions_df["risk_label"].value_counts().reset_index()
                chart_df.columns = ["risk_label", "days"]
                st.bar_chart(chart_df, x="risk_label", y="days")
            else:
                st.warning("Prediction data not found. Run: python -m src.model")

        elif chart_type == "events_over_time":
            if predictions_df is not None:
                chart_df = predictions_df.copy()
                chart_df["event_date"] = pd.to_datetime(
                    chart_df["event_date"],
                    errors="coerce",
                )
                st.line_chart(chart_df, x="event_date", y="total_events")
            else:
                st.warning("Prediction data not found. Run: python -m src.model")

        elif chart_type == "avg_magnitude":
            if predictions_df is not None:
                chart_df = predictions_df.copy()
                chart_df["event_date"] = pd.to_datetime(
                    chart_df["event_date"],
                    errors="coerce",
                )
                st.line_chart(chart_df, x="event_date", y="avg_magnitude")
            else:
                st.warning("Prediction data not found. Run: python -m src.model")

        elif chart_type == "summary":
            st.info(
                "Try asking: `Draw forecast risk graph`, "
                "`Draw feature importance graph`, "
                "`Draw total events over time`, or "
                "`Draw average magnitude graph`."
            )

    chat_box = st.container(height=520, border=True)

    with chat_box:
        for message in st.session_state.chat_messages:
            avatar = "🧑" if message["role"] == "user" else "🤖"

            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

                if message.get("chart"):
                    render_chart(message["chart"])

    user_question = st.chat_input("Message Earthquake AI...")

    if user_question:
        st.session_state.chat_messages.append(
            {"role": "user", "content": user_question}
        )

        try:
            from src.llm import ask_data

            answer = ask_data(
                user_question,
                chat_history=st.session_state.chat_messages,
            )

            if isinstance(answer, dict) and answer.get("type") == "chart":
                st.session_state.chat_messages.append(
                    {
                        "role": "assistant",
                        "content": answer["message"],
                        "chart": answer["chart"],
                    }
                )
            else:
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": answer}
                )

        except Exception as error:
            st.session_state.chat_messages.append(
                {
                    "role": "assistant",
                    "content": f"Assistant error: {error}",
                }
            )

        st.rerun()