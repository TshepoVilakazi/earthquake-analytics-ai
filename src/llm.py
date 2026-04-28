from pathlib import Path
from datetime import datetime
import pandas as pd

from src.config import GOLD_DIR


PREDICTIONS_FILE = GOLD_DIR / "earthquake_predictions.csv"
FORECAST_FILE = GOLD_DIR / "earthquake_forecast.csv"
FEATURE_IMPORTANCE_FILE = GOLD_DIR / "feature_importance.csv"
METRICS_FILE = GOLD_DIR / "earthquake_model_metrics.txt"


def load_csv(file_path):
    if Path(file_path).exists():
        return pd.read_csv(file_path)
    return None


def contains_any(text, keywords):
    return any(keyword in text for keyword in keywords)


def detect_chart_request(question):
    q = question.lower().strip()

    chart_words = ["draw", "graph", "chart", "plot", "visual", "visualize"]

    if not contains_any(q, chart_words):
        return None

    if contains_any(q, ["forecast risk", "risk score", "risk"]):
        return {
            "type": "chart",
            "chart": "forecast_risk",
            "message": "Sure — here is the forecast risk score graph.",
        }

    if contains_any(q, ["forecast events", "forecast total", "forecast volume"]):
        return {
            "type": "chart",
            "chart": "forecast_events",
            "message": "Sure — here is the forecasted event volume graph.",
        }

    if contains_any(q, ["feature", "importance", "driver"]):
        return {
            "type": "chart",
            "chart": "feature_importance",
            "message": "Sure — here is the model feature importance graph.",
        }

    if contains_any(q, ["high activity", "normal activity", "activity days"]):
        return {
            "type": "chart",
            "chart": "activity_distribution",
            "message": "Sure — here is the high vs normal activity distribution.",
        }

    if contains_any(q, ["events over time", "total events", "earthquakes over time"]):
        return {
            "type": "chart",
            "chart": "events_over_time",
            "message": "Sure — here is the total events over time graph.",
        }

    if contains_any(q, ["average magnitude", "avg magnitude", "magnitude over time"]):
        return {
            "type": "chart",
            "chart": "avg_magnitude",
            "message": "Sure — here is the average magnitude over time graph.",
        }

    return {
        "type": "chart",
        "chart": "summary",
        "message": "I can draw forecast risk, forecast events, feature importance, activity distribution, total events over time, and average magnitude graphs.",
    }


def answer_forecast(forecast):
    if forecast is None or forecast.empty:
        return "I do not see forecast data yet. Please run `python -m src.forecast` first."

    risk_summary = forecast["risk_label"].value_counts().reset_index()
    risk_summary.columns = ["risk_label", "days"]

    most_common_risk = risk_summary.iloc[0]["risk_label"]
    start_date = forecast["forecast_date"].min()
    end_date = forecast["forecast_date"].max()

    response = (
        f"The forecast period is from **{start_date}** to **{end_date}**. "
        f"The overall forecast risk is **{most_common_risk}**.\n\n"
        "Risk breakdown:\n"
    )

    for _, row in risk_summary.iterrows():
        response += f"- **{row['risk_label']}**: {row['days']} day(s)\n"

    if "risk_score" in forecast.columns:
        response += f"\nThe average risk score is **{forecast['risk_score'].mean():.2f}**."

    return response


def answer_feature_importance(feature_importance):
    if feature_importance is None or feature_importance.empty:
        return "I do not see feature importance data yet. Please run `python -m src.feature_importance` first."

    feature_importance = feature_importance.sort_values("importance", ascending=False)
    top_feature = feature_importance.iloc[0]

    response = (
        f"The most important feature is **{top_feature['feature']}**, "
        f"with an importance score of **{top_feature['importance']:.4f}**.\n\n"
        "Top 5 model drivers:\n"
    )

    for _, row in feature_importance.head(5).iterrows():
        response += f"- **{row['feature']}**: {row['importance']:.4f}\n"

    return response


def answer_high_activity(predictions):
    if predictions is None or predictions.empty:
        return "I do not see prediction data yet. Please run `python -m src.model` first."

    high_days = predictions[predictions["risk_label"] == "High Activity"]
    return f"There are **{len(high_days)} high activity days** in the prediction dataset."


def answer_most_events(predictions):
    if predictions is None or predictions.empty:
        return "I do not see prediction data yet. Please run `python -m src.model` first."

    max_day = predictions.loc[predictions["total_events"].idxmax()]

    return (
        f"The day with the most earthquake events was **{max_day['event_date']}**, "
        f"with **{int(max_day['total_events'])} events**."
    )


def answer_magnitude(predictions, question_lower):
    if predictions is None or predictions.empty:
        return "I do not see prediction data yet. Please run `python -m src.model` first."

    if contains_any(question_lower, ["average", "avg", "mean"]):
        return f"The average magnitude is **{predictions['avg_magnitude'].mean():.2f}**."

    if contains_any(question_lower, ["max", "maximum", "highest", "strongest"]):
        return f"The highest daily maximum magnitude is **{predictions['max_magnitude'].max():.2f}**."

    return "Ask me either average magnitude or highest magnitude."


def answer_generic(question_lower):
    if contains_any(question_lower, ["hi", "hello", "hey", "howzit", "sawubona"]):
        return "Hey 👋 I’m your offline earthquake analytics assistant. Ask me questions or ask me to draw graphs."

    if contains_any(question_lower, ["how are you", "are you okay"]):
        return "I’m good 😄 Ready to help you explore your earthquake analytics project."

    if contains_any(question_lower, ["thank you", "thanks"]):
        return "You’re welcome 🙌 This project is looking strong."

    if contains_any(question_lower, ["weather", "rain", "temperature", "wind"]):
        return "I do not have live weather data. This assistant currently focuses on earthquake analytics."

    if contains_any(question_lower, ["airflow", "orchestration", "dag"]):
        return "Airflow orchestrates Bronze extraction, Silver transformation, Gold analytics, model training, forecast creation, and feature importance generation."

    if contains_any(question_lower, ["medallion", "bronze", "silver", "gold"]):
        return "The medallion architecture has Bronze for raw data, Silver for cleaned data, and Gold for analytics-ready outputs."

    return (
        "I can answer project questions and draw graphs.\n\n"
        "Try asking:\n"
        "- Draw a graph of forecast risk\n"
        "- Draw feature importance graph\n"
        "- Draw total events over time\n"
        "- What is the forecast risk?\n"
        "- Which feature is most important?\n"
        "- How many high activity days are there?\n"
        "- Explain Airflow\n"
    )


def ask_data(question, chat_history=None):
    question_lower = question.lower().strip()

    chart_request = detect_chart_request(question)
    if chart_request:
        return chart_request

    predictions = load_csv(PREDICTIONS_FILE)
    forecast = load_csv(FORECAST_FILE)
    feature_importance = load_csv(FEATURE_IMPORTANCE_FILE)

    if contains_any(question_lower, ["forecast", "risk", "next 7 days"]):
        return answer_forecast(forecast)

    if contains_any(question_lower, ["feature", "important", "importance", "driver"]):
        return answer_feature_importance(feature_importance)

    if contains_any(question_lower, ["high activity", "high risk"]):
        return answer_high_activity(predictions)

    if contains_any(question_lower, ["most earthquakes", "max events", "highest events", "busiest day"]):
        return answer_most_events(predictions)

    if contains_any(question_lower, ["magnitude", "strongest"]):
        return answer_magnitude(predictions, question_lower)

    if contains_any(question_lower, ["summary", "overview", "what can you do", "help", "hello", "hi", "hey"]):
        return answer_generic(question_lower)

    return answer_generic(question_lower)