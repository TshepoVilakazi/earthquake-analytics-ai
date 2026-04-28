from pathlib import Path
from datetime import timedelta

import pandas as pd

from src.config import GOLD_DIR


PREDICTIONS_FILE = GOLD_DIR / "earthquake_predictions.csv"
FORECAST_FILE = GOLD_DIR / "earthquake_forecast.csv"


def create_next_7_days_forecast():
    print("Creating next 7 days earthquake activity forecast...")

    if not Path(PREDICTIONS_FILE).exists():
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_FILE}. Run python -m src.model first."
        )

    Path(GOLD_DIR).mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(PREDICTIONS_FILE)
    predictions["event_date"] = pd.to_datetime(predictions["event_date"], errors="coerce")
    predictions = predictions.dropna(subset=["event_date"])

    latest_date = predictions["event_date"].max()

    recent = predictions.sort_values("event_date").tail(14)

    avg_total_events = recent["total_events"].mean()
    avg_magnitude = recent["avg_magnitude"].mean()
    max_magnitude = recent["max_magnitude"].max()
    avg_depth_km = recent["avg_depth_km"].mean()
    max_depth_km = recent["max_depth_km"].max()
    avg_tsunami_events = recent["tsunami_events"].mean()

    high_activity_rate = recent["predicted_high_activity"].mean()

    forecast_rows = []

    for i in range(1, 8):
        forecast_date = latest_date + timedelta(days=i)

        risk_score = round(high_activity_rate * 100, 2)

        if risk_score >= 70:
            risk_label = "High Risk"
        elif risk_score >= 40:
            risk_label = "Medium Risk"
        else:
            risk_label = "Low Risk"

        forecast_rows.append(
            {
                "forecast_date": forecast_date.date(),
                "forecast_day": forecast_date.day_name(),
                "forecast_total_events": round(avg_total_events, 0),
                "forecast_avg_magnitude": round(avg_magnitude, 2),
                "forecast_max_magnitude": round(max_magnitude, 2),
                "forecast_avg_depth_km": round(avg_depth_km, 2),
                "forecast_max_depth_km": round(max_depth_km, 2),
                "forecast_tsunami_events": round(avg_tsunami_events, 0),
                "risk_score": risk_score,
                "risk_label": risk_label,
            }
        )

    forecast_df = pd.DataFrame(forecast_rows)
    forecast_df.to_csv(FORECAST_FILE, index=False)

    print(f"Forecast saved to: {FORECAST_FILE}")
    print(forecast_df)


if __name__ == "__main__":
    create_next_7_days_forecast()