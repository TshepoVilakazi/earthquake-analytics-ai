from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.config import SILVER_FILE, GOLD_DIR


MODEL_OUTPUT_FILE = GOLD_DIR / "earthquake_predictions.csv"
MODEL_METRICS_FILE = GOLD_DIR / "earthquake_model_metrics.txt"


def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Building daily ML features...")

    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df = df.dropna(subset=["event_time"])

    for col in ["magnitude", "depth_km", "tsunami"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    daily = (
        df.groupby(df["event_time"].dt.date)
        .agg(
            total_events=("earthquake_id", "count"),
            avg_magnitude=("magnitude", "mean"),
            max_magnitude=("magnitude", "max"),
            avg_depth_km=("depth_km", "mean"),
            max_depth_km=("depth_km", "max"),
            tsunami_events=("tsunami", "sum"),
        )
        .reset_index()
    )

    daily = daily.rename(columns={"event_time": "event_date"})
    daily["event_date"] = pd.to_datetime(daily["event_date"])

    daily["day_of_week"] = daily["event_date"].dt.dayofweek
    daily["month"] = daily["event_date"].dt.month
    daily["day"] = daily["event_date"].dt.day

    daily["events_previous_day"] = daily["total_events"].shift(1)
    daily["avg_magnitude_previous_day"] = daily["avg_magnitude"].shift(1)
    daily["max_magnitude_previous_day"] = daily["max_magnitude"].shift(1)

    daily = daily.dropna()

    threshold = daily["total_events"].quantile(0.75)
    daily["high_activity_day"] = (daily["total_events"] >= threshold).astype(int)

    return daily


def train_prediction_model():
    print("Training earthquake activity prediction model...")

    Path(GOLD_DIR).mkdir(parents=True, exist_ok=True)

    if not Path(SILVER_FILE).exists():
        raise FileNotFoundError(f"Silver file not found: {SILVER_FILE}")

    df = pd.read_csv(SILVER_FILE)

    daily = build_daily_features(df)

    feature_columns = [
        "avg_magnitude",
        "max_magnitude",
        "avg_depth_km",
        "max_depth_km",
        "tsunami_events",
        "day_of_week",
        "month",
        "day",
        "events_previous_day",
        "avg_magnitude_previous_day",
        "max_magnitude_previous_day",
    ]

    X = daily[feature_columns]
    y = daily["high_activity_day"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    daily["predicted_high_activity"] = model.predict(X)
    daily["risk_label"] = daily["predicted_high_activity"].map(
        {
            0: "Normal Activity",
            1: "High Activity",
        }
    )

    daily.to_csv(MODEL_OUTPUT_FILE, index=False)

    metrics_text = f"""
Earthquake Activity Prediction Model
====================================

Target:
High earthquake activity day

Definition:
A high activity day is any day where total earthquake events are greater than or equal to the 75th percentile.

Model:
RandomForestClassifier

Rows used:
{len(daily)}

Accuracy:
{accuracy:.4f}

Confusion Matrix:
{confusion_matrix(y_test, y_pred)}

Classification Report:
{classification_report(y_test, y_pred)}
"""

    with open(MODEL_METRICS_FILE, "w", encoding="utf-8") as file:
        file.write(metrics_text)

    print(f"Predictions saved to: {MODEL_OUTPUT_FILE}")
    print(f"Metrics saved to: {MODEL_METRICS_FILE}")
    print(f"Model accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    train_prediction_model()