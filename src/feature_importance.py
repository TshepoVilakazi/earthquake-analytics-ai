from pathlib import Path
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from src.config import SILVER_FILE, GOLD_DIR


OUTPUT_FILE = GOLD_DIR / "feature_importance.csv"


def build_features(df):
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
    daily["event_date"] = pd.to_datetime(daily.iloc[:, 0])

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


def calculate_feature_importance():
    print("Calculating feature importance...")

    Path(GOLD_DIR).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SILVER_FILE)
    daily = build_features(df)

    feature_cols = [
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

    X = daily[feature_cols]
    y = daily["high_activity_day"]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    importance.to_csv(OUTPUT_FILE, index=False)

    print(f"Feature importance saved to: {OUTPUT_FILE}")
    print(importance)


if __name__ == "__main__":
    calculate_feature_importance()