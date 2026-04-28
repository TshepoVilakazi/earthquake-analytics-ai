import pandas as pd
from src.config import SILVER_FILE, GOLD_DIR, GOLD_DAILY_FILE, GOLD_MAGNITUDE_FILE


def build_gold():
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    print("Building Gold analytics...")

    df = pd.read_csv(SILVER_FILE)
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")

    daily_summary = (
        df.groupby("event_date")
        .agg(
            total_earthquakes=("earthquake_id", "count"),
            average_magnitude=("magnitude", "mean"),
            max_magnitude=("magnitude", "max"),
            average_depth_km=("depth_km", "mean")
        )
        .reset_index()
    )

    magnitude_summary = (
        df.groupby("magnitude_band")
        .agg(
            total_earthquakes=("earthquake_id", "count"),
            average_magnitude=("magnitude", "mean"),
            max_magnitude=("magnitude", "max"),
            average_depth_km=("depth_km", "mean")
        )
        .reset_index()
        .sort_values("total_earthquakes", ascending=False)
    )

    daily_summary.to_csv(GOLD_DAILY_FILE, index=False)
    magnitude_summary.to_csv(GOLD_MAGNITUDE_FILE, index=False)

    print(f"Gold daily summary saved to: {GOLD_DAILY_FILE}")
    print(f"Gold magnitude summary saved to: {GOLD_MAGNITUDE_FILE}")