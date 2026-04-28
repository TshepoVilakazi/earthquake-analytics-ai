import json
from pathlib import Path

import pandas as pd

from src.config import BRONZE_FILE, SILVER_DIR, SILVER_FILE


def classify_magnitude(magnitude):
    if pd.isna(magnitude):
        return "Unknown"
    elif magnitude < 2:
        return "Micro"
    elif magnitude < 4:
        return "Minor"
    elif magnitude < 5:
        return "Light"
    elif magnitude < 6:
        return "Moderate"
    elif magnitude < 7:
        return "Strong"
    elif magnitude < 8:
        return "Major"
    else:
        return "Great"


def transform_bronze_to_silver():
    print("Transforming Bronze data to Silver...")

    Path(SILVER_DIR).mkdir(parents=True, exist_ok=True)

    if not Path(BRONZE_FILE).exists():
        raise FileNotFoundError(f"Bronze file not found: {BRONZE_FILE}")

    with open(BRONZE_FILE, "r", encoding="utf-8") as file:
        bronze_data = json.load(file)

    records = []

    for feature in bronze_data.get("features", []):
        properties = feature.get("properties", {})
        geometry = feature.get("geometry", {})

        coordinates = geometry.get("coordinates", [None, None, None])

        longitude = coordinates[0] if len(coordinates) > 0 else None
        latitude = coordinates[1] if len(coordinates) > 1 else None
        depth_km = coordinates[2] if len(coordinates) > 2 else None

        records.append(
            {
                "earthquake_id": feature.get("id"),
                "magnitude": properties.get("mag"),
                "place": properties.get("place"),
                "event_time": properties.get("time"),
                "updated_time": properties.get("updated"),
                "timezone_offset": properties.get("tz"),
                "url": properties.get("url"),
                "detail_url": properties.get("detail"),
                "felt": properties.get("felt"),
                "cdi": properties.get("cdi"),
                "mmi": properties.get("mmi"),
                "alert": properties.get("alert"),
                "status": properties.get("status"),
                "tsunami": properties.get("tsunami"),
                "significance": properties.get("sig"),
                "network": properties.get("net"),
                "code": properties.get("code"),
                "ids": properties.get("ids"),
                "sources": properties.get("sources"),
                "types": properties.get("types"),
                "nst": properties.get("nst"),
                "dmin": properties.get("dmin"),
                "rms": properties.get("rms"),
                "gap": properties.get("gap"),
                "mag_type": properties.get("magType"),
                "event_type": properties.get("type"),
                "title": properties.get("title"),
                "longitude": longitude,
                "latitude": latitude,
                "depth_km": depth_km,
            }
        )

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("No records found in Bronze file.")

    df["event_time"] = pd.to_datetime(df["event_time"], unit="ms", errors="coerce")
    df["updated_time"] = pd.to_datetime(df["updated_time"], unit="ms", errors="coerce")

    numeric_columns = [
        "magnitude",
        "felt",
        "cdi",
        "mmi",
        "tsunami",
        "significance",
        "nst",
        "dmin",
        "rms",
        "gap",
        "longitude",
        "latitude",
        "depth_km",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["magnitude_band"] = df["magnitude"].apply(classify_magnitude)

    df["event_date"] = df["event_time"].dt.date
    df["event_year"] = df["event_time"].dt.year
    df["event_month"] = df["event_time"].dt.month
    df["event_day"] = df["event_time"].dt.day
    df["event_hour"] = df["event_time"].dt.hour

    df = df.drop_duplicates(subset=["earthquake_id"])

    df = df.sort_values(by="event_time", ascending=False)

    df.to_csv(SILVER_FILE, index=False)

    print(f"Silver data saved to: {SILVER_FILE}")
    print(f"Total Silver records: {len(df)}")


if __name__ == "__main__":
    transform_bronze_to_silver()