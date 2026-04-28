import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

from src.config import (
    BASE_URL,
    START_DATE,
    END_DATE,
    BRONZE_DIR,
    BRONZE_FILE,
    CHUNK_DAYS,
    MAX_RETRIES,
    RETRY_DELAY,
)


def generate_date_chunks(start_date: str, end_date: str, chunk_days: int):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    current = start

    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)

        yield (
            current.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d"),
        )

        current = chunk_end


def fetch_usgs_data(start_date: str, end_date: str) -> dict:
    params = {
        "format": "geojson",
        "starttime": start_date,
        "endtime": end_date,
        "orderby": "time",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Fetching: {start_date} to {end_date} | Attempt {attempt}")

            response = requests.get(
                BASE_URL,
                params=params,
                timeout=60,
            )

            if response.status_code != 200:
                print("USGS API ERROR")
                print("Status code:", response.status_code)
                print("Response:", response.text)

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as error:
            print(f"Request failed: {error}")

            if attempt == MAX_RETRIES:
                raise

            print(f"Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)


def extract_earthquake_data():
    print("Extracting earthquake data from USGS API...")

    Path(BRONZE_DIR).mkdir(parents=True, exist_ok=True)

    all_features = []

    for chunk_start, chunk_end in generate_date_chunks(
        START_DATE,
        END_DATE,
        CHUNK_DAYS,
    ):
        data = fetch_usgs_data(chunk_start, chunk_end)

        features = data.get("features", [])
        print(f"Records fetched: {len(features)}")

        all_features.extend(features)

    bronze_payload = {
        "type": "FeatureCollection",
        "metadata": {
            "source": "USGS Earthquake API",
            "start_date": START_DATE,
            "end_date": END_DATE,
            "chunk_days": CHUNK_DAYS,
            "total_records": len(all_features),
            "extracted_at": datetime.now().isoformat(),
        },
        "features": all_features,
    }

    with open(BRONZE_FILE, "w", encoding="utf-8") as file:
        json.dump(bronze_payload, file, indent=4)

    print(f"Bronze data saved to: {BRONZE_FILE}")
    print(f"Total records saved: {len(all_features)}")


if __name__ == "__main__":
    extract_earthquake_data()