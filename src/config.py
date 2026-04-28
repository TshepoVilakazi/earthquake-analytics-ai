from pathlib import Path

# ================================
# 🌍 USGS API CONFIG
# ================================
BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# Start small for testing, expand later
START_DATE = "2024-01-01"
END_DATE = "2024-12-31"

# ================================
# 📂 PROJECT PATHS
# ================================
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"

BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"

# ================================
# 📄 FILE PATHS
# ================================
# Bronze
BRONZE_FILE = BRONZE_DIR / "earthquake_bronze.json"

# Silver
SILVER_FILE = SILVER_DIR / "earthquake_silver.csv"

# Gold (Detailed + Aggregations)
GOLD_FILE = GOLD_DIR / "earthquake_gold.csv"
GOLD_DAILY_FILE = GOLD_DIR / "earthquake_daily_summary.csv"
GOLD_MAGNITUDE_FILE = GOLD_DIR / "earthquake_magnitude_summary.csv"

# ================================
# ⚙️ PIPELINE SETTINGS
# ================================
CHUNK_DAYS = 30          # Avoid USGS 20k record limit
MAX_RETRIES = 3
RETRY_DELAY = 5          # seconds