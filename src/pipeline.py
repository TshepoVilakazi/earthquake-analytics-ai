from src.bronze import extract_earthquake_data
from src.silver import transform_to_silver
from src.gold import build_gold


def run_pipeline():
    print("Starting Earthquake Medallion Pipeline...")

    extract_earthquake_data()
    transform_to_silver()
    build_gold()

    print("Pipeline completed successfully.")