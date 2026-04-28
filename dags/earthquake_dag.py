from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


default_args = {
    "owner": "tshepo",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="earthquake_medallion_pipeline",
    default_args=default_args,
    description="USGS Earthquake Bronze to Silver to Gold to ML Forecast pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["earthquake", "medallion", "ml", "forecast", "airflow"],
) as dag:

    bronze_task = BashOperator(
        task_id="bronze_extract_usgs_api",
        bash_command="cd /opt/airflow/project && python -m src.bronze",
    )

    silver_task = BashOperator(
        task_id="silver_transform_data",
        bash_command="cd /opt/airflow/project && python -m src.silver",
    )

    gold_task = BashOperator(
        task_id="gold_build_analytics",
        bash_command="cd /opt/airflow/project && python -m src.gold",
    )

    model_task = BashOperator(
        task_id="train_earthquake_prediction_model",
        bash_command="cd /opt/airflow/project && python -m src.model",
    )

    forecast_task = BashOperator(
        task_id="create_next_7_days_forecast",
        bash_command="cd /opt/airflow/project && python -m src.forecast",
    )

    bronze_task >> silver_task >> gold_task >> model_task >> forecast_task