from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

BASE = "hdfs://namenode:9000/data/supplychain"

default_args = {
    "owner": "leftside-undergrads",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
}

def check_landing_zone(**context):
    import subprocess
    paths = [
        f"{BASE}/landing/production_lines",
        f"{BASE}/landing/inventory_levels",
        f"{BASE}/landing/equipment_sensors",
        f"{BASE}/landing/quality_inspections",
        f"{BASE}/landing/supplier_performance",
    ]
    missing = []
    for path in paths:
        result = subprocess.run(
            ["docker", "exec", "namenode", "hdfs", "dfs", "-test", "-d", path],
            capture_output=True,
        )
        if result.returncode != 0:
            missing.append(path)
    if missing:
        raise FileNotFoundError(f"Missing landing zone paths: {missing}")
    print("All landing zone paths exist ✓")

def data_quality_gate(**context):
    import subprocess
    paths = [
        f"{BASE}/landing/production_lines",
        f"{BASE}/landing/inventory_levels",
        f"{BASE}/landing/quality_inspections",
    ]
    for path in paths:
        result = subprocess.run(
            ["docker", "exec", "namenode", "hdfs", "dfs", "-count", path],
            capture_output=True, text=True,
        )
        output = result.stdout.strip().split()
        # output format: DIR_COUNT FILE_COUNT CONTENT_SIZE PATHNAME
        if len(output) >= 2 and int(output[1]) == 0:
            raise ValueError(f"Quality gate FAILED — no files found in {path}")
    print("Data quality gate passed ✓")

def validate_curated_output(**context):
    import subprocess
    paths = [
        f"{BASE}/curated/production_lines",
        f"{BASE}/curated/inventory_levels",
        f"{BASE}/curated/quality_inspections",
    ]
    for path in paths:
        result = subprocess.run(
            ["docker", "exec", "namenode", "hdfs", "dfs", "-test", "-d", path],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Curated output missing: {path}")
    print("Curated output validated ✓")

def send_success_alert(**context):
    print("✅ SupplyChain batch pipeline completed successfully!")

with DAG(
    dag_id="supplychain_batch_pipeline",
    default_args=default_args,
    schedule="0 6 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["supplychain", "batch", "stage4"],
) as dag:

    t1 = PythonOperator(
        task_id="check_landing_zone",
        python_callable=check_landing_zone,
        sla=timedelta(minutes=30),
    )

    t2 = PythonOperator(
        task_id="data_quality_gate",
        python_callable=data_quality_gate,
        sla=timedelta(minutes=20),
    )

    t3 = BashOperator(
        task_id="run_spark_transforms",
        bash_command=(
            "docker exec spark-master spark-submit "
            "--master spark://spark-master:7077 "
            "--executor-memory 1g "
            "/opt/spark/notebooks/02-spark-transforms.py"
        ),
        sla=timedelta(hours=1),
    )

    t4 = PythonOperator(
        task_id="validate_curated_output",
        python_callable=validate_curated_output,
        sla=timedelta(minutes=10),
    )

    t5 = PythonOperator(
        task_id="send_success_alert",
        python_callable=send_success_alert,
    )

    t1 >> t2 >> t3 >> t4 >> t5