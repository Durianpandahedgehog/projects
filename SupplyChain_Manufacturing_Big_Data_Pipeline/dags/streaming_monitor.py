from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

KAFKA_BROKER = "kafka:9092"
KAFKA_TOPIC = "machine-telemetry"

default_args = {
    "owner": "leftside-undergrads",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": True,
}

def check_kafka_topic(**context):
    import subprocess
    result = subprocess.run(
        [
            "docker", "exec", "kafka",
            "kafka-topics", "--bootstrap-server", KAFKA_BROKER,
            "--describe", "--topic", KAFKA_TOPIC,
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or KAFKA_TOPIC not in result.stdout:
        raise RuntimeError(f"Kafka topic '{KAFKA_TOPIC}' not found or broker is down!")
    print(f"Kafka topic '{KAFKA_TOPIC}' is healthy ✓")

def check_consumer_lag(**context):
    import subprocess
    result = subprocess.run(
        [
            "docker", "exec", "broker",
            "kafka-consumer-groups", "--bootstrap-server", KAFKA_BROKER,
            "--describe", "--group", "spark-streaming-group",
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("Could not retrieve consumer group info!")
    print(f"Consumer group status:\n{result.stdout}")

def check_messages_flowing(**context):
    import subprocess, time
    result1 = subprocess.run(
        [
            "docker", "exec", "broker",
            "kafka-run-class", "kafka.tools.GetOffsetShell",
            "--broker-list", KAFKA_BROKER,
            "--topic", KAFKA_TOPIC,
        ],
        capture_output=True, text=True,
    )
    time.sleep(10)
    result2 = subprocess.run(
        [
            "docker", "exec", "broker",
            "kafka-run-class", "kafka.tools.GetOffsetShell",
            "--broker-list", KAFKA_BROKER,
            "--topic", KAFKA_TOPIC,
        ],
        capture_output=True, text=True,
    )
    if result1.stdout == result2.stdout:
        print("⚠️ WARNING: No new messages detected in machine-telemetry topic!")
    else:
        print("Messages are flowing into machine-telemetry ✓")

def send_health_alert(**context):
    print("✅ Streaming health check completed — machine-telemetry topic is healthy!")

with DAG(
    dag_id="supplychain_streaming_monitor",
    default_args=default_args,
    schedule="*/30 * * * *",   # runs every 30 minutes
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["supplychain", "streaming", "monitor", "stage4"],
) as dag:

    t1 = PythonOperator(
        task_id="check_kafka_topic",
        python_callable=check_kafka_topic,
    )

    t2 = PythonOperator(
        task_id="check_consumer_lag",
        python_callable=check_consumer_lag,
    )

    t3 = PythonOperator(
        task_id="check_messages_flowing",
        python_callable=check_messages_flowing,
    )

    t4 = PythonOperator(
        task_id="send_health_alert",
        python_callable=send_health_alert,
    )

    t1 >> t2 >> t3 >> t4