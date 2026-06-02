"""
Stage 3 - Kafka Producer
Streams equipment sensor readings from equipment-sensors.csv.gz
to the 'machine-telemetry' Kafka topic, simulating real-time sensor data.
"""

import gzip
import csv
import json
import time
import sys
from kafka import KafkaProducer

KAFKA_BROKER  = "localhost:9092"
TOPIC_NAME    = "machine-telemetry"
DATA_FILE     = "../data/raw/equipment-sensors.csv.gz"
DELAY_SECONDS = 0.05

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

print(f"Connected to Kafka at {KAFKA_BROKER}")
print(f"Streaming to topic: '{TOPIC_NAME}'")
print("Press Ctrl+C to stop.\n")

sent = 0
try:
    with gzip.open(DATA_FILE, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            event = {
                "machine_id":            row.get("machine_id", ""),
                "factory_id":            row.get("factory_id", ""),
                "timestamp":             row.get("timestamp", ""),
                "temperature_c":         float(row["temperature_c"])        if row.get("temperature_c")        else None,
                "vibration_mm_s":        float(row["vibration_mm_s"])       if row.get("vibration_mm_s")       else None,
                "power_consumption_kw":  float(row["power_consumption_kw"]) if row.get("power_consumption_kw") else None,
                "oil_pressure_psi":      float(row["oil_pressure_psi"])     if row.get("oil_pressure_psi")     else None,
                "status":                row.get("status", ""),
            }
            producer.send(TOPIC_NAME, value=event)
            sent += 1
            if sent % 500 == 0:
                print(f"  Sent {sent} events... (machine={event['machine_id']}, temp={event['temperature_c']}, vib={event['vibration_mm_s']})")
            time.sleep(DELAY_SECONDS)

except KeyboardInterrupt:
    print(f"\nStopped after {sent} events.")
except FileNotFoundError:
    print(f"\nERROR: File not found at '{DATA_FILE}'")
    sys.exit(1)
finally:
    producer.flush()
    producer.close()
    print(f"Done. Total sent: {sent}")
