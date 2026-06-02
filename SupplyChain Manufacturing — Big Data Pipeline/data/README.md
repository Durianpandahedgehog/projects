# SupplyChain Manufacturing Data

## Data Source

The dataset for this project comes from the ISM 6562 course repository:

https://github.com/prof-tcsmith/ism6562s26-class/tree/main/final-projects/data/08-supplychain-manufacturing

---

## Required Files

Download the following files:

- production-lines.csv.gz
- inventory-levels.csv.gz
- equipment-sensors.csv.gz
- quality-inspections.json.gz
- supplier-performance.json.gz

---

## Folder Structure

After downloading, place the files in the following directory:

data/raw/

---

## Important Notes

- The raw data files are **not included in this repository**.
- They are excluded using `.gitignore` to avoid uploading large files.
- Spark can read `.csv.gz` and `.json.gz` files directly without unzipping them.

---

## Description of Each Dataset

### production-lines.csv.gz
Contains manufacturing production data including:
- line ID and factory ID
- timestamp of production activity
- product ID and batch ID
- units produced and defect count
- operator ID and shift (day, evening, night)
- machine temperature (machine_temp_c) and vibration level during production

---

### inventory-levels.csv.gz
Contains inventory data including:
- warehouse ID and product ID
- date of record
- quantity on hand and quantity reserved
- reorder point and lead time (in days)
- supplier ID and unit cost

---

### equipment-sensors.csv.gz
Contains machine sensor readings including:
- machine ID and factory ID
- timestamp of reading
- temperature (temperature_c) and vibration (vibration_mm_s)
- power consumption (power_consumption_kw)
- oil pressure (oil_pressure_psi)
- machine status (running, idle, maintenance)

---

### quality-inspections.json.gz
Contains product inspection data including:
- inspection ID, batch ID, and product ID
- timestamp of inspection
- inspector ID
- defect type and severity (critical, major, minor, none)
- nested measurement data:
  - dimension_mm
  - tolerance_mm
  - deviation_mm

---

### supplier-performance.json.gz
Contains supplier performance data including:
- supplier ID and country
- order ID
- expected delivery date and actual delivery date
- quantity ordered and quantity delivered
- quality score




