# Corn Price Predictor 🌽

An automated trading signal system for US corn futures. Every weekday morning it fetches the latest corn price, runs it through an ensemble machine learning model, and tells you whether to **BUY**, **SELL**, or **HOLD**.

---

## What It Does

- Pulls 20 years of US corn futures data (CBOT: ZC=F)
- Collects correlated assets: crude oil, soybeans, US Dollar Index
- Engineers 24 features: moving averages, RSI, MACD, Bollinger Bands, volatility, seasonality
- Trains an **Ensemble model (XGBoost + LSTM)** for daily BUY/SELL/HOLD signals
- Displays everything in a clean one-click visual dashboard
- Runs automatically every weekday at 6am UTC via Apache Airflow

---

## Results

Backtested on 4 years of data (2022–2026):

| Metric | Value |
|---|---|
| Starting capital | $50,000 |
| Ending capital | $58,175 |
| Total return | +16.35% |
| Win rate | 44.4% |
| Avg win | $5,006 |
| Avg loss | $2,370 |

During a period where corn prices fell ~20%, the model still generated positive returns.

---

## Dashboard

The dashboard shows:
- Today's BUY / SELL / HOLD signal with plain-English explanation
- Current price and daily/weekly/monthly performance
- Ensemble confidence bars
- Individual XGBoost vs LSTM model breakdown
- 180-day price chart with moving averages and Bollinger Bands

---

## Models

| Model | Accuracy | Notes |
|---|---|---|
| Decision Tree | 43% | Baseline — simple and interpretable |
| XGBoost | 43% | Better at handling tabular features |
| LSTM | 52% | Best for time series patterns |
| **Ensemble** | **48%** | XGBoost 40% + LSTM 60% weighted average |

The ensemble combines both models — XGBoost adds caution and stability, LSTM adds directional insight from price sequences.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Data collection, feature engineering, ML |
| PostgreSQL | Stores all historical price data |
| Apache Airflow | Automates daily data fetch + signal generation |
| TensorFlow / Keras | LSTM neural network |
| XGBoost | Gradient boosting model |
| Docker | Runs everything in containers |
| Yahoo Finance | Free source for corn futures data |

---

## Getting Started

### Requirements
- [Docker Desktop](https://www.docker.com/products/docker-desktop) — install and make sure it's running
- Git

---

### Windows

Open **Command Prompt** (search for it in the Start menu) and run:

```cmd
git clone https://github.com/Durianpandahedgehog/projects.git
cd projects\corn-predictor
setup.bat
```

Wait ~15 minutes. The dashboard will open automatically in your browser.

**Every day after:**
Double-click `run_dashboard.bat` in the `corn-predictor` folder.

---

### Mac

Open **Terminal** (press Cmd+Space and search for Terminal) and run:

```bash
git clone https://github.com/Durianpandahedgehog/projects.git
cd projects/corn-predictor
chmod +x setup.sh
bash setup.sh
```

Wait ~15 minutes. The dashboard will open automatically in your browser.

**Every day after:**
```bash
cd projects/corn-predictor
bash run_dashboard.sh
```

---

## How It Works

### Data
Every day the system fetches corn futures prices from Yahoo Finance and calculates:

- Moving averages (7, 30, 90 day)
- RSI, MACD, Bollinger Bands
- Price volatility
- Correlated assets: crude oil (CL=F), soybeans (ZS=F), US Dollar Index (DX-Y.NYB)
- Seasonal harvest indicators (Sep–Nov)

### Ensemble Model
Two models vote on the signal with weighted averaging:
- **XGBoost (40%)** — looks at today's features independently, conservative
- **LSTM (60%)** — looks at last 30 days as a sequence, captures trends

### Signal Logic
- **BUY** → ensemble predicts price will rise more than 2% in next 5 days
- **SELL** → ensemble predicts price will fall more than 2% in next 5 days
- **HOLD** → HOLD has highest confidence, or no strong movement expected

---

## Project Structure

corn-predictor/
├── collect_corn_data.py     # Fetches and stores price data + features
├── ml_model.py              # Decision Tree (baseline)
├── ml_model_xgb.py          # XGBoost model
├── ml_model_lstm.py         # LSTM neural network
├── ml_model_ensemble.py     # Trains XGBoost + LSTM ensemble
├── backtest.py              # Simulates trades on historical data
├── daily_signal.py          # Generates today's ensemble signal
├── dashboard.py             # Builds the visual HTML dashboard
├── dashboard.html           # Latest saved dashboard (open directly)
├── dags/
│   └── corn_pipeline.py     # Airflow DAG (daily automation)
├── Dockerfile               # Python + TensorFlow + XGBoost container
├── Dockerfile.airflow       # Airflow container with ML packages
├── docker-compose.yml       # Orchestrates all services
├── requirements.txt         # Python dependencies
├── setup.bat                # Windows: first time setup
├── run_dashboard.bat        # Windows: daily use
├── setup.sh                 # Mac: first time setup
└── run_dashboard.sh         # Mac: daily use

---

## Disclaimer

This project is for educational purposes only. It is **not financial advice**. Always do your own research before making any trading decisions. Past performance does not guarantee future results.

---

## Built With

- Python 3.12
- TensorFlow 2.16
- XGBoost
- Apache Airflow 2.9
- PostgreSQL 16
- Docker
- Yahoo Finance (yfinance)

