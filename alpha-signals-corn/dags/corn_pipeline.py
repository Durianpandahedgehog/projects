from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '/opt/airflow')

default_args = {
    "owner": "roland",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

def fetch_and_store():
    import yfinance as yf
    import pandas as pd
    from datetime import datetime, timedelta
    from collect_corn_data import calculate_features, get_engine
    from sqlalchemy import text

    end   = datetime.today()
    start = end - timedelta(days=200)

    # Fetch corn
    corn = yf.Ticker("ZC=F").history(start=start, end=end, interval="1d")
    corn = corn.rename(columns={
        "Open": "open_price", "High": "high_price",
        "Low":  "low_price",  "Close": "close_price", "Volume": "volume"
    })
    corn.index = pd.to_datetime(corn.index).tz_localize(None)
    corn.index.name = "date"
    corn = corn[["open_price", "high_price", "low_price", "close_price", "volume"]]

    # Fetch correlated assets
    def fetch_asset(ticker, col):
        df = yf.Ticker(ticker).history(start=start, end=end, interval="1d")
        if df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df[["Close"]].rename(columns={"Close": col})

    oil = fetch_asset("CL=F",     "oil_close")
    soy = fetch_asset("ZS=F",     "soy_close")
    dxy = fetch_asset("DX-Y.NYB", "dxy_close")

    df = corn.copy()
    for asset in [oil, soy, dxy]:
        if not asset.empty:
            df = df.join(asset, how="left")
    df = df.ffill()

    featured = calculate_features(df)
    latest   = featured.iloc[[-1]]

    engine = get_engine()
    with engine.begin() as conn:
        for idx, row in latest.iterrows():
            row_dict = row.to_dict()
            row_dict["date"] = idx
            for col in ["oil_close", "soy_close", "dxy_close",
                        "oil_pct_change", "soy_pct_change", "dxy_pct_change"]:
                row_dict.setdefault(col, None)

            conn.execute(text("""
                INSERT INTO corn_prices (date, open_price, close_price, high_price, low_price,
                    volume, moving_avg_7, moving_avg_30, moving_avg_90, price_change, pct_change,
                    volatility_30, daily_range, month, day_of_week, is_harvest, rsi_14, macd,
                    macd_signal, bb_upper, bb_lower, bb_width, volume_ma_20, oil_close, soy_close,
                    dxy_close, oil_pct_change, soy_pct_change, dxy_pct_change)
                VALUES (:date, :open_price, :close_price, :high_price, :low_price,
                    :volume, :moving_avg_7, :moving_avg_30, :moving_avg_90, :price_change, :pct_change,
                    :volatility_30, :daily_range, :month, :day_of_week, :is_harvest, :rsi_14, :macd,
                    :macd_signal, :bb_upper, :bb_lower, :bb_width, :volume_ma_20, :oil_close, :soy_close,
                    :dxy_close, :oil_pct_change, :soy_pct_change, :dxy_pct_change)
                ON CONFLICT (date) DO NOTHING
            """), row_dict)
    print(f"OK: Processed row for {latest.index[0].date()}")
def run_daily_signal():
    import sys
    sys.path.insert(0, '/opt/airflow')
    from daily_signal import generate_signal
    signal, proba, price = generate_signal()
    print(f"Today's signal: {signal} at ${price:.2f}/bushel")

def retrain_model():
    """Retrain model every 30 days to incorporate new data."""
    import sys
    sys.path.insert(0, '/opt/airflow')
    import os
    from daily_signal import load_data, train_and_save_model

    # Only retrain if model is older than 30 days
    MODEL_PATH = "/app/models/lstm_model.keras"
    if os.path.exists(MODEL_PATH):
        age_days = (datetime.now().timestamp() - os.path.getmtime(MODEL_PATH)) / 86400
        if age_days < 30:
            print(f"Model is {age_days:.0f} days old — skipping retrain")
            return

    print("Retraining model with latest data...")
    df = load_data()
    train_and_save_model(df)

with DAG(
    "corn_daily_pipeline",
    default_args=default_args,
    description="Fetch daily US corn prices and generate buy/sell signal",
    schedule_interval="0 6 * * 1-5",  # 6am UTC, Monday-Friday only
    start_date=datetime(2026, 1, 1),
    catchup=False,
) as dag:

    t1 = PythonOperator(
        task_id="fetch_and_store_price",
        python_callable=fetch_and_store,
    )

    t2 = PythonOperator(
        task_id="generate_daily_signal",
        python_callable=run_daily_signal,
    )

    t3 = PythonOperator(
        task_id="retrain_model_if_needed",
        python_callable=retrain_model,
    )

    t1 >> t2 >> t3