import ccxt
import pandas as pd
from datetime import datetime, timedelta
import os
import time
COIN = "Q/USDT:USDT"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "Q_USDT_5m.csv")
def fetch_ohlcv(symbol, timeframe="5m", days=180):
    exchange = ccxt.binance({"enableRateLimit": True})
    since    = exchange.parse8601(
        (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT00:00:00Z")
    )
    all_candles = []
    while True:
        candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        if not candles:
            break
        all_candles.extend(candles)
        since = candles[-1][0] + 1
        if len(candles) < 1000:
            break
        time.sleep(0.5)
    df = pd.DataFrame(all_candles, columns=["date","open","high","low","close","volume"])
    df["date"] = pd.to_datetime(df["date"], unit="ms")
    df = df.set_index("date")
    print(f"OK: {len(df)} candles fetched for {symbol}")
    return df
def fetch_btc(days=180):
    df = fetch_ohlcv("BTC/USDT", "5m", days)
    df = df[["close"]].rename(columns={"close": "btc_close"})
    return df
def merge_data(q_df, btc_df):
    df = q_df.join(btc_df, how="left")
    df = df.ffill()
    print(f"OK: Merged dataframe has {len(df)} rows")
    return df
def save_data(df):
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(CSV_PATH)
    print(f"OK: Saved {len(df)} rows to {CSV_PATH}")


if __name__ == "__main__":
    q_df   = fetch_ohlcv(COIN, "5m", days=180)
    btc_df = fetch_btc(days=180)
    df     = merge_data(q_df, btc_df)
    save_data(df)