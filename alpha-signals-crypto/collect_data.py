import pandas as pd
import requests
from datetime import datetime, timedelta
import os
import time

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH  = os.path.join(DATA_DIR, "Q_USDT_5m.csv")
BASE_URL  = "https://fapi.binance.com"   # futures public API


def fetch_ohlcv(symbol, timeframe="5m", days=180):
    print(f"Fetching {symbol}...")
    since  = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    symbol = symbol.replace("/", "").replace(":USDT", "")   # QUSDT

    all_candles = []
    while True:
        url    = f"{BASE_URL}/fapi/v1/klines"
        params = {
            "symbol"   : symbol,
            "interval" : timeframe,
            "startTime": since,
            "limit"    : 1000
        }
        resp    = requests.get(url, params=params, timeout=10)
        candles = resp.json()

        if not candles or not isinstance(candles, list):
            break

        all_candles.extend(candles)
        since = candles[-1][0] + 1

        if len(candles) < 1000:
            break
        time.sleep(0.3)

    df = pd.DataFrame(all_candles, columns=[
        "date", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df["date"]  = pd.to_datetime(df["date"], unit="ms")
    df          = df.set_index("date")
    df          = df.astype(float)
    df          = df[~df.index.duplicated(keep="last")]
    print(f"OK: {len(df)} candles fetched for {symbol}")
    return df


def fetch_btc(days=180):
    df = fetch_ohlcv("BTC/USDT:USDT", timeframe="5m", days=days)
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
    q_df   = fetch_ohlcv("Q/USDT:USDT", "5m", days=180)
    btc_df = fetch_btc(days=180)
    df     = merge_data(q_df, btc_df)
    save_data(df)