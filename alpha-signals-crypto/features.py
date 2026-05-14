import pandas as pd
import numpy as np
import os

DATA_DIR     = os.path.join(os.path.dirname(__file__), "data")
CSV_INPUT    = os.path.join(DATA_DIR, "Q_USDT_5m.csv")
CSV_OUTPUT   = os.path.join(DATA_DIR, "Q_USDT_features.csv")


def load_data():
    df = pd.read_csv(CSV_INPUT, index_col="date", parse_dates=True)
    print(f"OK: Loaded {len(df)} rows")
    return df


def calculate_features(df):
    print("Calculating features...")

    # RSI (7-period)
    delta        = df["close"].diff()
    gain         = delta.clip(lower=0).rolling(7).mean()
    loss         = (-delta.clip(upper=0)).rolling(7).mean()
    rs           = gain / loss
    df["rsi_7"]  = 100 - (100 / (1 + rs))

    # EMAs
    df["ema_9"]  = df["close"].ewm(span=9).mean()
    df["ema_21"] = df["close"].ewm(span=21).mean()
    df["ema_50"] = df["close"].ewm(span=50).mean()

    # MACD
    ema12              = df["close"].ewm(span=12).mean()
    ema26              = df["close"].ewm(span=26).mean()
    df["macd"]         = ema12 - ema26
    df["macd_signal"]  = df["macd"].ewm(span=9).mean()

    # Bollinger Bands (20-period)
    bb_mid         = df["close"].rolling(20).mean()
    bb_std         = df["close"].rolling(20).std()
    df["bb_upper"] = bb_mid + (2 * bb_std)
    df["bb_lower"] = bb_mid - (2 * bb_std)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid

    # Volume spike
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    # Candle body size as % of price
    df["candle_body"] = abs(df["close"] - df["open"]) / df["open"] * 100

    # BTC and Q % changes
    df["btc_pct_change"]   = df["btc_close"].pct_change() * 100
    df["price_pct_change"] = df["close"].pct_change() * 100

    df = df.dropna()
    print(f"OK: {len(df)} rows after feature calculation")
    return df


def save_features(df):
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(CSV_OUTPUT)
    print(f"OK: Saved {len(df)} rows to {CSV_OUTPUT}")


if __name__ == "__main__":
    df = load_data()
    df = calculate_features(df)
    save_features(df)