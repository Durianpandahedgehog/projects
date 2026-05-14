import pandas as pd
import numpy as np
import os

DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
CSV_INPUT   = os.path.join(DATA_DIR, "Q_USDT_features.csv")
CSV_OUTPUT  = os.path.join(DATA_DIR, "Q_USDT_signals.csv")


def load_features():
    df = pd.read_csv(CSV_INPUT, index_col="date", parse_dates=True)
    print(f"OK: Loaded {len(df)} rows")
    return df


def generate_signals(df):
    print("Generating signals...")
    df["signal"] = "HOLD"

    # BUY conditions
    buy = (
        (df["rsi_7"] < 30) &
        (df["close"] > df["ema_21"]) &
        (df["close"] > df["ema_50"]) & 
        (df["macd"] > df["macd_signal"]) &
        (df["volume_ratio"] > 1.5) &
        (df["btc_pct_change"] > -0.5)
    )

    # SELL conditions
    sell = (
        (df["rsi_7"] > 65) &
        (df["close"] < df["ema_21"]) &
        (df["macd"] < df["macd_signal"]) &
        (df["volume_ratio"] > 1.5) &
        (df["btc_pct_change"] < 0.5)
    )

    df.loc[buy,  "signal"] = "BUY"
    df.loc[sell, "signal"] = "SELL"

    print(f"  BUY  signals: {(df['signal'] == 'BUY').sum()}")
    print(f"  SELL signals: {(df['signal'] == 'SELL').sum()}")
    print(f"  HOLD signals: {(df['signal'] == 'HOLD').sum()}")
    return df


def save_signals(df):
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(CSV_OUTPUT)
    print(f"OK: Saved to {CSV_OUTPUT}")


if __name__ == "__main__":
    df = load_features()
    df = generate_signals(df)
    save_signals(df)