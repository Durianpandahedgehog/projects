import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from xgboost import XGBClassifier
import os, pickle

MODEL_DIR   = "/app/models"
LOOKBACK    = 30
XGB_WEIGHT  = 0.4
LSTM_WEIGHT = 0.6

FEATURES = [
    "close_price","moving_avg_7","moving_avg_30","moving_avg_90",
    "price_change","pct_change","volatility_30","daily_range",
    "month","day_of_week","is_harvest","rsi_14","macd","macd_signal",
    "bb_upper","bb_lower","bb_width","volume_ma_20",
    "oil_close","soy_close","dxy_close",
    "oil_pct_change","soy_pct_change","dxy_pct_change",
]

def get_engine():
    host = os.getenv("DB_HOST","localhost")
    port = os.getenv("DB_PORT","5432")
    name = os.getenv("DB_NAME","corn_db")
    user = os.getenv("DB_USER","corn_user")
    pw   = os.getenv("DB_PASS","corn_pass")
    return create_engine(f"postgresql://{user}:{pw}@{host}:{port}/{name}")

def load_data():
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM corn_prices ORDER BY date ASC", engine)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df

def generate_signal():
    df        = load_data()
    lstm_model= load_model(f"{MODEL_DIR}/lstm_model.keras")
    xgb_model = XGBClassifier()
    xgb_model.load_model(f"{MODEL_DIR}/xgb_model.json")

    with open(f"{MODEL_DIR}/scaler.pkl","rb") as f:
        scaler, available = pickle.load(f)

    X_raw  = df[available].values
    X_scaled = scaler.transform(X_raw)

    # XGBoost — use last row
    xgb_proba  = xgb_model.predict_proba(X_scaled[-1:])

    # LSTM — use last 30 rows as sequence
    seq        = X_scaled[-LOOKBACK:].reshape(1, LOOKBACK, len(available))
    lstm_proba = lstm_model.predict(seq, verbose=0)

    # Combine
    combined   = (XGB_WEIGHT * xgb_proba) + (LSTM_WEIGHT * lstm_proba)
    hold_p     = float(combined[0][0])
    buy_p      = float(combined[0][1])
    sell_p     = float(combined[0][2])

    if hold_p >= buy_p and hold_p >= sell_p:
        signal = 0
    elif buy_p >= 0.25 and buy_p >= sell_p:
        signal = 1
    else:
        signal = 2

    labels = {0:"HOLD", 1:"BUY", 2:"SELL"}
    today  = df.index[-1].date()
    price  = float(df["close_price"].iloc[-1])

    print("\n" + "="*50)
    print(f"  CORN DAILY SIGNAL — {today}")
    print("="*50)
    print(f"  Signal     : {labels[signal]}")
    print(f"  Price      : ${price:.2f} / bushel")
    print(f"  Confidence : HOLD={hold_p:.0%} | BUY={buy_p:.0%} | SELL={sell_p:.0%}")
    print(f"  XGB proba  : HOLD={xgb_proba[0][0]:.0%} | BUY={xgb_proba[0][1]:.0%} | SELL={xgb_proba[0][2]:.0%}")
    print(f"  LSTM proba : HOLD={lstm_proba[0][0]:.0%} | BUY={lstm_proba[0][1]:.0%} | SELL={lstm_proba[0][2]:.0%}")
    print("="*50)

    return labels[signal], combined[0], price, xgb_proba[0], lstm_proba[0]

if __name__ == "__main__":
    generate_signal()