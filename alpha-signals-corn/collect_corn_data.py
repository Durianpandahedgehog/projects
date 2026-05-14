import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import os
import time

# ── 1. DATABASE CONNECTION ────────────────────────────────────────────────────

def get_engine():
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "corn_db")
    user = os.getenv("DB_USER", "corn_user")
    pw   = os.getenv("DB_PASS", "corn_pass")
    return create_engine(f"postgresql://{user}:{pw}@{host}:{port}/{name}")

def create_table(engine):
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS corn_prices"))
        conn.execute(text("""
            CREATE TABLE corn_prices (
                id              SERIAL PRIMARY KEY,
                date            DATE NOT NULL UNIQUE,
                open_price      DECIMAL(10,2),
                close_price     DECIMAL(10,2),
                high_price      DECIMAL(10,2),
                low_price       DECIMAL(10,2),
                volume          DECIMAL(15,2),
                moving_avg_7    DECIMAL(10,2),
                moving_avg_30   DECIMAL(10,2),
                moving_avg_90   DECIMAL(10,2),
                price_change    DECIMAL(10,2),
                pct_change      DECIMAL(10,4),
                volatility_30   DECIMAL(10,4),
                daily_range     DECIMAL(10,2),
                month           INTEGER,
                day_of_week     INTEGER,
                is_harvest      INTEGER,
                rsi_14          DECIMAL(10,4),
                macd            DECIMAL(10,4),
                macd_signal     DECIMAL(10,4),
                bb_upper        DECIMAL(10,2),
                bb_lower        DECIMAL(10,2),
                bb_width        DECIMAL(10,4),
                volume_ma_20    DECIMAL(15,2),
                oil_close       DECIMAL(10,2),
                soy_close       DECIMAL(10,2),
                dxy_close       DECIMAL(10,4),
                oil_pct_change  DECIMAL(10,4),
                soy_pct_change  DECIMAL(10,4),
                dxy_pct_change  DECIMAL(10,4),
                created_at      TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.commit()
    print("OK: Table created")

# ── 2. FETCH CORRELATED ASSETS ────────────────────────────────────────────────

def fetch_asset(ticker, start, end, col_name):
    df = yf.Ticker(ticker).history(start=start, end=end, interval="1d")
    if df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[["Close"]].rename(columns={"Close": col_name})

# ── 3. FETCH CORN + ALL FEATURES ─────────────────────────────────────────────

def fetch_corn_prices(years=20):
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=365 * years)
    print(f"Fetching data from {start_date.date()} to {end_date.date()}...")

    # Corn futures
    corn = yf.Ticker("ZC=F").history(start=start_date, end=end_date, interval="1d")
    corn = corn.rename(columns={
        "Open":   "open_price",
        "High":   "high_price",
        "Low":    "low_price",
        "Close":  "close_price",
        "Volume": "volume",
    })
    corn.index = pd.to_datetime(corn.index).tz_localize(None)
    corn.index.name = "date"
    corn = corn[["open_price", "high_price", "low_price", "close_price", "volume"]]
    print(f"OK: {len(corn)} corn trading days retrieved")

    # Correlated assets
    oil = fetch_asset("CL=F",      start_date, end_date, "oil_close")
    soy = fetch_asset("ZS=F",      start_date, end_date, "soy_close")
    dxy = fetch_asset("DX-Y.NYB",  start_date, end_date, "dxy_close")

    # Merge all on date index
    df = corn.copy()
    for asset in [oil, soy, dxy]:
        if not asset.empty:
            df = df.join(asset, how="left")

    df = df.ffill()  # forward fill any missing days
    print("OK: Correlated assets merged")
    return df

# ── 4. ENGINEER FEATURES ─────────────────────────────────────────────────────

def calculate_features(df):
    print("Calculating features...")

    # Basic features
    df["moving_avg_7"]  = df["close_price"].rolling(7).mean()
    df["moving_avg_30"] = df["close_price"].rolling(30).mean()
    df["moving_avg_90"] = df["close_price"].rolling(90).mean()
    df["price_change"]  = df["close_price"].diff()
    df["pct_change"]    = df["close_price"].pct_change() * 100
    df["volatility_30"] = df["close_price"].rolling(30).std()
    df["daily_range"]   = df["high_price"] - df["low_price"]
    df["month"]         = df.index.month
    df["day_of_week"]   = df.index.dayofweek
    df["is_harvest"]    = df["month"].isin([9, 10, 11]).astype(int)

    # RSI (14-day)
    delta     = df["close_price"].diff()
    gain      = delta.clip(lower=0).rolling(14).mean()
    loss      = (-delta.clip(upper=0)).rolling(14).mean()
    rs        = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD (12/26 EMA, 9 signal)
    ema12          = df["close_price"].ewm(span=12).mean()
    ema26          = df["close_price"].ewm(span=26).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()

    # Bollinger Bands (20-day)
    bb_mid          = df["close_price"].rolling(20).mean()
    bb_std          = df["close_price"].rolling(20).std()
    df["bb_upper"]  = bb_mid + (2 * bb_std)
    df["bb_lower"]  = bb_mid - (2 * bb_std)
    df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / bb_mid

    # Volume MA
    df["volume_ma_20"] = df["volume"].rolling(20).mean()

    # Correlated asset % changes
    for col in ["oil_close", "soy_close", "dxy_close"]:
        if col in df.columns:
            pct_col = col.replace("_close", "_pct_change")
            df[pct_col] = df[col].pct_change() * 100

    df = df.dropna(subset=["moving_avg_90", "rsi_14"])
    print(f"OK: {len(df)} clean rows ready")
    return df

# ── 5. SAVE TO POSTGRES ───────────────────────────────────────────────────────

def save_to_db(df, engine):
    print("Saving to PostgreSQL...")
    df.to_sql("corn_prices", engine, if_exists="append",
              index=True, method="multi", chunksize=500)
    print("OK: Data saved")

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Waiting for database...")
    time.sleep(5)
    engine      = get_engine()
    create_table(engine)
    raw_df      = fetch_corn_prices(years=20)
    featured_df = calculate_features(raw_df)
    save_to_db(featured_df, engine)
    print(f"Done! {len(featured_df)} rows saved to database.")