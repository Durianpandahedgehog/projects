import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import os

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────

def get_engine():
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "corn_db")
    user = os.getenv("DB_USER", "corn_user")
    pw   = os.getenv("DB_PASS", "corn_pass")
    return create_engine(f"postgresql://{user}:{pw}@{host}:{port}/{name}")

def load_data():
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM corn_prices ORDER BY date ASC", engine)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    print(f"OK: Loaded {len(df)} rows from database")
    return df

# ── 2. CREATE LABELS ──────────────────────────────────────────────────────────

def create_labels(df, threshold=2.0):
    future_return = df["close_price"].shift(-5) / df["close_price"] - 1
    future_return *= 100

    df["signal"] = 0
    df.loc[future_return >  threshold, "signal"] = 1
    df.loc[future_return < -threshold, "signal"] = 2
    df = df.dropna()

    print(f"OK: Labels created")
    print(f"    BUY  signals: {(df['signal'] == 1).sum()}")
    print(f"    SELL signals: {(df['signal'] == 2).sum()}")
    print(f"    HOLD signals: {(df['signal'] == 0).sum()}")
    return df

# ── 3. FEATURES ───────────────────────────────────────────────────────────────

FEATURES = [
    "close_price",
    "moving_avg_7",
    "moving_avg_30",
    "moving_avg_90",
    "price_change",
    "pct_change",
    "volatility_30",
    "daily_range",
    "month",
    "day_of_week",
    "is_harvest",
    "rsi_14",
    "macd",
    "macd_signal",
    "bb_upper",
    "bb_lower",
    "bb_width",
    "volume_ma_20",
    "oil_close",
    "soy_close",
    "dxy_close",
    "oil_pct_change",
    "soy_pct_change",
    "dxy_pct_change",
]

def prepare_features(df):
    # Only keep features that exist in the dataframe
    available = [f for f in FEATURES if f in df.columns]
    missing   = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"NOTE: Missing features skipped: {missing}")
    X = df[available]
    y = df["signal"]
    return X, y

# ── 4. TRAIN/TEST SPLIT ───────────────────────────────────────────────────────

def split_data(X, y):
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    print(f"OK: Train size: {len(X_train)} | Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test

# ── 5. TRAIN XGBOOST ──────────────────────────────────────────────────────────

def train_model(X_train, y_train):
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        eval_metric="mlogloss",
        random_state=42,
    )

    model.fit(X_train, y_train, sample_weight=sample_weights)
    print("OK: XGBoost model trained")
    return model

# ── 6. EVALUATE ───────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("\n── Classification Report ────────────────────────────")
    print(classification_report(y_test, y_pred,
          target_names=["HOLD", "BUY", "SELL"]))

    print("── Confusion Matrix ─────────────────────────────────")
    print(confusion_matrix(y_test, y_pred))

    print("\n── Top Feature Importances ──────────────────────────")
    importances = pd.Series(model.feature_importances_, index=X_test.columns)
    print(importances.sort_values(ascending=False).to_string())

# ── 7. LATEST SIGNAL ─────────────────────────────────────────────────────────

def show_latest_signal(model, df):
    available = [f for f in FEATURES if f in df.columns]
    latest    = df[available].iloc[[-1]]
    pred      = model.predict(latest)[0]
    proba     = model.predict_proba(latest)[0]
    labels    = {0: "HOLD", 1: "BUY", 2: "SELL"}

    print(f"\n── Latest Signal ({df.index[-1].date()}) ──────────────────────")
    print(f"    Signal     : {labels[pred]}")
    print(f"    Confidence : HOLD={proba[0]:.0%} | BUY={proba[1]:.0%} | SELL={proba[2]:.0%}")

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df                               = load_data()
    df                               = create_labels(df, threshold=2.0)
    X, y                             = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model                            = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    show_latest_signal(model, df)