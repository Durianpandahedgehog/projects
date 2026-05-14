import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, confusion_matrix
import os

# ── 1. LOAD DATA FROM POSTGRES ────────────────────────────────────────────────

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

# ── 2. CREATE TARGET LABEL ────────────────────────────────────────────────────

def create_labels(df, threshold=2.0):
    """
    BUY  (1) = price rises more than threshold% in next 5 days
    SELL (2) = price drops more than threshold% in next 5 days
    HOLD (0) = everything else
    """
    future_return = df["close_price"].shift(-5) / df["close_price"] - 1
    future_return *= 100  # convert to percentage

    df["signal"] = 0  # HOLD
    df.loc[future_return >  threshold, "signal"] = 1  # BUY
    df.loc[future_return < -threshold, "signal"] = 2  # SELL

    df = df.dropna()
    print(f"OK: Labels created")
    print(f"    BUY  signals: {(df['signal'] == 1).sum()}")
    print(f"    SELL signals: {(df['signal'] == 2).sum()}")
    print(f"    HOLD signals: {(df['signal'] == 0).sum()}")
    return df

# ── 3. PREPARE FEATURES ───────────────────────────────────────────────────────

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
]

def prepare_features(df):
    X = df[FEATURES]
    y = df["signal"]
    return X, y

# ── 4. TRAIN/TEST SPLIT (80/20) ───────────────────────────────────────────────

def split_data(X, y):
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    print(f"OK: Train size: {len(X_train)} | Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test

# ── 5. TRAIN DECISION TREE ────────────────────────────────────────────────────

def train_model(X_train, y_train):
    model = DecisionTreeClassifier(
        max_depth=5,       # keep it simple to avoid overfitting
        min_samples_leaf=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("OK: Model trained")
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
    importances = pd.Series(model.feature_importances_, index=FEATURES)
    print(importances.sort_values(ascending=False).to_string())

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df                              = load_data()
    df                              = create_labels(df, threshold=2.0)
    X, y                            = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model                           = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)