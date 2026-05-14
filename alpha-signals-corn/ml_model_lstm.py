import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
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

# ── 4. BUILD SEQUENCES ────────────────────────────────────────────────────────
# LSTM needs sequences — for each day, look back 30 days of history

def build_sequences(X, y, lookback=30):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

# ── 5. PREPARE DATA ───────────────────────────────────────────────────────────

def prepare_data(df, lookback=30):
    available = [f for f in FEATURES if f in df.columns]
    X_raw = df[available].values
    y_raw = df["signal"].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Build sequences
    X_seq, y_seq = build_sequences(X_scaled, y_raw, lookback)

    # Train/test split (80/20, time-ordered)
    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    # One-hot encode labels for keras
    y_train_cat = to_categorical(y_train, num_classes=3)
    y_test_cat  = to_categorical(y_test,  num_classes=3)

    print(f"OK: Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"OK: Sequence shape: {X_train.shape}")
    return X_train, X_test, y_train, y_test, y_train_cat, y_test_cat, scaler, available

# ── 6. BUILD LSTM MODEL ───────────────────────────────────────────────────────

def build_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(3, activation="softmax")  # 3 classes: HOLD, BUY, SELL
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model

# ── 7. TRAIN ──────────────────────────────────────────────────────────────────

def train_model(model, X_train, y_train_cat, X_test, y_test_cat):
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    # Compute class weights to fix imbalance
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.array([0, 1, 2])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    print(f"OK: Class weights: {class_weight_dict}")

    history = model.fit(
        X_train, y_train_cat,
        epochs=100,
        batch_size=64,
        validation_data=(X_test, y_test_cat),
        callbacks=[early_stop],
        class_weight=class_weight_dict,
        verbose=1
    )

    print("OK: LSTM model trained")
    return model, history

# ── 8. EVALUATE ───────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test):
    y_pred_proba = model.predict(X_test)
    y_pred       = np.argmax(y_pred_proba, axis=1)

    print("\n── Classification Report ────────────────────────────")
    print(classification_report(y_test, y_pred,
          target_names=["HOLD", "BUY", "SELL"]))

    print("── Confusion Matrix ─────────────────────────────────")
    print(confusion_matrix(y_test, y_pred))

# ── 9. LATEST SIGNAL ─────────────────────────────────────────────────────────

def show_latest_signal(model, df, scaler, available, lookback=30):
    X_raw    = df[available].values
    X_scaled = scaler.transform(X_raw)
    sequence = X_scaled[-lookback:].reshape(1, lookback, len(available))

    proba  = model.predict(sequence)[0]
    pred   = np.argmax(proba)
    labels = {0: "HOLD", 1: "BUY", 2: "SELL"}

    print(f"\n── Latest Signal ({df.index[-1].date()}) ──────────────────────")
    print(f"    Signal     : {labels[pred]}")
    print(f"    Confidence : HOLD={proba[0]:.0%} | BUY={proba[1]:.0%} | SELL={proba[2]:.0%}")

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()
    df = create_labels(df, threshold=2.0)

    X_train, X_test, y_train, y_test, \
    y_train_cat, y_test_cat, scaler, available = prepare_data(df, lookback=30)

    model          = build_model(input_shape=(30, len(available)))
    model, history = train_model(model, X_train, y_train_cat, X_test, y_test_cat)

    evaluate_model(model, X_test, y_test)
    show_latest_signal(model, df, scaler, available)