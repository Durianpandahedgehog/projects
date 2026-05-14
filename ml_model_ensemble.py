import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import pickle, os

def get_engine():
    host = os.getenv("DB_HOST","localhost")
    port = os.getenv("DB_PORT","5432")
    name = os.getenv("DB_NAME","corn_db")
    user = os.getenv("DB_USER","corn_user")
    pw   = os.getenv("DB_PASS","corn_pass")
    return create_engine(f"postgresql://{user}:{pw}@{host}:{port}/{name}")

FEATURES = [
    "close_price","moving_avg_7","moving_avg_30","moving_avg_90",
    "price_change","pct_change","volatility_30","daily_range",
    "month","day_of_week","is_harvest","rsi_14","macd","macd_signal",
    "bb_upper","bb_lower","bb_width","volume_ma_20",
    "oil_close","soy_close","dxy_close",
    "oil_pct_change","soy_pct_change","dxy_pct_change",
]

LOOKBACK     = 30
XGB_WEIGHT   = 0.4
LSTM_WEIGHT  = 0.6
MODEL_DIR    = "/app/models"

def load_data():
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM corn_prices ORDER BY date ASC", engine)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    print(f"OK: Loaded {len(df)} rows")
    return df

def create_labels(df, threshold=2.0):
    future_return = df["close_price"].shift(-5) / df["close_price"] - 1
    future_return *= 100
    df = df.copy()
    df["signal"] = 0
    df.loc[future_return >  threshold, "signal"] = 1
    df.loc[future_return < -threshold, "signal"] = 2
    df = df.dropna()
    print(f"OK: BUY={( df['signal']==1).sum()} SELL={(df['signal']==2).sum()} HOLD={(df['signal']==0).sum()}")
    return df

def build_sequences(X, y, lookback=30):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i-lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

def prepare(df):
    available = [f for f in FEATURES if f in df.columns]
    X_raw  = df[available].values
    y_raw  = df["signal"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    split = int(len(X_raw) * 0.8)

    # For XGBoost (flat features)
    X_xgb_train = X_scaled[:split]
    X_xgb_test  = X_scaled[split:]
    y_xgb_train = y_raw[:split]
    y_xgb_test  = y_raw[split:]

    # For LSTM (sequences)
    X_seq, y_seq = build_sequences(X_scaled, y_raw, LOOKBACK)
    seq_split    = int(len(X_seq) * 0.8)
    X_lstm_train = X_seq[:seq_split]
    X_lstm_test  = X_seq[seq_split:]
    y_lstm_train = y_seq[:seq_split]
    y_lstm_test  = y_seq[seq_split:]
    y_lstm_train_cat = to_categorical(y_lstm_train, 3)
    y_lstm_test_cat  = to_categorical(y_lstm_test, 3)

    print(f"OK: XGB train={len(X_xgb_train)} test={len(X_xgb_test)}")
    print(f"OK: LSTM train={len(X_lstm_train)} test={len(X_lstm_test)}")
    return (X_xgb_train, X_xgb_test, y_xgb_train, y_xgb_test,
            X_lstm_train, X_lstm_test, y_lstm_train, y_lstm_test,
            y_lstm_train_cat, y_lstm_test_cat, scaler, available, seq_split)

def train_xgboost(X_train, y_train):
    print("\nTraining XGBoost...")
    weights = compute_sample_weight(class_weight="balanced", y=y_train)
    model = XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, gamma=0.1,
        eval_metric="mlogloss", random_state=42
    )
    model.fit(X_train, y_train, sample_weight=weights)
    print("OK: XGBoost trained")
    return model

def train_lstm(X_train, y_train, y_train_cat, X_test, y_test_cat, n_features):
    print("\nTraining LSTM...")
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(LOOKBACK, n_features)),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(3, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    classes = np.array([0,1,2])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(
        X_train, y_train_cat,
        epochs=100, batch_size=64,
        validation_data=(X_test, y_test_cat),
        callbacks=[early_stop],
        class_weight=dict(zip(classes, weights)),
        verbose=1
    )
    print("OK: LSTM trained")
    return model

def ensemble_predict(xgb_model, lstm_model, X_xgb, X_lstm):
    xgb_proba  = xgb_model.predict_proba(X_xgb)
    lstm_proba = lstm_model.predict(X_lstm, verbose=0)

    # Align lengths (LSTM has fewer rows due to lookback)
    min_len   = min(len(xgb_proba), len(lstm_proba))
    xgb_proba = xgb_proba[-min_len:]
    lstm_proba= lstm_proba[-min_len:]

    combined  = (XGB_WEIGHT * xgb_proba) + (LSTM_WEIGHT * lstm_proba)
    return combined

def evaluate(name, y_true, y_pred):
    print(f"\n── {name} ────────────────────────────────────────")
    print(classification_report(y_true, y_pred, target_names=["HOLD","BUY","SELL"]))

def save_models(xgb_model, lstm_model, scaler, available):
    os.makedirs(MODEL_DIR, exist_ok=True)
    xgb_model.save_model(f"{MODEL_DIR}/xgb_model.json")
    lstm_model.save(f"{MODEL_DIR}/lstm_model.keras")
    with open(f"{MODEL_DIR}/scaler.pkl","wb") as f:
        pickle.dump((scaler, available), f)
    print(f"\nOK: All models saved to {MODEL_DIR}")

if __name__ == "__main__":
    df  = load_data()
    df  = create_labels(df)
    out = prepare(df)
    (X_xgb_train, X_xgb_test, y_xgb_train, y_xgb_test,
     X_lstm_train, X_lstm_test, y_lstm_train, y_lstm_test,
     y_lstm_train_cat, y_lstm_test_cat,
     scaler, available, seq_split) = out

    xgb_model  = train_xgboost(X_xgb_train, y_xgb_train)
    lstm_model = train_lstm(X_lstm_train, y_lstm_train, y_lstm_train_cat,
                            X_lstm_test, y_lstm_test_cat, len(available))

    # Evaluate each individually
    xgb_pred  = xgb_model.predict(X_xgb_test)
    lstm_pred = np.argmax(lstm_model.predict(X_lstm_test, verbose=0), axis=1)
    evaluate("XGBoost", y_xgb_test[-len(lstm_pred):], xgb_pred[-len(lstm_pred):])
    evaluate("LSTM", y_lstm_test, lstm_pred)

    # Evaluate ensemble
    combined_proba = ensemble_predict(xgb_model, lstm_model, X_xgb_test, X_lstm_test)
    ensemble_pred  = np.argmax(combined_proba, axis=1)
    evaluate("Ensemble (XGB 40% + LSTM 60%)", y_lstm_test, ensemble_pred)

    save_models(xgb_model, lstm_model, scaler, available)
    print("\nDone! Run daily_signal.py to get today's ensemble signal.")