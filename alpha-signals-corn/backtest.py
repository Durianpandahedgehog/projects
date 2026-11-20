import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import os

def get_engine():
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "corn_db")
    user = os.getenv("DB_USER", "corn_user")
    pw   = os.getenv("DB_PASS", "corn_pass")
    return create_engine(f"postgresql://{user}:{pw}@{host}:{port}/{name}")

FEATURES = [
    "close_price", "moving_avg_7", "moving_avg_30", "moving_avg_90",
    "price_change", "pct_change", "volatility_30", "daily_range",
    "month", "day_of_week", "is_harvest", "rsi_14", "macd", "macd_signal",
    "bb_upper", "bb_lower", "bb_width", "volume_ma_20",
    "oil_close", "soy_close", "dxy_close",
    "oil_pct_change", "soy_pct_change", "dxy_pct_change",
]

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
    df["signal"] = 0
    df.loc[future_return >  threshold, "signal"] = 1
    df.loc[future_return < -threshold, "signal"] = 2
    df = df.dropna()
    return df

def build_sequences(X, y, lookback=30):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

def prepare_data(df, lookback=30):
    available = [f for f in FEATURES if f in df.columns]
    X_raw     = df[available].values
    y_raw     = df["signal"].values
    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(X_raw)
    X_seq, y_seq = build_sequences(X_scaled, y_raw, lookback)
    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]
    y_train_cat = to_categorical(y_train, num_classes=3)
    y_test_cat  = to_categorical(y_test,  num_classes=3)
    dates      = df.index[lookback:]
    test_dates = dates[split:]
    print(f"OK: Train size: {len(X_train)} | Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test, y_train_cat, y_test_cat, scaler, available, test_dates

def build_and_train_model(X_train, y_train, y_train_cat, X_test, y_test_cat, n_features):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(30, n_features)),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(3, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    classes = np.array([0, 1, 2])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))

    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(
        X_train, y_train_cat,
        epochs=100, batch_size=64,
        validation_data=(X_test, y_test_cat),
        callbacks=[early_stop],
        class_weight=class_weight_dict,
        verbose=1
    )
    print("OK: Model trained")
    return model

def run_backtest(model, X_test, test_dates, df, starting_capital=50000.0):
    print("\nRunning backtest on test period...")

    y_pred_proba = model.predict(X_test)

    # Fire BUY/SELL when confidence exceeds threshold
    BUY_THRESHOLD  = 0.25
    SELL_THRESHOLD = 0.25

    signals = []
    for proba in y_pred_proba:
        hold_p, buy_p, sell_p = proba[0], proba[1], proba[2]
        if buy_p >= BUY_THRESHOLD and buy_p >= sell_p:
            signals.append(1)   # BUY
        elif sell_p >= SELL_THRESHOLD and sell_p >= buy_p:
            signals.append(2)   # SELL
        else:
            signals.append(0)   # HOLD

    test_df = df.loc[test_dates].copy()
    test_df["signal"] = signals

    print(f"  Test period : {test_df.index[0].date()} to {test_df.index[-1].date()}")
    print(f"  BUY signals : {(test_df['signal'] == 1).sum()}")
    print(f"  SELL signals: {(test_df['signal'] == 2).sum()}")
    print(f"  HOLD signals: {(test_df['signal'] == 0).sum()}")

    capital      = starting_capital
    position     = 0
    entry_price  = 0.0
    trades       = []
    equity_curve = []

    for date, row in test_df.iterrows():
        price  = float(row["close_price"])
        signal = int(row["signal"])

        if signal == 1 and position == 0:
            cost = 100 * price
            if cost <= capital:
                position    = 100
                entry_price = price
                capital    -= cost
                trades.append({
                    "date": date, "action": "BUY",
                    "price": price, "bushels": 100,
                    "pnl": None, "capital": capital
                })

        elif signal == 2 and position > 0:
            pnl      = (price - entry_price) * position
            capital += position * price
            trades.append({
                "date": date, "action": "SELL",
                "price": price, "bushels": position,
                "pnl": pnl, "capital": capital
            })
            position = 0

        equity_curve.append({"date": date, "equity": capital + position * price})

    if position > 0:
        last_price = float(test_df.iloc[-1]["close_price"])
        pnl        = (last_price - entry_price) * position
        capital   += position * last_price
        trades.append({
            "date": test_df.index[-1], "action": "SELL (close)",
            "price": last_price, "bushels": position,
            "pnl": pnl, "capital": capital
        })

    return pd.DataFrame(trades), pd.DataFrame(equity_curve), capital

def print_results(trades_df, equity_df, starting_capital, final_capital):
    print("\n── Backtest Results ─────────────────────────────────")
    print(f"  Starting capital : ${starting_capital:,.2f}")
    print(f"  Final capital    : ${final_capital:,.2f}")
    total_return = (final_capital - starting_capital) / starting_capital * 100
    print(f"  Total return     : {total_return:+.2f}%")

    if trades_df.empty:
        print("\n  No trades executed.")
        return

    sells   = trades_df[trades_df["action"].str.contains("SELL")].dropna(subset=["pnl"])
    winning = sells[sells["pnl"] > 0]
    losing  = sells[sells["pnl"] <= 0]

    print(f"\n  Total trades   : {len(sells)}")
    print(f"  Winning trades : {len(winning)}")
    print(f"  Losing trades  : {len(losing)}")
    if len(sells) > 0:
        print(f"  Win rate       : {len(winning)/len(sells)*100:.1f}%")
    if not winning.empty:
        print(f"  Avg win        : ${winning['pnl'].mean():,.2f}")
    if not losing.empty:
        print(f"  Avg loss       : ${losing['pnl'].mean():,.2f}")

    print("\n── All Trades ───────────────────────────────────────")
    print(trades_df.to_string(index=False))

    print("\n── Equity Curve (every 50 days) ─────────────────────")
    print(equity_df.iloc[::50][["date", "equity"]].to_string(index=False))

if __name__ == "__main__":
    STARTING_CAPITAL = 50000.0

    df = load_data()
    df = create_labels(df, threshold=2.0)

    X_train, X_test, y_train, y_test, \
    y_train_cat, y_test_cat, scaler, available, test_dates = prepare_data(df)

    model = build_and_train_model(
        X_train, y_train, y_train_cat,
        X_test, y_test_cat,
        n_features=len(available)
    )

    trades_df, equity_df, final_capital = run_backtest(
        model, X_test, test_dates, df,
        starting_capital=STARTING_CAPITAL
    )

    print_results(trades_df, equity_df, STARTING_CAPITAL, final_capital)