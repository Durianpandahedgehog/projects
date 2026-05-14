import pandas as pd
import numpy as np
import os

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
CSV_INPUT  = os.path.join(DATA_DIR, "Q_USDT_signals.csv")

STARTING_CAPITAL = 1000.0
FEE              = 0.001   # 0.1% per trade (Binance standard)
SLIPPAGE         = 0.0005  # 0.05% slippage estimate


def load_signals():
    df = pd.read_csv(CSV_INPUT, index_col="date", parse_dates=True)
    print(f"OK: Loaded {len(df)} rows")
    return df


def run_backtest(df):
    print("\nRunning backtest...")
    capital      = STARTING_CAPITAL
    position     = 0.0
    entry_price  = 0.0
    trades       = []
    equity_curve = []

    for date, row in df.iterrows():
        price  = float(row["close"])
        signal = row["signal"]

        # Stop loss — exit if price drops 3% below entry
        stop_loss_price = entry_price * 0.92

        if position > 0 and price <= stop_loss_price:
            proceeds = position * price * (1 - FEE - SLIPPAGE)
            pnl      = proceeds - (position * entry_price)
            capital  = proceeds
            trades.append({
                "date": date, "action": "STOP LOSS",
                "price": price, "pnl": pnl, "capital": capital
            })
            position = 0

        if signal == "BUY" and position == 0:
            cost        = capital * (1 - FEE - SLIPPAGE)
            position    = cost / price
            entry_price = price
            capital     = 0
            trades.append({
                "date": date, "action": "BUY",
                "price": price, "capital": capital
            })

        elif signal == "SELL" and position > 0:
            proceeds = position * price * (1 - FEE - SLIPPAGE)
            pnl      = proceeds - (position * entry_price)
            capital  = proceeds
            trades.append({
                "date": date, "action": "SELL",
                "price": price, "pnl": pnl, "capital": capital
            })
            position = 0

        equity = capital + (position * price if position > 0 else 0)
        equity_curve.append({"date": date, "equity": equity})

    # Close any open position at end
    if position > 0:
        last_price = float(df.iloc[-1]["close"])
        proceeds   = position * last_price * (1 - FEE - SLIPPAGE)
        pnl        = proceeds - (position * entry_price)
        capital    = proceeds
        trades.append({
            "date": df.index[-1], "action": "SELL (close)",
            "price": last_price, "pnl": pnl, "capital": capital
        })

    return pd.DataFrame(trades), pd.DataFrame(equity_curve), capital


def print_results(trades_df, equity_df, final_capital):
    print("\n-- Backtest Results ---------------------")
    print(f"  Starting capital : ${STARTING_CAPITAL:,.2f}")
    print(f"  Final capital    : ${final_capital:,.2f}")
    total_return = (final_capital - STARTING_CAPITAL) / STARTING_CAPITAL * 100
    print(f"  Total return     : {total_return:+.2f}%")

    sells   = trades_df[trades_df["action"].str.contains("SELL")].dropna(subset=["pnl"])
    winning = sells[sells["pnl"] > 0]
    losing  = sells[sells["pnl"] <= 0]

    print(f"\n  Total trades   : {len(sells)}")
    print(f"  Winning trades : {len(winning)}")
    print(f"  Losing trades  : {len(losing)}")
    if len(sells) > 0:
        print(f"  Win rate       : {len(winning)/len(sells)*100:.1f}%")
    if not winning.empty:
        print(f"  Avg win        : ${winning['pnl'].mean():,.4f}")
    if not losing.empty:
        print(f"  Avg loss       : ${losing['pnl'].mean():,.4f}")

    print("\n-- All Trades ---------------------------")
    print(trades_df.to_string(index=False))


if __name__ == "__main__":
    df                              = load_signals()
    trades_df, equity_df, final_cap = run_backtest(df)
    print_results(trades_df, equity_df, final_cap)