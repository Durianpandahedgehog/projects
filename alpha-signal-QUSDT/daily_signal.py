from collect_data import fetch_ohlcv, fetch_btc, merge_data
from features import calculate_features
from strategy import generate_signals


def get_signal():
    print("Fetching latest data...")
    q_df   = fetch_ohlcv("Q/USDT:USDT", "5m", days=60)
    btc_df = fetch_btc(days=60)
    df     = merge_data(q_df, btc_df)
    df     = calculate_features(df)
    df     = generate_signals(df)

    latest = df.iloc[-1]
    signal = latest["signal"]

    print("\n-- Latest Signal -------------------------")
    print(f"  Date   : {df.index[-1]}")
    print(f"  Price  : {latest['close']:.6f} USDT")
    print(f"  RSI 7  : {latest['rsi_7']:.2f}")
    print(f"  MACD   : {latest['macd']:.8f}")
    print(f"  Volume : {latest['volume_ratio']:.2f}x average")
    print(f"  BTC    : {latest['btc_pct_change']:+.2f}%")
    print(f"\n  SIGNAL : {signal}")
    print("------------------------------------------")


if __name__ == "__main__":
    get_signal()