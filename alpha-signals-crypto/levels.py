import pandas as pd
import numpy as np


def find_support_resistance(df, lookback=500, window=5):
    """
    Find support and resistance levels from recent price action.
    lookback = how many candles to look back
    window   = how many candles on each side to confirm a level
    """
    recent = df.tail(lookback).copy()

    # Local lows — support
    lows = []
    for i in range(window, len(recent) - window):
        candle = recent.iloc[i]
        left   = recent.iloc[i - window:i]
        right  = recent.iloc[i + 1:i + window + 1]
        if candle["low"] <= left["low"].min() and candle["low"] <= right["low"].min():
            lows.append((recent.index[i], candle["low"]))

    # Local highs — resistance
    highs = []
    for i in range(window, len(recent) - window):
        candle = recent.iloc[i]
        left   = recent.iloc[i - window:i]
        right  = recent.iloc[i + 1:i + window + 1]
        if candle["high"] >= left["high"].max() and candle["high"] >= right["high"].max():
            highs.append((recent.index[i], candle["high"]))

    return lows, highs


def get_trade_levels(df, min_rr=1.5):
    """
    Calculate entry, take profit, stop loss and risk/reward.
    Only returns a valid setup if RR >= min_rr.
    """
    current_price = float(df.iloc[-1]["close"])
    lows, highs   = find_support_resistance(df)

    # Nearest support below current price
    support_levels = [price for _, price in lows if price < current_price]
    if not support_levels:
        return None
    stop_loss = max(support_levels)   # closest one below

    # Nearest resistance above current price
    resistance_levels = [price for _, price in highs if price > current_price]
    if not resistance_levels:
        return None
    take_profit = min(resistance_levels)   # closest one above

    # Risk / Reward
    risk   = current_price - stop_loss
    reward = take_profit - current_price

    if risk <= 0 or reward <= 0:
        return None

    rr = reward / risk

    if rr < min_rr:
        return None

    return {
        "entry"       : current_price,
        "take_profit" : take_profit,
        "stop_loss"   : stop_loss,
        "risk"        : risk,
        "reward"      : reward,
        "rr"          : rr,
        "tp_pct"      : (take_profit - current_price) / current_price * 100,
        "sl_pct"      : (current_price - stop_loss)   / current_price * 100,
    }