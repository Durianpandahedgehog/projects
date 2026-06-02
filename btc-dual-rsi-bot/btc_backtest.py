"""
BTC 3-Strategy Backtest — Grid Search
Strategies: A=Supertrend 4h | B=EMA+MACD 1h | C=BB Breakout 1h
pip install pandas numpy yfinance
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import math
import numpy as np
import pandas as pd
import yfinance as yf
from itertools import product
from multiprocessing import Pool, cpu_count
from datetime import datetime, timedelta, timezone

# ══════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIG
# ══════════════════════════════════════════════════════════════════

MONTHS     = 20
FEE        = 0.0004
SLIP       = 0.0005
CAPITAL    = 1000.0
LEVERAGE   = 10
MARGIN_PCT = 0.20

GRIDS = {
    "A": {  # Supertrend 4h
        "st_period":    [7, 10, 14],
        "st_mult":      [2.5, 3.0, 3.5],
        "adx_thresh":   [20, 25, 30],
        "sl_mult":      [1.2, 1.5, 2.0],
        "tp_mult":      [3.0, 3.5, 4.5],
    },
    "B": {  # EMA 9/21/50 + MACD 1h
        "adx_thresh":   [0, 20, 25],
        "ema200_filter":[True, False],
        "sl_mult":      [0.8, 1.2, 1.5],
        "tp_mult":      [2.0, 2.5, 3.0],
    },
    "C": {  # Bollinger Breakout 1h
        "bb_period":    [14, 20, 25],
        "bb_std":       [1.8, 2.0, 2.3],
        "vol_mult":     [1.3, 1.5, 1.8],
        "sl_mult":      [1.0, 1.5, 2.0],
        "tp_mult":      [2.0, 2.5, 3.0],
    },
}

# ══════════════════════════════════════════════════════════════════
# SECTION 2 — DATA
# ══════════════════════════════════════════════════════════════════

def fetch(interval: str, months: int = MONTHS) -> pd.DataFrame:
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=min(months * 30, 729))
    tf_label = "4h" if interval == "4h" else "1h"
    print(f"  Fetching BTC-USD {tf_label} data ({months}mo)...")
    df = yf.download("BTC-USD", start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"), interval=interval,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df = df[["open","high","low","close","volume"]].dropna()
    print(f"  {len(df)} candles | {df.index[0].date()} → {df.index[-1].date()}")
    return df

# ══════════════════════════════════════════════════════════════════
# SECTION 3 — FEATURES
# ══════════════════════════════════════════════════════════════════

def _atr(df, period=14):
    hl  = df["high"] - df["low"]
    hpc = (df["high"] - df["close"].shift()).abs()
    lpc = (df["low"]  - df["close"].shift()).abs()
    return pd.concat([hl, hpc, lpc], axis=1).max(axis=1).ewm(alpha=1/period, adjust=False).mean()

def _adx(df, period=14):
    plus_dm  = df["high"].diff().clip(lower=0)
    minus_dm = (-df["low"].diff()).clip(lower=0)
    ov = (plus_dm > 0) & (minus_dm > 0)
    plus_dm[ov & (plus_dm <= minus_dm)] = 0
    minus_dm[ov & (minus_dm < plus_dm)] = 0
    atr14    = _atr(df, period)
    plus_di  = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr14
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr14
    dx       = (100 * (plus_di - minus_di).abs() /
                (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    return dx.ewm(alpha=1/period, adjust=False).mean()

def features_A(df, st_period, st_mult):
    d = df.copy()
    d["atr"]    = _atr(d, st_period)
    d["ema200"] = d["close"].ewm(span=200, adjust=False).mean()
    d["adx"]    = _adx(d)
    # RSI
    delta = d["close"].diff()
    gain  = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    # Supertrend
    hl2  = (d["high"] + d["low"]) / 2
    up   = hl2 - st_mult * d["atr"]
    dn   = hl2 + st_mult * d["atr"]
    st   = pd.Series(np.nan, index=d.index)
    trend= pd.Series(1,      index=d.index)
    for i in range(1, len(d)):
        up.iloc[i]  = max(up.iloc[i],  up.iloc[i-1])  if d["close"].iloc[i-1] > up.iloc[i-1]  else up.iloc[i]
        dn.iloc[i]  = min(dn.iloc[i],  dn.iloc[i-1])  if d["close"].iloc[i-1] < dn.iloc[i-1]  else dn.iloc[i]
        if   d["close"].iloc[i] > dn.iloc[i-1]: trend.iloc[i] = 1
        elif d["close"].iloc[i] < up.iloc[i-1]: trend.iloc[i] = -1
        else: trend.iloc[i] = trend.iloc[i-1]
        st.iloc[i] = up.iloc[i] if trend.iloc[i] == 1 else dn.iloc[i]
    d["st"]       = st
    d["st_trend"] = trend
    return d.dropna()

def features_B(df):
    d = df.copy()
    d["ema9"]   = d["close"].ewm(span=9,   adjust=False).mean()
    d["ema21"]  = d["close"].ewm(span=21,  adjust=False).mean()
    d["ema50"]  = d["close"].ewm(span=50,  adjust=False).mean()
    d["ema200"] = d["close"].ewm(span=200, adjust=False).mean()
    d["atr"]    = _atr(d)
    d["adx"]    = _adx(d)
    mf = d["close"].ewm(span=12, adjust=False).mean()
    ms = d["close"].ewm(span=26, adjust=False).mean()
    d["macd"]      = mf - ms
    d["macd_sig"]  = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_sig"]
    d["hist_slope"]= d["macd_hist"].diff()
    delta = d["close"].diff()
    gain  = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    d["rsi"]     = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    d["vol_ma20"]= d["volume"].rolling(20).mean()
    d["slope9"]  = d["ema9"].diff(2)
    return d.dropna()

def features_C(df, bb_period, bb_std, vol_mult):
    d = df.copy()
    d["atr"]    = _atr(d)
    d["ema200"] = d["close"].ewm(span=200, adjust=False).mean()
    sma         = d["close"].rolling(bb_period).mean()
    std         = d["close"].rolling(bb_period).std()
    d["bb_up"]  = sma + bb_std * std
    d["bb_dn"]  = sma - bb_std * std
    d["bb_mid"] = sma
    d["vol_ma"] = d["volume"].rolling(20).mean()
    d["vol_ok"] = (d["volume"] > d["vol_ma"] * vol_mult).astype(int)
    delta = d["close"].diff()
    gain  = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    return d.dropna()

# ══════════════════════════════════════════════════════════════════
# SECTION 4 — BACKTEST ENGINE (shared)
# ══════════════════════════════════════════════════════════════════

def _score(trades, balance):
    if len(trades) < 15: return None
    t   = pd.DataFrame(trades)
    ret = (balance - CAPITAL) / CAPITAL * 100
    cum = CAPITAL + t["pnl"].cumsum()
    dd  = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    r   = t["pnl"] / CAPITAL
    sh  = (r.mean() / r.std() * math.sqrt(8760)) if r.std() > 0 else 0
    pf_w = t[t["pnl"] > 0]["pnl"].sum()
    pf_l = t[t["pnl"] < 0]["pnl"].abs().sum()
    pf   = pf_w / pf_l if pf_l > 0 else 999
    score = sh * (1 - abs(dd) / 100)
    return {"n": len(t), "ret": round(ret,2), "dd": round(dd,2),
            "wr": round(t["win"].mean()*100,1),
            "sharpe": round(sh,2), "pf": round(pf,2), "score": round(score,3)}

def _sim(df, signals):
    """signals: list of (i, direction). Returns score dict."""
    balance  = CAPITAL
    trades   = []
    in_trade = False
    direction = entry_px = sl = tp = qty = 0.0
    sig_map  = {s[0]: s for s in signals}

    for i in range(len(df)):
        if in_trade:
            hi, lo = float(df.iloc[i]["high"]), float(df.iloc[i]["low"])
            reason = exit_px = None
            if direction == 1:
                if lo <= sl:   reason, exit_px = "SL", sl
                elif hi >= tp: reason, exit_px = "TP", tp
            else:
                if hi >= sl:   reason, exit_px = "SL", sl
                elif lo <= tp: reason, exit_px = "TP", tp
            if reason:
                gross   = (exit_px - entry_px) * direction * qty
                net     = gross - (qty * entry_px * FEE * 2)
                balance+= net
                trades.append({"pnl": net, "win": net > 0})
                in_trade = False

        if not in_trade and i in sig_map:
            _, d, atr, sl_m, tp_m = sig_map[i]
            fill     = float(df.iloc[i]["open"]) * (1 + SLIP * d)
            margin   = balance * MARGIN_PCT
            qty      = (margin * LEVERAGE * 0.95) / fill
            sl       = fill - atr * sl_m * d
            tp       = fill + atr * tp_m * d
            entry_px = fill
            direction= d
            in_trade = True

    return _score(trades, balance)

# ══════════════════════════════════════════════════════════════════
# SECTION 5 — STRATEGY SIGNAL GENERATORS
# ══════════════════════════════════════════════════════════════════

def signals_A(df, adx_thresh, sl_mult, tp_mult):
    sigs = []
    prev_trend = df["st_trend"].iloc[0]
    for i in range(1, len(df)):
        p = df.iloc[i-1]
        cur_trend = int(p["st_trend"])
        flip = cur_trend != prev_trend
        prev_trend = cur_trend
        if not flip: continue
        adx_ok = float(p["adx"]) > adx_thresh
        # Long: trend flipped up, price > EMA200, RSI not OB
        if cur_trend == 1 and adx_ok and float(p["close"]) > float(p["ema200"]) and float(p["rsi"]) < 70:
            sigs.append((i, 1, float(p["atr"]), sl_mult, tp_mult))
        # Short: trend flipped down, price < EMA200, RSI not OS
        elif cur_trend == -1 and adx_ok and float(p["close"]) < float(p["ema200"]) and float(p["rsi"]) > 30:
            sigs.append((i, -1, float(p["atr"]), sl_mult, tp_mult))
    return sigs

def signals_B(df, adx_thresh, ema200_filter, sl_mult, tp_mult):
    sigs = []
    for i in range(1, len(df)):
        p = df.iloc[i-1]
        close  = float(p["close"])
        e9,e21,e50,e200 = float(p["ema9"]),float(p["ema21"]),float(p["ema50"]),float(p["ema200"])
        macd,msig,hist,hslope = float(p["macd"]),float(p["macd_sig"]),float(p["macd_hist"]),float(p["hist_slope"])
        adx_ok = float(p["adx"]) > adx_thresh if adx_thresh > 0 else True
        rsi    = float(p["rsi"])
        slope  = float(p["slope9"])
        long_ema  = e9 > e21 > e50
        short_ema = e9 < e21 < e50
        long_sig  = long_ema  and macd > msig and hist > 0 and hslope > 0 and slope > 0 and 40 < rsi < 65 and adx_ok
        short_sig = short_ema and macd < msig and hist < 0 and hslope < 0 and slope < 0 and 35 < rsi < 60 and adx_ok
        if ema200_filter:
            long_sig  = long_sig  and close > e200
            short_sig = short_sig and close < e200
        if long_sig:  sigs.append((i,  1, float(p["atr"]), sl_mult, tp_mult))
        if short_sig: sigs.append((i, -1, float(p["atr"]), sl_mult, tp_mult))
    return sigs

def signals_C(df, sl_mult, tp_mult):
    sigs = []
    for i in range(1, len(df)):
        p = df.iloc[i-1]
        c = float(p["close"]); bb_up=float(p["bb_up"]); bb_dn=float(p["bb_dn"])
        rsi=float(p["rsi"]); vol_ok=bool(p["vol_ok"]); e200=float(p["ema200"])
        prev_c = float(df.iloc[i-2]["close"]) if i >= 2 else c
        long_sig  = prev_c < bb_up and c > bb_up and vol_ok and rsi < 72 and c > e200
        short_sig = prev_c > bb_dn and c < bb_dn and vol_ok and rsi > 28 and c < e200
        if long_sig:  sigs.append((i,  1, float(p["atr"]), sl_mult, tp_mult))
        if short_sig: sigs.append((i, -1, float(p["atr"]), sl_mult, tp_mult))
    return sigs

# ══════════════════════════════════════════════════════════════════
# SECTION 6 — WORKERS
# ══════════════════════════════════════════════════════════════════

def _worker_A(args):
    df, p = args
    feat = features_A(df, p["st_period"], p["st_mult"])
    sigs = signals_A(feat, p["adx_thresh"], p["sl_mult"], p["tp_mult"])
    res  = _sim(feat, sigs)
    return {**p, **res} if res else None

def _worker_B(args):
    df, p = args
    df   = features_B(df)
    sigs = signals_B(df, p["adx_thresh"], p["ema200_filter"], p["sl_mult"], p["tp_mult"])
    res  = _sim(df, sigs)
    return {**p, **res} if res else None

def _worker_C(args):
    df, p = args
    feat = features_C(df, p["bb_period"], p["bb_std"], p["vol_mult"])
    sigs = signals_C(feat, p["sl_mult"], p["tp_mult"])
    res  = _sim(feat, sigs)
    return {**p, **res} if res else None

def grid_search(worker, df, grid, label):
    keys   = list(grid.keys())
    combos = [dict(zip(keys, v)) for v in product(*grid.values())]
    cores  = cpu_count()
    print(f"\n  Strategy {label}: {len(combos)} combos on {cores} cores...")
    with Pool(processes=cores) as pool:
        raw = pool.map(worker, [(df, p) for p in combos])
    results = [r for r in raw if r is not None]
    print(f"  {len(results)} valid results.")
    return pd.DataFrame(results) if results else pd.DataFrame()

# ══════════════════════════════════════════════════════════════════
# SECTION 7 — RESULTS
# ══════════════════════════════════════════════════════════════════

def print_top(label, name, df_res):
    if df_res.empty:
        print(f"\n  Strategy {label} — No valid results."); return
    top = df_res.sort_values("score", ascending=False).head(5)
    print(f"\n{'='*95}")
    print(f"  Strategy {label} — {name}  |  Top 5 by Score")
    print(f"{'='*95}")
    param_cols = [c for c in top.columns if c not in ["n","ret","dd","wr","sharpe","pf","score"]]
    param_str  = "  ".join(f"{c[:8]:<8}" for c in param_cols)
    print(f"  {'#':<3} {param_str}  {'Ret%':<10} {'DD%':<8} {'WR%':<6} {'Sharpe':<8} {'PF':<6} {'Score':<8} N")
    print(f"  {'-'*90}")
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        p_vals = "  ".join(f"{str(r[c])[:8]:<8}" for c in param_cols)
        print(f"  {rank:<3} {p_vals}  {r.ret:>+8.1f}%  {r.dd:>6.1f}%  {r.wr:>5.1f}%  "
              f"{r.sharpe:>6.2f}  {r.pf:>5.2f}  {r.score:>7.3f}  {int(r.n)}")
    b = top.iloc[0]
    print(f"\n  BEST: Ret={b.ret:+.1f}% | DD={b.dd:.1f}% | WR={b.wr:.1f}% | "
          f"Sharpe={b.sharpe:.2f} | PF={b.pf:.2f} | Score={b.score:.3f} | N={int(b.n)}")

def print_winner(results):
    print(f"\n{'#'*95}")
    print("  WINNER COMPARISON — Best of each strategy")
    print(f"{'#'*95}")
    print(f"  {'Strat':<8} {'Ret%':<10} {'DD%':<8} {'WR%':<6} {'Sharpe':<8} {'PF':<6} {'Score':<8} N")
    print(f"  {'-'*70}")
    winner = None
    for label, name, df_res in results:
        if df_res.empty: continue
        b = df_res.sort_values("score", ascending=False).iloc[0]
        tag = f"{label} ({name})"
        print(f"  {tag:<20} {b.ret:>+8.1f}%  {b.dd:>6.1f}%  {b.wr:>5.1f}%  "
              f"{b.sharpe:>6.2f}  {b.pf:>5.2f}  {b.score:>7.3f}  {int(b.n)}")
        if winner is None or b.score > winner[1]:
            winner = (f"{label} — {name}", b.score, b)
    if winner:
        print(f"\n  🏆 WINNER: {winner[0]}  |  Score={winner[1]:.3f}")
    print(f"{'#'*95}")

# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  BTC 3-Strategy Backtest")
    print(f"  Capital=${CAPITAL} | Leverage={LEVERAGE}x | Margin={MARGIN_PCT*100:.0f}%")
    print(f"  Fee={FEE*100:.2f}% | Slip={SLIP*100:.2f}% | Period={MONTHS}mo")
    print("="*60)

    print("\n[1/3] Loading data...")
    df_4h = fetch("4h")
    df_1h = fetch("1h")

    print("\n[2/3] Running grid searches...")
    res_A = grid_search(_worker_A, df_4h, GRIDS["A"], "A (Supertrend 4h)")
    res_B = grid_search(_worker_B, df_1h, GRIDS["B"], "B (EMA+MACD 1h)")
    res_C = grid_search(_worker_C, df_1h, GRIDS["C"], "C (BB Breakout 1h)")

    print("\n[3/3] Results...")
    print_top("A", "Supertrend 4h", res_A)
    print_top("B", "EMA 9/21/50 + MACD 1h", res_B)
    print_top("C", "BB Breakout 1h", res_C)
    print_winner([("A","Supertrend 4h",res_A),
                  ("B","EMA+MACD 1h",  res_B),
                  ("C","BB Breakout 1h",res_C)])

    for label, df_res in [("A_supertrend",res_A),("B_ema_macd",res_B),("C_bb_breakout",res_C)]:
        if not df_res.empty:
            df_res.sort_values("score", ascending=False).to_csv(f"btc_{label}_results.csv", index=False)
    print("\n  CSVs saved. Done.")
