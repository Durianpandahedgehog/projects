"""
BTC Round 2 — 3 Strategy Grid Search
A: 4h EMA Cross + RSI Pullback
B: 4h Chandelier Exit + Momentum
C: Dual RSI (4h trend + 1h entry)
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
    "A": {  # 4h EMA Cross + RSI Pullback
        "ema_fast":   [9, 13, 21],
        "ema_slow":   [50, 100],
        "rsi_entry":  [35, 40, 45],    # pullback RSI min (long)
        "sl_mult":    [1.0, 1.5, 2.0],
        "tp_mult":    [2.5, 3.5, 4.5],
    },
    "B": {  # 4h Chandelier Exit + Momentum
        "ce_period":  [14, 22],
        "ce_mult":    [2.0, 3.0, 4.0],
        "mom_bars":   [3, 6],
        "mom_thresh": [0.005, 0.010],
        "sl_mult":    [1.0, 1.5],
        "tp_mult":    [2.5, 3.5],
    },
    "C": {  # Dual RSI
        "rsi4h_bull": [55, 60],        # 4h RSI above = bull regime
        "rsi4h_bear": [40, 45],        # 4h RSI below = bear regime
        "rsi1h_long": [30, 35, 40],    # 1h RSI oversold entry (long)
        "rsi1h_short":[60, 65, 70],    # 1h RSI overbought entry (short)
        "sl_mult":    [1.0, 1.5],
        "tp_mult":    [2.5, 3.0],
    },
}

# ══════════════════════════════════════════════════════════════════
# SECTION 2 — DATA
# ══════════════════════════════════════════════════════════════════

def fetch(interval: str, months: int = MONTHS) -> pd.DataFrame:
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=min(months * 30, 729))
    print(f"  Fetching BTC-USD {interval} ({months}mo)...")
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
# SECTION 3 — SHARED INDICATORS
# ══════════════════════════════════════════════════════════════════

def _atr(df, period=14):
    hl  = df["high"] - df["low"]
    hpc = (df["high"] - df["close"].shift()).abs()
    lpc = (df["low"]  - df["close"].shift()).abs()
    return pd.concat([hl,hpc,lpc],axis=1).max(axis=1).ewm(alpha=1/period,adjust=False).mean()

def _rsi(close, period=14):
    d = close.diff()
    g = d.clip(lower=0).ewm(span=period,adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(span=period,adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

# ══════════════════════════════════════════════════════════════════
# SECTION 4 — STRATEGY FEATURES
# ══════════════════════════════════════════════════════════════════

def features_A(df, ema_fast, ema_slow):
    d = df.copy()
    d["ema_fast"] = d["close"].ewm(span=ema_fast, adjust=False).mean()
    d["ema_slow"] = d["close"].ewm(span=ema_slow, adjust=False).mean()
    d["ema200"]   = d["close"].ewm(span=200,      adjust=False).mean()
    d["atr"]      = _atr(d)
    d["rsi"]      = _rsi(d["close"])
    d["slope_fast"]= d["ema_fast"].diff(3)
    return d.dropna()

def features_B(df, ce_period, ce_mult):
    d = df.copy()
    d["atr"]  = _atr(d, ce_period)
    d["rsi"]  = _rsi(d["close"])
    d["mom"]  = d["close"].pct_change(1)
    # Chandelier Exit levels
    highest_high = d["high"].rolling(ce_period).max()
    lowest_low   = d["low"].rolling(ce_period).min()
    d["ce_long"]  = highest_high - ce_mult * d["atr"]   # long trail stop
    d["ce_short"] = lowest_low  + ce_mult * d["atr"]    # short trail stop
    # Chandelier trend: build as numpy array to avoid SettingWithCopyWarning
    close_arr  = d["close"].values
    ce_long_arr= d["ce_long"].values
    ce_short_arr=d["ce_short"].values
    trend_arr  = np.zeros(len(d), dtype=int)
    trend = 0
    for i in range(1, len(d)):
        if   close_arr[i] > ce_long_arr[i]:  trend =  1
        elif close_arr[i] < ce_short_arr[i]: trend = -1
        trend_arr[i] = trend
    d = d.copy()
    d["ce_trend"] = trend_arr
    d["ce_flip"]  = d["ce_trend"].diff().fillna(0)
    return d.dropna()

def features_C_4h(df) -> pd.DataFrame:
    d = df.copy()
    d["rsi4h"] = _rsi(d["close"])
    d["atr4h"] = _atr(d)
    return d.dropna()

def features_C_1h(df, df4h) -> pd.DataFrame:
    d = df.copy()
    d["rsi1h"] = _rsi(d["close"])
    d["atr1h"] = _atr(d)
    d["ema200"]= d["close"].ewm(span=200, adjust=False).mean()
    # Align 4h RSI → 1h
    d["rsi4h"] = df4h["rsi4h"].reindex(d.index, method="ffill")
    d["atr4h"] = df4h["atr4h"].reindex(d.index, method="ffill")
    return d.dropna()

# ══════════════════════════════════════════════════════════════════
# SECTION 5 — SIGNAL GENERATORS
# ══════════════════════════════════════════════════════════════════

def signals_A(df, rsi_entry, sl_mult, tp_mult):
    """
    Long:  ema_fast > ema_slow > ema200 (uptrend)
           AND close pulled back to within 0.5% of ema_fast
           AND RSI bounced from rsi_entry (was below, now above)
           AND ema_fast slope positive
    Short: mirror
    """
    sigs = []
    for i in range(2, len(df)):
        p    = df.iloc[i-1]
        prev = df.iloc[i-2]
        c    = float(p["close"])
        ef   = float(p["ema_fast"]); es = float(p["ema_slow"]); e200 = float(p["ema200"])
        rsi  = float(p["rsi"]); prev_rsi = float(prev["rsi"])
        atr  = float(p["atr"]); slp = float(p["slope_fast"])

        pullback_long  = abs(c - ef) / ef < 0.008   # within 0.8% of fast EMA
        pullback_short = abs(c - ef) / ef < 0.008

        long_sig  = (ef > es > e200 and slp > 0
                     and prev_rsi < rsi_entry and rsi >= rsi_entry
                     and pullback_long)
        short_sig = (ef < es < e200 and slp < 0
                     and prev_rsi > (100 - rsi_entry) and rsi <= (100 - rsi_entry)
                     and pullback_short)

        if long_sig:  sigs.append((i,  1, atr, sl_mult, tp_mult))
        if short_sig: sigs.append((i, -1, atr, sl_mult, tp_mult))
    return sigs

def signals_B(df, mom_bars, mom_thresh, sl_mult, tp_mult):
    """
    Long:  Chandelier trend flips to bull (ce_flip > 0)
           AND momentum (mom_bars pct change) > mom_thresh
           AND RSI not overbought
    Short: mirror
    """
    sigs = []
    for i in range(mom_bars + 1, len(df)):
        p   = df.iloc[i-1]
        atr = float(p["atr"])
        flip= float(p["ce_flip"])
        rsi = float(p["rsi"])
        mom = float(df["close"].iloc[i-1]) / float(df["close"].iloc[i-1-mom_bars]) - 1

        long_sig  = flip > 0 and mom >  mom_thresh and rsi < 70
        short_sig = flip < 0 and mom < -mom_thresh and rsi > 30

        if long_sig:  sigs.append((i,  1, atr, sl_mult, tp_mult))
        if short_sig: sigs.append((i, -1, atr, sl_mult, tp_mult))
    return sigs

def signals_C(df, rsi4h_bull, rsi4h_bear, rsi1h_long, rsi1h_short, sl_mult, tp_mult):
    """
    Long:  4h RSI > rsi4h_bull (bull regime)
           AND 1h RSI dips below rsi1h_long then bounces back above
    Short: 4h RSI < rsi4h_bear (bear regime)
           AND 1h RSI spikes above rsi1h_short then falls back below
    """
    sigs = []
    for i in range(2, len(df)):
        p    = df.iloc[i-1]
        prev = df.iloc[i-2]
        r4   = float(p["rsi4h"]) if not pd.isna(p["rsi4h"]) else 50.0
        r1   = float(p["rsi1h"]); prev_r1 = float(prev["rsi1h"])
        atr  = float(p["atr1h"]); e200 = float(p["ema200"])
        c    = float(p["close"])

        long_sig  = (r4 > rsi4h_bull
                     and prev_r1 < rsi1h_long and r1 >= rsi1h_long
                     and c > e200)
        short_sig = (r4 < rsi4h_bear
                     and prev_r1 > rsi1h_short and r1 <= rsi1h_short
                     and c < e200)

        if long_sig:  sigs.append((i,  1, atr, sl_mult, tp_mult))
        if short_sig: sigs.append((i, -1, atr, sl_mult, tp_mult))
    return sigs

# ══════════════════════════════════════════════════════════════════
# SECTION 6 — BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════

def run_sim(df, signals):
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
            entry_px = fill; direction = d; in_trade = True

    if len(trades) < 10: return None
    t   = pd.DataFrame(trades)
    ret = (balance - CAPITAL) / CAPITAL * 100
    cum = CAPITAL + t["pnl"].cumsum()
    dd  = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    r   = t["pnl"] / CAPITAL
    bars_per_year = 8760 if "1h" in str(df.index.freq or "") else 2190
    sh  = (r.mean() / r.std() * math.sqrt(bars_per_year)) if r.std() > 0 else 0
    pf_w = t[t["pnl"]>0]["pnl"].sum(); pf_l = t[t["pnl"]<0]["pnl"].abs().sum()
    pf   = pf_w / pf_l if pf_l > 0 else 999
    score= sh * (1 - abs(dd)/100)
    return {"n": len(t), "ret": round(ret,2), "dd": round(dd,2),
            "wr": round(t["win"].mean()*100,1),
            "sharpe": round(sh,2), "pf": round(pf,2), "score": round(score,3)}

# ══════════════════════════════════════════════════════════════════
# SECTION 7 — WORKERS
# ══════════════════════════════════════════════════════════════════

def _worker_A(args):
    df, p = args
    feat = features_A(df, p["ema_fast"], p["ema_slow"])
    sigs = signals_A(feat, p["rsi_entry"], p["sl_mult"], p["tp_mult"])
    res  = run_sim(feat, sigs)
    return {**p, **res} if res else None

def _worker_B(args):
    df, p = args
    feat = features_B(df, p["ce_period"], p["ce_mult"])
    sigs = signals_B(feat, p["mom_bars"], p["mom_thresh"], p["sl_mult"], p["tp_mult"])
    res  = run_sim(feat, sigs)
    return {**p, **res} if res else None

def _worker_C(args):
    df1h, df4h, p = args
    f4   = features_C_4h(df4h)
    feat = features_C_1h(df1h, f4)
    sigs = signals_C(feat, p["rsi4h_bull"], p["rsi4h_bear"],
                     p["rsi1h_long"], p["rsi1h_short"], p["sl_mult"], p["tp_mult"])
    res  = run_sim(feat, sigs)
    return {**p, **res} if res else None

def grid_search(worker, data, grid, label):
    keys   = list(grid.keys())
    combos = [dict(zip(keys, v)) for v in product(*grid.values())]
    cores  = cpu_count()
    print(f"  {label}: {len(combos)} combos on {cores} cores...")
    if isinstance(data, tuple):
        tasks = [(data[0], data[1], p) for p in combos]
    else:
        tasks = [(data, p) for p in combos]
    with Pool(processes=cores) as pool:
        raw = pool.map(worker, tasks)
    results = [r for r in raw if r is not None]
    print(f"  → {len(results)} valid results")
    return pd.DataFrame(results) if results else pd.DataFrame()

# ══════════════════════════════════════════════════════════════════
# SECTION 8 — RESULTS
# ══════════════════════════════════════════════════════════════════

def print_top(label, name, df_res):
    if df_res.empty:
        print(f"\n  Strategy {label} — No valid results."); return
    top = df_res.sort_values("score", ascending=False).head(5)
    param_cols = [c for c in top.columns
                  if c not in ["n","ret","dd","wr","sharpe","pf","score"]]
    print(f"\n{'='*100}")
    print(f"  Strategy {label} — {name}  |  Top 5 by Score")
    print(f"{'='*100}")
    hdr = "  " + " ".join(f"{c[:9]:<10}" for c in param_cols)
    hdr+= f"  {'Ret%':<10}{'DD%':<8}{'WR%':<7}{'Sharpe':<9}{'PF':<7}{'Score':<9}N"
    print(hdr); print("  " + "-"*95)
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        pv = " ".join(f"{str(r[c])[:9]:<10}" for c in param_cols)
        print(f"  {pv}  {r.ret:>+8.1f}%  {r.dd:>6.1f}%  {r.wr:>5.1f}%  "
              f"{r.sharpe:>7.2f}  {r.pf:>4.2f}  {r.score:>7.3f}  {int(r.n)}")
    b = top.iloc[0]
    print(f"\n  BEST: Ret={b.ret:+.1f}% | DD={b.dd:.1f}% | WR={b.wr:.1f}% | "
          f"Sharpe={b.sharpe:.2f} | PF={b.pf:.2f} | Score={b.score:.3f} | N={int(b.n)}")

def print_winner(results):
    print(f"\n{'#'*100}")
    print("  WINNER COMPARISON")
    print(f"{'#'*100}")
    print(f"  {'Strategy':<28}{'Ret%':<11}{'DD%':<9}{'WR%':<7}{'Sharpe':<9}{'PF':<7}{'Score':<9}N")
    print(f"  {'-'*90}")
    winner = None
    for label, name, df_res in results:
        if df_res.empty: continue
        b   = df_res.sort_values("score", ascending=False).iloc[0]
        tag = f"{label} ({name})"
        print(f"  {tag:<28}{b.ret:>+9.1f}%  {b.dd:>6.1f}%  {b.wr:>5.1f}%  "
              f"{b.sharpe:>7.2f}  {b.pf:>4.2f}  {b.score:>7.3f}  {int(b.n)}")
        if winner is None or b.score > winner[1]:
            winner = (f"{label} — {name}", b.score, b)
    if winner:
        print(f"\n  🏆 WINNER: {winner[0]}  |  Score={winner[1]:.3f}")
    print(f"{'#'*100}")

# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*65)
    print("  BTC Round 2 — 3 Strategy Grid Search")
    print(f"  Capital=${CAPITAL} | {LEVERAGE}x | Margin={MARGIN_PCT*100:.0f}% | {MONTHS}mo")
    print("="*65)

    print("\nFetching data...")
    df_4h = fetch("4h")
    df_1h = fetch("1h")

    print("\nRunning grid searches...")
    res_A = grid_search(_worker_A, df_4h, GRIDS["A"], "A: EMA Cross + RSI Pullback (4h)")
    res_B = grid_search(_worker_B, df_4h, GRIDS["B"], "B: Chandelier Exit + Momentum (4h)")
    res_C = grid_search(_worker_C, (df_1h, df_4h), GRIDS["C"], "C: Dual RSI 4h/1h")

    print_top("A", "EMA Cross + RSI Pullback 4h", res_A)
    print_top("B", "Chandelier Exit + Momentum 4h", res_B)
    print_top("C", "Dual RSI 4h/1h", res_C)
    print_winner([("A","EMA Pullback 4h", res_A),
                  ("B","Chandelier 4h",   res_B),
                  ("C","Dual RSI",        res_C)])

    for lbl, df_r in [("A_ema_pullback",res_A),
                      ("B_chandelier",  res_B),
                      ("C_dual_rsi",    res_C)]:
        if not df_r.empty:
            df_r.sort_values("score",ascending=False).to_csv(
                f"btc_{lbl}_results.csv", index=False)
    print("\n  CSVs saved. Done.")
