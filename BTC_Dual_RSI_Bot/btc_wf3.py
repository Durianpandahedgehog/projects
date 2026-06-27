"""
BTC Dual RSI — Walk-Forward Validation
4h RSI regime + 1h RSI entry | 4 windows + robustness grid
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
N_WINDOWS  = 4

# Best params from grid search
BEST = dict(rsi4h_bull=55, rsi4h_bear=40,
            rsi1h_long=30, rsi1h_short=65,
            sl_mult=1.5,   tp_mult=3.0)

# Robustness grid — params close to best
ROBUST_GRID = {
    "rsi4h_bull":  [50, 55, 60],
    "rsi4h_bear":  [35, 40, 45],
    "rsi1h_long":  [25, 30, 35],
    "rsi1h_short": [60, 65, 70],
    "sl_mult":     [1.0, 1.5, 2.0],
    "tp_mult":     [2.5, 3.0, 3.5],
}

# ══════════════════════════════════════════════════════════════════
# SECTION 2 — DATA
# ══════════════════════════════════════════════════════════════════

def fetch(interval: str, months: int = MONTHS) -> pd.DataFrame:
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=min(months * 30, 729))
    df = yf.download("BTC-USD", start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"), interval=interval,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df = df[["open","high","low","close","volume"]].dropna()
    return df

# ══════════════════════════════════════════════════════════════════
# SECTION 3 — FEATURES
# ══════════════════════════════════════════════════════════════════

def _rsi(close, period=14):
    d = close.diff()
    g = d.clip(lower=0).ewm(span=period, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(span=period, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def _atr(df, period=14):
    hl  = df["high"] - df["low"]
    hpc = (df["high"] - df["close"].shift()).abs()
    lpc = (df["low"]  - df["close"].shift()).abs()
    return pd.concat([hl,hpc,lpc],axis=1).max(axis=1).ewm(alpha=1/period,adjust=False).mean()

def build_features(df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> pd.DataFrame:
    d = df_1h.copy()
    d["rsi1h"]  = _rsi(d["close"])
    d["atr1h"]  = _atr(d)
    d["ema200"] = d["close"].ewm(span=200, adjust=False).mean()

    # 4h RSI — compute then forward-fill onto 1h index
    f4          = df_4h.copy()
    f4["rsi4h"] = _rsi(f4["close"])
    d["rsi4h"]  = f4["rsi4h"].reindex(d.index, method="ffill")

    return d.dropna()

# ══════════════════════════════════════════════════════════════════
# SECTION 4 — SIGNALS
# ══════════════════════════════════════════════════════════════════

def get_signals(df, rsi4h_bull, rsi4h_bear,
                rsi1h_long, rsi1h_short, sl_mult, tp_mult):
    sigs = []
    rsi4  = df["rsi4h"].to_numpy()
    rsi1  = df["rsi1h"].to_numpy()
    atr   = df["atr1h"].to_numpy()
    close = df["close"].to_numpy()
    ema200= df["ema200"].to_numpy()

    for i in range(1, len(df)):
        r4       = rsi4[i]  if not np.isnan(rsi4[i])  else 50.0
        r1       = rsi1[i];  prev_r1 = rsi1[i-1]
        a        = atr[i];   c = close[i]; e200 = ema200[i]

        long_sig  = (r4 > rsi4h_bull
                     and prev_r1 < rsi1h_long and r1 >= rsi1h_long
                     and c > e200)
        short_sig = (r4 < rsi4h_bear
                     and prev_r1 > rsi1h_short and r1 <= rsi1h_short
                     and c < e200)

        if long_sig:  sigs.append((i,  1, a, sl_mult, tp_mult))
        if short_sig: sigs.append((i, -1, a, sl_mult, tp_mult))
    return sigs

# ══════════════════════════════════════════════════════════════════
# SECTION 5 — BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════

def run_sim(df, signals):
    if not signals: return None
    balance  = CAPITAL
    trades   = []
    in_trade = False
    direction = entry_px = sl = tp = qty = 0.0
    sig_map  = {s[0]: s for s in signals}
    opens    = df["open"].to_numpy()
    highs    = df["high"].to_numpy()
    lows     = df["low"].to_numpy()

    for i in range(len(df)):
        if in_trade:
            hi, lo = highs[i], lows[i]
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
            fill     = opens[i] * (1 + SLIP * d)
            margin   = balance * MARGIN_PCT
            qty      = (margin * LEVERAGE * 0.95) / fill
            sl       = fill - atr * sl_m * d
            tp       = fill + atr * tp_m * d
            entry_px = fill; direction = d; in_trade = True

    if len(trades) < 4: return None
    t   = pd.DataFrame(trades)
    ret = (balance - CAPITAL) / CAPITAL * 100
    cum = CAPITAL + t["pnl"].cumsum()
    dd  = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    r   = t["pnl"] / CAPITAL
    sh  = (r.mean() / r.std() * math.sqrt(8760)) if r.std() > 0 else 0
    pfw = t[t["pnl"]>0]["pnl"].sum(); pfl = t[t["pnl"]<0]["pnl"].abs().sum()
    pf  = pfw / pfl if pfl > 0 else 999
    return {"n": len(t), "ret": round(ret,2), "dd": round(dd,2),
            "wr": round(t["win"].mean()*100,1),
            "sharpe": round(sh,2), "pf": round(pf,2)}

# ══════════════════════════════════════════════════════════════════
# SECTION 6 — WORKER + GRID
# ══════════════════════════════════════════════════════════════════

def _worker(args):
    df1h_win, df4h_win, p = args
    df   = build_features(df1h_win, df4h_win)
    sigs = get_signals(df, p["rsi4h_bull"], p["rsi4h_bear"],
                       p["rsi1h_long"], p["rsi1h_short"],
                       p["sl_mult"], p["tp_mult"])
    res  = run_sim(df, sigs)
    return {**p, **res} if res else None

def run_grid(df1h_win, df4h_win):
    keys   = list(ROBUST_GRID.keys())
    combos = [dict(zip(keys, v)) for v in product(*ROBUST_GRID.values())]
    with Pool(processes=cpu_count()) as pool:
        raw = pool.map(_worker, [(df1h_win, df4h_win, p) for p in combos])
    results = [r for r in raw if r is not None]
    return pd.DataFrame(results) if results else pd.DataFrame()

# ══════════════════════════════════════════════════════════════════
# SECTION 7 — WINDOWS
# ══════════════════════════════════════════════════════════════════

def make_windows(df, n):
    sz = len(df) // n
    out = []
    for i in range(n):
        s = i * sz
        e = (i+1)*sz if i < n-1 else len(df)
        c = df.iloc[s:e]
        out.append((i+1, c,
                    c.index[0].strftime("%Y-%m-%d"),
                    c.index[-1].strftime("%Y-%m-%d")))
    return out

def slice_4h(df4h, start, end):
    return df4h.loc[start:end]

# ══════════════════════════════════════════════════════════════════
# SECTION 8 — PRINT + VERDICT
# ══════════════════════════════════════════════════════════════════

def print_window(wnum, start, end, res_best, res_grid):
    print(f"\n  Window {wnum}  [{start} → {end}]")
    print(f"  {'─'*65}")
    if res_best:
        print(f"  Best Params → Ret={res_best['ret']:>+7.1f}% | DD={res_best['dd']:>6.1f}% | "
              f"WR={res_best['wr']:>5.1f}% | Sharpe={res_best['sharpe']:>6.2f} | "
              f"PF={res_best['pf']:>4.2f} | N={res_best['n']}")
    else:
        print("  Best Params → No trades in this window")

    verdict = "⚠️  NO DATA"
    if not res_grid.empty:
        pos     = (res_grid["ret"] > 0).sum()
        total   = len(res_grid)
        pct_pos = pos / total * 100
        med_ret = res_grid["ret"].median()
        med_dd  = res_grid["dd"].median()
        print(f"  Grid ({total} valid) → Positive={pos}/{total} ({pct_pos:.0f}%) | "
              f"Median Ret={med_ret:>+6.1f}% | Median DD={med_dd:>5.1f}%")

        # Show best and worst in grid
        best_r = res_grid.sort_values("ret", ascending=False).iloc[0]
        worst_r= res_grid.sort_values("ret", ascending=True).iloc[0]
        print(f"  Grid best:  Ret={best_r['ret']:>+6.1f}% "
              f"(4h:{int(best_r['rsi4h_bull'])}/{int(best_r['rsi4h_bear'])} "
              f"1h:{int(best_r['rsi1h_long'])}/{int(best_r['rsi1h_short'])} "
              f"SL={best_r['sl_mult']} TP={best_r['tp_mult']})")
        print(f"  Grid worst: Ret={worst_r['ret']:>+6.1f}%")

        verdict = ("✅ ROBUST"  if pct_pos >= 60 and med_ret > 0  else
                   "⚠️  FRAGILE" if pct_pos >= 40 or med_ret > -5  else
                   "❌ FAILS")
    print(f"  Verdict: {verdict}")
    return verdict, res_best

def final_verdict(window_results, all_grids):
    print(f"\n{'#'*70}")
    print("  DUAL RSI WALK-FORWARD SUMMARY")
    print(f"{'#'*70}")

    verdicts  = [v for v, _ in window_results]
    robust_n  = sum(1 for v in verdicts if "ROBUST"  in v)
    fragile_n = sum(1 for v in verdicts if "FRAGILE" in v)
    fail_n    = sum(1 for v in verdicts if "FAILS"   in v)
    nodata_n  = sum(1 for v in verdicts if "NO DATA" in v)

    valid = [r for _, r in window_results if r is not None]
    if valid:
        avg_ret = sum(r["ret"] for r in valid) / len(valid)
        avg_dd  = sum(r["dd"]  for r in valid) / len(valid)
        avg_wr  = sum(r["wr"]  for r in valid) / len(valid)
        print(f"\n  Avg (best params across windows):")
        print(f"  Return={avg_ret:>+.1f}% | DD={avg_dd:.1f}% | WR={avg_wr:.1f}%")

    # Best param combo averaged across all windows
    if all_grids:
        combined = pd.concat(all_grids)
        group_cols = ["rsi4h_bull","rsi4h_bear","rsi1h_long","rsi1h_short","sl_mult","tp_mult"]
        avg = (combined.groupby(group_cols)["ret"]
               .agg(["mean","count"]).reset_index()
               .query("count >= 2")
               .sort_values("mean", ascending=False))
        if not avg.empty:
            b = avg.iloc[0]
            print(f"\n  Most consistent param combo (seen in ≥2 windows):")
            print(f"  4h RSI bull={int(b['rsi4h_bull'])} bear={int(b['rsi4h_bear'])} | "
                  f"1h RSI long={int(b['rsi1h_long'])} short={int(b['rsi1h_short'])} | "
                  f"SL={b['sl_mult']} TP={b['tp_mult']} → Avg Ret={b['mean']:+.1f}%")

    print(f"\n  Windows: ✅ Robust={robust_n}  ⚠️ Fragile={fragile_n}  "
          f"❌ Fails={fail_n}  ❓ No data={nodata_n}")

    if robust_n >= 3:
        print("\n  🟢 DEPLOY: Consistent across time. Build the bot.")
    elif robust_n >= 2 and fail_n == 0:
        print("\n  🟡 MOSTLY GOOD: Reduce size to 10% margin, paper first.")
    elif robust_n >= 2 and fail_n == 1:
        print("\n  🟡 CAUTION: Good in trending markets, add regime filter before deploying.")
    else:
        print("\n  🔴 DO NOT DEPLOY: Does not hold up across all market conditions.")
    print(f"{'#'*70}")

# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    n_combos = len(list(product(*ROBUST_GRID.values())))
    print("\n" + "="*70)
    print("  BTC Dual RSI — Walk-Forward Validation")
    print(f"  {N_WINDOWS} windows | {n_combos} robustness combos each")
    print(f"  Best: 4h({BEST['rsi4h_bull']}/{BEST['rsi4h_bear']}) "
          f"1h({BEST['rsi1h_long']}/{BEST['rsi1h_short']}) "
          f"SL={BEST['sl_mult']} TP={BEST['tp_mult']}")
    print("="*70)

    print("\nFetching data...")
    df_1h = fetch("1h")
    df_4h = fetch("4h")
    print(f"  1h: {len(df_1h)} bars | 4h: {len(df_4h)} bars")

    windows = make_windows(df_1h, N_WINDOWS)
    print(f"\nRunning {N_WINDOWS} windows × {n_combos} combos each...")

    window_results = []
    all_grids      = []

    for wnum, df_win, start, end in windows:
        df4h_win = slice_4h(df_4h, start, end)

        # Best params test
        df_feat   = build_features(df_win, df4h_win)
        sigs_best = get_signals(df_feat, **BEST)
        res_best  = run_sim(df_feat, sigs_best)

        # Robustness grid
        res_grid = run_grid(df_win, df4h_win)
        if not res_grid.empty:
            all_grids.append(res_grid)

        v, r = print_window(wnum, start, end, res_best, res_grid)
        window_results.append((v, r))

    final_verdict(window_results, all_grids)

    # Save full grid results per window
    for i, (_, df_win, start, end) in enumerate(windows):
        if i < len(all_grids) and not all_grids[i].empty:
            fname = f"btc_dualrsi_wf_window{i+1}.csv"
            all_grids[i].sort_values("ret", ascending=False).to_csv(fname, index=False)
    print("\n  CSVs saved. Done.")
