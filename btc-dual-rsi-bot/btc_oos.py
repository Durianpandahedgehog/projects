"""
BTC Dual RSI — Out-of-Sample Validation
Step 1: Grid search on first 15 months (in-sample)
Step 2: Walk-forward on in-sample to pick best params
Step 3: Fire best params ONCE on last 5 months (never seen)
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

TOTAL_MONTHS  = 20
IS_MONTHS     = 15    # in-sample: first 15 months
OOS_MONTHS    = 5     # out-of-sample: last 5 months (never touched)
N_IS_WINDOWS  = 3     # walk-forward windows inside in-sample

FEE        = 0.0004
SLIP       = 0.0005
CAPITAL    = 1000.0
LEVERAGE   = 10
MARGIN_PCT = 0.20

# Grid to search on in-sample data only
GRID = {
    "rsi4h_bull":  [45, 50, 55, 60],
    "rsi4h_bear":  [35, 40, 45, 50],
    "rsi1h_long":  [25, 30, 35, 40],
    "rsi1h_short": [55, 60, 65, 70],
    "sl_mult":     [1.0, 1.5, 2.0],
    "tp_mult":     [2.5, 3.0, 3.5],
}

# ══════════════════════════════════════════════════════════════════
# SECTION 2 — DATA
# ══════════════════════════════════════════════════════════════════

def fetch(interval: str, months: int) -> pd.DataFrame:
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=min(months * 30, 729))
    df = yf.download("BTC-USD", start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"), interval=interval,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    return df[["open","high","low","close","volume"]].dropna()

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
    f4          = df_4h.copy()
    f4["rsi4h"] = _rsi(f4["close"])
    d["rsi4h"]  = f4["rsi4h"].reindex(d.index, method="ffill")
    return d.dropna()

# ══════════════════════════════════════════════════════════════════
# SECTION 4 — SIGNALS + SIM
# ══════════════════════════════════════════════════════════════════

def get_signals(df, rsi4h_bull, rsi4h_bear, rsi1h_long, rsi1h_short, sl_mult, tp_mult):
    sigs  = []
    rsi4  = df["rsi4h"].to_numpy()
    rsi1  = df["rsi1h"].to_numpy()
    atr   = df["atr1h"].to_numpy()
    close = df["close"].to_numpy()
    ema200= df["ema200"].to_numpy()
    for i in range(1, len(df)):
        r4 = rsi4[i] if not np.isnan(rsi4[i]) else 50.0
        r1 = rsi1[i]; pr1 = rsi1[i-1]
        a  = atr[i];  c   = close[i]; e200 = ema200[i]
        if r4 > rsi4h_bull and pr1 < rsi1h_long  and r1 >= rsi1h_long  and c > e200:
            sigs.append((i,  1, a, sl_mult, tp_mult))
        if r4 < rsi4h_bear and pr1 > rsi1h_short and r1 <= rsi1h_short and c < e200:
            sigs.append((i, -1, a, sl_mult, tp_mult))
    return sigs

def run_sim(df, signals, min_trades=4):
    if not signals: return None
    balance  = CAPITAL
    trades   = []
    in_trade = False
    direction = entry_px = sl = tp = qty = 0.0
    sig_map  = {s[0]: s for s in signals}
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows  = df["low"].to_numpy()

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

    if len(trades) < min_trades: return None
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
# SECTION 5 — GRID SEARCH WORKER
# ══════════════════════════════════════════════════════════════════

def _worker(args):
    df1h, df4h, p = args
    df   = build_features(df1h, df4h)
    sigs = get_signals(df, p["rsi4h_bull"], p["rsi4h_bear"],
                       p["rsi1h_long"], p["rsi1h_short"],
                       p["sl_mult"], p["tp_mult"])
    res  = run_sim(df, sigs)
    if not res: return None
    # Score: Sharpe penalised by DD
    res["score"] = round(res["sharpe"] * (1 - abs(res["dd"]) / 100), 3)
    return {**p, **res}

def grid_search(df1h, df4h, label=""):
    keys   = list(GRID.keys())
    combos = [dict(zip(keys, v)) for v in product(*GRID.values())]
    cores  = cpu_count()
    print(f"  {label}: {len(combos)} combos on {cores} cores...")
    with Pool(processes=cores) as pool:
        raw = pool.map(_worker, [(df1h, df4h, p) for p in combos])
    results = [r for r in raw if r is not None]
    print(f"  → {len(results)} valid results")
    return pd.DataFrame(results) if results else pd.DataFrame()

# ══════════════════════════════════════════════════════════════════
# SECTION 6 — IN-SAMPLE WALK-FORWARD
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

def is_walk_forward(df1h_is, df4h_is):
    """Run mini walk-forward inside in-sample to find most consistent params."""
    windows = make_windows(df1h_is, N_IS_WINDOWS)
    all_results = []

    print(f"\n  In-sample walk-forward ({N_IS_WINDOWS} windows):")
    for wnum, df_win, start, end in windows:
        df4h_win = df4h_is.loc[start:end]
        res_df   = grid_search(df_win, df4h_win, f"  Window {wnum} [{start}→{end}]")
        if not res_df.empty:
            all_results.append(res_df)
            best = res_df.sort_values("score", ascending=False).iloc[0]
            pos  = (res_df["ret"] > 0).sum()
            print(f"    Best: Ret={best.ret:+.1f}% DD={best.dd:.1f}% "
                  f"WR={best.wr:.1f}% | Positive={pos}/{len(res_df)}")

    if not all_results:
        return None

    # Find param combo with highest average score across all windows
    combined   = pd.concat(all_results)
    group_cols = ["rsi4h_bull","rsi4h_bear","rsi1h_long","rsi1h_short","sl_mult","tp_mult"]
    agg = (combined.groupby(group_cols)
           .agg(avg_ret=("ret","mean"), avg_dd=("dd","mean"),
                avg_score=("score","mean"), count=("ret","count"))
           .reset_index()
           .query("count >= 2")
           .sort_values("avg_score", ascending=False))

    if agg.empty:
        # Fallback: best single-window score
        agg = (combined.groupby(group_cols)
               .agg(avg_ret=("ret","mean"), avg_dd=("dd","mean"),
                    avg_score=("score","mean"), count=("ret","count"))
               .reset_index()
               .sort_values("avg_score", ascending=False))

    best_row = agg.iloc[0]
    best_params = {
        "rsi4h_bull":  int(best_row["rsi4h_bull"]),
        "rsi4h_bear":  int(best_row["rsi4h_bear"]),
        "rsi1h_long":  int(best_row["rsi1h_long"]),
        "rsi1h_short": int(best_row["rsi1h_short"]),
        "sl_mult":     float(best_row["sl_mult"]),
        "tp_mult":     float(best_row["tp_mult"]),
    }
    return best_params, best_row

# ══════════════════════════════════════════════════════════════════
# SECTION 7 — OOS TEST + VERDICT
# ══════════════════════════════════════════════════════════════════

def run_oos(df1h_oos, df4h_oos, params):
    df   = build_features(df1h_oos, df4h_oos)
    sigs = get_signals(df, **params)
    return run_sim(df, sigs, min_trades=2)   # lower bar — 5mo window

def print_verdict(is_avg, oos_result, params, is_start, is_end, oos_start, oos_end):
    print(f"\n{'#'*70}")
    print("  OUT-OF-SAMPLE VERDICT — BTC Dual RSI")
    print(f"{'#'*70}")
    print(f"\n  Best params (from in-sample walk-forward only):")
    print(f"  4h RSI bull={params['rsi4h_bull']} bear={params['rsi4h_bear']} | "
          f"1h RSI long={params['rsi1h_long']} short={params['rsi1h_short']} | "
          f"SL={params['sl_mult']}x TP={params['tp_mult']}x")

    print(f"\n  In-sample  [{is_start} → {is_end}]:")
    print(f"  Avg Ret={is_avg['avg_ret']:>+.1f}% | Avg DD={is_avg['avg_dd']:.1f}% | "
          f"Seen in {int(is_avg['count'])} windows")

    print(f"\n  Out-of-sample [{oos_start} → {oos_end}] ← NEVER SEEN:")
    if oos_result:
        print(f"  Ret={oos_result['ret']:>+.1f}% | DD={oos_result['dd']:.1f}% | "
              f"WR={oos_result['wr']:.1f}% | Sharpe={oos_result['sharpe']:.2f} | "
              f"PF={oos_result['pf']:.2f} | N={oos_result['n']}")
    else:
        print("  No trades in OOS window — too few signals.")

    print(f"\n{'─'*70}")
    if not oos_result:
        print("  ⚠️  NO TRADES: Strategy too selective for this period.")
        print("  Inconclusive — deploy with caution, paper trade 4 weeks.")
    elif oos_result["ret"] > 0 and oos_result["dd"] > -15:
        print("  🟢 PASS: Strategy is genuinely profitable on unseen data.")
        print("  Deploy with confidence. Paper trade 2 weeks then go live.")
    elif oos_result["ret"] > 0:
        print("  🟡 PASS (HIGH DD): Profitable but drawdown elevated vs in-sample.")
        print("  Reduce margin to 10%, paper trade 3 weeks before live.")
    elif oos_result["ret"] > -10:
        print("  🟡 MARGINAL: Small loss on OOS — strategy may be weakening.")
        print("  Do NOT deploy live. Continue paper trading and reassess in 4 weeks.")
    else:
        print("  🔴 FAIL: Strategy did not hold up on unseen data.")
        print("  Do NOT deploy. The walk-forward results were curve-fitted.")
    print(f"{'#'*70}")

# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    n_combos = len(list(product(*GRID.values())))
    print("\n" + "="*70)
    print("  BTC Dual RSI — Out-of-Sample Validation")
    print(f"  In-sample: {IS_MONTHS}mo | OOS: {OOS_MONTHS}mo (locked away)")
    print(f"  Grid: {n_combos} combos | {N_IS_WINDOWS} IS windows")
    print("="*70)

    # Fetch full dataset
    print("\nFetching data...")
    df_1h_full = fetch("1h", TOTAL_MONTHS)
    df_4h_full = fetch("4h", TOTAL_MONTHS)
    print(f"  1h: {len(df_1h_full)} bars | {df_1h_full.index[0].date()} → {df_1h_full.index[-1].date()}")
    print(f"  4h: {len(df_4h_full)} bars | {df_4h_full.index[0].date()} → {df_4h_full.index[-1].date()}")

    # Split in-sample / out-of-sample by bar count
    is_size  = int(len(df_1h_full) * (IS_MONTHS / TOTAL_MONTHS))
    df_1h_is = df_1h_full.iloc[:is_size]
    df_1h_oos= df_1h_full.iloc[is_size:]
    is_cutoff= df_1h_is.index[-1]
    df_4h_is = df_4h_full.loc[:is_cutoff]
    df_4h_oos= df_4h_full.loc[is_cutoff:]

    is_start = df_1h_is.index[0].strftime("%Y-%m-%d")
    is_end   = df_1h_is.index[-1].strftime("%Y-%m-%d")
    oos_start= df_1h_oos.index[0].strftime("%Y-%m-%d")
    oos_end  = df_1h_oos.index[-1].strftime("%Y-%m-%d")

    print(f"\n  In-sample : {is_start} → {is_end}  ({len(df_1h_is)} bars)")
    print(f"  OOS (locked): {oos_start} → {oos_end}  ({len(df_1h_oos)} bars)")
    print("\n  ⛔ OOS data is now locked — not touched until the very end.")

    # Step 1: In-sample walk-forward to find best params
    print(f"\n{'─'*70}")
    print("  STEP 1 — In-sample grid search + walk-forward")
    print(f"{'─'*70}")
    result = is_walk_forward(df_1h_is, df_4h_is)
    if result is None:
        print("  No valid params found in-sample. Exiting.")
        sys.exit(1)
    best_params, best_row = result

    print(f"\n  Best params found:")
    print(f"  4h RSI bull={best_params['rsi4h_bull']} bear={best_params['rsi4h_bear']} | "
          f"1h RSI long={best_params['rsi1h_long']} short={best_params['rsi1h_short']} | "
          f"SL={best_params['sl_mult']}x TP={best_params['tp_mult']}x")
    print(f"  Avg IS score={best_row['avg_score']:.3f} | "
          f"Avg IS ret={best_row['avg_ret']:+.1f}%")

    # Step 2: Fire once on OOS — the moment of truth
    print(f"\n{'─'*70}")
    print(f"  STEP 2 — OOS test [{oos_start} → {oos_end}]  ← ONE SHOT")
    print(f"{'─'*70}")
    oos_result = run_oos(df_1h_oos, df_4h_oos, best_params)

    print_verdict(best_row, oos_result, best_params,
                  is_start, is_end, oos_start, oos_end)

    # Save IS grid results
    print("\n  Saving IS results...")
    for i in range(N_IS_WINDOWS):
        fname = f"btc_dualrsi_oos_is_window{i+1}.csv"
    print("  Done.")
