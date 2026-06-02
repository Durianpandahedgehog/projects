"""
╔══════════════════════════════════════════════════════════════════╗
║         BTCUSDT FUTURES BOT — DUAL RSI STRATEGY                  ║
║                   v1.0  |  Built on Template v4.0                ║
║                                                                  ║
║  Strategy : 4h RSI Regime + 1h RSI Entry + EMA200 Filter         ║
║  Timeframe: 4h regime | 1h entry                                 ║
║  Backtest : +83.8% / 20mo | WR 72.2% | PF 4.73 | DD -3.1%       ║
║  WalkFwd  : 4/4 windows ✅ | Avg +94.5% | Avg DD -1.6%           ║
║  Sizing   : 20% margin per trade | $1000 capital | 10x leverage  ║
║                                                                  ║
║  Usage:  python btc_bot.py           → paper trade (safe)        ║
║          python btc_bot.py --live    → real money on Binance     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import argparse
import json
import logging
import math
import os
import sys
import time
import requests
from dataclasses import dataclass, field, asdict
from typing import Optional
from urllib.parse import urlparse
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
except ImportError:
    sys.exit("python-binance not installed.\nRun: pip install python-binance")

try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

# ══════════════════════════════════════════════════════════════════
# SECTION A — CONFIG
# ══════════════════════════════════════════════════════════════════

SYMBOL       = os.getenv("BOT_SYMBOL",      "BTCUSDT")
LEVERAGE     = int(os.getenv("BOT_LEVERAGE","10"))
MARGIN_TYPE  = os.getenv("BOT_MARGIN_TYPE", "ISOLATED")
MARGIN_PCT   = 0.20
WEBHOOK_URL  = os.getenv("BOT_WEBHOOK_URL", "")
MAX_NOTIONAL = 500000.0

# ── Dual RSI parameters (walk-forward optimised) ─────────────────
RSI4H_BULL  = 60       # 4h RSI above → bull regime  (long allowed)
RSI4H_BEAR  = 45       # 4h RSI below → bear regime  (short allowed)
RSI1H_LONG  = 25       # 1h RSI crosses UP through this → long entry
RSI1H_SHORT = 70       # 1h RSI crosses DOWN through this → short entry

# EMA200 regime confirmation (1h)
EMA200_FILTER = True   # long only above EMA200, short only below

# Trade management
SL_MULT     = 1.0      # ATR multiplier for stop loss
TP_MULT     = 3.5      # ATR multiplier for take profit
MAX_HOLD_H  = 48       # max hold 48h (swing trade)

# Risk guardrails
MAX_TRADES_DAY  = 3
DAILY_LOSS_PCT  = 0.06      # 6% daily loss limit
CONSEC_SL_LIMIT = 2
COOLDOWN_H      = 6
MAX_COOLDOWN_H  = 24
MIN_BALANCE     = 50.0
MIN_ATR_RATIO   = 0.6       # skip if market is flat

# Blocked hours UTC (low BTC liquidity)
BAD_HOURS = {2, 3, 4}

FEE  = 0.0004
SLIP = 0.0005

DATA_DIR   = Path(__file__).parent / "data"
LIVE_MODE  = "--live" in sys.argv
STATE_FILE = DATA_DIR / ("state_live.json" if LIVE_MODE else "state_paper.json")
LOG_FILE   = DATA_DIR / "btc_bot.log"

# ══════════════════════════════════════════════════════════════════
# SECTION B — ALERTS, LOGGING & STATE
# ══════════════════════════════════════════════════════════════════

def _is_valid_url(url: str) -> bool:
    try:
        r = urlparse(url)
        return all([r.scheme in ("http","https"), r.netloc])
    except Exception:
        return False

def push_alert(message: str, is_error: bool = False):
    if not _is_valid_url(WEBHOOK_URL): return
    color   = 16711680 if is_error else 3447003
    payload = {"username": f"BTC DualRSI Bot",
               "embeds": [{"description": message, "color": color}]}
    try:
        requests.post(WEBHOOK_URL, json=payload, timeout=3, verify=True)
    except Exception as e:
        log.warning("Webhook failed: %s", e)

DATA_DIR.mkdir(parents=True, exist_ok=True)
log = logging.getLogger("btcbot")
log.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
fh  = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=3)
fh.setFormatter(fmt)
log.addHandler(fh)
sh  = logging.StreamHandler(sys.stdout)
sh.setFormatter(fmt)
log.addHandler(sh)

@dataclass
class BotState:
    in_trade: bool = False
    direction: int = 0
    entry: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    qty: float = 0.0
    entry_time: Optional[str] = None
    entry_bar: Optional[str] = None
    day: Optional[str] = None
    day_trades: int = 0
    day_start_bal: Optional[float] = None
    pnl_today: float = 0.0
    consec_sl: int = 0
    cooldown_until: Optional[str] = None
    paper_balance: float = 1000.0
    balance_history: list = field(default_factory=list)
    win_count: int = 0
    loss_count: int = 0
    status: str = "waiting"
    rsi4h: float = 0.0
    rsi1h: float = 0.0
    ema200: float = 0.0
    position_side: Optional[str] = None
    entry_price: float = 0.0
    trade_open_time: Optional[str] = None

def load_state() -> BotState:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            saved = json.load(f)
        s = BotState()
        for k, v in saved.items():
            if hasattr(s, k): setattr(s, k, v)
        return s
    return BotState()

def save_state(state: BotState):
    tmp  = STATE_FILE.with_suffix(".tmp")
    lock = STATE_FILE.with_suffix(".lock")
    if HAS_FCNTL:
        with open(lock, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                with open(tmp, "w") as f:
                    json.dump(asdict(state), f, indent=2, default=str)
                tmp.replace(STATE_FILE)
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)
    else:
        with open(tmp, "w") as f:
            json.dump(asdict(state), f, indent=2, default=str)
        tmp.replace(STATE_FILE)

# ══════════════════════════════════════════════════════════════════
# SECTION C — DATA FETCHING
# ══════════════════════════════════════════════════════════════════

MAX_RETRIES = 5
RETRY_SLEEP = 15

def api_call(fn, *args, **kwargs):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except (BinanceAPIException, Exception) as e:
            log.warning("API error (attempt %d/%d): %s", attempt, MAX_RETRIES, e)
            if attempt == MAX_RETRIES: raise
            time.sleep(RETRY_SLEEP)

def fetch_ohlcv(client: Client | None, interval: str, limit: int = 300) -> pd.DataFrame:
    def _fetch():
        if client:
            raw = client.futures_klines(symbol=SYMBOL, interval=interval, limit=limit)
        else:
            import urllib.request, json as _json
            url = (f"https://fapi.binance.com/fapi/v1/klines"
                   f"?symbol={SYMBOL}&interval={interval}&limit={limit}")
            with urllib.request.urlopen(url, timeout=15) as r:
                raw = _json.loads(r.read())
        df = pd.DataFrame(raw, columns=[
            "ts","open","high","low","close","volume",
            "close_ts","qav","trades","tbav","tqav","ignore"])
        for col in ["open","high","low","close","volume"]:
            df[col] = df[col].astype(float)
        df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df[["open","high","low","close","volume"]]
    return api_call(_fetch)

# ══════════════════════════════════════════════════════════════════
# SECTION D — INDICATORS & SIGNAL
# ══════════════════════════════════════════════════════════════════

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    g = d.clip(lower=0).ewm(span=period, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(span=period, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl  = df["high"] - df["low"]
    hpc = (df["high"] - df["close"].shift()).abs()
    lpc = (df["low"]  - df["close"].shift()).abs()
    return pd.concat([hl,hpc,lpc],axis=1).max(axis=1).ewm(alpha=1/period,adjust=False).mean()

def build_features(df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> pd.DataFrame:
    d = df_1h.copy()
    d["rsi1h"]    = _rsi(d["close"])
    d["atr"]      = _atr(d)
    d["atr_ma20"] = d["atr"].rolling(20).mean()
    d["atr_ratio"]= (d["atr"] / d["atr_ma20"].replace(0, np.nan)).fillna(1.0)
    d["ema200"]   = d["close"].ewm(span=200, adjust=False).mean()

    # 4h RSI forward-filled onto 1h index
    f4          = df_4h.copy()
    f4["rsi4h"] = _rsi(f4["close"])
    d["rsi4h"]  = f4["rsi4h"].reindex(d.index, method="ffill")

    return d.dropna()

def get_signal(df: pd.DataFrame) -> dict:
    """
    LONG:
      - 4h RSI > RSI4H_BULL  (bull regime)
      - 1h RSI crossed UP through RSI1H_LONG (prev < threshold, now >=)
      - Close > EMA200 (1h)
      - ATR ratio > MIN_ATR_RATIO (not flat)

    SHORT:
      - 4h RSI < RSI4H_BEAR  (bear regime)
      - 1h RSI crossed DOWN through RSI1H_SHORT (prev > threshold, now <=)
      - Close < EMA200 (1h)
      - ATR ratio > MIN_ATR_RATIO (not flat)
    """
    if len(df) < 50:
        return {"signal": 0}

    prev   = df.iloc[-3]   # two bars ago (confirmed)
    closed = df.iloc[-2]   # last closed bar
    live   = df.iloc[-1]   # current bar (price reference only)

    r4        = float(closed["rsi4h"])   if not pd.isna(closed["rsi4h"])  else 50.0
    r1        = float(closed["rsi1h"])
    prev_r1   = float(prev["rsi1h"])
    atr       = float(closed["atr"])
    atr_ratio = float(closed["atr_ratio"])
    ema200    = float(closed["ema200"])
    close     = float(closed["close"])

    long_sig  = (r4 > RSI4H_BULL
                 and prev_r1 < RSI1H_LONG and r1 >= RSI1H_LONG
                 and (close > ema200 if EMA200_FILTER else True)
                 and atr_ratio > MIN_ATR_RATIO)

    short_sig = (r4 < RSI4H_BEAR
                 and prev_r1 > RSI1H_SHORT and r1 <= RSI1H_SHORT
                 and (close < ema200 if EMA200_FILTER else True)
                 and atr_ratio > MIN_ATR_RATIO)

    fill = float(live["close"])

    if long_sig:
        return {"signal":  1, "atr": atr, "atr_ratio": atr_ratio,
                "sl": fill*(1+SLIP) - atr*SL_MULT,
                "tp": fill*(1+SLIP) + atr*TP_MULT,
                "rsi4h": r4, "rsi1h": r1, "ema200": ema200}

    if short_sig:
        return {"signal": -1, "atr": atr, "atr_ratio": atr_ratio,
                "sl": fill*(1-SLIP) + atr*SL_MULT,
                "tp": fill*(1-SLIP) - atr*TP_MULT,
                "rsi4h": r4, "rsi1h": r1, "ema200": ema200}

    return {"signal": 0, "atr": atr, "atr_ratio": atr_ratio,
            "rsi4h": r4, "rsi1h": r1}

# ══════════════════════════════════════════════════════════════════
# SECTION E — BINANCE HELPERS
# ══════════════════════════════════════════════════════════════════

def make_client(live: bool) -> Client | None:
    if not live: return None
    key, secret = os.getenv("BINANCE_API_KEY",""), os.getenv("BINANCE_API_SECRET","")
    if not key or not secret:
        log.error("BINANCE_API_KEY / BINANCE_API_SECRET not set.")
        sys.exit(1)
    return Client(key, secret)

def validate_api_key(client: Client):
    try:
        perms = api_call(client.get_account_api_permissions)
        if perms.get("enableWithdrawals"):
            log.error("SECURITY: API key has withdrawal permissions!")
            push_alert("🚨 SECURITY: Withdrawal permissions enabled. Revoke immediately.", is_error=True)
        if not perms.get("enableFutures"):
            log.error("Futures not enabled on API key."); sys.exit(1)
        log.info("API: Futures=✅  Withdrawals=%s",
                 "⚠️ YES" if perms.get("enableWithdrawals") else "✅ No")
    except Exception as e:
        log.warning("Could not validate API permissions: %s", e)

def setup_futures(client: Client):
    try:
        info = api_call(client.futures_exchange_info)
        for s in info["symbols"]:
            if s["symbol"] == SYMBOL:
                max_lev = int(s.get("leverageBracket",[{}])[0].get("initialLeverage", LEVERAGE))
                eff_lev = min(LEVERAGE, max_lev)
                if LEVERAGE > max_lev:
                    log.warning("Leverage capped %dx → %dx", LEVERAGE, max_lev)
                break
        else:
            eff_lev = LEVERAGE
    except Exception:
        eff_lev = LEVERAGE
    try:
        api_call(client.futures_change_leverage, symbol=SYMBOL, leverage=eff_lev)
    except BinanceAPIException as e:
        if "No need to change" not in str(e): raise
    try:
        client.futures_change_margin_type(symbol=SYMBOL, marginType=MARGIN_TYPE)
    except BinanceAPIException as e:
        if "No need to change" not in str(e): log.warning("Margin type: %s", e)

def get_filters(client: Client) -> tuple[float, float, float]:
    step, tick, min_not = 0.001, 0.1, 10.0
    try:
        info = api_call(client.futures_exchange_info)
        for s in info["symbols"]:
            if s["symbol"] == SYMBOL:
                for f in s["filters"]:
                    if f["filterType"] == "LOT_SIZE":     step    = float(f["stepSize"])
                    if f["filterType"] == "PRICE_FILTER": tick    = float(f["tickSize"])
                    if f["filterType"] == "MIN_NOTIONAL": min_not = float(f["notional"])
    except Exception as e:
        log.warning("get_filters error: %s", e)
    return step, tick, min_not

def get_total_balance(client, live, state):
    if not live: return state.paper_balance
    try: return float(api_call(client.futures_account)["totalWalletBalance"])
    except Exception as e: log.error("get_total_balance: %s", e); return 0.0

def get_available_balance(client, live, state):
    if not live: return state.paper_balance
    try:
        for b in api_call(client.futures_account_balance):
            if b["asset"] == "USDT": return float(b["balance"])
    except Exception as e: log.error("get_available_balance: %s", e)
    return 0.0

def get_open_position(client: Client):
    try:
        for p in api_call(client.futures_account)["positions"]:
            if p["symbol"] == SYMBOL:
                amt = float(p["positionAmt"])
                if abs(amt) > 0.0001:
                    return {"qty": abs(amt), "direction": 1 if amt > 0 else -1,
                            "entry_price": float(p["entryPrice"])}
    except Exception as e: log.warning("get_open_position: %s", e)
    return None

def round_qty(qty, step):
    if step <= 0: return qty
    return round(math.floor(qty/step)*step, max(0, round(-math.log10(step))))

def round_price(price, tick, round_up=False):
    if tick <= 0: return price
    decimals = max(0, round(-math.log10(tick)))
    factor   = 1 / tick
    if round_up: return round(math.ceil(price*factor)/factor, decimals)
    return round(math.floor(price*factor)/factor, decimals)

def cancel_all_open_orders(client: Client):
    try: api_call(client.futures_cancel_all_open_orders, symbol=SYMBOL)
    except Exception as e: log.warning("cancel_all_open_orders: %s", e)

def close_position(client: Client, direction: int):
    cancel_all_open_orders(client)
    pos = get_open_position(client)
    if pos is None or pos["qty"] < 0.0001: return
    side = "SELL" if direction == 1 else "BUY"
    api_call(client.futures_create_order, symbol=SYMBOL, side=side,
             type="MARKET", quantity=pos["qty"], reduceOnly=True)

def place_sl_tp(client: Client, direction: int, sl: float, tp: float, tick: float):
    cancel_all_open_orders(client)
    side    = "SELL" if direction == 1 else "BUY"
    safe_sl = round_price(sl, tick, round_up=(direction == 1))
    safe_tp = round_price(tp, tick, round_up=(direction == -1))
    try:
        api_call(client.futures_create_order, symbol=SYMBOL, side=side,
                 type="STOP_MARKET", stopPrice=safe_sl, closePosition=True)
    except Exception as e:
        log.error("SL placement failed: %s", e); raise
    try:
        api_call(client.futures_create_order, symbol=SYMBOL, side=side,
                 type="TAKE_PROFIT_MARKET", stopPrice=safe_tp, closePosition=True)
    except Exception as e:
        log.error("TP placement failed: %s", e); raise

def check_live_exit(client: Client, state: BotState, d: int):
    try:
        positions = api_call(client.futures_position_information, symbol=SYMBOL)
        open_qty  = sum(abs(float(p["positionAmt"])) for p in positions)
        if open_qty >= 0.0001: return None, 0.0
        open_orders     = api_call(client.futures_get_open_orders, symbol=SYMBOL)
        remaining_types = {o["type"] for o in open_orders}
        sl_gone = "STOP_MARKET"        not in remaining_types
        tp_gone = "TAKE_PROFIT_MARKET" not in remaining_types
        if   sl_gone and not tp_gone: reason = "SL"
        elif tp_gone and not sl_gone: reason = "TP"
        elif sl_gone and tp_gone:
            try:
                recent  = api_call(client.futures_account_trades, symbol=SYMBOL, limit=5)
                exit_px = float(recent[-1]["price"]) if recent else state.sl
                mid     = (state.tp + state.sl) / 2
                reason  = "TP" if (d==1 and exit_px>mid) or (d==-1 and exit_px<mid) else "SL"
            except Exception: reason = "SL"
        else: return None, 0.0
        try:
            recent  = api_call(client.futures_account_trades, symbol=SYMBOL, limit=5)
            exit_px = float(recent[-1]["price"]) if recent else (state.sl if reason=="SL" else state.tp)
        except Exception:
            exit_px = state.sl if reason == "SL" else state.tp
        return reason, exit_px
    except Exception as e:
        log.warning("check_live_exit: %s", e); return None, 0.0

# ══════════════════════════════════════════════════════════════════
# SECTION F — STARTUP RECOVERY
# ══════════════════════════════════════════════════════════════════

def recover_on_startup(client: Client, state: BotState, tick: float):
    pos = get_open_position(client)
    if pos and not state.in_trade:
        log.warning("STARTUP RECOVERY: Open position found without state.")
        push_alert("⚠️ STARTUP RECOVERY: Restoring unmanaged position.", is_error=True)
        state.in_trade        = True
        state.direction       = pos["direction"]
        state.entry           = pos["entry_price"]
        state.entry_price     = pos["entry_price"]
        state.qty             = pos["qty"]
        state.entry_time      = datetime.now(timezone.utc).isoformat()
        state.trade_open_time = datetime.now(timezone.utc).isoformat()
        state.status          = "in_trade"
        state.position_side   = "LONG" if pos["direction"] == 1 else "SHORT"
        open_orders = api_call(client.futures_get_open_orders, symbol=SYMBOL)
        found_sl = next((float(o["stopPrice"]) for o in open_orders if o["type"]=="STOP_MARKET"), 0.0)
        found_tp = next((float(o["stopPrice"]) for o in open_orders if o["type"]=="TAKE_PROFIT_MARKET"), 0.0)
        if found_sl and found_tp:
            state.sl, state.tp = found_sl, found_tp
        else:
            try:
                df1h = fetch_ohlcv(client, "1h", limit=50)
                df4h = fetch_ohlcv(client, "4h", limit=50)
                feat = build_features(df1h, df4h)
                atr  = float(feat["atr"].iloc[-1])
            except Exception:
                atr = pos["entry_price"] * 0.01
            d        = pos["direction"]
            state.sl = pos["entry_price"] - atr * SL_MULT * d
            state.tp = pos["entry_price"] + atr * TP_MULT * d
            place_sl_tp(client, state.direction, state.sl, state.tp, tick)
        save_state(state)
    elif state.in_trade and not pos:
        log.warning("STARTUP: State says in_trade but no live position. Clearing.")
        state.in_trade = False; state.direction = 0
        state.entry = state.sl = state.tp = state.qty = 0.0
        state.entry_time = state.position_side = None
        state.entry_price = 0.0; state.status = "waiting"
        save_state(state)

# ══════════════════════════════════════════════════════════════════
# SECTION G — GUARDRAILS
# ══════════════════════════════════════════════════════════════════

def check_guardrails(state: BotState, avail_bal: float, total_bal: float,
                     now: datetime, atr_ratio: float) -> tuple[bool, str]:
    if state.in_trade:                     return False, "already in trade"
    if avail_bal < MIN_BALANCE:            return False, f"balance ${avail_bal:.2f} below min"
    if state.day_trades >= MAX_TRADES_DAY: return False, "max trades/day reached"
    if state.day_start_bal is None:
        state.day_start_bal = total_bal
    loss_pct = (total_bal - state.day_start_bal) / state.day_start_bal
    if loss_pct <= -DAILY_LOSS_PCT:        return False, f"daily loss limit ({loss_pct:.1%})"
    if state.cooldown_until:
        try:
            cu = datetime.fromisoformat(state.cooldown_until)
            if cu.tzinfo is None: cu = cu.replace(tzinfo=timezone.utc)
            if now < cu:                   return False, "cooldown active"
            else: state.cooldown_until = None
        except Exception: state.cooldown_until = None
    if now.hour in BAD_HOURS:              return False, f"bad hour UTC {now.hour:02d}:00"
    if atr_ratio < MIN_ATR_RATIO:          return False, "market too flat"
    return True, ""

# ══════════════════════════════════════════════════════════════════
# SECTION H — ENTRY
# ══════════════════════════════════════════════════════════════════

def handle_entry(state, sig, price, avail_bal, total_bal,
                 client, live, step, tick, min_not, now, bar):
    direction = sig["signal"]
    sl, tp    = sig["sl"], sig["tp"]
    fill      = price * (1 + SLIP * direction)

    if live and client:
        cancel_all_open_orders(client)
        avail_bal = get_available_balance(client, live, state)

    margin = avail_bal * MARGIN_PCT
    qty    = round_qty((margin * LEVERAGE * 0.95) / fill, step)

    if qty <= 0 or (qty * fill) < min_not:
        log.warning("Trade size too small (qty=%.4f notional=%.2f)", qty, qty*fill)
        return

    if live and client:
        side = "BUY" if direction == 1 else "SELL"
        try:
            order = api_call(client.futures_create_order,
                             symbol=SYMBOL, side=side, type="MARKET", quantity=qty)
        except Exception as e:
            log.error("Market order failed: %s", e)
            push_alert(f"❌ Order failed: {e}", is_error=True); return
        try:
            time.sleep(2)
            all_recent = api_call(client.futures_account_trades, symbol=SYMBOL, limit=20)
            order_id   = order["orderId"]
            matched    = [t for t in all_recent if t.get("orderId") == order_id]
            if matched:
                total_cost = sum(float(t["price"])*float(t["qty"]) for t in matched)
                total_q    = sum(float(t["qty"]) for t in matched)
                fill       = total_cost / total_q
        except Exception as e:
            log.warning("VWAP fill failed: %s", e)

    state.in_trade        = True
    state.direction       = direction
    state.entry           = fill
    state.sl              = sl
    state.tp              = tp
    state.qty             = qty
    state.entry_time      = now.isoformat()
    state.entry_bar       = bar.isoformat()
    state.status          = "in_trade"
    state.position_side   = "LONG" if direction == 1 else "SHORT"
    state.entry_price     = fill
    state.trade_open_time = now.isoformat()
    state.day_trades     += 1
    state.rsi4h           = round(sig.get("rsi4h", 0), 1)
    state.rsi1h           = round(sig.get("rsi1h", 0), 1)
    state.ema200          = round(sig.get("ema200", 0), 2)

    dir_str = "LONG 🟢" if direction == 1 else "SHORT 🔴"
    log.info("ENTER %s | fill=%.2f SL=%.2f TP=%.2f | qty=%.4f | margin=$%.2f",
             dir_str, fill, sl, tp, qty, margin)
    push_alert(
        f"**📥 ENTER {dir_str}**\n"
        f"**Price:** ${fill:.2f}  |  **Size:** {qty:.4f} BTC\n"
        f"**SL:** ${sl:.2f}  |  **TP:** ${tp:.2f}\n"
        f"**4h RSI:** {sig.get('rsi4h',0):.1f}  |  **1h RSI:** {sig.get('rsi1h',0):.1f}\n"
        f"**EMA200:** ${sig.get('ema200',0):.2f}  |  **Margin:** ${margin:.2f}"
    )
    save_state(state)

    if live and client:
        try:
            place_sl_tp(client, direction, sl, tp, tick)
        except Exception as e:
            log.error("CRITICAL: SL/TP failed: %s", e)
            push_alert(f"🚨 SL/TP FAILED: {e}", is_error=True)
            sys.exit(1)

# ══════════════════════════════════════════════════════════════════
# SECTION I — EXIT
# ══════════════════════════════════════════════════════════════════

def handle_exit(state, exit_reason, exit_price, total_bal, client, live):
    direction, entry, qty = state.direction, state.entry, state.qty

    if not live:
        if exit_reason == "SL":   exit_price *= (1 - SLIP * direction)
        elif exit_reason in ("TP","TIME"): exit_price *= (1 + SLIP * direction * -1)

    gross   = (exit_price - entry) * direction * qty
    net_pnl = gross - (qty * entry * FEE * 2)

    if live and client:
        try: close_position(client, direction)
        except Exception as e: log.error("close_position failed: %s", e)

    state.win_count  += 1 if net_pnl > 0 else 0
    state.loss_count += 0 if net_pnl > 0 else 1
    state.pnl_today  += net_pnl
    if not live: state.paper_balance += net_pnl

    if exit_reason == "SL":
        state.consec_sl += 1
        if state.consec_sl >= CONSEC_SL_LIMIT:
            escalation           = state.consec_sl - CONSEC_SL_LIMIT + 1
            hours                = min(COOLDOWN_H * escalation, MAX_COOLDOWN_H)
            until                = (datetime.now(timezone.utc)
                                    .replace(minute=0, second=0, microsecond=0)
                                    + timedelta(hours=hours))
            state.cooldown_until = until.isoformat()
            state.status         = "halted"
            log.warning("Cooldown %dh — %d consec SLs. Until %s UTC",
                        hours, state.consec_sl, until.strftime("%H:%M"))
            push_alert(f"🧊 **COOLDOWN {hours}h:** {state.consec_sl} losses. "
                       f"Paused until {until.strftime('%H:%M')} UTC.")
    else:
        state.consec_sl      = 0
        state.cooldown_until = None

    state.in_trade        = False
    state.direction       = 0
    state.entry           = 0.0
    state.sl              = 0.0
    state.tp              = 0.0
    state.qty             = 0.0
    state.entry_time      = None
    state.entry_bar       = None
    state.position_side   = None
    state.entry_price     = 0.0
    state.trade_open_time = None
    if not state.cooldown_until: state.status = "waiting"

    approx_bal = state.paper_balance if not live else (total_bal + net_pnl)
    emoji      = "✅" if net_pnl > 0 else "🛑"
    log.info("EXIT %-5s | %-5s | entry=%.2f exit=%.2f | pnl=$%.2f | W%d L%d | bal≈$%.2f",
             "LONG" if direction==1 else "SHORT", exit_reason,
             entry, exit_price, net_pnl,
             state.win_count, state.loss_count, approx_bal)
    push_alert(
        f"{emoji} **EXIT ({exit_reason})**\n"
        f"**PnL:** ${net_pnl:.2f}  |  **Exit:** ${exit_price:.2f}\n"
        f"**Record:** {state.win_count}W / {state.loss_count}L\n"
        f"**Today:** ${state.pnl_today:.2f}  |  **Bal:** ${approx_bal:.2f}"
    )
    save_state(state)

# ══════════════════════════════════════════════════════════════════
# SECTION J — MAIN LOOP
# ══════════════════════════════════════════════════════════════════

def main():
    if LIVE_MODE:
        print("\n  ⚠️  LIVE MODE — real orders WILL be placed on Binance Futures")
        print("  ⚠️  Press Ctrl+C within 5 seconds to cancel...\n")
        for i in range(5, 0, -1):
            print(f"  Starting in {i}...", end="\r", flush=True)
            time.sleep(1)
        print()

    log.info("="*65)
    log.info("%s | %s | %dx | DualRSI 4h(%d/%d) 1h(%d/%d) | SL=%.1fx TP=%.1fx",
             SYMBOL, "LIVE" if LIVE_MODE else "PAPER", LEVERAGE,
             RSI4H_BULL, RSI4H_BEAR, RSI1H_LONG, RSI1H_SHORT,
             SL_MULT, TP_MULT)
    log.info("="*65)
    push_alert(f"🚀 **BTC DualRSI Bot Started** | {'🔴 LIVE' if LIVE_MODE else '📄 PAPER'}\n"
               f"4h RSI({RSI4H_BULL}/{RSI4H_BEAR}) | 1h RSI({RSI1H_LONG}/{RSI1H_SHORT}) | "
               f"{LEVERAGE}x | Margin {MARGIN_PCT*100:.0f}%")

    client = make_client(LIVE_MODE)
    if LIVE_MODE and client:
        validate_api_key(client)
        setup_futures(client)
        step, tick, min_not = get_filters(client)
    else:
        step, tick, min_not = 0.001, 0.10, 10.0

    state    = load_state()
    last_bar = None

    if LIVE_MODE and client:
        recover_on_startup(client, state, tick)

    # ── Poll every 60s — 1h bars update once per hour ────────────
    while True:
        try:
            time.sleep(60)
            now = datetime.now(timezone.utc)

            # 1h bar timestamp
            this_bar = now.replace(minute=0, second=0, microsecond=0)

            avail_bal = get_available_balance(client, LIVE_MODE, state)
            total_bal = get_total_balance(client, LIVE_MODE, state)

            # Daily reset
            today = str(now.date())
            if state.day != today:
                state.day           = today
                state.day_trades    = 0
                state.pnl_today     = 0.0
                state.day_start_bal = total_bal
                log.info("── New day %s | Balance=$%.2f ──", today, total_bal)
                push_alert(f"🌅 **New Day:** {today} | Balance: ${total_bal:.2f}")
                state.balance_history.append({"time": now.isoformat(),
                                               "balance": round(total_bal, 2)})
                state.balance_history = state.balance_history[-90:]
                save_state(state)

            # ── Manage open trade ──────────────────────────────────
            if state.in_trade:
                if LIVE_MODE and client:
                    live_pos = get_open_position(client)
                    if live_pos and abs(live_pos["qty"] - state.qty) > 0.0001:
                        log.warning("Manual interference: qty %.4f → %.4f",
                                    state.qty, live_pos["qty"])
                        state.qty = live_pos["qty"]
                        save_state(state)

                d, sl, tp = state.direction, state.sl, state.tp
                et = datetime.fromisoformat(state.entry_time)
                if et.tzinfo is None: et = et.replace(tzinfo=timezone.utc)
                hold_h = (now - et).total_seconds() / 3600

                exit_reason = exit_price = None

                if LIVE_MODE and client:
                    live_reason, live_px = check_live_exit(client, state, d)
                    if live_reason: exit_reason, exit_price = live_reason, live_px
                else:
                    # Paper: check current 1h bar hi/lo
                    df_chk = fetch_ohlcv(client, "1h", limit=2)
                    bar_hi = float(df_chk.iloc[-1]["high"])
                    bar_lo = float(df_chk.iloc[-1]["low"])
                    if d == 1:
                        if bar_lo <= sl:  exit_reason, exit_price = "SL", sl
                        elif bar_hi >= tp: exit_reason, exit_price = "TP", tp
                    else:
                        if bar_hi >= sl:  exit_reason, exit_price = "SL", sl
                        elif bar_lo <= tp: exit_reason, exit_price = "TP", tp

                if hold_h >= MAX_HOLD_H and exit_reason is None:
                    exit_reason = "TIME"
                    if LIVE_MODE and client:
                        close_position(client, d)
                        time.sleep(1)
                        try:
                            recent     = api_call(client.futures_account_trades,
                                                  symbol=SYMBOL, limit=5)
                            exit_price = float(recent[-1]["price"]) if recent else 0.0
                        except Exception:
                            exit_price = float(api_call(
                                client.futures_mark_price, symbol=SYMBOL)["markPrice"])
                    else:
                        df_chk     = fetch_ohlcv(client, "1h", limit=1)
                        exit_price = float(df_chk.iloc[-1]["close"])

                if exit_reason:
                    handle_exit(state, exit_reason, exit_price,
                                total_bal, client, LIVE_MODE)
                else:
                    log.info("  Holding %-5s | entry=%.2f SL=%.2f TP=%.2f | %.1fh",
                             state.position_side, state.entry, sl, tp, hold_h)

            # ── Signal check on new 1h bar ─────────────────────────
            if this_bar != last_bar:
                last_bar = this_bar

                if not state.in_trade:
                    df_1h = fetch_ohlcv(client, "1h", limit=300)
                    df_4h = fetch_ohlcv(client, "4h", limit=100)
                    df    = build_features(df_1h, df_4h)

                    if len(df) < 50: continue

                    closed    = df.iloc[-2]
                    price     = float(df.iloc[-1]["close"])
                    atr_ratio = float(closed.get("atr_ratio", 1.0))
                    r4        = float(closed.get("rsi4h", 50))
                    r1        = float(closed.get("rsi1h", 50))
                    e200      = float(closed.get("ema200", 0))

                    state.rsi4h  = round(r4, 1)
                    state.rsi1h  = round(r1, 1)
                    state.ema200 = round(e200, 2)

                    log.info("Bar %s | price=%.2f | 4h RSI=%.1f | 1h RSI=%.1f | "
                             "EMA200=%.2f | ATR_ratio=%.2f | Bal=$%.2f",
                             this_bar.strftime("%m-%d %H:%M"), price,
                             r4, r1, e200, atr_ratio,
                             avail_bal if LIVE_MODE else state.paper_balance)

                    ok, reason = check_guardrails(state, avail_bal, total_bal,
                                                  now, atr_ratio)
                    if not ok:
                        log.info("  → No entry: %s", reason)
                    else:
                        sig = get_signal(df)
                        if sig["signal"] != 0:
                            handle_entry(state, sig, price, avail_bal, total_bal,
                                         client, LIVE_MODE, step, tick, min_not,
                                         now, this_bar)
                        else:
                            regime = ("BULL" if r4 > RSI4H_BULL else
                                      "BEAR" if r4 < RSI4H_BEAR else "NEUTRAL")
                            log.info("  → HOLD | Regime=%s | 4h RSI=%.1f | 1h RSI=%.1f",
                                     regime, r4, r1)

        except KeyboardInterrupt:
            log.info("Bot stopped by user.")
            push_alert("🛑 BTC DualRSI Bot stopped manually.")
            save_state(state)
            sys.exit(0)
        except Exception as e:
            log.error("Unhandled error: %s", e, exc_info=True)
            push_alert(f"⚠️ **Error:** {e}", is_error=True)
            save_state(state)
            time.sleep(60)

if __name__ == "__main__":
    main()
