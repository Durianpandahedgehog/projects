import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from collect_data import fetch_ohlcv, fetch_btc, merge_data
from features import calculate_features
from strategy import generate_signals
from levels import get_trade_levels
import time

st.set_page_config(
    page_title="Alpha Signals — Q/USDT",
    page_icon="📡",
    layout="wide"
)

# ── STYLES ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0a0f;
    color: #e2e8f0;
}

.signal-buy {
    background: linear-gradient(135deg, #00ff88, #00cc6a);
    color: #000;
    font-family: 'Space Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    padding: 1.5rem 3rem;
    border-radius: 12px;
    text-align: center;
    letter-spacing: 0.2em;
    box-shadow: 0 0 40px rgba(0,255,136,0.4);
}

.signal-sell {
    background: linear-gradient(135deg, #ff4466, #cc0033);
    color: #fff;
    font-family: 'Space Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    padding: 1.5rem 3rem;
    border-radius: 12px;
    text-align: center;
    letter-spacing: 0.2em;
    box-shadow: 0 0 40px rgba(255,68,102,0.4);
}

.signal-hold {
    background: linear-gradient(135deg, #334155, #1e293b);
    color: #94a3b8;
    font-family: 'Space Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    padding: 1.5rem 3rem;
    border-radius: 12px;
    text-align: center;
    letter-spacing: 0.2em;
    border: 1px solid #334155;
}

.metric-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
}

.metric-label {
    font-size: 0.75rem;
    color: #64748b;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: #f1f5f9;
}

.header-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    color: #00ff88;
    letter-spacing: 0.3em;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)


# ── DATA ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)   # refresh every 5 minutes
def load_live_data():
    q_df   = fetch_ohlcv("Q/USDT:USDT", "5m", days=60)
    btc_df = fetch_btc(days=60)
    df     = merge_data(q_df, btc_df)
    df     = calculate_features(df)
    df     = generate_signals(df)
    return df


# ── LAYOUT ────────────────────────────────────────────────────────────────────

st.markdown('<p class="header-title">📡 Alpha Signals — Q/USDT Swing Strategy</p>', unsafe_allow_html=True)
st.markdown("---")

with st.spinner("Fetching live data from Binance..."):
    df = load_live_data()

with st.spinner("Fetching live data from Binance..."):
    df = load_live_data()

if df is None or len(df) == 0:
    st.error("Could not fetch data from Binance. Please try again in a moment.")
    st.stop()
    
if df is None or len(df) == 0:
    st.error("Could not fetch data from Binance. Please try again in a moment.")
    st.stop()

latest = df.iloc[-1]
signal = latest["signal"]
price  = latest["close"]
rsi    = latest["rsi_7"]
macd   = latest["macd"]
vol    = latest["volume_ratio"]
btc    = latest["btc_pct_change"]

# ── SIGNAL BADGE ──────────────────────────────────────────────────────────────

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    css_class = f"signal-{signal.lower()}"
    st.markdown(f'<div class="{css_class}">{signal}</div>', unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center;color:#64748b;font-size:0.8rem;margin-top:0.5rem'>{df.index[-1].strftime('%Y-%m-%d %H:%M')} UTC · refreshes every 5 min</p>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TRADE SETUP ───────────────────────────────────────────────────────────────

if signal in ["BUY", "SELL"]:
    levels = get_trade_levels(df)

    if levels:
        st.markdown("### Trade Setup")
        t1, t2, t3, t4 = st.columns(4)

        with t1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Entry</div>
                <div class="metric-value">{levels['entry']:.6f}</div>
            </div>""", unsafe_allow_html=True)

        with t2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Take Profit</div>
                <div class="metric-value" style="color:#00ff88">
                    {levels['take_profit']:.6f}
                    <span style="font-size:0.8rem;color:#00cc6a"> +{levels['tp_pct']:.1f}%</span>
                </div>
            </div>""", unsafe_allow_html=True)

        with t3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Stop Loss</div>
                <div class="metric-value" style="color:#ff4466">
                    {levels['stop_loss']:.6f}
                    <span style="font-size:0.8rem;color:#cc0033"> -{levels['sl_pct']:.1f}%</span>
                </div>
            </div>""", unsafe_allow_html=True)

        with t4:
            rr_color = "#00ff88" if levels['rr'] >= 2 else "#fbbf24"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Risk / Reward</div>
                <div class="metric-value" style="color:{rr_color}">
                    1 : {levels['rr']:.1f}
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    else:
        st.info("Signal detected but no clean R/R setup found. Wait for better entry.") 

# ── METRICS ───────────────────────────────────────────────────────────────────

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Price</div>
        <div class="metric-value">{price:.6f}</div>
    </div>""", unsafe_allow_html=True)

with c2:
    rsi_color = "#00ff88" if rsi < 35 else "#ff4466" if rsi > 65 else "#f1f5f9"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">RSI 7</div>
        <div class="metric-value" style="color:{rsi_color}">{rsi:.1f}</div>
    </div>""", unsafe_allow_html=True)

with c3:
    macd_color = "#00ff88" if macd > 0 else "#ff4466"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">MACD</div>
        <div class="metric-value" style="color:{macd_color}">{macd:.6f}</div>
    </div>""", unsafe_allow_html=True)

with c4:
    vol_color = "#00ff88" if vol > 1.5 else "#f1f5f9"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Volume Ratio</div>
        <div class="metric-value" style="color:{vol_color}">{vol:.2f}x</div>
    </div>""", unsafe_allow_html=True)

with c5:
    btc_color = "#00ff88" if btc > 0 else "#ff4466"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">BTC 5m</div>
        <div class="metric-value" style="color:{btc_color}">{btc:+.2f}%</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── CANDLESTICK CHART ─────────────────────────────────────────────────────────

st.markdown("### Price Chart (Last 3 Days)")
chart_df = df.tail(864)   # 864 x 5min = 3 days

fig = go.Figure(data=[
    go.Candlestick(
        x=chart_df.index,
        open=chart_df["open"],
        high=chart_df["high"],
        low=chart_df["low"],
        close=chart_df["close"],
        increasing_line_color="#00ff88",
        decreasing_line_color="#ff4466",
        name="Q/USDT"
    ),
    go.Scatter(
        x=chart_df.index,
        y=chart_df["ema_21"],
        line=dict(color="#fbbf24", width=1),
        name="EMA 21"
    ),
    go.Scatter(
        x=chart_df.index,
        y=chart_df["ema_9"],
        line=dict(color="#818cf8", width=1),
        name="EMA 9"
    )
])

# Add BUY/SELL markers
buys  = chart_df[chart_df["signal"] == "BUY"]
sells = chart_df[chart_df["signal"] == "SELL"]

if not buys.empty:
    fig.add_trace(go.Scatter(
        x=buys.index, y=buys["low"] * 0.995,
        mode="markers",
        marker=dict(symbol="triangle-up", size=14, color="#00ff88"),
        name="BUY"
    ))

if not sells.empty:
    fig.add_trace(go.Scatter(
        x=sells.index, y=sells["high"] * 1.005,
        mode="markers",
        marker=dict(symbol="triangle-down", size=14, color="#ff4466"),
        name="SELL"
    ))

fig.update_layout(
    paper_bgcolor="#0a0a0f",
    plot_bgcolor="#0d1117",
    font=dict(color="#94a3b8"),
    xaxis=dict(gridcolor="#1e293b", showgrid=True),
    yaxis=dict(gridcolor="#1e293b", showgrid=True),
    xaxis_rangeslider_visible=False,
    height=450,
    margin=dict(l=0, r=0, t=20, b=0),
    legend=dict(bgcolor="#111827", bordercolor="#1e293b", borderwidth=1)
)

st.plotly_chart(fig, use_container_width=True)

# ── SIGNAL HISTORY ────────────────────────────────────────────────────────────

st.markdown("### Recent Signals")
signals_df = df[df["signal"] != "HOLD"][["close", "rsi_7", "volume_ratio", "signal"]].tail(10)
signals_df = signals_df.rename(columns={
    "close": "Price",
    "rsi_7": "RSI 7",
    "volume_ratio": "Volume Ratio",
    "signal": "Signal"
})
signals_df.index = signals_df.index.strftime("%Y-%m-%d %H:%M")
st.dataframe(signals_df, use_container_width=True)

# ── AUTO REFRESH ──────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("<p style='color:#334155;font-size:0.75rem;text-align:center'>Alpha Signals · Q/USDT Swing · Binance Futures · Auto-refresh 5min</p>", unsafe_allow_html=True)

time.sleep(300)
st.rerun()