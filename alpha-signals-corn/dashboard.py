import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from xgboost import XGBClassifier
import os, pickle, json

MODEL_PATH  = "/app/models/lstm_model.keras"
SCALER_PATH = "/app/models/scaler.pkl"
OUTPUT_PATH = "/app/dashboard.html"
LOOKBACK    = 30
XGB_WEIGHT  = 0.4
LSTM_WEIGHT = 0.6

FEATURES = [
    "close_price","moving_avg_7","moving_avg_30","moving_avg_90",
    "price_change","pct_change","volatility_30","daily_range",
    "month","day_of_week","is_harvest","rsi_14","macd","macd_signal",
    "bb_upper","bb_lower","bb_width","volume_ma_20",
    "oil_close","soy_close","dxy_close",
    "oil_pct_change","soy_pct_change","dxy_pct_change",
]

def get_engine():
    host = os.getenv("DB_HOST","localhost")
    port = os.getenv("DB_PORT","5432")
    name = os.getenv("DB_NAME","corn_db")
    user = os.getenv("DB_USER","corn_user")
    pw   = os.getenv("DB_PASS","corn_pass")
    return create_engine(f"postgresql://{user}:{pw}@{host}:{port}/{name}")

def load_data():
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM corn_prices ORDER BY date ASC", engine)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df

def generate_signal(df):
    lstm_model = load_model(MODEL_PATH)
    xgb_model  = XGBClassifier()
    xgb_model.load_model(f"{os.path.dirname(MODEL_PATH)}/xgb_model.json")

    with open(SCALER_PATH,"rb") as f:
        scaler, available = pickle.load(f)

    X_raw    = df[available].values
    X_scaled = scaler.transform(X_raw)

    xgb_proba  = xgb_model.predict_proba(X_scaled[-1:])
    seq        = X_scaled[-LOOKBACK:].reshape(1, LOOKBACK, len(available))
    lstm_proba = lstm_model.predict(seq, verbose=0)

    combined = (XGB_WEIGHT * xgb_proba) + (LSTM_WEIGHT * lstm_proba)
    hold_p   = float(combined[0][0])
    buy_p    = float(combined[0][1])
    sell_p   = float(combined[0][2])

    if hold_p >= buy_p and hold_p >= sell_p:
        signal = "HOLD"
    elif buy_p >= 0.25 and buy_p >= sell_p:
        signal = "BUY"
    else:
        signal = "SELL"

    return signal, hold_p, buy_p, sell_p, xgb_proba[0], lstm_proba[0]

def build_html(df, signal, hold_p, buy_p, sell_p, xgb_p, lstm_p):
    chart_df   = df.tail(180).copy()
    dates      = chart_df.index.strftime("%Y-%m-%d").tolist()
    prices     = [round(x,2) for x in chart_df["close_price"].tolist()]
    ma30       = [round(x,2) if not pd.isna(x) else None for x in chart_df["moving_avg_30"].tolist()]
    ma90       = [round(x,2) if not pd.isna(x) else None for x in chart_df["moving_avg_90"].tolist()]
    bb_upper   = [round(x,2) if not pd.isna(x) else None for x in chart_df["bb_upper"].tolist()]
    bb_lower   = [round(x,2) if not pd.isna(x) else None for x in chart_df["bb_lower"].tolist()]

    today      = df.index[-1].strftime("%B %d, %Y")
    price      = round(float(df["close_price"].iloc[-1]), 2)
    prev_price = round(float(df["close_price"].iloc[-2]), 2)
    change     = round(price - prev_price, 2)
    change_pct = round((change / prev_price) * 100, 2)
    week_ago   = round(float(df["close_price"].iloc[-6]), 2) if len(df) >= 6 else price
    month_ago  = round(float(df["close_price"].iloc[-22]), 2) if len(df) >= 22 else price
    week_chg   = round(((price - week_ago) / week_ago) * 100, 2)
    month_chg  = round(((price - month_ago) / month_ago) * 100, 2)

    signal_color = {"BUY":"#16a34a","SELL":"#dc2626","HOLD":"#d97706"}[signal]
    signal_bg    = {"BUY":"#f0fdf4","SELL":"#fef2f2","HOLD":"#fffbeb"}[signal]
    signal_desc  = {
        "BUY":  "The model predicts corn prices are likely to rise over the next few days. This may be a good time to buy.",
        "SELL": "The model predicts corn prices are likely to fall over the next few days. This may be a good time to sell.",
        "HOLD": "The model sees no strong price movement coming. Stay with your current position for now."
    }[signal]
    signal_icon  = {"BUY":"ti-trending-up","SELL":"ti-trending-down","HOLD":"ti-minus"}[signal]
    change_color = "#16a34a" if change >= 0 else "#dc2626"
    change_sign  = "+" if change >= 0 else ""

    xgb_buy  = round(float(xgb_p[1])*100)
    xgb_hold = round(float(xgb_p[0])*100)
    xgb_sell = round(float(xgb_p[2])*100)
    lstm_buy  = round(float(lstm_p[1])*100)
    lstm_hold = round(float(lstm_p[0])*100)
    lstm_sell = round(float(lstm_p[2])*100)

    data_json = json.dumps({
        "dates":dates,"prices":prices,"ma30":ma30,
        "ma90":ma90,"bb_upper":bb_upper,"bb_lower":bb_lower
    })

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Corn Signal — {today}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/tabler-icons.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f8fafc;color:#0f172a;padding:32px;min-height:100vh}}
.container{{max-width:960px;margin:0 auto}}
.top{{display:flex;justify-content:space-between;align-items:center;margin-bottom:28px}}
.app-name{{font-size:13px;font-weight:600;color:#64748b;letter-spacing:1px;text-transform:uppercase}}
.date-label{{font-size:13px;color:#94a3b8}}
.signal-banner{{background:{signal_bg};border:1.5px solid {signal_color}40;border-radius:16px;padding:28px 32px;margin-bottom:20px;display:flex;align-items:center;gap:24px}}
.signal-icon{{width:64px;height:64px;background:{signal_color};border-radius:12px;display:flex;align-items:center;justify-content:center;flex-shrink:0}}
.signal-icon i{{font-size:32px;color:white}}
.signal-text h1{{font-size:32px;font-weight:700;color:{signal_color};letter-spacing:-0.5px}}
.signal-text p{{font-size:15px;color:#475569;margin-top:6px;line-height:1.5;max-width:580px}}
.grid3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:20px}}
.card{{background:white;border:1px solid #e2e8f0;border-radius:12px;padding:20px}}
.card-label{{font-size:11px;font-weight:600;color:#94a3b8;letter-spacing:1px;text-transform:uppercase;margin-bottom:10px}}
.card-value{{font-size:28px;font-weight:700;color:#0f172a;letter-spacing:-0.5px}}
.card-sub{{font-size:13px;margin-top:4px}}
.conf-row{{display:flex;flex-direction:column;gap:10px;margin-top:16px}}
.conf-item label{{display:flex;justify-content:space-between;font-size:13px;color:#64748b;margin-bottom:5px}}
.conf-item label span:last-child{{font-weight:600;color:#0f172a}}
.bar-bg{{background:#f1f5f9;border-radius:99px;height:8px;overflow:hidden}}
.bar-fill{{height:100%;border-radius:99px}}
.chart-card{{background:white;border:1px solid #e2e8f0;border-radius:12px;padding:24px;margin-bottom:20px}}
.chart-header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px}}
.chart-title{{font-size:15px;font-weight:600;color:#0f172a}}
.legend{{display:flex;gap:16px}}
.leg-item{{display:flex;align-items:center;gap:6px;font-size:12px;color:#64748b}}
.leg-dot{{width:10px;height:4px;border-radius:2px}}
.model-grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}}
.model-col-label{{font-size:11px;font-weight:600;color:#94a3b8;letter-spacing:1px;text-transform:uppercase;margin-bottom:12px}}
.mini-bar-row{{display:flex;flex-direction:column;gap:8px}}
.mini-bar-item{{}}
.mini-bar-header{{display:flex;justify-content:space-between;font-size:12px;color:#64748b;margin-bottom:4px}}
.mini-bar-header span:last-child{{font-weight:600;color:#0f172a}}
.mini-bg{{background:#f1f5f9;border-radius:99px;height:6px}}
.mini-fill{{height:100%;border-radius:99px}}
.footer{{text-align:center;font-size:12px;color:#94a3b8;margin-top:24px}}
.footer strong{{color:#64748b}}
</style>
</head>
<body>
<div class="container">

  <div class="top">
    <span class="app-name">Corn Trading Signal</span>
    <span class="date-label">{today}</span>
  </div>

  <div class="signal-banner">
    <div class="signal-icon"><i class="ti {signal_icon}"></i></div>
    <div class="signal-text">
      <h1>{signal}</h1>
      <p>{signal_desc}</p>
    </div>
  </div>

  <div class="grid3">
    <div class="card">
      <div class="card-label">Current Price</div>
      <div class="card-value">${price}</div>
      <div class="card-sub" style="color:{change_color}">
        {change_sign}{change} ({change_sign}{change_pct}%) today
      </div>
    </div>

    <div class="card">
      <div class="card-label">Performance</div>
      <div style="margin-top:4px">
        <div style="display:flex;justify-content:space-between;margin-bottom:10px">
          <span style="font-size:13px;color:#64748b">This week</span>
          <span style="font-size:15px;font-weight:700;color:{'#16a34a' if week_chg>=0 else '#dc2626'}">{'+'if week_chg>=0 else ''}{week_chg}%</span>
        </div>
        <div style="display:flex;justify-content:space-between">
          <span style="font-size:13px;color:#64748b">This month</span>
          <span style="font-size:15px;font-weight:700;color:{'#16a34a' if month_chg>=0 else '#dc2626'}">{'+'if month_chg>=0 else ''}{month_chg}%</span>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-label">Ensemble Confidence</div>
      <div class="conf-row">
        <div class="conf-item">
          <label><span>BUY</span><span>{round(buy_p*100)}%</span></label>
          <div class="bar-bg"><div class="bar-fill" style="width:{round(buy_p*100)}%;background:#16a34a"></div></div>
        </div>
        <div class="conf-item">
          <label><span>HOLD</span><span>{round(hold_p*100)}%</span></label>
          <div class="bar-bg"><div class="bar-fill" style="width:{round(hold_p*100)}%;background:#d97706"></div></div>
        </div>
        <div class="conf-item">
          <label><span>SELL</span><span>{round(sell_p*100)}%</span></label>
          <div class="bar-bg"><div class="bar-fill" style="width:{round(sell_p*100)}%;background:#dc2626"></div></div>
        </div>
      </div>
    </div>
  </div>

  <div class="chart-card">
    <div class="chart-header">
      <span class="chart-title">Corn Price — Last 180 Days</span>
      <div class="legend">
        <div class="leg-item"><div class="leg-dot" style="background:#3b82f6"></div>Price</div>
        <div class="leg-item"><div class="leg-dot" style="background:#f59e0b"></div>30-day avg</div>
        <div class="leg-item"><div class="leg-dot" style="background:#8b5cf6"></div>90-day avg</div>
        <div class="leg-item"><div class="leg-dot" style="background:#e2e8f0;width:16px"></div>Bollinger Bands</div>
      </div>
    </div>
    <div style="position:relative;height:320px">
      <canvas id="priceChart" role="img" aria-label="Corn price chart for last 180 days">Corn price data over the last 180 trading days.</canvas>
    </div>
  </div>

  <div class="chart-card">
    <div class="chart-header">
      <span class="chart-title">Model Breakdown</span>
      <span style="font-size:12px;color:#94a3b8">XGBoost 40% + LSTM 60% weighted ensemble</span>
    </div>
    <div class="model-grid">

      <div>
        <div class="model-col-label">XGBoost</div>
        <div class="mini-bar-row">
          <div class="mini-bar-item">
            <div class="mini-bar-header"><span>BUY</span><span>{xgb_buy}%</span></div>
            <div class="mini-bg"><div class="mini-fill" style="width:{xgb_buy}%;background:#16a34a"></div></div>
          </div>
          <div class="mini-bar-item">
            <div class="mini-bar-header"><span>HOLD</span><span>{xgb_hold}%</span></div>
            <div class="mini-bg"><div class="mini-fill" style="width:{xgb_hold}%;background:#d97706"></div></div>
          </div>
          <div class="mini-bar-item">
            <div class="mini-bar-header"><span>SELL</span><span>{xgb_sell}%</span></div>
            <div class="mini-bg"><div class="mini-fill" style="width:{xgb_sell}%;background:#dc2626"></div></div>
          </div>
        </div>
      </div>

      <div>
        <div class="model-col-label">LSTM</div>
        <div class="mini-bar-row">
          <div class="mini-bar-item">
            <div class="mini-bar-header"><span>BUY</span><span>{lstm_buy}%</span></div>
            <div class="mini-bg"><div class="mini-fill" style="width:{lstm_buy}%;background:#16a34a"></div></div>
          </div>
          <div class="mini-bar-item">
            <div class="mini-bar-header"><span>HOLD</span><span>{lstm_hold}%</span></div>
            <div class="mini-bg"><div class="mini-fill" style="width:{lstm_hold}%;background:#d97706"></div></div>
          </div>
          <div class="mini-bar-item">
            <div class="mini-bar-header"><span>SELL</span><span>{lstm_sell}%</span></div>
            <div class="mini-bg"><div class="mini-fill" style="width:{lstm_sell}%;background:#dc2626"></div></div>
          </div>
        </div>
      </div>

      <div>
        <div class="model-col-label">Ensemble</div>
        <div class="mini-bar-row">
          <div class="mini-bar-item">
            <div class="mini-bar-header"><span>BUY</span><span>{round(buy_p*100)}%</span></div>
            <div class="mini-bg"><div class="mini-fill" style="width:{round(buy_p*100)}%;background:#16a34a"></div></div>
          </div>
          <div class="mini-bar-item">
            <div class="mini-bar-header"><span>HOLD</span><span>{round(hold_p*100)}%</span></div>
            <div class="mini-bg"><div class="mini-fill" style="width:{round(hold_p*100)}%;background:#d97706"></div></div>
          </div>
          <div class="mini-bar-item">
            <div class="mini-bar-header"><span>SELL</span><span>{round(sell_p*100)}%</span></div>
            <div class="mini-bg"><div class="mini-fill" style="width:{round(sell_p*100)}%;background:#dc2626"></div></div>
          </div>
        </div>
      </div>

    </div>
  </div>

  <div class="footer">
    Powered by XGBoost + LSTM ensemble trained on 20 years of CBOT corn futures data &nbsp;·&nbsp;
    <strong>Not financial advice.</strong> Always do your own research before trading.
  </div>

</div>
<script>
const d = {data_json};
const ctx = document.getElementById('priceChart').getContext('2d');
new Chart(ctx, {{
  type: 'line',
  data: {{
    labels: d.dates,
    datasets: [
      {{
        label: 'Bollinger Upper',
        data: d.bb_upper,
        borderColor: '#e2e8f0',
        backgroundColor: 'rgba(226,232,240,0.3)',
        borderWidth: 1,
        borderDash: [4,4],
        pointRadius: 0,
        fill: '+1',
        tension: 0.3
      }},
      {{
        label: 'Bollinger Lower',
        data: d.bb_lower,
        borderColor: '#e2e8f0',
        borderWidth: 1,
        borderDash: [4,4],
        pointRadius: 0,
        fill: false,
        tension: 0.3
      }},
      {{
        label: '90-day avg',
        data: d.ma90,
        borderColor: '#8b5cf6',
        borderWidth: 2,
        pointRadius: 0,
        fill: false,
        tension: 0.3
      }},
      {{
        label: '30-day avg',
        data: d.ma30,
        borderColor: '#f59e0b',
        borderWidth: 2,
        pointRadius: 0,
        fill: false,
        tension: 0.3
      }},
      {{
        label: 'Price',
        data: d.prices,
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59,130,246,0.08)',
        borderWidth: 2.5,
        pointRadius: 0,
        fill: false,
        tension: 0.3
      }}
    ]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        callbacks: {{
          label: ctx => ctx.dataset.label + ': $' + ctx.parsed.y.toFixed(2)
        }}
      }}
    }},
    scales: {{
      x: {{
        ticks: {{ maxTicksLimit: 8, color: '#94a3b8', font: {{ size: 11 }} }},
        grid: {{ color: '#f1f5f9' }}
      }},
      y: {{
        ticks: {{ color: '#94a3b8', font: {{ size: 11 }}, callback: v => '$' + v }},
        grid: {{ color: '#f1f5f9' }}
      }}
    }}
  }}
}});
</script>
</body>
</html>"""

    with open(OUTPUT_PATH, "w") as f:
        f.write(html)
    print(f"OK: Dashboard saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print("Generating signal...")
    signal, hold_p, buy_p, sell_p, xgb_p, lstm_p = generate_signal(df)
    print(f"Signal: {signal}")
    build_html(df, signal, hold_p, buy_p, sell_p, xgb_p, lstm_p)
    print("Done!")