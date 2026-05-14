\# Alpha Signals — Q/USDT Swing Strategy



A rule-based trading signal system for Q/USDT (Quack AI) on Binance Futures.

Tells you whether to \*\*BUY\*\*, \*\*SELL\*\*, or \*\*HOLD\*\* based on technical indicators,

with support/resistance-based take profit and stop loss levels.



\---



\## What It Does



\- Fetches 5-minute OHLCV candles from Binance Futures (free, no API key needed)

\- Engineers technical features: RSI, MACD, EMA, Bollinger Bands, volume ratio

\- Generates BUY/SELL/HOLD signals using a rule-based swing strategy

\- Calculates take profit and stop loss from real support/resistance levels

\- Runs a backtest with fees and slippage included

\- Shows everything on a live local dashboard that refreshes every 5 minutes



\---



\## Backtest Results (Nov 2025 – May 2026)



| Metric | Result |

|---|---|

| Starting Capital | $1,000 |

| Final Capital | $1,829 |

| Total Return | +83% |

| Win Rate | 75% |

| Total Trades | 8 |

| Avg Win | $231 |

| Avg Loss | $-272 |



> Note: 8 trades over 6 months. Q/USDT launched Sep 2025 so data is limited.

> The April 2026 crypto market crash affected results.



\---



\## Project Structure

alpha-signals-crypto/

├── collect\_data.py      # Fetch Q/USDT + BTC/USDT 5m candles from Binance

├── features.py          # Calculate RSI, MACD, EMA, Bollinger Bands, volume

├── strategy.py          # Generate BUY/SELL/HOLD signals

├── backtest.py          # Backtest strategy with fees + slippage

├── daily\_signal.py      # Print today's signal in terminal

├── levels.py            # Calculate support/resistance, TP and SL levels

├── app.py               # Streamlit live dashboard

└── requirements.txt



\---



\## Setup



\*\*1. Clone the repo:\*\*

```bash

git clone https://github.com/Durianpandahedgehog/projects

cd projects/alpha-signals-crypto

```



\*\*2. Install dependencies:\*\*

```bash

pip install -r requirements.txt

```



\*\*3. Fetch data:\*\*

```bash

python collect\_data.py

```



\*\*4. Run backtest:\*\*

```bash

python features.py

python strategy.py

python backtest.py

```



\*\*5. Get today's signal:\*\*

```bash

python daily\_signal.py

```



\*\*6. Launch dashboard:\*\*

```bash

streamlit run app.py

```



Opens at `http://localhost:8501`



\---



\## Strategy Logic



\*\*BUY when all conditions are true:\*\*

\- RSI-7 < 35 — oversold

\- Price above EMA-21 — uptrend

\- Price above EMA-50 — macro uptrend

\- MACD above signal line — momentum turning up

\- Volume ratio > 1.5x — volume confirming move

\- BTC not crashing (> -0.5%) — market not in freefall



\*\*SELL when all conditions are true:\*\*

\- RSI-7 > 65 — overbought

\- Price below EMA-21 — downtrend

\- MACD below signal line — momentum turning down

\- Volume ratio > 1.5x — volume confirming move

\- BTC not pumping (< +0.5%)



\*\*Take Profit / Stop Loss:\*\*

Calculated from nearest resistance and support levels

in the last 500 candles. Only shows a trade setup

if risk/reward ratio is at least 1.5:1.



\---



\## Tech Stack



\- Python 3

\- ccxt — Binance Futures data

\- pandas — data processing

\- Streamlit — live dashboard

\- Plotly — candlestick charts



\---



\## Part of Alpha Signals



This project is part of the

\[Alpha Signals](https://github.com/Durianpandahedgehog/projects)

portfolio — a collection of rule-based and ML trading signal systems.



Also see: \*\*alpha-signals-corn\*\* — swing trading signals for US corn futures

using XGBoost + LSTM ensemble models.

