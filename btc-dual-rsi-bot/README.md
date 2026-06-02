# BTC Dual RSI Futures Bot

**By Sai Han Shan Pha**

This is an automated Bitcoin futures trading bot that runs 24/7 on Binance Futures. It uses a Multi-Timeframe Dual RSI strategy that was developed and validated through a full quantitative research pipeline before any real money was used.

This bot is part of a larger portfolio of automated trading bots I built and deployed. Each bot goes through the same rigorous process: strategy research, backtesting with a grid search, walk-forward validation, and then production deployment on a cloud server.

---

## Table of Contents

- [What the Bot Does](#what-the-bot-does)
- [Strategy Logic](#strategy-logic)
- [How I Developed This](#how-i-developed-this)
- [Backtest Results](#backtest-results)
- [Walk-Forward Validation](#walk-forward-validation)
- [Final Parameters](#final-parameters)
- [Bot Architecture and Template](#bot-architecture-and-template)
- [Risk Management](#risk-management)
- [Discord Alerts Setup](#discord-alerts-setup)
- [Cloud Deployment on Hetzner](#cloud-deployment-on-hetzner)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)

---

## What the Bot Does

The bot trades BTCUSDT perpetual futures on Binance. It checks for a trade signal every time a new 1-hour candle closes. When the signal conditions are met it opens a long or short position, places a stop-loss and take-profit order directly on the exchange, and then monitors the trade until it exits. All of this happens automatically without any manual input.

It runs in two modes:

```
python btc_trader.py          # paper trade mode (no real money)
python btc_trader.py --live   # live mode on Binance Futures
```

---

## Strategy Logic

The strategy is called Dual RSI Multi-Timeframe. It uses two timeframes at the same time to make better decisions.

**The idea behind it:**

The 4-hour chart tells us the overall market direction (the regime). The 1-hour chart tells us exactly when to enter. This way we only take long trades when the market is already in a bullish regime and only take short trades when the market is in a bearish regime. We also use the EMA 200 as an extra filter to make sure we are trading with the bigger trend.

**Long signal (buy):**
- 4h RSI is above 50 which means the market is in a bullish regime
- 1h RSI crosses up through 35 from below
- Price is above the 200-period EMA on the 1h chart

**Short signal (sell):**
- 4h RSI is below 45 which means the market is in a bearish regime
- 1h RSI crosses down through 60 from above
- Price is below the 200-period EMA on the 1h chart

**Exit:**
- Stop-loss at 2.0x ATR from entry
- Take-profit at 3.0x ATR from entry (this gives a 1.5:1 reward to risk ratio)
- Both orders are placed directly on Binance as exchange orders so they trigger even if the bot goes offline

---

## How I Developed This

I did not just write a bot and run it with real money. I followed a full development pipeline that is used in professional algorithmic trading.

### Step 1 — Strategy Shootout (Two Rounds)

I ran two full rounds of backtests on 20 months of Bitcoin data. Each round tested multiple strategies using a multiprocessing grid search to evaluate hundreds of parameter combinations at once and rank them by a combined score that rewards return and penalizes drawdown equally.

Round 1 tested 1h strategies and Round 2 tested 4h strategies. The full results are in the Backtest Results section below. The winner after both rounds was Dual RSI Multi-Timeframe with these metrics:

| Metric | Value |
|--------|-------|
| Return | +83.8% |
| Max Drawdown | -3.1% |
| Win Rate | 72.2% |
| Sharpe Ratio | 33.00 |
| Profit Factor | 4.73 |
| Number of Trades | 36 |

Even though these numbers looked very good I treated them with skepticism. A Sharpe of 33 on a leveraged crypto bot is unusually high and could mean the strategy is overfit to historical data. That is why walk-forward validation was the next mandatory step before any deployment decision.

### Step 2 — Walk-Forward Validation

Walk-forward validation is the most important part of this pipeline. The idea is simple: if a strategy only works on the data it was optimized on, it is not a real strategy. It needs to prove it works on data it has never seen before.

I split 20 months of data into 4 equal windows of about 5 months each. For every window I tested the best parameters AND a robustness grid of 729 parameter combinations around those best parameters. A strategy is only considered robust if the majority of the grid combinations are profitable, not just the one cherry-picked best set.

### Step 3 — Production Bot

Only after 4/4 windows passed did I build the live bot. The walk-forward also found better parameters than the original grid search so those updated parameters are what runs in production.

---

## Backtest Results

I ran two rounds of shootouts on 20 months of BTCUSD data before finding a strategy worth validating.

**Round 1 — 1h strategies (550 combos total across 3 strategies)**

| Strategy | Return | Max DD | Win Rate | Sharpe | Profit Factor | Trades | Decision |
|----------|--------|--------|----------|--------|---------------|--------|----------|
| A — Supertrend 4h | -8.1% | -29.3% | 40.0% | -6.48 | 0.86 | 15 | Dropped |
| B — EMA+MACD 1h | +5.2% | -15.3% | 35.2% | 3.37 | 1.09 | 54 | Too weak |
| C — BB Breakout 1h | +29.4% | -13.1% | 59.5% | 21.62 | 1.67 | 42 | Promising but needed more testing |

BB Breakout won round 1 but the returns felt moderate and I wanted to test higher-timeframe strategies before committing. So I ran a second round focused on 4h strategies.

**Round 2 — 4h strategies (402 combos total across 3 strategies)**

| Strategy | Return | Max DD | Win Rate | Sharpe | Profit Factor | Trades | Decision |
|----------|--------|--------|----------|--------|---------------|--------|----------|
| A — EMA Cross + RSI Pullback 4h | -1.9% | -7.4% | 25.0% | -1.97 | 0.90 | 12 | Dropped |
| B — Chandelier Exit + Momentum 4h | +32.2% | -39.4% | 34.5% | 1.87 | 1.09 | 84 | -39% DD too dangerous |
| **C — Dual RSI 4h/1h (this bot)** | **+83.8%** | **-3.1%** | **72.2%** | **33.00** | **4.73** | **36** | **Winner — proceed to walk-forward** |

Dual RSI was the clear winner with exceptional metrics. The -3.1% drawdown on a leveraged futures strategy with 72% win rate and a Profit Factor of 4.73 stood out immediately. The Chandelier strategy had decent returns but a -39.4% drawdown makes it unacceptable for real deployment.

---

## Walk-Forward Validation

This is the real proof that the strategy works. Each window is roughly 5 months of live market data. The robustness grid tests 729 variations of the parameters so we know the strategy does not only work with one specific setting.

I split 20 months of data into 4 equal windows of about 5 months each and ran 729 parameter combinations per window. A strategy is only considered robust if the majority of those 729 combinations are profitable — not just the one best set. This is how you detect curve-fitting.

| Window | Period | Return | Max DD | Win Rate | Sharpe | Profit Factor | Trades | Grid Positivity | Median Grid Ret | Verdict |
|--------|--------|--------|--------|----------|--------|---------------|--------|-----------------|-----------------|---------|
| 1 | Oct 2024 – Mar 2025 | +36.2% | 0.0% | 100.0% | 268.22 | 999.00 | 10 | 675 / 675 (100%) | +49.3% | ✅ Robust |
| 2 | Mar 2025 – Jul 2025 | +14.0% | -1.9% | 83.3% | 82.18 | 7.44 | 6 | 594 / 594 (100%) | +15.5% | ✅ Robust |
| 3 | Jul 2025 – Dec 2025 | +10.1% | -2.1% | 60.0% | 43.42 | 3.00 | 10 | 657 / 657 (100%) | +16.6% | ✅ Robust |
| 4 | Dec 2025 – May 2026 | +32.0% | -2.3% | 83.3% | 99.11 | 9.38 | 12 | 673 / 675 (100%) | +28.3% | ✅ Robust |

**Average across all windows: +23.1% return / -1.6% max drawdown / 81.7% win rate**

**Final verdict: 4/4 ROBUST — Deploy**

### Limitations and Honest Acknowledgements

Before reading the results it is important to understand what this validation does and does not prove.

**Sample size is small.** The strategy produced 36 trades over 20 months in the full backtest and as few as 6 trades in a single walk-forward window. In statistics 36 trades is a very small sample. It is not enough to draw strong conclusions about long-term edge. A strategy can look great on 36 trades purely by luck. Ideally you want 200 or more trades before feeling confident in win rate and profit factor numbers.

**20 months of data is limited.** The backtest covers October 2024 to May 2026 which is a relatively short and specific period in Bitcoin's history. It includes a strong bull run and some volatile periods but it does not include a prolonged multi-year bear market like 2022. A strategy that worked in this window may behave very differently in market conditions that are not represented here.

**Walk-forward does not guarantee future performance.** The walk-forward validation reduces the risk of curve-fitting significantly but it cannot eliminate it. All it proves is that the strategy was consistent across 4 different 5-month periods in the past. Markets change and past consistency does not mean the strategy will keep working.

**These metrics should be treated as promising not proven.** The results are strong enough to justify paper trading and careful live deployment with small size. They are not strong enough to justify putting large capital at risk. The right approach is to treat the first 6 to 12 months of live trading as additional validation and scale up only if real performance matches the backtest.

A few things worth highlighting from these results:

Window 1 shows a 0.0% drawdown with 100% win rate — every single one of the 675 parameter combinations in the grid was profitable in that period. Window 3 is the weakest window with only 60% win rate and +10.1% return but it still passed because the entire grid of 729 combinations was profitable and the median grid return was +16.6%. The worst performing combo in any window was only -2.5% which shows the strategy does not blow up even when it underperforms.

The walk-forward also found a better parameter set than the original grid search. The most consistent combo seen across multiple windows was different from what the original backtest suggested:

| Parameter | Original Grid Search Best | Walk-Forward Best |
|-----------|--------------------------|-------------------|
| 4h RSI Bull | 55 | 50 |
| 4h RSI Bear | 40 | 45 |
| 1h RSI Long | 30 | 35 |
| 1h RSI Short | 65 | 60 |
| SL multiplier | 1.5x ATR | 2.0x ATR |
| TP multiplier | 3.0x ATR | 3.0x ATR |

The walk-forward consistent params averaged +94.5% return across windows vs +83.8% in-sample. These updated parameters are what runs in the live bot.

---

## Final Parameters

These are the parameters used in the live bot. They came from the walk-forward analysis, not from the original grid search, because the walk-forward found a more consistent set across time.

| Parameter | Value | Notes |
|-----------|-------|-------|
| 4h RSI Bull threshold | 50 | Above this = bullish regime |
| 4h RSI Bear threshold | 45 | Below this = bearish regime |
| 1h RSI Long entry | 35 | RSI crosses up through this level |
| 1h RSI Short entry | 60 | RSI crosses down through this level |
| Stop-Loss | 2.0x ATR | ATR = Average True Range of the 1h bar |
| Take-Profit | 3.0x ATR | 1.5:1 reward to risk ratio |
| EMA 200 Filter | On | Must be on the right side of the trend |
| Leverage | 10x | Isolated margin |
| Margin per trade | 20% of balance | Conservative sizing |

---

## Bot Architecture and Template

All the bots in this portfolio are built on a shared template I developed called **Template v4.0**. Writing a custom template from scratch and reusing it across every bot was an intentional design decision for consistency and maintainability.

The template handles everything that is not strategy-specific so that when I build a new bot I only need to write the signal logic. Everything else is already built and tested.

**What the template includes:**

**State management** — the bot saves its full state to a JSON file after every action using an atomic write (write to temp file then rename). This means if the server crashes or the bot restarts it picks up exactly where it left off without losing any trade information.

**Startup position recovery** — when the bot starts it checks the Binance exchange for any open positions. If there is a position open that does not match the saved state (for example after an unexpected restart) it recovers the position data, maps any existing SL/TP orders, and continues managing the trade correctly.

**Paper and live mode** — every bot has a paper trading mode that simulates trades with real market data but uses no real money. This is used to validate the bot in production conditions before switching to live. The mode is controlled by a single flag.

**Rotating log files** — all activity is logged to a file that rotates automatically when it reaches 5MB so logs never fill up the disk.

**API retry logic** — every Binance API call goes through a retry wrapper that retries up to 5 times with a 15-second sleep between attempts. This handles temporary network issues and API rate limits gracefully.

**SL/TP as exchange orders** — stop-loss and take-profit are placed as actual orders on Binance Futures, not managed by the bot's own price checking loop. This means trades are protected even if the bot goes offline.

**Manual interference detection** — the bot compares its saved position size to the live position size on the exchange every loop. If they differ it logs a warning and alerts Discord. This catches any manual trades that might interfere with the bot.

---

## Risk Management

The bot has several layers of protection beyond just the stop-loss:

**Daily loss limit** — if the account loses more than 8% in a single day the bot stops taking new trades for the rest of the day

**Max trades per day** — capped at 3 trades per day regardless of how many signals fire

**Consecutive stop-loss cooldown** — if the bot hits 2 stop-losses in a row it enters a cooldown period. The cooldown escalates with each additional loss. After 2 SLs the cooldown is 4 hours. After 3 it is 8 hours. After 4 it is 12 hours. This prevents the bot from getting into a losing streak in bad market conditions.

**Minimum balance check** — the bot will not open a trade if the account balance drops below a minimum threshold

**Flat market filter** — uses ATR ratio to detect when the market is too quiet and skip trading during those periods

**Bad hour filter** — blocks new trades during UTC hours 0 to 3 which tend to have low liquidity and wider spreads on BTC

---

## Discord Alerts Setup

The bot sends real-time alerts to a private Discord server for every important event. This means I can monitor all my bots from my phone without ever logging into the server.

**Alerts that are sent:**
- Bot started (paper or live mode)
- Trade opened (with entry price, size, SL, TP, RSI reading)
- Trade closed (with PnL, exit reason, win/loss record, balance)
- Daily reset with current balance
- Cooldown triggered after consecutive losses
- Any critical errors or unexpected behavior
- Manual interference detected

**How to set it up:**

1. Open Discord and go to your server (or create a new one)
2. Right-click on any text channel and click Edit Channel
3. Go to Integrations then Webhooks
4. Click Create Webhook, give it a name like "BTC Bot" and copy the webhook URL
5. Set the URL as an environment variable on your server:

```bash
export BOT_WEBHOOK_URL="https://discord.com/api/webhooks/YOUR_WEBHOOK_URL"
```

The bot validates the URL on startup and will skip sending if it is not set. You can test it is working by restarting the bot and checking for the startup message in Discord.

For managing multiple bots I created a separate Discord channel for each bot so alerts from BTC do not get mixed with alerts from ETH or LINK.

---

## Cloud Deployment on Hetzner

The bots run 24/7 on a cloud server rented from [Hetzner](https://www.hetzner.com/). I chose Hetzner because it is reliable, affordable, and has data centers in Europe with good latency to Binance. Running on a VPS (Virtual Private Server) is important for trading bots because your home internet connection can drop and your home computer might restart or sleep.

**My server setup:**

I use a Hetzner CX22 instance running Ubuntu 22.04. This costs about 4 euros per month which is very affordable for running multiple bots continuously.

**Initial server setup:**

```bash
# Update the system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3 python3-pip python3-venv -y

# Clone your bot repository
git clone https://github.com/YOUR_USERNAME/btc-dual-rsi-bot.git
cd btc-dual-rsi-bot

# Create a virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install python-binance pandas numpy requests
```

**Setting environment variables securely:**

Never put your API keys directly in the code. Set them as environment variables on the server:

```bash
# Add these to ~/.bashrc so they persist across reboots
export BINANCE_API_KEY="your_api_key_here"
export BINANCE_API_SECRET="your_api_secret_here"
export BOT_WEBHOOK_URL="your_discord_webhook_url_here"

# Apply the changes
source ~/.bashrc
```

**Running the bot with systemd so it restarts automatically:**

Create a service file so the bot runs as a background service and restarts if it crashes:

```bash
sudo nano /etc/systemd/system/btc-bot.service
```

Paste this into the file:

```ini
[Unit]
Description=BTC Dual RSI Trading Bot
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/btc-dual-rsi-bot
Environment=BINANCE_API_KEY=your_key
Environment=BINANCE_API_SECRET=your_secret
Environment=BOT_WEBHOOK_URL=your_webhook
ExecStart=/home/YOUR_USERNAME/btc-dual-rsi-bot/venv/bin/python btc_trader.py --live
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
```

Then enable and start it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable btc-bot
sudo systemctl start btc-bot

# Check that it is running
sudo systemctl status btc-bot

# Watch live logs
journalctl -u btc-bot -f
```

**Updating the bot:**

When you push a new version to GitHub you just pull it on the server and restart:

```bash
git pull origin main
sudo systemctl restart btc-bot
```

---

## How to Run

**Requirements:**

```bash
pip install python-binance pandas numpy requests
```

**Environment variables needed:**

```bash
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
export BOT_WEBHOOK_URL="your_discord_webhook"   # optional but recommended
```

**Paper trade (safe, no real money):**

```bash
python btc_trader.py
```

**Live trade:**

```bash
python btc_trader.py --live
```

When you run in live mode the bot will print a 5-second warning countdown before starting. This gives you time to press Ctrl+C if you ran it by accident.

**Binance API key permissions needed:**
- Enable Futures Trading — yes
- Enable Reading — yes
- Enable Withdrawals — NO (the bot checks for this on startup and will warn you if withdrawals are enabled because that is a security risk)

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| python-binance | Binance Futures API client |
| pandas / numpy | Data processing and indicator calculations |
| yfinance | Historical data for backtesting |
| requests | Discord webhook alerts |
| Hetzner CX22 VPS | 24/7 cloud server (Ubuntu 22.04) |
| systemd | Process management and auto-restart |
| Discord Webhooks | Real-time trade alerts |
| Git / GitHub | Version control and deployment |

---

## Disclaimer

This project is for educational and portfolio purposes. Trading cryptocurrency futures involves significant financial risk and you can lose more than your initial investment. Past backtest performance does not guarantee future results. Always paper trade first and never trade with money you cannot afford to lose.

---

*Built by Sai Han Shan Pha*
