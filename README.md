# Autonomous Trading Bot - Binance USDT-M Futures

This repository contains a production-grade autonomous trading bot for Binance USDT-Margined Futures. It uses an ensemble of machine learning models (LightGBM & XGBoost), sophisticated risk management (Half Kelly, circuit breakers), and a GMM-based market regime filter.

## Architecture Overview

```ascii
[ CSV Data ] --> [ Data Pipeline ] --> [ Features ] --> [ Regime Filter ] --> [ Ensemble Model ] --> [ Risk Manager ]
      ^                |                    |                |                      |                  |
      |                v                    v                v                      v                  v
  OHLCV, BTC,     Load & Merge         77 Technical     GMM-based Market      LightGBM + XGBoost    Half Kelly, SL/TP,
  Funding Rates   Clean NaNs           Indicators       State Classifier      Optuna Optimized      Circuit Breakers
```

## Features & Implementation

### 1. Data Pipeline & Feature Engineering
- **Modular Ingestion**: Efficient loading and merging of OHLCV and funding rate data.
- **77 Features**: Comprehensive coverage across Trend, Momentum, Volatility, Volume, Microstructure, Temporal, and Meta categories.
- **Strict Causality**: Rolling z-score normalization (lookback=168) excluding prices/ATR/targets to prevent look-ahead bias.
- **Target Engineering**: Triple Barrier Method (TBM) with dynamic ATR-based barriers and risk-adjusted continuous targets.

### 2. Model Architecture
- **Ensemble System**: Combined LightGBM and XGBoost classifiers. Execution requires directional agreement and a 0.58 confidence threshold.
- **Regime Filter**: Unsupervised Gaussian Mixture Model (GMM) classifying markets into TRENDING, RANGING, and VOLATILE.
- **Walk-Forward Framework**: Robust validation scheme (2000-candle train, 500-candle test) with periodic Optuna hyperparameter re-optimization (targeting Calmar ratio).

### 3. Risk Management Engine
- **Fractional Kelly**: Position sizing using Half Kelly Criterion based on trailing 50-trade performance.
- **Dynamic Leverage**: Regime-based base leverage (Trending 10x, Ranging 5x, Volatile 3x) scaled by model confidence.
- **Circuit Breakers**: Multi-tier protection including Daily (-3%) / Weekly (-7%) loss limits and Drawdown-based sizing reductions.
- **Advanced Exits**: ATR-based dynamic SL (1.5x) and TP (2.5x), trailing stops, and 24h time-based exits.

### 4. Backtesting & Robustness
- **Event-Driven Engine**: Candle-by-candle simulation using High/Low for realistic SL/TP checks.
- **Correct Metrics**: Annualized Sharpe (5% risk-free adjusted), CAGR, Sortino, Calmar, and Drawdown analysis.
- **Robustness Tests**: 1000-iteration Monte Carlo sequence reshuffling and SL/TP sensitivity heatmaps.

### 5. Live Trading Infrastructure
- **Async Execution**: Hourly loop synchronized with candle close (:01 past).
- **Order Management**: Automatic precision and lot-size formatting based on exchange rules.
- **Failure Resilience**: 3-tier retry logic with exponential backoff and instant Telegram alerts.
- **Persistent Audit**: SQLite-based logging of every trade and model prediction.

---

## Installation Guide

### 1. Prerequisites
- Python 3.10 or higher
- A Binance account with Futures enabled
- A Telegram Bot (optional, for notifications)

### 2. Setup Environment
```bash
# Clone the repository (or extract files)
cd autonomous-trading-bot

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Copy the template and fill in your details:
```bash
cp config.yaml.template config.yaml
```
Edit `config.yaml`:
- **binance**: API keys (Use Testnet first if desired)
- **telegram**: Bot token and Chat ID
- **trading**: Adjust asset pairs and initial equity

---

## Usage Instructions

### 1. Data Preparation
Place your OHLCV and funding rate CSV files in the root directory. Expected format: `[ASSET]_1h_klines.csv` and `[ASSET]_funding_8h.csv`.

### 2. Run Backtest & Analysis
To verify the strategy and generate performance reports:
```bash
python phase4_runner.py
```
This will produce:
- `backtest_trades.csv`: Detailed trade log.
- `backtest_equity_curve.csv`: Periodic equity balance.
- `backtest_report.png`: Equity curve and drawdown plots.
- `monthly_heatmap.png`: Returns by month/year.
- `sensitivity_heatmap.png`: Parameter stability analysis.

### 3. Run Live Bot
**WARNING: Ensure you have tested the strategy thoroughly and have configured `config.yaml` correctly.**

To start the bot in live/dry-run mode (set `is_live` in config):
```bash
python main.py
```
The bot will initialize, fetch the latest balance, and enter its hourly loop.

---

## Deployment (Production)

### Systemd Service (Linux)
To ensure the bot runs continuously and restarts on failure, create a systemd service:

1. Create `/etc/systemd/system/trading-bot.service`:
```ini
[Unit]
Description=Binance Trading Bot
After=network.target

[Service]
User=your-user
WorkingDirectory=/path/to/bot
ExecStart=/path/to/bot/venv/bin/python main.py
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
```

2. Start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable trading-bot
sudo systemctl start trading-bot
```

### Docker
(Optional) A `Dockerfile` can be provided for containerized deployment.

---

## Risk Warning
This software is for educational and research purposes only. Trading cryptocurrency futures involves significant risk of capital loss. The developers are not responsible for any financial losses incurred through the use of this bot. **Never trade money you cannot afford to lose.**
