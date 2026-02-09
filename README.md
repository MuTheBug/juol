# Autonomous Trading Bot - Phase 1: Data Pipeline & Feature Engineering

This repository contains the first phase of a production-grade autonomous trading bot for Binance USDT-Margined Futures.

## Architecture Overview (Phase 1)

```ascii
[ CSV Data ] --> [ data_pipeline.py ] --> [ features.py ] --> [ Preprocessing ] --> [ Analysis/Output ]
      ^                |                         |                  |                      |
      |                v                         v                  v                      v
  OHLCV, BTC,     Load & Merge            Calculate 70+        Rolling Z-Score,      MI & Permutation
  Funding Rates   Clean NaNs              Technical Indicators  Correlation Drop      Importance
```

## Modular Components

### 1. `config.py`
Centralized configuration for all hyperparameters, including EMA periods, ATR lookbacks, risk parameters, and preprocessing thresholds. This ensures consistency across the entire pipeline.

### 2. `data_pipeline.py`
- **Data Ingestion**: Loads 1H klines and 8H funding rates. Automatically merges secondary asset data (BTCUSDT) for market regime reference.
- **Target Engineering**:
    - **Triple Barrier Method**: Implements a forward-looking labeling system. Barriers are set at $\pm 1.5 \times ATR(14)$, with a 12-hour timeout.
    - **Risk-Adjusted Return**: Calculates (12H Forward Return / Current ATR) as a continuous target for potential regression models.
- **Preprocessing**:
    - **Rolling Z-Scores**: Features are standardized using a 168-hour (1 week) rolling window. This is critical to prevent "future look-ahead bias" that occurs with global standardization.
    - **Correlation Filter**: Automatically removes features with a Pearson correlation > 0.95 to reduce redundancy and model complexity.

### 3. `features.py`
Engineers 77 distinct features across 7 categories:
- **Trend**: EMA crosses, Linear Regression slopes, ADX, Supertrend, VWAP distance.
- **Momentum**: RSI (with divergence detection), MACD, Stochastic RSI, ROC, Williams %R.
- **Volatility**: ATR, Bollinger Bands, Keltner Channels, Parkinson Volatility, Volatility Regime Classifier.
- **Volume**: OBV, Volume SMA ratio, VPT, Accumulation/Distribution.
- **Microstructure**: Funding rate MA/Z-score, BTC correlation, BTC RSI.
- **Temporal**: Cyclical encoding of hours/days, Session one-hot encoding, Time since extremes.
- **Meta**: Rolling Sharpe ratios, Distance from 24H High/Low, Consecutive candle counts.

### 4. `phase1_runner.py`
The orchestration script that runs the full pipeline, prints feature/sample counts, and performs feature importance analysis using:
- **Mutual Information**: Captures non-linear dependencies between features and the Triple Barrier label.
- **Permutation Importance**: Measures feature impact using a baseline Random Forest classifier.

## Design Decisions

- **Avoidance of Look-ahead Bias**: All calculations are strictly causal. Rolling z-scores and shifted targets ensure the model only trains on information that would have been available at the time.
- **ATR-Based Barriers**: Using ATR for barriers instead of fixed percentages makes the labels volatility-aware, which is essential for crypto markets where volatility is non-stationary.
- **Standardization**: Rolling z-scores handle non-stationary features better than global scaling, especially for indicators that aren't naturally bounded (like volume or VPT).

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Phase 1 pipeline:
   ```bash
   python phase1_runner.py
   ```
   This will generate a `feature_correlation.png` and output the top features by importance.

3. Run unit tests:
   ```bash
   python test_pipeline.py
   ```

## Risk Warning
This is experimental software. Trading cryptocurrencies involves significant risk of capital loss.
