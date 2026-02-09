# Autonomous Trading Bot - Binance USDT-M Futures

This repository contains a production-grade autonomous trading bot for Binance USDT-Margined Futures, developed in multiple phases.

## Architecture Overview

```ascii
[ CSV Data ] --> [ Data Pipeline ] --> [ Features ] --> [ Regime Filter ] --> [ Ensemble Model ] --> [ Risk Manager ]
      ^                |                    |                |                      |                  |
      |                v                    v                v                      v                  v
  OHLCV, BTC,     Load & Merge         77 Technical     GMM-based Market      LightGBM + XGBoost    Half Kelly, SL/TP,
  Funding Rates   Clean NaNs           Indicators       State Classifier      Optuna Optimized      Circuit Breakers
```

## Completed Phases

### Phase 1: Data Pipeline & Feature Engineering
- **Modular Ingestion**: Efficient loading and merging of OHLCV and funding rate data.
- **77 Features**: Comprehensive coverage of Trend, Momentum, Volatility, Volume, Microstructure, Temporal, and Meta categories.
- **Strict Causality**: Rolling z-score normalization (lookback=168) and purged K-fold logic to prevent look-ahead bias.
- **Target Engineering**: Triple Barrier Method (TBM) with ATR-based barriers.

### Phase 2: Model Architecture
- **Ensemble System**: Combined LightGBM and XGBoost classifiers. Execution requires directional agreement and a 0.58 confidence threshold.
- **Regime Filter**: Unsupervised Gaussian Mixture Model (GMM) classifying markets into TRENDING, RANGING, and VOLATILE/CHAOTIC.
- **Walk-Forward Framework**: 2000-candle train, 500-candle test, and 500-candle step windows with periodic Optuna hyperparameter re-optimization.
- **Sample Weighting**: Inverse ATR weighting to prioritize signals in cleaner, low-volatility environments.

### Phase 3: Risk Management Engine
- **Fractional Kelly**: Position sizing using Half Kelly Criterion based on trailing performance.
- **Dynamic Leverage**: Regime-based base leverage (up to 10x) further scaled by model confidence.
- **Circuit Breakers**: Daily/Weekly loss limits and two-tier drawdown protection.
- **Advanced Exits**: ATR-based dynamic SL/TP, trailing stops, and time-based exits.

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
2. Run Phase 1 analysis:
   ```bash
   python phase1_runner.py
   ```
3. Run Phase 2 Walk-Forward backtest:
   ```bash
   python phase2_runner.py
   ```

## Roadmap

- **Phase 4: Event-Driven Backtester** (Current): Candle-by-candle simulation with fee and funding rate modeling, Monte Carlo, and sensitivity analysis.
- **Phase 5: Live Trading Infrastructure**: Async execution, API integration, and Telegram notifications.

## Risk Warning
This is experimental software. Trading cryptocurrencies involves significant risk of capital loss.
