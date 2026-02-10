"""Configuration for the Tori Trades Trendline Strategy bot."""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Binance API (live USDT-M futures — NOT testnet)
# ---------------------------------------------------------------------------
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
BASE_URL = "https://fapi.binance.com"

# ---------------------------------------------------------------------------
# Trading universe
# ---------------------------------------------------------------------------
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]

# ---------------------------------------------------------------------------
# Strategy parameters (aligned with Tori Trades Playbook)
# ---------------------------------------------------------------------------
# The playbook specifies the 4-hour timeframe
TIMEFRAME = "4h"

# Swing detection: number of bars on each side to confirm a swing point
SWING_ORDER = 5

# Minimum bars a trendline must span (1 week on 4h = 7*6 = 42 bars)
MIN_TRENDLINE_BARS = 42

# Tolerance for a candle to count as "touching" the trendline (ATR multiple)
TOUCH_TOLERANCE_ATR = 0.25

# Threshold: a close beyond the line by this ATR multiple counts as a "break"
BREAK_THRESHOLD_ATR = 0.10

# Buffer added to stop-loss beyond the trendline to avoid wick stop-outs
WICK_BUFFER_ATR = 0.40

# Maximum allowed distance between entry and safety line (ATR multiple).
# If the distance exceeds this, the trade is skipped (playbook says "skip
# if risk is too far").
MAX_RISK_ATR = 3.0

# ATR look-back period (number of 4h bars)
ATR_PERIOD = 14

# Maximum slope — reject near-vertical trendlines (price % per bar)
MAX_SLOPE_PCT_PER_BAR = 0.015  # 1.5 % per 4-hour bar

# Maximum number of close-violations allowed while building a trendline
MAX_VIOLATIONS = 2

# ---------------------------------------------------------------------------
# Risk / position sizing
# ---------------------------------------------------------------------------
RISK_PER_TRADE_PCT = 1.0        # risk 1 % of account per trade
DEFAULT_LEVERAGE = 5
MARGIN_TYPE = "CROSSED"         # CROSSED or ISOLATED

# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------
INITIAL_CAPITAL = 10_000.0      # USDT starting balance
COMMISSION_PCT = 0.04           # Binance futures taker fee 0.04 %
FUNDING_RATE_ENABLED = True     # account for funding rates in backtest

# ---------------------------------------------------------------------------
# Live bot
# ---------------------------------------------------------------------------
POLL_INTERVAL_SECONDS = 15      # how often the bot checks for new candles
KLINE_LIMIT = 500               # candles to fetch from Binance per request
