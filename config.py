import os
import yaml
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR
CONFIG_PATH = BASE_DIR / "config.yaml"

# Default configuration
DEFAULT_CONFIG = {
    'binance': {
        'api_key': os.getenv('BINANCE_API_KEY', ''),
        'api_secret': os.getenv('BINANCE_API_SECRET', '')
    },
    'telegram': {
        'token': os.getenv('TELEGRAM_TOKEN', ''),
        'chat_id': os.getenv('TELEGRAM_CHAT_ID', '')
    },
    'trading': {
        'primary_asset': 'XRPUSDT',
        'secondary_asset': 'BTCUSDT',
        'timeframe': '1h',
        'initial_equity': 1000.0,
        'is_live': False
    }
}

def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            user_config = yaml.safe_load(f)
            # Deep merge could be better, but simple update for now
            return {**DEFAULT_CONFIG, **user_config}
    return DEFAULT_CONFIG

config_data = load_config()

# Assets
PRIMARY_ASSET = config_data['trading']['primary_asset']
SECONDARY_ASSET = config_data['trading']['secondary_asset']
TIMEFRAME = config_data['trading']['timeframe']

# API Keys
BINANCE_API_KEY = config_data['binance']['api_key']
BINANCE_API_SECRET = config_data['binance']['api_secret']
TELEGRAM_TOKEN = config_data['telegram']['token']
TELEGRAM_CHAT_ID = config_data['telegram']['chat_id']

# Feature Parameters
EMA_PAIRS = [(8, 21), (21, 55), (55, 200)]
LINREG_LOOKBACKS = [20, 50, 100]
ROC_LOOKBACKS = [6, 12, 24, 48]
ATR_PERIOD = 14
RSI_PERIOD = 14
MACD_PARAMS = (12, 26, 9)
BB_PARAMS = (20, 2)
SUPERTREND_PARAMS = [(10, 3), (20, 5)]
ROLLING_Z_LOOKBACK = 168
BTC_CORR_LOOKBACKS = [24, 72, 168]
SHARPE_LOOKBACKS = [24, 72, 168]

# Target Parameters
BARRIER_ATR_MULT = 1.5
BARRIER_TIME_LIMIT = 12

# Preprocessing
CORRELATION_THRESHOLD = 0.95

# Model Training
PURGE_GAP = 12
TRAIN_WINDOW = 2000
TEST_WINDOW = 500
STEP_SIZE = 500

# Columns that should NEVER be standardized
EXCLUDE_COLS = [
    'open', 'high', 'low', 'close', 'volume',
    'market_open', 'market_high', 'market_low', 'market_close', 'market_volume',
    'fundingRate', 'vwap', 'atr_barrier', 'atr',
    'last_24h_high', 'last_24h_low',
    'triple_barrier_label', 'forward_return_12h', 'forward_price_diff', 'risk_adj_return',
    'signal', 'confidence', 'regime'
]

# Risk Management
MAX_LEVERAGE = 10
MAX_POSITION_SIZE = 0.30
TAKER_FEE = 0.0004
SLIPPAGE = 0.0001
ROUND_TRIP_COST = (TAKER_FEE + SLIPPAGE) * 2
MIN_NOTIONAL = 5.0
