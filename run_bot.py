#!/usr/bin/env python3
"""Entry point for the Tori Trades live trading bot.

Set your Binance API credentials in a .env file or as environment variables:

    BINANCE_API_KEY=your_api_key
    BINANCE_API_SECRET=your_api_secret

Then run:

    python run_bot.py

The bot connects to Binance LIVE USDT-M Futures (NOT testnet) and trades
BTC, ETH, SOL, XRP on the 4-hour timeframe using the Tori Trades
Trendline Strategy.

Press Ctrl+C to stop gracefully.
"""

import logging
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from tori_trades_bot.config import BINANCE_API_KEY, BINANCE_API_SECRET, SYMBOLS
from tori_trades_bot.live_bot import ToriTradesBot


def main() -> None:
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("tori_bot.log", mode="a"),
        ],
    )

    # Validate API keys
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        print("ERROR: BINANCE_API_KEY and BINANCE_API_SECRET must be set.")
        print("Create a .env file with:")
        print("  BINANCE_API_KEY=your_key")
        print("  BINANCE_API_SECRET=your_secret")
        sys.exit(1)

    print("=" * 60)
    print("  Tori Trades Trendline Strategy — Live Bot")
    print("  Instruments: " + ", ".join(SYMBOLS))
    print("  Exchange:    Binance USDT-M Futures (LIVE)")
    print("  Strategy:    4H Trendline Bounce + Break")
    print("=" * 60)
    print()

    bot = ToriTradesBot(symbols=SYMBOLS)
    bot.run()


if __name__ == "__main__":
    main()
