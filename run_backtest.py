#!/usr/bin/env python3
"""Run the Tori Trades Trendline Strategy backtest on all available data."""

import sys
import pathlib

# Ensure the project root is on the path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from tori_trades_bot.backtester import run_backtest, build_report


def main() -> None:
    data_dir = pathlib.Path(__file__).resolve().parent
    print("Tori Trades Trendline Strategy — Backtester")
    print("=" * 50)
    print(f"Data directory: {data_dir}")
    print()

    results = run_backtest(data_dir=data_dir)
    report = build_report(results)
    print(report)

    # Save report to file
    report_path = data_dir / "backtest_report.txt"
    report_path.write_text(report)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
