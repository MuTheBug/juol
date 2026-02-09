import sqlite3
import pandas as pd
from datetime import datetime

class TradeLogger:
    def __init__(self, db_path: str = "trades.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    timestamp TEXT,
                    asset TEXT,
                    direction TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    size REAL,
                    leverage REAL,
                    fee REAL,
                    funding_paid REAL,
                    pnl_gross REAL,
                    pnl_net REAL,
                    confidence REAL,
                    regime TEXT,
                    hold_duration INTEGER,
                    exit_reason TEXT,
                    equity_after REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    timestamp TEXT,
                    asset TEXT,
                    signal INTEGER,
                    confidence REAL,
                    regime TEXT
                )
            """)

    def log_trade(self, trade_data: dict):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO trades VALUES (
                    :timestamp, :asset, :direction, :entry_price, :exit_price,
                    :size, :leverage, :fee, :funding_paid, :pnl_gross, :pnl_net,
                    :confidence, :regime, :hold_duration, :exit_reason, :equity_after
                )
            """, trade_data)

    def log_prediction(self, pred_data: dict):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO predictions VALUES (
                    :timestamp, :asset, :signal, :confidence, :regime
                )
            """, pred_data)
