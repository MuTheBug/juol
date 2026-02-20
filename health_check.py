import os
import sys
import pandas as pd
import sqlite3
from datetime import datetime, timedelta

def check_bot_health():
    print("=== BOT HEALTH CHECK ===")

    # 1. Check Log File
    if os.path.exists("bot.log"):
        last_modified = datetime.fromtimestamp(os.path.getmtime("bot.log"))
        print(f"Log file last updated: {last_modified}")
        if datetime.now() - last_modified > timedelta(hours=2):
            print("CRITICAL: Log file has not been updated in over 2 hours!")
    else:
        print("WARNING: Log file not found.")

    # 2. Check SQLite DB
    if os.path.exists("trades.db"):
        with sqlite3.connect("trades.db") as conn:
            preds = pd.read_sql("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 5", conn)
            if not preds.empty:
                print(f"Latest prediction: {preds.iloc[0]['timestamp']} (Signal: {preds.iloc[0]['signal']})")
            else:
                print("WARNING: No predictions found in database.")
    else:
        print("WARNING: Trade database not found.")

if __name__ == "__main__":
    check_bot_health()
