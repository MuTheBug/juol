import requests
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.enabled = bool(token and chat_id)
        if not self.enabled:
            logger.warning("Telegram Notifier disabled: token or chat_id missing.")

    def send_message(self, message: str):
        if not self.enabled:
            return

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': 'Markdown'
        }

        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")

    def notify_trade_open(self, asset: str, side: str, price: float, size: float,
                          leverage: float, sl: float, tp: float, confidence: float, regime: str):
        msg = (
            f"🚀 *TRADE OPENED: {asset}*\n"
            f"───────────────────\n"
            f"Side: `{side}`\n"
            f"Price: `{price:.4f}`\n"
            f"Size: `${size:.2f}`\n"
            f"Leverage: `{leverage}x`\n"
            f"SL: `{sl:.4f}` | TP: `{tp:.4f}`\n"
            f"Confidence: `{confidence:.2%}`\n"
            f"Regime: `{regime}`"
        )
        self.send_message(msg)

    def notify_trade_close(self, asset: str, exit_price: float, pnl_usd: float,
                           pnl_pct: float, duration: int, reason: str):
        emoji = "✅" if pnl_usd > 0 else "❌"
        msg = (
            f"{emoji} *TRADE CLOSED: {asset}*\n"
            f"───────────────────\n"
            f"Reason: `{reason}`\n"
            f"Exit Price: `{exit_price:.4f}`\n"
            f"P&L: `${pnl_usd:.2f}` (`{pnl_pct:.2%}`)\n"
            f"Duration: `{duration}h`"
        )
        self.send_message(msg)

    def notify_error(self, error: str):
        msg = f"⚠️ *BOT ERROR*\n───────────────────\n`{error}`"
        self.send_message(msg)

    def notify_status(self, equity: float, daily_pnl: float, regime: str, drawdown: float):
        msg = (
            f"📊 *DAILY STATUS UPDATE*\n"
            f"───────────────────\n"
            f"Equity: `${equity:.2f}`\n"
            f"Daily P&L: `${daily_pnl:.2f}`\n"
            f"Current Regime: `{regime}`\n"
            f"Current Drawdown: `{drawdown:.2%}`"
        )
        self.send_message(msg)
