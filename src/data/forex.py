"""
Forex Data Collector Module

Collects forex data from Yahoo Finance and ExchangeRate-API.
Supports major currency pairs and cross-rate calculations.

Features:
- Real-time forex quotes
- Historical forex data
- Currency conversion
- Cross-rate calculations
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List

import pandas as pd
import numpy as np
import yfinance as yf
import requests

from src.core.utils import retry_with_backoff
from src.core.exceptions import DataCollectionError

logger = logging.getLogger(__name__)

FOREX_PAIRS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "NZDUSD": "NZDUSD=X",
    "EURGBP": "EURGBP=X",
    "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
    "EURAUD": "EURAUD=X",
    "EURCAD": "EURCAD=X",
    "EURCHF": "EURCHF=X",
    "AUDJPY": "AUDJPY=X",
    "CADJPY": "CADJPY=X",
    "CHFJPY": "CHFJPY=X",
    "XAUUSD": "XAUUSD=X",
    "XAGUSD": "XAGUSD=X",
}

CURRENCY_SYMBOLS = {
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "CHF": "Fr",
    "CAD": "C$",
    "AUD": "A$",
    "NZD": "NZ$",
}


class ForexDataCollector:
    """Collects forex data from various sources."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Forex Data Collector.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

    def _normalize_pair(self, pair: str) -> str:
        """Normalize forex pair to Yahoo Finance format."""
        pair = pair.upper().replace("/", "").replace("-", "")

        if pair in FOREX_PAIRS:
            return FOREX_PAIRS[pair]

        if "=" not in pair:
            pair = pair + "=X"

        return pair

    def get_quote(self, pair: str) -> Dict:
        """
        Get real-time quote for a forex pair.

        Args:
            pair: Forex pair (e.g., 'EURUSD', 'EUR/USD')

        Returns:
            Dictionary with quote data
        """
        yf_pair = self._normalize_pair(pair)

        try:
            ticker = yf.Ticker(yf_pair)
            info = ticker.info

            current = info.get("currentPrice") or info.get("previousClose")
            bid = info.get("bid")
            ask = info.get("ask")
            day_high = info.get("dayHigh")
            day_low = info.get("dayLow")
            volume = info.get("volume")

            return {
                "pair": pair.upper().replace("=X", ""),
                "symbol": yf_pair,
                "current": current,
                "bid": bid,
                "ask": ask,
                "spread": (ask - bid) if (ask and bid) else None,
                "high": day_high,
                "low": day_low,
                "volume": volume,
                "previous_close": info.get("previousClose"),
            }

        except Exception as e:
            logger.error(f"Error fetching forex quote for {pair}: {e}")
            return {"pair": pair.upper(), "error": str(e)}

    def get_historical(
        self, pair: str, period: str = "1y", interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get historical forex data.

        Args:
            pair: Forex pair
            period: Period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Interval (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)

        Returns:
            DataFrame with OHLCV data
        """
        yf_pair = self._normalize_pair(pair)

        try:
            ticker = yf.Ticker(yf_pair)
            hist = ticker.history(period=period, interval=interval)

            if hist.empty:
                return pd.DataFrame()

            hist.columns = [c.lower() for c in hist.columns]
            return hist

        except Exception as e:
            logger.error(f"Error fetching forex history for {pair}: {e}")
            return pd.DataFrame()

    def get_multiple_quotes(self, pairs: List[str]) -> Dict[str, Dict]:
        """
        Get quotes for multiple forex pairs.

        Args:
            pairs: List of forex pairs

        Returns:
            Dictionary mapping pair to quote data
        """
        results = {}

        for pair in pairs:
            try:
                quote = self.get_quote(pair)
                if "error" not in quote:
                    results[pair.upper()] = quote
            except Exception as e:
                logger.warning(f"Failed to fetch {pair}: {e}")
                results[pair.upper()] = {"pair": pair.upper(), "error": str(e)}

        return results

    def calculate_cross_rate(
        self, base: str, target: str, rates: Optional[Dict] = None
    ) -> Optional[float]:
        """
        Calculate cross exchange rate.

        Args:
            base: Base currency (e.g., 'EUR')
            target: Target currency (e.g., 'GBP')
            rates: Optional cached rates

        Returns:
            Cross rate or None
        """
        if rates is None:
            rates = self.get_all_rates()

        if base not in rates or target not in rates:
            return None

        base_usd = 1 / rates[base] if base != "USD" else 1
        target_usd = 1 / rates[target] if target != "USD" else 1

        return target_usd / base_usd

    def get_all_rates(self, base: str = "USD") -> Dict[str, float]:
        """
        Get all exchange rates relative to a base currency.

        Args:
            base: Base currency (default: 'USD')

        Returns:
            Dictionary mapping currency to rate
        """
        major_pairs = [
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "USDCHF",
            "AUDUSD",
            "USDCAD",
            "NZDUSD",
        ]
        quotes = self.get_multiple_quotes(major_pairs)

        rates = {base: 1.0}

        for pair, data in quotes.items():
            current = data.get("current")
            if current:
                base_curr, target_curr = pair[:3], pair[3:6]
                if base == "USD":
                    if target_curr == "USD":
                        rates[base_curr] = 1 / current
                    else:
                        rates[target_curr] = current
                else:
                    if target_curr == base:
                        rates[base_curr] = current
                    elif base_curr == base:
                        rates[target_curr] = 1 / current

        return rates

    def convert(
        self, amount: float, from_currency: str, to_currency: str
    ) -> Optional[float]:
        """
        Convert amount from one currency to another.

        Args:
            amount: Amount to convert
            from_currency: Source currency
            to_currency: Target currency

        Returns:
            Converted amount or None
        """
        from_currency = from_currency.upper()
        to_currency = to_currency.upper()

        if from_currency == to_currency:
            return amount

        if from_currency == "USD":
            quote = self.get_quote(f"USD{to_currency}")
            rate = quote.get("current")
        elif to_currency == "USD":
            quote = self.get_quote(f"USD{from_currency}")
            current = quote.get("current")
            rate = 1 / current if current is not None and current != 0 else None
        else:
            cross_rate = self.calculate_cross_rate(from_currency, to_currency)
            rate = cross_rate

        if rate:
            return amount * rate

        return None

    def get_gold_price(self) -> Dict:
        """
        Get current gold (XAU) price.

        Returns:
            Dictionary with gold price data
        """
        return self.get_quote("XAUUSD")

    def get_silver_price(self) -> Dict:
        """
        Get current silver (XAG) price.

        Returns:
            Dictionary with silver price data
        """
        return self.get_quote("XAGUSD")


def get_forex_quote(pair: str) -> Dict:
    """
    Convenience function to get forex quote.

    Args:
        pair: Forex pair (e.g., 'EURUSD')

    Returns:
        Dictionary with quote data
    """
    collector = ForexDataCollector()
    return collector.get_quote(pair)


def get_forex_history(pair: str, period: str = "1y") -> pd.DataFrame:
    """
    Convenience function to get forex historical data.

    Args:
        pair: Forex pair
        period: Period

    Returns:
        DataFrame with OHLCV data
    """
    collector = ForexDataCollector()
    return collector.get_historical(pair, period)
