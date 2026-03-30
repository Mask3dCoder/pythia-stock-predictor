"""
Crypto Data Collector Module

Collects cryptocurrency data from CoinGecko API.
Free API with no API key required for basic usage.

Features:
- Historical OHLCV data
- Market data (cap, volume, supply)
- Coin metadata
- Exchange rates between crypto pairs
- Multiple timeframes
"""

import logging
from datetime import datetime
from typing import Optional, Dict, List, Union

import pandas as pd
import numpy as np
import requests

from src.core.utils import retry_with_backoff
from src.core.exceptions import DataCollectionError
from src.core.cache import cache

logger = logging.getLogger(__name__)

COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

MAPPING = {
    "bitcoin": "btc",
    "ethereum": "eth",
    "tether": "usdt",
    "binancecoin": "bnb",
    "solana": "sol",
    "ripple": "xrp",
    "usd-coin": "usdc",
    "cardano": "ada",
    "avalanche-2": "avax",
    "dogecoin": "doge",
    "polkadot": "dot",
    "chainlink": "link",
    "matic-network": "matic",
    "litecoin": "ltc",
    "shiba-inu": "shib",
}


class CryptoDataCollector:
    """Collects cryptocurrency data from CoinGecko."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Crypto Data Collector.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.base_url = COINGECKO_BASE_URL
        self.session = requests.Session()
        self.session.headers.update(
            {"Accept": "application/json", "User-Agent": "Pythia/1.0"}
        )

    def _get_coin_id(self, symbol: str) -> Optional[str]:
        """Convert symbol to CoinGecko coin ID."""
        symbol = symbol.lower()

        if symbol in MAPPING:
            return MAPPING[symbol]

        symbol_to_id = {
            "btc": "bitcoin",
            "eth": "ethereum",
            "usdt": "tether",
            "bnb": "binancecoin",
            "sol": "solana",
            "xrp": "ripple",
            "usdc": "usd-coin",
            "ada": "cardano",
            "avax": "avalanche-2",
            "doge": "dogecoin",
            "dot": "polkadot",
            "link": "chainlink",
            "matic": "matic-network",
            "ltc": "litecoin",
            "shib": "shiba-inu",
            "btc-usd": "bitcoin",
            "eth-usd": "ethereum",
        }

        return symbol_to_id.get(symbol)

    @retry_with_backoff(
        max_retries=3, backoff_factor=2.0, exceptions=(requests.RequestException,)
    )
    def get_market_data(self, coin_id: str, currency: str = "usd") -> Dict:
        """
        Get market data for a specific coin.

        Args:
            coin_id: CoinGecko coin ID (e.g., 'bitcoin')
            currency: Currency for prices (default: 'usd')

        Returns:
            Dictionary with market data
        """
        cache_key = f"crypto:market:{coin_id}:{currency}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        url = f"{self.base_url}/coins/{coin_id}"
        params = {
            "localization": "false",
            "tickers": "false",
            "community_data": "false",
            "developer_data": "false",
        }

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            market_data = {
                "symbol": data.get("symbol", "").upper(),
                "name": data.get("name", ""),
                "current_price": data.get("market_data", {})
                .get("current_price", {})
                .get(currency),
                "market_cap": data.get("market_data", {})
                .get("market_cap", {})
                .get(currency),
                "market_cap_rank": data.get("market_cap_rank"),
                "total_volume": data.get("market_data", {})
                .get("total_volume", {})
                .get(currency),
                "high_24h": data.get("market_data", {})
                .get("high_24h", {})
                .get(currency),
                "low_24h": data.get("market_data", {}).get("low_24h", {}).get(currency),
                "price_change_24h": data.get("market_data", {}).get("price_change_24h"),
                "price_change_percentage_24h": data.get("market_data", {}).get(
                    "price_change_percentage_24h"
                ),
                "price_change_percentage_7d": data.get("market_data", {}).get(
                    "price_change_percentage_7d_in_currency"
                ),
                "price_change_percentage_30d": data.get("market_data", {}).get(
                    "price_change_percentage_30d_in_currency"
                ),
                "circulating_supply": data.get("market_data", {}).get(
                    "circulating_supply"
                ),
                "total_supply": data.get("market_data", {}).get("total_supply"),
                "max_supply": data.get("market_data", {}).get("max_supply"),
                "ath": data.get("market_data", {}).get("ath", {}).get(currency),
                "ath_change_percentage": data.get("market_data", {})
                .get("ath_change_percentage", {})
                .get(currency),
                "atl": data.get("market_data", {}).get("atl", {}).get(currency),
                "atl_change_percentage": data.get("market_data", {})
                .get("atl_change_percentage", {})
                .get(currency),
                "last_updated": data.get("market_data", {}).get("last_updated"),
            }

            cache.set(cache_key, market_data, ttl=60)

            return market_data

        except requests.RequestException as e:
            logger.error(f"Error fetching market data for {coin_id}: {e}")
            raise DataCollectionError(f"Failed to fetch market data: {e}")

    def get_historical_data(
        self, coin_id: str, days: int = 365, currency: str = "usd"
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data for a coin.

        Args:
            coin_id: CoinGecko coin ID
            days: Number of days of historical data
            currency: Currency for prices

        Returns:
            DataFrame with OHLCV data
        """
        url = f"{self.base_url}/coins/{coin_id}/ohlc"
        params = {
            "vs_currency": currency,
            "days": days,
        }

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(
                data, columns=["timestamp", "open", "high", "low", "close"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df = df.sort_index()

            df["volume"] = 0

            return df

        except requests.RequestException as e:
            logger.error(f"Error fetching historical data for {coin_id}: {e}")
            raise DataCollectionError(f"Failed to fetch historical data: {e}")

    def get_market_chart(
        self, coin_id: str, days: int = 365, currency: str = "usd"
    ) -> pd.DataFrame:
        """
        Get market chart data with volume.

        Args:
            coin_id: CoinGecko coin ID
            days: Number of days
            currency: Currency for prices

        Returns:
            DataFrame with OHLCV data
        """
        url = f"{self.base_url}/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": currency,
            "days": days,
        }

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            prices = pd.DataFrame(data["prices"], columns=["timestamp", "close"])
            volumes = pd.DataFrame(
                data["total_volumes"], columns=["timestamp", "volume"]
            )
            market_caps = pd.DataFrame(
                data["market_caps"], columns=["timestamp", "market_cap"]
            )

            prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit="ms")
            volumes["timestamp"] = pd.to_datetime(volumes["timestamp"], unit="ms")
            market_caps["timestamp"] = pd.to_datetime(
                market_caps["timestamp"], unit="ms"
            )

            df = prices.merge(volumes, on="timestamp").merge(
                market_caps, on="timestamp"
            )
            df.set_index("timestamp", inplace=True)
            df = df.sort_index()

            df["open"] = df["close"]
            df["high"] = df["close"]
            df["low"] = df["close"]

            df["high"] = df["close"].cummax()
            df["low"] = df["close"].cummin()

            return df[["open", "high", "low", "close", "volume", "market_cap"]]

        except requests.RequestException as e:
            logger.error(f"Error fetching market chart for {coin_id}: {e}")
            raise DataCollectionError(f"Failed to fetch market chart: {e}")

    def get_top_coins(self, limit: int = 100, currency: str = "usd") -> List[Dict]:
        """
        Get top cryptocurrencies by market cap.

        Args:
            limit: Number of coins to return
            currency: Currency for prices

        Returns:
            List of dictionaries with coin data
        """
        url = f"{self.base_url}/coins/markets"
        params = {
            "vs_currency": currency,
            "order": "market_cap_desc",
            "per_page": limit,
            "page": 1,
            "sparkline": "false",
        }

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            coins = []
            for coin in data:
                coins.append(
                    {
                        "id": coin.get("id"),
                        "symbol": coin.get("symbol", "").upper(),
                        "name": coin.get("name"),
                        "current_price": coin.get("current_price"),
                        "market_cap": coin.get("market_cap"),
                        "market_cap_rank": coin.get("market_cap_rank"),
                        "total_volume": coin.get("total_volume"),
                        "price_change_24h": coin.get("price_change_24h"),
                        "price_change_percentage_24h": coin.get(
                            "price_change_percentage_24h"
                        ),
                        "circulating_supply": coin.get("circulating_supply"),
                    }
                )

            return coins

        except requests.RequestException as e:
            logger.error(f"Error fetching top coins: {e}")
            raise DataCollectionError(f"Failed to fetch top coins: {e}")

    def get_exchange_rate(self, base: str, target: str = "usd") -> Optional[float]:
        """
        Get exchange rate between two currencies.

        Args:
            base: Base currency (e.g., 'btc', 'eth')
            target: Target currency (default: 'usd')

        Returns:
            Exchange rate or None if failed
        """
        coin_id = self._get_coin_id(base)
        if not coin_id:
            return None

        try:
            data = self.get_market_data(coin_id, target.lower())
            return data.get("current_price")
        except Exception:
            return None

    def get_multiple_coins(
        self, symbols: List[str], currency: str = "usd"
    ) -> Dict[str, Dict]:
        """
        Get market data for multiple coins.

        Args:
            symbols: List of coin symbols
            currency: Currency for prices

        Returns:
            Dictionary mapping symbol to market data
        """
        results = {}

        for symbol in symbols:
            coin_id = self._get_coin_id(symbol)
            if coin_id:
                try:
                    data = self.get_market_data(coin_id, currency)
                    results[symbol.upper()] = data
                except Exception as e:
                    logger.warning(f"Failed to fetch data for {symbol}: {e}")

        return results

    def get_coin_info(self, symbol: str) -> Optional[Dict]:
        """
        Get detailed coin information.

        Args:
            symbol: Coin symbol (e.g., 'btc', 'bitcoin')

        Returns:
            Dictionary with coin info
        """
        coin_id = self._get_coin_id(symbol)
        if not coin_id:
            return None

        url = f"{self.base_url}/coins/{coin_id}"
        params = {
            "localization": "false",
            "tickers": "false",
            "community_data": "true",
            "developer_data": "false",
        }

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            return {
                "id": data.get("id"),
                "symbol": data.get("symbol", "").upper(),
                "name": data.get("name"),
                "description": data.get("description", {}).get("en", ""),
                "image": data.get("image", {}).get("large"),
                "homepage": data.get("links", {}).get("homepage", [None])[0],
                "genesis_date": data.get("genesis_date"),
                "market_cap_rank": data.get("market_cap_rank"),
                "community_score": data.get("community_score"),
                "developer_score": data.get("developer_score"),
            }

        except requests.RequestException as e:
            logger.error(f"Error fetching coin info for {symbol}: {e}")
            return None


def get_crypto_quote(symbol: str) -> Dict:
    """
    Convenience function to get crypto quote.

    Args:
        symbol: Crypto symbol (e.g., 'BTC', 'bitcoin')

    Returns:
        Dictionary with quote data
    """
    collector = CryptoDataCollector()
    coin_id = collector._get_coin_id(symbol)

    if not coin_id:
        return {"error": f"Unknown symbol: {symbol}"}

    return collector.get_market_data(coin_id)


def get_crypto_history(symbol: str, days: int = 365) -> pd.DataFrame:
    """
    Convenience function to get crypto historical data.

    Args:
        symbol: Crypto symbol
        days: Number of days

    Returns:
        DataFrame with OHLCV data
    """
    collector = CryptoDataCollector()
    coin_id = collector._get_coin_id(symbol)

    if not coin_id:
        return pd.DataFrame()

    return collector.get_historical_data(coin_id, days)
