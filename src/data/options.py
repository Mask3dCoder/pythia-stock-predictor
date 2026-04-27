"""
Options Data Module

Collects options data and calculates Greeks.
Uses Black-Scholes model for options pricing.

Features:
- Options chain data from Yahoo Finance
- Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
- Implied volatility analysis
- Options strategy analysis
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm

from src.core.exceptions import DataCollectionError

logger = logging.getLogger(__name__)


@dataclass
class Option:
    """Represents a single option."""
    strike: float
    expiration: str
    type: str
    last: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0


class OptionsDataCollector:
    """Collects options data and calculates Greeks."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Options Data Collector.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.risk_free_rate = self.config.get('risk_free_rate', 0.05)

    def get_options_chain(
        self,
        symbol: str,
        expiration: Optional[str] = None
    ) -> Dict:
        """
        Get options chain for a symbol.

        Args:
            symbol: Stock symbol
            expiration: Specific expiration date (YYYY-MM-DD)

        Returns:
            Dictionary with calls, puts, and underlying data
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            current_price = info.get('currentPrice') or info.get('previousClose')

            if current_price is None:
                return {'error': f'Could not get price for {symbol}'}

            try:
                expirations = ticker.options
            except Exception:
                expirations = []

            if not expirations:
                return {'error': f'No options available for {symbol}'}

            if expiration is None:
                target_date = datetime.now() + timedelta(days=30)
                best_exp = None
                best_diff = float('inf')
                for exp in expirations:
                    exp_date = datetime.strptime(exp, '%Y-%m-%d')
                    diff = abs((exp_date - target_date).total_seconds())
                    if diff < best_diff:
                        best_diff = diff
                        best_exp = exp
                expiration = best_exp if best_exp is not None else expirations[0]

            opt = ticker.option_chain(expiration)
            calls = opt.calls
            puts = opt.puts

            def process_options(df):
                if df.empty:
                    return []
                return [
                    Option(
                        strike=row['strike'],
                        expiration=expiration,
                        type='call' if 'call' in df.name.lower() else 'put',
                        last=row.get('lastPrice', 0),
                        bid=row.get('bid', 0),
                        ask=row.get('ask', 0),
                        volume=int(row.get('volume', 0)),
                        open_interest=int(row.get('openInterest', 0)),
                        implied_volatility=row.get('impliedVolatility', 0),
                    )
                    for _, row in df.iterrows()
                ]

            call_options = process_options(calls)
            put_options = process_options(puts)

            for opt in call_options + put_options:
                self._calculate_greeks(opt, current_price, self.risk_free_rate)

            return {
                'symbol': symbol,
                'current_price': current_price,
                'expiration': expiration,
                'expirations': expirations,
                'calls': call_options,
                'puts': put_options,
            }

        except Exception as e:
            logger.error(f"Error fetching options chain for {symbol}: {e}")
            return {'error': str(e)}

    def _calculate_greeks(
        self,
        option: Option,
        spot_price: float,
        risk_free_rate: float,
        time_to_expiry: Optional[float] = None
    ):
        """
        Calculate Greeks for an option using Black-Scholes.

        Args:
            option: Option object
            spot_price: Current stock price
            risk_free_rate: Risk-free interest rate
            time_to_expiry: Time to expiry in years
        """
        K = option.strike
        sigma = option.implied_volatility

        if sigma <= 0:
            sigma = 0.3

        if time_to_expiry is None:
            if option.expiration:
                try:
                    exp_date = datetime.strptime(option.expiration, '%Y-%m-%d')
                    days_to_expiry = (exp_date - datetime.now()).days
                    time_to_expiry = max(days_to_expiry / 365, 0.001)
                except Exception:
                    time_to_expiry = 0.25
            else:
                time_to_expiry = 0.25

        T = time_to_expiry
        r = risk_free_rate
        q = 0

        d1 = (math.log(spot_price / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if option.type == 'call':
            option.delta = math.exp(-q * T) * norm.cdf(d1)
            option.rho = K * T * math.exp(-r * T) * norm.cdf(d2)
        else:
            option.delta = -math.exp(-q * T) * norm.cdf(-d1)
            option.rho = -K * T * math.exp(-r * T) * norm.cdf(-d2)

        option.gamma = math.exp(-q * T) * norm.pdf(d1) / (spot_price * sigma * math.sqrt(T))
        option.vega = spot_price * math.exp(-q * T) * norm.pdf(d1) * math.sqrt(T) / 100

        if option.type == 'call':
            option.theta = (-(spot_price * sigma * math.exp(-q * T) * norm.pdf(d1)) / (2 * math.sqrt(T))
                          - r * K * math.exp(-r * T) * norm.cdf(d2)
                          + q * spot_price * math.exp(-q * T) * norm.cdf(d1)) / 365
        else:
            option.theta = (-(spot_price * sigma * math.exp(-q * T) * norm.pdf(d1)) / (2 * math.sqrt(T))
                          + r * K * math.exp(-r * T) * norm.cdf(-d2)
                          - q * spot_price * math.exp(-q * T) * norm.cdf(-d1)) / 365

    def get_near_the_money(
        self,
        symbol: str,
        num_strikes: int = 10
    ) -> Dict:
        """
        Get near-the-money options.

        Args:
            symbol: Stock symbol
            num_strikes: Number of strikes above/below ATM

        Returns:
            Dictionary with ATM options
        """
        chain = self.get_options_chain(symbol)

        if 'error' in chain:
            return chain

        current_price = chain['current_price']

        atm_strike = round(current_price / 5) * 5

        calls = [c for c in chain['calls']
                if abs(c.strike - atm_strike) <= num_strikes * 5]
        puts = [p for p in chain['puts']
               if abs(p.strike - atm_strike) <= num_strikes * 5]

        return {
            'symbol': symbol,
            'current_price': current_price,
            'atm_strike': atm_strike,
            'calls': calls,
            'puts': puts,
        }

    def calculate_option_price(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        rate: float = 0.05,
        option_type: str = 'call'
    ) -> float:
        """
        Calculate option price using Black-Scholes.

        Args:
            spot: Current stock price
            strike: Option strike price
            time_to_expiry: Time to expiry in years
            volatility: Volatility (decimal)
            rate: Risk-free rate
            option_type: 'call' or 'put'

        Returns:
            Option price
        """
        if time_to_expiry <= 0 or volatility <= 0:
            return 0.0

        d1 = (math.log(spot / strike) + (rate + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * math.sqrt(time_to_expiry))
        d2 = d1 - volatility * math.sqrt(time_to_expiry)

        if option_type == 'call':
            price = spot * norm.cdf(d1) - strike * math.exp(-rate * time_to_expiry) * norm.cdf(d2)
        else:
            price = strike * math.exp(-rate * time_to_expiry) * norm.cdf(-d2) - spot * norm.cdf(-d1)

        return price

    def analyze_strategy(
        self,
        symbol: str,
        strategy: str = 'covered_call'
    ) -> Dict:
        """
        Analyze common options strategies.

        Args:
            symbol: Stock symbol
            strategy: Strategy name

        Returns:
            Dictionary with strategy analysis
        """
        chain = self.get_options_chain(symbol)

        if 'error' in chain:
            return chain

        current_price = chain['current_price']
        atm_strike = round(current_price / 5) * 5

        analysis = {
            'symbol': symbol,
            'current_price': current_price,
            'strategy': strategy,
        }

        if strategy == 'covered_call':
            analysis['description'] = 'Own stock + sell call'
            analysis['max_profit'] = 'Unlimited'
            analysis['max_loss'] = 'Stock price goes to zero'
            analysis['breakeven'] = current_price * 0.95

        elif strategy == 'protective_put':
            analysis['description'] = 'Own stock + buy put'
            analysis['max_profit'] = 'Unlimited'
            analysis['max_loss'] = 'Strike price'
            analysis['breakeven'] = current_price * 0.95

        elif strategy == 'straddle':
            analysis['description'] = 'Buy call + buy put at same strike'
            analysis['max_profit'] = 'Unlimited'
            analysis['max_loss'] = 'Sum of premiums'

        elif strategy == 'strangle':
            analysis['description'] = 'Buy OTM call + buy OTM put'
            analysis['max_profit'] = 'Unlimited'
            analysis['max_loss'] = 'Sum of premiums'

        return analysis


def get_options_chain(symbol: str) -> Dict:
    """
    Convenience function to get options chain.

    Args:
        symbol: Stock symbol

    Returns:
        Dictionary with options data
    """
    collector = OptionsDataCollector()
    return collector.get_options_chain(symbol)


def calculate_greeks(symbol: str, expiration: Optional[str] = None) -> Dict:
    """
    Convenience function to get options with Greeks.

    Args:
        symbol: Stock symbol
        expiration: Expiration date

    Returns:
        Dictionary with options and Greeks
    """
    collector = OptionsDataCollector()
    return collector.get_options_chain(symbol, expiration)
