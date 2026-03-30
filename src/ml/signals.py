"""
Trading Signals Module

Generates multi-factor trading signals including:
- Technical indicator signals (MA, RSI, MACD, Bollinger Bands)
- Signal combination strategies
- Signal strength scoring
- Buy/Sell/Hold recommendations
"""

import logging
from typing import Optional, Dict, List
from datetime import datetime

import yfinance as yf
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Generates trading signals from multiple indicators."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    def generate_signals(self, symbol: str) -> Dict:
        """
        Generate comprehensive trading signals for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with signal analysis
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1y')

            if hist.empty or len(hist) < 50:
                return {'error': 'Insufficient data'}

            signals = {}

            signals['moving_averages'] = self._ma_signals(hist)
            signals['rsi'] = self._rsi_signals(hist)
            signals['macd'] = self._macd_signals(hist)
            signals['bollinger'] = self._bollinger_signals(hist)
            signals['momentum'] = self._momentum_signals(hist)

            combined = self._combine_signals(signals)

            return {
                'symbol': symbol,
                'current_price': hist['Close'].iloc[-1],
                'signals': signals,
                'combined': combined,
                'recommendation': self._get_recommendation(combined),
            }

        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
            return {'error': str(e)}

    def _ma_signals(self, df: pd.DataFrame) -> Dict:
        """Generate moving average signals."""
        close = df['Close']

        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean() if len(close) >= 200 else None

        current_price = close.iloc[-1]
        current_ma20 = ma20.iloc[-1]
        current_ma50 = ma50.iloc[-1]

        signals = []

        if current_price > current_ma20:
            signals.append(('Price vs MA20', 'bullish', 0.3))
        else:
            signals.append(('Price vs MA20', 'bearish', -0.3))

        if current_ma20 > current_ma50:
            signals.append(('MA20 vs MA50', 'bullish', 0.5))
        else:
            signals.append(('MA20 vs MA50', 'bearish', -0.5))

        if ma200 is not None:
            current_ma200 = ma200.iloc[-1]
            if current_price > current_ma200:
                signals.append(('Price vs MA200', 'bullish', 0.4))
            else:
                signals.append(('Price vs MA200', 'bearish', -0.4))

        return {
            'signals': signals,
            'score': sum(s[2] for s in signals),
        }

    def _rsi_signals(self, df: pd.DataFrame) -> Dict:
        """Generate RSI signals."""
        close = df['Close']
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        current_rsi = rsi.iloc[-1]

        signals = []
        strength = 0

        if current_rsi < 30:
            signals.append(('RSI', 'oversold', 0.8))
            strength = 0.8
        elif current_rsi > 70:
            signals.append(('RSI', 'overbought', -0.8))
            strength = -0.8
        elif 30 <= current_rsi <= 50:
            signals.append(('RSI', 'bullish', 0.3))
            strength = 0.3
        elif 50 <= current_rsi <= 70:
            signals.append(('RSI', 'neutral', 0))
            strength = 0
        else:
            signals.append(('RSI', 'bearish', -0.3))
            strength = -0.3

        return {
            'rsi': round(current_rsi, 2),
            'signals': signals,
            'score': strength,
        }

    def _macd_signals(self, df: pd.DataFrame) -> Dict:
        """Generate MACD signals."""
        close = df['Close']

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal

        current_macd = macd.iloc[-1]
        current_signal = signal.iloc[-1]
        current_hist = histogram.iloc[-1]

        signals = []
        strength = 0

        if current_macd > current_signal:
            signals.append(('MACD', 'bullish', 0.5))
            strength = 0.5
        else:
            signals.append(('MACD', 'bearish', -0.5))
            strength = -0.5

        if current_hist > 0:
            signals.append(('MACD Histogram', 'bullish', 0.3))
            strength += 0.3
        else:
            signals.append(('MACD Histogram', 'bearish', -0.3))
            strength -= 0.3

        return {
            'macd': round(current_macd, 4),
            'signal': round(current_signal, 4),
            'histogram': round(current_hist, 4),
            'signals': signals,
            'score': strength,
        }

    def _bollinger_signals(self, df: pd.DataFrame) -> Dict:
        """Generate Bollinger Bands signals."""
        close = df['Close']

        sma = close.rolling(20).mean()
        std = close.rolling(20).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)

        current_price = close.iloc[-1]
        current_upper = upper.iloc[-1]
        current_lower = lower.iloc[-1]

        signals = []
        strength = 0

        if current_price < current_lower:
            signals.append(('Bollinger', 'oversold', 0.7))
            strength = 0.7
        elif current_price > current_upper:
            signals.append(('Bollinger', 'overbought', -0.7))
            strength = -0.7
        else:
            position = (current_price - current_lower) / (current_upper - current_lower)
            if position < 0.3:
                signals.append(('Bollinger', 'bullish', 0.3))
                strength = 0.3
            elif position > 0.7:
                signals.append(('Bollinger', 'bearish', -0.3))
                strength = -0.3
            else:
                signals.append(('Bollinger', 'neutral', 0))
                strength = 0

        return {
            'upper': round(current_upper, 2),
            'middle': round(sma.iloc[-1], 2),
            'lower': round(current_lower, 2),
            'position': round((current_price - current_lower) / (current_upper - current_lower), 2),
            'signals': signals,
            'score': strength,
        }

    def _momentum_signals(self, df: pd.DataFrame) -> Dict:
        """Generate momentum signals."""
        close = df['Close']

        roc_10 = ((close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]) * 100 if len(close) >= 10 else 0
        roc_20 = ((close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]) * 100 if len(close) >= 20 else 0

        signals = []
        strength = 0

        if roc_10 > 5:
            signals.append(('ROC 10d', 'strong_up', 0.6))
            strength = 0.6
        elif roc_10 < -5:
            signals.append(('ROC 10d', 'strong_down', -0.6))
            strength = -0.6
        elif roc_10 > 0:
            signals.append(('ROC 10d', 'up', 0.3))
            strength = 0.3
        else:
            signals.append(('ROC 10d', 'down', -0.3))
            strength = -0.3

        if roc_20 > 10:
            signals.append(('ROC 20d', 'strong_up', 0.4))
            strength += 0.4
        elif roc_20 < -10:
            signals.append(('ROC 20d', 'strong_down', -0.4))
            strength += -0.4

        return {
            'roc_10': round(roc_10, 2),
            'roc_20': round(roc_20, 2),
            'signals': signals,
            'score': strength,
        }

    def _combine_signals(self, signals: Dict) -> Dict:
        """Combine all signals into a score."""
        total_score = 0
        total_weight = 0

        for indicator, data in signals.items():
            if isinstance(data, dict) and 'score' in data:
                weight = {
                    'moving_averages': 0.3,
                    'rsi': 0.2,
                    'macd': 0.2,
                    'bollinger': 0.15,
                    'momentum': 0.15,
                }.get(indicator, 0.1)

                total_score += data['score'] * weight
                total_weight += weight

        combined_score = total_score / total_weight if total_weight > 0 else 0

        return {
            'score': round(combined_score, 2),
            'normalized': round((combined_score + 1) / 2 * 100, 1),
        }

    def _get_recommendation(self, combined: Dict) -> Dict:
        """Get trading recommendation."""
        score = combined.get('score', 0)

        if score > 0.5:
            rating = 'Strong Buy'
            action = 'Buy'
        elif score > 0.2:
            rating = 'Buy'
            action = 'Buy'
        elif score > -0.2:
            rating = 'Hold'
            action = 'Hold'
        elif score > -0.5:
            rating = 'Sell'
            action = 'Sell'
        else:
            rating = 'Strong Sell'
            action = 'Sell'

        return {
            'rating': rating,
            'action': action,
            'confidence': abs(score) * 100,
        }


def generate_signals(symbol: str) -> Dict:
    """Convenience function to generate signals."""
    generator = SignalGenerator()
    return generator.generate_signals(symbol)
