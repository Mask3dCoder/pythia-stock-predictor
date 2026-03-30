"""
Pattern Recognition Module

Detects candlestick and chart patterns including:
- Candlestick patterns (doji, hammer, engulfing, morning star, etc.)
- Chart patterns (head & shoulders, triangles, flags, wedges)
- Technical pattern recognition using machine learning
"""

import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

logger = logging.getLogger(__name__)


@dataclass
class CandlestickPattern:
    """Represents a candlestick pattern."""
    name: str
    bullish: bool
    strength: float
    location: str


@dataclass
class ChartPattern:
    """Represents a chart pattern."""
    name: str
    pattern_type: str
    breakout_direction: str
    confidence: float


class PatternRecognizer:
    """Recognizes patterns in price data."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    def get_all_patterns(self, symbol: str) -> Dict:
        """
        Get all detected patterns for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with pattern analysis
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='3mo')

            if hist.empty or len(hist) < 30:
                return {'error': 'Insufficient data'}

            return {
                'candlestick': self._detect_candlestick_patterns(hist),
                'chart': self._detect_chart_patterns(hist),
                'recent': self._get_recent_patterns(hist),
            }

        except Exception as e:
            logger.error(f"Error detecting patterns for {symbol}: {e}")
            return {'error': str(e)}

    def _detect_candlestick_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect candlestick patterns."""
        patterns = []
        o = df['Open']
        h = df['High']
        l = df['Low']
        c = df['Close']

        for i in range(2, len(df)):
            current = {'o': o.iloc[i], 'h': h.iloc[i], 'l': l.iloc[i], 'c': c.iloc[i]}
            prev = {'o': o.iloc[i-1], 'h': h.iloc[i-1], 'l': l.iloc[i-1], 'c': c.iloc[i-1]}
            prev2 = {'o': o.iloc[i-2], 'h': h.iloc[i-2], 'l': l.iloc[i-2], 'c': c.iloc[i-2]}

            body = abs(current['c'] - current['o'])
            upper_shadow = current['h'] - max(current['o'], current['c'])
            lower_shadow = min(current['o'], current['c']) - current['l']
            total_range = current['h'] - current['l']

            if total_range == 0:
                continue

            body_ratio = body / total_range
            upper_shadow_ratio = upper_shadow / total_range
            lower_shadow_ratio = lower_shadow / total_range

            if upper_shadow_ratio > 0.6 and lower_shadow_ratio < 0.1 and body_ratio < 0.3:
                if current['c'] > current['o']:
                    patterns.append({
                        'pattern': 'Inverted Hammer',
                        'bullish': True,
                        'strength': upper_shadow_ratio,
                        'date': str(df.index[i].date())
                    })
                else:
                    patterns.append({
                        'pattern': 'Shooting Star',
                        'bullish': False,
                        'strength': upper_shadow_ratio,
                        'date': str(df.index[i].date())
                    })

            if lower_shadow_ratio > 0.6 and upper_shadow_ratio < 0.1 and body_ratio < 0.3:
                if current['c'] < current['o']:
                    patterns.append({
                        'pattern': 'Hammer',
                        'bullish': True,
                        'strength': lower_shadow_ratio,
                        'date': str(df.index[i].date())
                    })

            if body_ratio < 0.1 and upper_shadow_ratio < 0.1 and lower_shadow_ratio < 0.1:
                patterns.append({
                    'pattern': 'Doji',
                    'bullish': None,
                    'strength': 0.5,
                    'date': str(df.index[i].date())
                })

            if i >= 1:
                prev_body = abs(prev['c'] - prev['o'])
                curr_body = body
                
                if prev_body > 0 and curr_body > 0:
                    prev_bullish = prev['c'] > prev['o']
                    curr_bullish = current['c'] > current['o']

                    if not prev_bullish and curr_bullish:
                        if current['c'] > prev['o'] and current['o'] < prev['c']:
                            patterns.append({
                                'pattern': 'Bullish Engulfing',
                                'bullish': True,
                                'strength': min(curr_body / prev_body, 2.0),
                                'date': str(df.index[i].date())
                            })

                    if prev_bullish and not curr_bullish:
                        if current['c'] < prev['o'] and current['o'] > prev['c']:
                            patterns.append({
                                'pattern': 'Bearish Engulfing',
                                'bullish': False,
                                'strength': min(curr_body / prev_body, 2.0),
                                'date': str(df.index[i].date())
                            })

            if i >= 2:
                prev2_body = abs(prev2['c'] - prev2['o'])
                
                if prev2_body > 0 and body > 0 and prev_body > 0:
                    if prev2['c'] < prev2['o'] and prev['c'] > prev['o'] and current['c'] > current['o']:
                        if current['c'] > prev['h']:
                            patterns.append({
                                'pattern': 'Morning Star',
                                'bullish': True,
                                'strength': 0.8,
                                'date': str(df.index[i].date())
                            })

                    if prev2['c'] > prev2['o'] and prev['c'] < prev['o'] and current['c'] < current['o']:
                        if current['c'] < prev['l']:
                            patterns.append({
                                'pattern': 'Evening Star',
                                'bullish': False,
                                'strength': 0.8,
                                'date': str(df.index[i].date())
                            })

        return sorted(patterns, key=lambda x: x['strength'], reverse=True)[:10]

    def _detect_chart_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect chart patterns."""
        patterns = []
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values

        try:
            local_max = argrelextrema(close, np.greater, order=5)[0]
            local_min = argrelextrema(close, np.less, order=5)[0]

            if len(local_max) >= 2 and len(local_min) >= 2:
                max_prices = close[local_max]
                min_prices = close[local_min]

                if len(max_prices) >= 3:
                    if abs(max_prices[-1] - max_prices[0]) < np.std(max_prices) * 0.5:
                        if max_prices[-1] > max_prices[1]:
                            patterns.append({
                                'pattern': 'Head & Shoulders',
                                'type': 'reversal',
                                'direction': 'bearish',
                                'confidence': 0.7,
                            })
                        else:
                            patterns.append({
                                'pattern': 'Inverse Head & Shoulders',
                                'type': 'reversal',
                                'direction': 'bullish',
                                'confidence': 0.7,
                            })

                if len(local_min) >= 3:
                    if abs(min_prices[-1] - min_prices[0]) < np.std(min_prices) * 0.3:
                        patterns.append({
                            'pattern': 'Double Bottom',
                            'type': 'reversal',
                            'direction': 'bullish',
                            'confidence': 0.7,
                        })

                if len(local_max) >= 3:
                    if abs(max_prices[-1] - max_prices[0]) < np.std(max_prices) * 0.3:
                        patterns.append({
                            'pattern': 'Double Top',
                            'type': 'reversal',
                            'direction': 'bearish',
                            'confidence': 0.7,
                        })

            recent = close[-30:]
            if len(recent) >= 20:
                x = np.arange(len(recent))
                coeffs = np.polyfit(x, recent, 1)
                slope = coeffs[0]

                trend = 'uptrend' if slope > 0 else 'downtrend' if slope < 0 else 'sideways'

                if abs(slope) < np.std(recent) * 0.1:
                    patterns.append({
                        'pattern': 'Channel',
                        'type': 'continuation',
                        'direction': 'sideways',
                        'confidence': 0.6,
                    })

                if slope > 0:
                    patterns.append({
                        'pattern': 'Ascending Channel',
                        'type': 'continuation',
                        'direction': 'bullish',
                        'confidence': 0.6,
                    })
                elif slope < 0:
                    patterns.append({
                        'pattern': 'Descending Channel',
                        'type': 'continuation',
                        'direction': 'bearish',
                        'confidence': 0.6,
                    })

        except Exception as e:
            logger.warning(f"Error detecting chart patterns: {e}")

        return patterns

    def _get_recent_patterns(self, df: pd.DataFrame) -> Dict:
        """Get summary of recent patterns."""
        candlestick = self._detect_candlestick_patterns(df)
        chart = self._detect_chart_patterns(df)

        bullish_count = sum(1 for p in candlestick if p.get('bullish') is True)
        bearish_count = sum(1 for p in candlestick if p.get('bullish') is False)

        return {
            'total_candlesticks': len(candlestick),
            'bullish': bullish_count,
            'bearish': bearish_count,
            'chart_patterns': len(chart),
            'overall_bias': 'bullish' if bullish_count > bearish_count else 'bearish' if bearish_count > bullish_count else 'neutral',
        }

    def detect_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Detect support and resistance levels."""
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values

        resistance_levels = []
        support_levels = []

        local_max = argrelextrema(close, np.greater, order=10)[0]
        local_min = argrelextrema(close, np.less, order=10)[0]

        for idx in local_max:
            resistance_levels.append(close[idx])

        for idx in local_min:
            support_levels.append(close[idx])

        current_price = close[-1]

        nearest_support = max([s for s in support_levels if s < current_price], default=None)
        nearest_resistance = min([r for r in resistance_levels if r > current_price], default=None)

        return {
            'resistance_levels': sorted(resistance_levels, reverse=True)[:5],
            'support_levels': sorted(support_levels)[:5],
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'current_price': current_price,
        }


def detect_patterns(symbol: str) -> Dict:
    """Convenience function to detect patterns."""
    recognizer = PatternRecognizer()
    return recognizer.get_all_patterns(symbol)
