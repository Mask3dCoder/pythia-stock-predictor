"""
Market Breadth Module

Provides market breadth analysis including:
- Advance/Decline ratio
- New highs/New lows
- McClellan Oscillator
- Put/Call ratio
- VIX analysis
- Arms Index (TRIN)
- Percent above moving averages
"""

import logging
from typing import Optional, Dict, List
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import numpy as np

from src.core.exceptions import DataCollectionError

logger = logging.getLogger(__name__)

MAJOR_INDICES = ['^GSPC', '^DJI', '^IXIC', '^RUT']

HIGH_LOW_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'UNH',
    'JNJ', 'WMT', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'PEP', 'KO',
    'COST', 'AVGO', 'TMO', 'MCD', 'DIS', 'CSCO', 'ACN', 'ABT', 'DHR', 'NKE',
    'ADBE', 'CRM', 'TXN', 'NEE', 'PM', 'UPS', 'MS', 'QCOM', 'RTX', 'LOW',
]


class MarketBreadthAnalyzer:
    """Analyzes market breadth indicators."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    def get_breadth_summary(self) -> Dict:
        """
        Get comprehensive market breadth summary.

        Returns:
            Dictionary with breadth indicators
        """
        return {
            'ad_ratio': self.get_ad_ratio(),
            'ad_line': self.get_ad_line(),
            'mcclellan': self.get_mcclellan_oscillator(),
            'new_highs_lows': self.get_new_highs_lows(),
            'put_call_ratio': self.get_put_call_ratio(),
            'vix_analysis': self.get_vix_analysis(),
            'percent_above_ma': self.get_percent_above_ma(),
            'Arms Index': self.get_arms_index(),
        }

    def get_ad_ratio(self, period: int = 20) -> Dict:
        """
        Get Advance/Decline ratio.

        Args:
            period: Number of days to analyze

        Returns:
            Dictionary with AD ratio data
        """
        try:
            sp500 = yf.Ticker('^GSPC')
            hist = sp500.history(period=f'{period + 10}d')

            if hist.empty or len(hist) < 5:
                return {'error': 'Insufficient data'}

            adv = hist['Close'].diff().fillna(0) > 0
            dec = hist['Close'].diff().fillna(0) < 0

            dec_sum = dec.rolling(period).sum().replace(0, np.nan)
            ad_ratio = adv.rolling(period).sum() / dec_sum

            current_ratio = ad_ratio.iloc[-1]
            avg_ratio = ad_ratio.mean()

            return {
                'current': current_ratio,
                'average': avg_ratio,
                'interpretation': 'overbought' if current_ratio > 2 else 'oversold' if current_ratio < 0.5 else 'neutral',
            }

        except Exception as e:
            logger.error(f"Error calculating AD ratio: {e}")
            return {'error': str(e)}

    def get_ad_line(self, period: int = 50) -> Dict:
        """
        Get Advance/Decline line.

        Args:
            period: Lookback period

        Returns:
            Dictionary with AD line data
        """
        try:
            sp500 = yf.Ticker('^GSPC')
            hist = sp500.history(period=f'{period + 20}d')

            if hist.empty:
                return {'error': 'Insufficient data'}

            diffs = hist['Close'].diff()
            ad_line = diffs.where(diffs > 0, 0).cumsum() - diffs.where(diffs < 0, 0).cumsum()

            return {
                'value': ad_line.iloc[-1],
                'rising': ad_line.iloc[-1] > ad_line.iloc[-5],
                'signal': 'bullish' if ad_line.iloc[-1] > 0 else 'bearish',
            }

        except Exception as e:
            logger.error(f"Error calculating AD line: {e}")
            return {'error': str(e)}

    def get_mcclellan_oscillator(self, period: int = 19) -> Dict:
        """
        Get McClellan Oscillator.

        Args:
            period: EMA period

        Returns:
            Dictionary with oscillator data
        """
        try:
            sp500 = yf.Ticker('^GSPC')
            hist = sp500.history(period='6mo')

            if hist.empty:
                return {'error': 'Insufficient data'}

            diffs = hist['Close'].diff()
            adv = diffs.where(diffs > 0, 0).rolling(26).sum()
            dec = diffs.where(diffs < 0, 0).rolling(26).sum()

            net_adv = adv - dec
            ema_19 = net_adv.ewm(span=19).mean()
            ema_39 = net_adv.ewm(span=39).mean()

            oscillator = ema_19 - ema_39
            current = oscillator.iloc[-1]

            return {
                'value': current,
                'signal': 'overbought' if current > 50 else 'oversold' if current < -50 else 'neutral',
                'histogram': (ema_19 - ema_39).iloc[-1],
            }

        except Exception as e:
            logger.error(f"Error calculating McClellan: {e}")
            return {'error': str(e)}

    def get_new_highs_lows(self) -> Dict:
        """
        Get New Highs/Lows analysis.

        Returns:
            Dictionary with highs/lows data
        """
        try:
            highs = 0
            lows = 0

            for ticker in HIGH_LOW_TICKERS[:20]:
                try:
                    t = yf.Ticker(ticker)
                    hist = t.history(period='1y')
                    if not hist.empty:
                        current = hist['Close'].iloc[-1]
                        high_52w = hist['High'].max()
                        low_52w = hist['Low'].min()

                        if current >= high_52w * 0.98:
                            highs += 1
                        if current <= low_52w * 1.02:
                            lows += 1
                except Exception:
                    pass

            ratio = highs / max(lows, 1)

            return {
                'highs': highs,
                'lows': lows,
                'ratio': ratio,
                'interpretation': 'strong_bullish' if ratio > 3 else 'bullish' if ratio > 1.5 else 'bearish' if ratio < 0.5 else 'neutral',
            }

        except Exception as e:
            logger.error(f"Error calculating highs/lows: {e}")
            return {'error': str(e)}

    def get_put_call_ratio(self) -> Dict:
        """
        Get Put/Call ratio from major indices.

        Returns:
            Dictionary with ratio data
        """
        try:
            etf = yf.Ticker('SPY')
            info = etf.info

            put_call = info.get('putCallRatio', 1.0)

            return {
                'ratio': put_call,
                'interpretation': 'very_bearish' if put_call > 1.5 else 'bearish' if put_call > 1.1 else 'bullish' if put_call < 0.7 else 'very_bullish' if put_call < 0.5 else 'neutral',
            }

        except Exception as e:
            logger.error(f"Error getting put/call ratio: {e}")
            return {'error': str(e)}

    def get_vix_analysis(self) -> Dict:
        """
        Get VIX analysis.

        Returns:
            Dictionary with VIX data
        """
        try:
            vix = yf.Ticker('^VIX')
            vix_hist = vix.history(period='3mo')

            spy = yf.Ticker('^GSPC')
            spy_hist = spy.history(period='3mo')

            if vix_hist.empty or spy_hist.empty:
                return {'error': 'Insufficient data'}

            current_vix = vix_hist['Close'].iloc[-1]
            vix_mean = vix_hist['Close'].mean()
            vix_high = vix_hist['Close'].max()
            vix_low = vix_hist['Close'].min()

            spy_current = spy_hist['Close'].iloc[-1]
            spy_ma20 = spy_hist['Close'].rolling(20).mean().iloc[-1]
            spy_ma50 = spy_hist['Close'].rolling(50).mean().iloc[-1]

            return {
                'current': current_vix,
                'mean': vix_mean,
                'high': vix_high,
                'low': vix_low,
                'regime': 'high_volatility' if current_vix > 25 else 'normal' if current_vix > 15 else 'low_volatility',
                'fear_level': 'extreme_fear' if current_vix > 30 else 'fear' if current_vix > 20 else 'neutral' if current_vix > 15 else 'complacency' if current_vix > 10 else 'extreme_complacency',
                'market_above_ma20': spy_current > spy_ma20,
                'market_above_ma50': spy_current > spy_ma50,
            }

        except Exception as e:
            logger.error(f"Error analyzing VIX: {e}")
            return {'error': str(e)}

    def get_percent_above_ma(self, symbols: List[str] = None) -> Dict:
        """
        Get percent of stocks above moving averages.

        Args:
            symbols: List of symbols to analyze

        Returns:
            Dictionary with MA analysis
        """
        if symbols is None:
            symbols = HIGH_LOW_TICKERS[:30]

        above_ma20 = 0
        above_ma50 = 0
        above_ma200 = 0

        for symbol in symbols:
            try:
                t = yf.Ticker(symbol)
                hist = t.history(period='1y')

                if hist.empty or len(hist) < 200:
                    continue

                current = hist['Close'].iloc[-1]
                ma20 = hist['Close'].rolling(20).mean().iloc[-1]
                ma50 = hist['Close'].rolling(50).mean().iloc[-1]
                ma200 = hist['Close'].rolling(200).mean().iloc[-1]

                if current > ma20:
                    above_ma20 += 1
                if current > ma50:
                    above_ma50 += 1
                if current > ma200:
                    above_ma200 += 1

            except Exception:
                pass

        total = len(symbols)

        return {
            'above_ma20_pct': (above_ma20 / total) * 100 if total > 0 else 0,
            'above_ma50_pct': (above_ma50 / total) * 100 if total > 0 else 0,
            'above_ma200_pct': (above_ma200 / total) * 100 if total > 0 else 0,
            'interpretation': 'strong_bullish' if above_ma20 / total > 0.7 else 'bearish' if above_ma20 / total < 0.3 else 'neutral',
        }

    def get_arms_index(self, period: int = 20) -> Dict:
        """
        Get Arms Index (TRIN).

        Args:
            period: Analysis period

        Returns:
            Dictionary with TRIN data
        """
        try:
            sp500 = yf.Ticker('^GSPC')
            hist = sp500.history(period=f'{period + 5}d')

            if hist.empty:
                return {'error': 'Insufficient data'}

            returns = hist['Close'].pct_change().dropna()
            volume = hist['Volume']

            advancing = returns > 0
            declining = returns < 0

            adv_volume = volume.where(advancing, 0).rolling(period).sum()
            dec_volume = volume.where(declining, 0).rolling(period).sum()

            adv_count = advancing.rolling(period).sum()
            dec_count = declining.rolling(period).sum()

            trin = (adv_count / dec_count) / (adv_volume / dec_volume.replace(0, 1))

            current_trin = trin.iloc[-1]

            return {
                'value': current_trin,
                'interpretation': 'oversold' if current_trin > 2 else 'overbought' if current_trin < 0.5 else 'neutral',
            }

        except Exception as e:
            logger.error(f"Error calculating TRIN: {e}")
            return {'error': str(e)}

    def get_market_score(self) -> Dict:
        """
        Calculate overall market score based on breadth indicators.

        Returns:
            Dictionary with market score
        """
        score = 0
        max_score = 10
        signals = {}

        ad = self.get_ad_ratio()
        if 'current' in ad:
            if ad['current'] > 1.5:
                score += 2
                signals['ad_ratio'] = 'bullish'
            elif ad['current'] < 0.8:
                score -= 1
                signals['ad_ratio'] = 'bearish'
            else:
                signals['ad_ratio'] = 'neutral'

        hl = self.get_new_highs_lows()
        if 'ratio' in hl:
            if hl['ratio'] > 2:
                score += 2
                signals['highs_lows'] = 'bullish'
            elif hl['ratio'] < 0.8:
                score -= 1
                signals['highs_lows'] = 'bearish'
            else:
                signals['highs_lows'] = 'neutral'

        pc = self.get_put_call_ratio()
        if 'ratio' in pc:
            if pc['ratio'] < 0.8:
                score += 2
                signals['put_call'] = 'bullish'
            elif pc['ratio'] > 1.3:
                score -= 1
                signals['put_call'] = 'bearish'
            else:
                signals['put_call'] = 'neutral'

        vix = self.get_vix_analysis()
        if 'current' in vix:
            if vix['current'] < 15:
                score += 2
                signals['vix'] = 'bullish'
            elif vix['current'] > 25:
                score -= 2
                signals['vix'] = 'bearish'
            else:
                signals['vix'] = 'neutral'

        ma = self.get_percent_above_ma()
        if 'above_ma20_pct' in ma:
            if ma['above_ma20_pct'] > 60:
                score += 2
                signals['ma_percent'] = 'bullish'
            elif ma['above_ma20_pct'] < 40:
                score -= 2
                signals['ma_percent'] = 'bearish'
            else:
                signals['ma_percent'] = 'neutral'

        if score >= 7:
            overall = 'Strong Buy'
        elif score >= 4:
            overall = 'Buy'
        elif score >= 1:
            overall = 'Hold'
        elif score >= -1:
            overall = 'Sell'
        else:
            overall = 'Strong Sell'

        return {
            'score': score,
            'max_score': max_score,
            'rating': overall,
            'signals': signals,
        }


def get_market_breadth() -> Dict:
    """Convenience function to get market breadth."""
    analyzer = MarketBreadthAnalyzer()
    return analyzer.get_breadth_summary()


def get_market_score() -> Dict:
    """Convenience function to get market score."""
    analyzer = MarketBreadthAnalyzer()
    return analyzer.get_market_score()
