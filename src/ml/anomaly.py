"""
Anomaly Detection Module

Detects anomalies in price and volume data including:
- Statistical anomaly detection (Z-score, IQR)
- Volatility spikes
- Volume anomalies
- Price gaps
- Unusual price movements
"""

import logging
from typing import Optional, Dict, List
from datetime import datetime

import yfinance as yf
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detects anomalies in financial data."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.z_threshold = self.config.get('z_threshold', 2.5)

    def detect_all_anomalies(self, symbol: str) -> Dict:
        """
        Detect all types of anomalies for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with anomaly analysis
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='3mo')

            if hist.empty or len(hist) < 30:
                return {'error': 'Insufficient data'}

            return {
                'price_anomalies': self._detect_price_anomalies(hist),
                'volume_anomalies': self._detect_volume_anomalies(hist),
                'volatility_anomalies': self._detect_volatility_anomalies(hist),
                'gap_anomalies': self._detect_gap_anomalies(hist),
                'summary': self._get_anomaly_summary(hist),
            }

        except Exception as e:
            logger.error(f"Error detecting anomalies for {symbol}: {e}")
            return {'error': str(e)}

    def _detect_price_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect price anomalies using Z-score."""
        anomalies = []
        close = df['Close']
        returns = close.pct_change().dropna()

        if returns.empty:
            return anomalies

        mean = returns.mean()
        std = returns.std()

        for i, (date, ret) in enumerate(returns.items()):
            z_score = (ret - mean) / std if std > 0 else 0

            if abs(z_score) > self.z_threshold:
                anomalies.append({
                    'date': str(date.date()),
                    'type': 'price',
                    'direction': 'up' if ret > 0 else 'down',
                    'magnitude': abs(ret) * 100,
                    'z_score': round(z_score, 2),
                    'severity': 'extreme' if abs(z_score) > 4 else 'high' if abs(z_score) > 3 else 'moderate',
                })

        return sorted(anomalies, key=lambda x: abs(x['z_score']), reverse=True)[:10]

    def _detect_volume_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect volume anomalies."""
        anomalies = []
        volume = df['Volume']

        if volume.empty:
            return anomalies

        mean_vol = volume.mean()
        std_vol = volume.std()

        for i, (date, vol) in enumerate(volume.items()):
            if mean_vol > 0:
                z_score = (vol - mean_vol) / std_vol if std_vol > 0 else 0

                if z_score > self.z_threshold:
                    price = df['Close'].iloc[i]
                    prev_price = df['Close'].iloc[i-1] if i > 0 else price
                    price_change = ((price - prev_price) / prev_price) * 100 if prev_price > 0 else 0

                    anomalies.append({
                        'date': str(date.date()),
                        'type': 'volume',
                        'volume': int(vol),
                        'volume_ratio': round(vol / mean_vol, 2),
                        'z_score': round(z_score, 2),
                        'price_change': round(price_change, 2),
                        'severity': 'extreme' if z_score > 4 else 'high' if z_score > 3 else 'moderate',
                    })

        return sorted(anomalies, key=lambda x: x['z_score'], reverse=True)[:10]

    def _detect_volatility_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect volatility anomalies."""
        anomalies = []
        close = df['Close']
        returns = close.pct_change().dropna()

        if returns.empty or len(returns) < 20:
            return anomalies

        rolling_vol = returns.rolling(20).std() * np.sqrt(252)
        historical_vol = returns.std() * np.sqrt(252)

        for i, (date, vol) in enumerate(rolling_vol.items()):
            if pd.notna(vol) and historical_vol > 0:
                vol_ratio = vol / historical_vol

                if vol_ratio > 2.0:
                    price = close.iloc[i] if i < len(close) else close.iloc[-1]

                    anomalies.append({
                        'date': str(date.date()),
                        'type': 'volatility',
                        'volatility': round(vol * 100, 2),
                        'volatility_ratio': round(vol_ratio, 2),
                        'price': round(price, 2),
                        'severity': 'extreme' if vol_ratio > 4 else 'high' if vol_ratio > 3 else 'moderate',
                    })

        return sorted(anomalies, key=lambda x: x['volatility_ratio'], reverse=True)[:10]

    def _detect_gap_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect price gap anomalies."""
        anomalies = []
        close = df['Close']

        for i in range(1, len(close)):
            current = close.iloc[i]
            previous = close.iloc[i-1]

            gap_pct = ((current - previous) / previous) * 100

            if abs(gap_pct) > 3:
                date = close.index[i]

                anomalies.append({
                    'date': str(date.date()),
                    'type': 'gap',
                    'direction': 'up' if gap_pct > 0 else 'down',
                    'gap_pct': round(gap_pct, 2),
                    'from_price': round(previous, 2),
                    'to_price': round(current, 2),
                    'severity': 'extreme' if abs(gap_pct) > 10 else 'high' if abs(gap_pct) > 5 else 'moderate',
                })

        return sorted(anomalies, key=lambda x: abs(x['gap_pct']), reverse=True)[:10]

    def _get_anomaly_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary of all anomalies."""
        price = self._detect_price_anomalies(df)
        volume = self._detect_volume_anomalies(df)
        volatility = self._detect_volatility_anomalies(df)
        gaps = self._detect_gap_anomalies(df)

        extreme_count = sum(1 for a in price + volume + volatility + gaps if a.get('severity') == 'extreme')
        high_count = sum(1 for a in price + volume + volatility + gaps if a.get('severity') == 'high')

        return {
            'total_anomalies': len(price) + len(volume) + len(volatility) + len(gaps),
            'price_anomalies': len(price),
            'volume_anomalies': len(volume),
            'volatility_anomalies': len(volatility),
            'gap_anomalies': len(gaps),
            'extreme_count': extreme_count,
            'high_count': high_count,
            'risk_level': 'high' if extreme_count > 2 or high_count > 5 else 'medium' if extreme_count > 0 or high_count > 2 else 'low',
            'recent_activity': 'elevated' if extreme_count + high_count > 3 else 'normal',
        }

    def detect_outliers(self, df: pd.DataFrame, column: str = 'Close') -> pd.Series:
        """Detect outliers using IQR method."""
        data = df[column]
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        return (data < lower_bound) | (data > upper_bound)


def detect_anomalies(symbol: str) -> Dict:
    """Convenience function to detect anomalies."""
    detector = AnomalyDetector()
    return detector.detect_all_anomalies(symbol)
