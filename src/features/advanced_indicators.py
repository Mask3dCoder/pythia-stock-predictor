"""
Advanced Technical Indicators

Comprehensive technical analysis indicators including:
- Volatility regimes (low, medium, high volatility detection)
- Order flow metrics (delta, imbalance, cumulative volume delta)
- Spectral features (FFT, wavelets)
- Market microstructure (VWAP, TWAP, spread)
"""

import logging
from typing import Optional, List, Union
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq

logger = logging.getLogger(__name__)

# Optional imports
try:
    from PyEMD import EMD

    PYEMD_AVAILABLE = True
except ImportError:
    PYEMD_AVAILABLE = False
    logger.warning("PyEMD not installed. Wavelet features will be limited.")


class VolatilityIndicators:
    """
    Volatility-based indicators and regime detection.
    """

    @staticmethod
    def calculate_atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """
        Calculate Average True Range (ATR).

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period

        Returns:
            ATR series
        """
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()

        return atr

    @staticmethod
    def calculate_keltner_channels(
        close: pd.Series, atr: pd.Series, ema_period: int = 20, multiplier: float = 2.0
    ) -> pd.DataFrame:
        """
        Calculate Keltner Channels.

        Args:
            close: Close prices
            atr: Average True Range
            ema_period: EMA period for middle line
            multiplier: ATR multiplier for bands

        Returns:
            DataFrame with upper, middle, lower channels
        """
        middle = close.ewm(span=ema_period, adjust=False).mean()
        upper = middle + multiplier * atr
        lower = middle - multiplier * atr

        return pd.DataFrame({"upper": upper, "middle": middle, "lower": lower})

    @staticmethod
    def calculate_donchian_channels(
        high: pd.Series, low: pd.Series, period: int = 20
    ) -> pd.DataFrame:
        """
        Calculate Donchian Channels.

        Args:
            high: High prices
            low: Low prices
            period: Channel period

        Returns:
            DataFrame with upper, middle, lower channels
        """
        upper = high.rolling(window=period).max()
        lower = low.rolling(window=period).min()
        middle = (upper + lower) / 2

        return pd.DataFrame({"upper": upper, "middle": middle, "lower": lower})

    @staticmethod
    def detect_volatility_regime(
        close: pd.Series,
        lookback: int = 20,
        low_threshold: float = 0.5,
        high_threshold: float = 1.5,
    ) -> pd.Series:
        """
        Detect volatility regime (low, medium, high).

        Uses rolling standard deviation normalized by mean.

        Args:
            close: Close prices
            lookback: Rolling window size
            low_threshold: Threshold for low volatility (percentile)
            high_threshold: Threshold for high volatility (percentile)

        Returns:
            Series with regime labels (0=low, 1=medium, 2=high)
        """
        returns = close.pct_change()
        rolling_std = returns.rolling(window=lookback).std()
        rolling_mean = returns.rolling(window=lookback).mean()

        # Normalize volatility
        volatility_ratio = rolling_std / rolling_mean.abs()

        # Calculate thresholds from data (percentiles between 0 and 1)
        low_thresh = volatility_ratio.quantile(min(max(low_threshold, 0.1), 0.5))
        high_thresh = volatility_ratio.quantile(min(max(high_threshold, 0.5), 0.9))

        # Assign regimes
        regime = pd.Series(index=close.index, dtype=int)
        regime[volatility_ratio <= low_thresh] = 0  # Low
        regime[(volatility_ratio > low_thresh) & (volatility_ratio <= high_thresh)] = (
            1  # Medium
        )
        regime[volatility_ratio > high_thresh] = 2  # High

        return regime

    @staticmethod
    def calculate_volatility_percentile(
        close: pd.Series, lookback: int = 252, current_window: int = 20
    ) -> pd.Series:
        """
        Calculate volatility percentile rank.

        Args:
            close: Close prices
            lookback: Historical lookback period
            current_window: Window for current volatility calculation

        Returns:
            Series with percentile ranks
        """
        returns = close.pct_change()
        historical_vol = returns.rolling(window=current_window).std()

        # Calculate percentile rank
        def percentile_rank(x):
            return (x < x.iloc[-1]).mean() if len(x) > 1 else 0.5

        vol_percentile = historical_vol.rolling(window=lookback).apply(
            percentile_rank, raw=False
        )

        return vol_percentile


class OrderFlowIndicators:
    """
    Order flow and volume-based indicators.
    """

    @staticmethod
    def calculate_volume_delta(
        close: pd.Series, volume: pd.Series, period: int = 14
    ) -> pd.DataFrame:
        """
        Calculate volume delta and cumulative volume delta.

        Args:
            close: Close prices
            volume: Volume
            period: Smoothing period

        Returns:
            DataFrame with delta, cumulative delta, and imbalance
        """
        # Calculate price change direction
        direction = np.sign(close.diff())

        # Calculate delta (volume at up ticks vs down ticks)
        delta = direction * volume
        delta = delta.fillna(0)

        # Cumulative volume delta
        cumulative_delta = delta.cumsum()

        # Volume imbalance
        up_volume = volume.where(close.diff() > 0, 0)
        down_volume = volume.where(close.diff() < 0, 0)

        total = up_volume + down_volume
        imbalance = (up_volume - down_volume) / total.replace(0, 1)

        # Smoothed versions
        delta_smooth = delta.ewm(span=period).mean()
        imbalance_smooth = imbalance.ewm(span=period).mean()

        return pd.DataFrame(
            {
                "delta": delta,
                "cumulative_delta": cumulative_delta,
                "delta_smooth": delta_smooth,
                "imbalance": imbalance,
                "imbalance_smooth": imbalance_smooth,
                "up_volume": up_volume,
                "down_volume": down_volume,
            }
        )

    @staticmethod
    def calculate_vwap(
        high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
    ) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume

        Returns:
            VWAP series
        """
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()

        return vwap

    @staticmethod
    def calculate_twap(close: pd.Series, period: str = "D") -> pd.Series:
        """
        Calculate Time Weighted Average Price.

        Args:
            close: Close prices
            period: Time period for TWAP calculation

        Returns:
            TWAP series
        """
        twap = close.expanding().mean()

        return twap

    @staticmethod
    def calculate_money_flow(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 14,
    ) -> pd.DataFrame:
        """
        Calculate Chaikin Money Flow.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume
            period: CMF period

        Returns:
            DataFrame with money flow and CMF
        """
        # Money Flow Multiplier
        typical_price = (high + low + close) / 3
        money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
        money_flow_multiplier = money_flow_multiplier.fillna(0)

        # Money Flow Volume
        money_flow_volume = money_flow_multiplier * volume

        # Chaikin Money Flow
        cmf = (
            money_flow_volume.rolling(window=period).sum()
            / volume.rolling(window=period).sum()
        )

        return pd.DataFrame(
            {
                "money_flow_multiplier": money_flow_multiplier,
                "money_flow_volume": money_flow_volume,
                "cmf": cmf,
            }
        )


class SpectralIndicators:
    """
    Spectral analysis indicators using FFT and wavelets.
    """

    @staticmethod
    def calculate_fft_features(close: pd.Series, n_components: int = 5) -> pd.DataFrame:
        """
        Calculate FFT-based spectral features.

        Args:
            close: Close prices
            n_components: Number of frequency components to extract

        Returns:
            DataFrame with spectral features
        """
        features = pd.DataFrame(index=close.index)

        window_size = 60  # FFT window
        step = 10

        fft_amplitudes = []
        fft_frequencies = []

        for i in range(window_size, len(close), step):
            window = close.iloc[i - window_size : i].values

            # Apply FFT
            fft_result = fft(window)
            frequencies = fftfreq(window_size)

            # Get positive frequencies only
            positive_mask = frequencies > 0
            amplitudes = np.abs(fft_result[positive_mask])
            freqs = frequencies[positive_mask]

            # Sort by amplitude
            sorted_indices = np.argsort(amplitudes)[::-1]

            # Extract top n components
            top_amplitudes = amplitudes[sorted_indices[:n_components]]
            top_freqs = freqs[sorted_indices[:n_components]]

            fft_amplitudes.append(top_amplitudes)
            fft_frequencies.append(top_freqs)

        # Create features
        for i in range(n_components):
            col_name = f"fft_amp_{i}"
            features[col_name] = 0.0
            col_freq = f"fft_freq_{i}"
            features[col_freq] = 0.0

            for j, amp in enumerate(fft_amplitudes):
                if i < len(amp):
                    idx = window_size + j * step
                    if idx < len(features):
                        features.loc[features.index[idx], col_name] = amp[i]
                        features.loc[features.index[idx], col_freq] = fft_frequencies[
                            j
                        ][i]

        # Forward fill
        features = features.ffill().fillna(0)

        return features

    @staticmethod
    def calculate_wavelet_features(
        close: pd.Series, wavelet: str = "db4", level: int = 5
    ) -> pd.DataFrame:
        """
        Calculate wavelet decomposition features.

        Args:
            close: Close prices
            wavelet: Wavelet type
            level: Decomposition level

        Returns:
            DataFrame with wavelet coefficients
        """
        try:
            import pywt

            features = pd.DataFrame(index=close.index)

            # Process in windows
            window_size = 2 ** (level + 1)
            step = 10

            for i in range(window_size, len(close), step):
                window = close.iloc[i - window_size : i].values

                # Wavelet decomposition
                coeffs = pywt.wavedec(window, wavelet, level=level)

                # Extract features from coefficients
                for j, coeff in enumerate(coeffs):
                    col_name = f"wavelet_{j}"
                    features.loc[features.index[i], col_name] = np.mean(np.abs(coeff))

            features = features.ffill().fillna(0)

        except ImportError:
            logger.warning("PyWavelets not installed. Using placeholder features.")
            features = pd.DataFrame(index=close.index)

        return features

    @staticmethod
    def calculate_spectral_entropy(close: pd.Series, window: int = 60) -> pd.Series:
        """
        Calculate spectral entropy of price series.

        Args:
            close: Close prices
            window: Rolling window size

        Returns:
            Series with spectral entropy
        """

        def entropy(x):
            # FFT
            fft_vals = np.abs(fft(x))
            # Normalize
            psd = fft_vals / np.sum(fft_vals)
            # Remove zero values
            psd = psd[psd > 0]
            # Calculate entropy
            return -np.sum(psd * np.log2(psd))

        spectral_entropy = close.rolling(window=window).apply(entropy, raw=True)

        return spectral_entropy


class MarketMicrostructure:
    """
    Market microstructure indicators.
    """

    @staticmethod
    def calculate_spread(high: pd.Series, low: pd.Series) -> pd.DataFrame:
        """
        Calculate bid-ask spread metrics.

        Args:
            high: High prices (ask)
            low: Low prices (bid)

        Returns:
            DataFrame with spread metrics
        """
        spread = high - low
        spread_pct = (spread / high) * 10000  # In basis points

        # Rolling spread
        spread_ma = spread.rolling(window=20).mean()
        spread_std = spread.rolling(window=20).std()

        # Z-score of spread
        spread_zscore = (spread - spread_ma) / spread_std.replace(0, 1)

        return pd.DataFrame(
            {
                "spread": spread,
                "spread_bps": spread_pct,
                "spread_ma": spread_ma,
                "spread_zscore": spread_zscore,
            }
        )

    @staticmethod
    def calculate_price_impact(
        close: pd.Series, volume: pd.Series, period: int = 20
    ) -> pd.DataFrame:
        """
        Estimate price impact of volume.

        Args:
            close: Close prices
            volume: Volume
            period: Rolling period

        Returns:
            DataFrame with price impact metrics
        """
        returns = close.pct_change()

        # Volume-weighted returns
        vwap_return = (returns * volume).rolling(window=period).sum() / volume.rolling(
            window=period
        ).sum()

        # Price impact coefficient (regression of returns on volume)
        def calc_impact(window):
            if len(window) < 5:
                return 0
            corr = np.corrcoef(window[:, 0], window[:, 1])[0, 1]
            return corr if not np.isnan(corr) else 0

        impact = (
            pd.DataFrame({"returns": returns, "volume": volume})
            .rolling(window=period)
            .apply(lambda x: calc_impact(x.values), raw=True)
        )

        return pd.DataFrame(
            {
                "returns": returns,
                "vwap_return": vwap_return,
                "price_impact": impact["returns"],
            }
        )

    @staticmethod
    def calculate_amihud_illiquidity(
        returns: pd.Series, volume: pd.Series, period: int = 20
    ) -> pd.Series:
        """
        Calculate Amihud illiquidity ratio.

        Args:
            returns: Returns series
            volume: Volume series
            period: Rolling period

        Returns:
            Illiquidity ratio series
        """
        # Absolute return / volume
        illiquidity = np.abs(returns) / volume

        # Rolling mean
        illiquidity_ratio = illiquidity.rolling(window=period).mean()

        return illiquidity_ratio


class AdvancedIndicators:
    """
    Composite indicator class combining all advanced indicators.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all advanced indicators.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with all indicators added
        """
        df = df.copy()

        # Volatility indicators
        atr = VolatilityIndicators.calculate_atr(df["high"], df["low"], df["close"])
        df["atr"] = atr

        keltner = VolatilityIndicators.calculate_keltner_channels(df["close"], atr)
        df["keltner_upper"] = keltner["upper"]
        df["keltner_middle"] = keltner["middle"]
        df["keltner_lower"] = keltner["lower"]

        df["volatility_regime"] = VolatilityIndicators.detect_volatility_regime(
            df["close"]
        )

        df["volatility_percentile"] = (
            VolatilityIndicators.calculate_volatility_percentile(df["close"])
        )

        # Order flow indicators
        flow_data = OrderFlowIndicators.calculate_volume_delta(
            df["close"], df["volume"]
        )
        for col in flow_data.columns:
            df[col] = flow_data[col]

        df["vwap"] = OrderFlowIndicators.calculate_vwap(
            df["high"], df["low"], df["close"], df["volume"]
        )

        cmf = OrderFlowIndicators.calculate_money_flow(
            df["high"], df["low"], df["close"], df["volume"]
        )
        df["cmf"] = cmf["cmf"]

        # Microstructure
        spread_data = MarketMicrostructure.calculate_spread(df["high"], df["low"])
        for col in spread_data.columns:
            df[col] = spread_data[col]

        returns = df["close"].pct_change()
        df["amihud_illiquidity"] = MarketMicrostructure.calculate_amihud_illiquidity(
            returns, df["volume"]
        )

        return df
