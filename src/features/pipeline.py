"""
Feature Pipeline

Comprehensive feature engineering pipeline that combines:
- Basic technical indicators
- Advanced technical indicators
- Alternative data (macroeconomic, market indices)
- Sentiment features
"""

import logging
from typing import Optional, Dict, List, Union
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Complete feature engineering pipeline.
    
    Integrates all feature sources into a unified interface.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feature pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Feature configuration
        self.basic_indicators = self.config.get('basic_indicators', True)
        self.advanced_indicators = self.config.get('advanced_indicators', True)
        self.alternative_data = self.config.get('alternative_data', False)
        self.sentiment_features = self.config.get('sentiment_features', False)
        
        # Feature storage
        self.feature_names = []
        self.scaler = None
        
    def transform(
        self,
        df: pd.DataFrame,
        include_targets: bool = True
    ) -> pd.DataFrame:
        """
        Transform raw data into features.
        
        Args:
            df: Raw OHLCV DataFrame
            include_targets: Include target variables
            
        Returns:
            DataFrame with all features
        """
        logger.info("Starting feature transformation...")
        
        df = df.copy()
        
        # 1. Basic technical indicators
        if self.basic_indicators:
            df = self._add_basic_indicators(df)
            
        # 2. Advanced technical indicators
        if self.advanced_indicators:
            df = self._add_advanced_indicators(df)
            
        # 3. Lag features
        df = self._add_lag_features(df)
            
        # 4. Rolling statistics
        df = self._add_rolling_statistics(df)
            
        # 5. Target variable
        if include_targets:
            df = self._add_target(df)
            
        # 6. Clean up
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Store feature names
        self.feature_names = [col for col in df.columns 
                             if col not in ['open', 'high', 'low', 'close', 'volume', 'target']]
        
        logger.info(f"Created {len(self.feature_names)} features")
        
        return df
    
    def _add_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators."""
        
        # Moving averages
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
            
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        for period in [7, 14, 21]:
            avg_gain = gain.ewm(span=period, adjust=False).mean()
            avg_loss = loss.ewm(span=period, adjust=False).mean()
            rs = avg_gain / avg_loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        for period in [10, 20]:
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            df[f'bb_upper_{period}'] = sma + 2 * std
            df[f'bb_lower_{period}'] = sma - 2 * std
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
            
        # Average True Range
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        for period in [7, 14, 21]:
            df[f'atr_{period}'] = tr.ewm(span=period, adjust=False).mean()
            
        # Stochastic
        for period in [14, 21]:
            low_min = df['low'].rolling(period).min()
            high_max = df['high'].rolling(period).max()
            df[f'stoch_k_{period}'] = 100 * (df['close'] - low_min) / (high_max - low_min)
            df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(3).mean()
            
        # On-Balance Volume
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # VWAP
        df['vwap'] = ((df['close'] * df['volume']).cumsum() / df['volume'].cumsum())
        
        return df
    
    def _add_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced technical indicators."""
        
        try:
            from src.features.advanced_indicators import (
                VolatilityIndicators,
                OrderFlowIndicators,
                MarketMicrostructure
            )
            
            # Volatility regime
            df['volatility_regime'] = VolatilityIndicators.detect_volatility_regime(df['close'])
            df['volatility_percentile'] = VolatilityIndicators.calculate_volatility_percentile(df['close'])
            
            # Order flow
            flow_data = OrderFlowIndicators.calculate_volume_delta(df['close'], df['volume'])
            for col in ['delta', 'imbalance', 'cmf']:
                if col in flow_data.columns:
                    df[f'flow_{col}'] = flow_data[col]
                    
            # Money flow
            cmf_data = OrderFlowIndicators.calculate_money_flow(
                df['high'], df['low'], df['close'], df['volume']
            )
            if 'cmf' in cmf_data.columns:
                df['money_flow'] = cmf_data['cmf']
                
            # Spread
            spread_data = MarketMicrostructure.calculate_spread(df['high'], df['low'])
            for col in ['spread', 'spread_bps']:
                if col in spread_data.columns:
                    df[col] = spread_data[col]
                    
            # Amihud illiquidity
            returns = df['close'].pct_change()
            df['amihud_illiquidity'] = MarketMicrostructure.calculate_amihud_illiquidity(
                returns, df['volume']
            )
            
        except ImportError:
            logger.warning("Advanced indicators not available")
            
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag features."""
        
        for col in ['close', 'volume']:
            if col in df.columns:
                for lag in [1, 2, 3, 5, 10, 20]:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                    
        # Returns
        for period in [1, 2, 3, 5, 10, 20]:
            df[f'return_{period}d'] = df['close'].pct_change(period)
            
        return df
    
    def _add_rolling_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistics."""
        
        for col in ['close', 'volume']:
            if col not in df.columns:
                continue
                
            for window in [5, 10, 20, 50]:
                df[f'{col}_mean_{window}'] = df[col].rolling(window).mean()
                df[f'{col}_std_{window}'] = df[col].rolling(window).std()
                df[f'{col}_min_{window}'] = df[col].rolling(window).min()
                df[f'{col}_max_{window}'] = df[col].rolling(window).max()
                
        # Price momentum
        for window in [5, 10, 20]:
            df[f'momentum_{window}'] = df['close'] - df['close'].shift(window)
            df[f'roc_{window}'] = (df['close'] - df['close'].shift(window)) / df['close'].shift(window)
            
        return df
    
    def _add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add target variable."""
        
        # Next day return
        df['target'] = df['close'].shift(-1) / df['close'] - 1
        
        # Binary target: 1 if price goes up, 0 otherwise
        df['target_direction'] = (df['target'] > 0).astype(int)
        
        return df
    
    def select_features(
        self,
        X: pd.DataFrame,
        method: str = 'variance',
        n_features: Optional[int] = None,
        threshold: float = 0.01
    ) -> List[str]:
        """
        Select most important features.
        
        Args:
            X: Feature DataFrame
            method: Selection method
            n_features: Number of features to select
            threshold: Threshold for variance method
            
        Returns:
            List of selected feature names
        """
        if method == 'variance':
            # Remove low variance features
            variances = X.var()
            selected = variances[variances > threshold].index.tolist()
            
        elif method == 'correlation':
            # Remove highly correlated features
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            selected = [col for col in upper.columns 
                       if not any(upper[col] > 0.95)]
            
        elif method == 'importance':
            # This would require a fitted model
            logger.warning("Importance method requires a fitted model, using variance instead")
            return self.select_features(X, 'variance', n_features, threshold)
            
        else:
            selected = X.columns.tolist()
            
        # Limit number of features
        if n_features and len(selected) > n_features:
            selected = selected[:n_features]
            
        return selected
    
    def get_feature_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'mutual_info'
    ) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            X: Feature DataFrame
            y: Target series
            method: Method to use
            
        Returns:
            DataFrame with feature importance
        """
        if method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_regression
            
            # Handle missing values
            X_clean = X.fillna(0)
            
            mi_scores = mutual_info_regression(X_clean, y)
            
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': mi_scores
            }).sort_values('importance', ascending=False)
            
        elif method == 'permutation':
            from sklearn.inspection import permutation_importance
            
            # Would need a fitted model
            logger.warning("Permutation importance requires a fitted model")
            return pd.DataFrame()
            
        else:
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': 0
            })
            
        return importance
    
    def save_feature_config(self, path: Path) -> None:
        """Save feature configuration."""
        import json
        
        config = {
            'feature_names': self.feature_names,
            'basic_indicators': self.basic_indicators,
            'advanced_indicators': self.advanced_indicators,
            'alternative_data': self.alternative_data,
            'sentiment_features': self.sentiment_features
        }
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
            
    def load_feature_config(self, path: Path) -> None:
        """Load feature configuration."""
        import json
        
        with open(path, 'r') as f:
            config = json.load(f)
            
        self.feature_names = config.get('feature_names', [])
        self.basic_indicators = config.get('basic_indicators', True)
        self.advanced_indicators = config.get('advanced_indicators', True)
        self.alternative_data = config.get('alternative_data', False)
        self.sentiment_features = config.get('sentiment_features', False)
