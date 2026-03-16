"""
Data Preprocessor Module

Handles data cleaning, normalization, and technical indicator calculation.
"""

import logging
from typing import Optional, List, Dict
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# FIX: Moved import to module level for PEP 8 compliance

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocesses stock data for model training."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Data Preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess stock data.
        
        Args:
            df: Raw stock data
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df = df.ffill()  # Forward fill
        df = df.bfill()  # Backward fill for any remaining NaNs
        
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        
        # Sort by date
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()
        elif 'date' in df.columns.lower():
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            
        logger.info(f"Cleaned data: {len(df)} rows, {df.isnull().sum().sum()} NaN values remaining")
        
        return df
    
    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        std_threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Remove outliers using z-score method.
        
        Args:
            df: DataFrame with stock data
            columns: Columns to check for outliers
            std_threshold: Standard deviation threshold
            
        Returns:
            DataFrame with outliers removed
        """
        df = df.copy()
        
        if columns is None:
            columns = ['open', 'high', 'low', 'close', 'volume']
            
        columns = [col for col in columns if col in df.columns]
        
        for col in columns:
            if df[col].dtype in [np.float64, np.int64]:
                mean = df[col].mean()
                std = df[col].std()
                
                lower_bound = mean - std_threshold * std
                upper_bound = mean + std_threshold * std
                
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                
        logger.info(f"Removed outliers: {len(df)} rows remaining")
        
        return df
    
    def normalize_data(
        self,
        df: pd.DataFrame,
        method: str = "log_returns",
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Normalize stock data using various methods.
        
        Args:
            df: DataFrame with stock data
            method: Normalization method (log_returns, zscore, minmax, none)
            columns: Columns to normalize
            
        Returns:
            DataFrame with normalized values
        """
        df = df.copy()
        
        if method == "none" or method is None:
            return df
            
        if columns is None:
            columns = ['open', 'high', 'low', 'close']
            
        columns = [col for col in columns if col in df.columns]
        
        if method == "log_returns":
            for col in columns:
                df[f'{col}_log_return'] = np.log(df[col] / df[col].shift(1))
                
        elif method == "zscore":
            for col in columns:
                mean = df[col].mean()
                std = df[col].std()
                df[f'{col}_zscore'] = (df[col] - mean) / std
                
        elif method == "minmax":
            for col in columns:
                min_val = df[col].min()
                max_val = df[col].max()
                df[f'{col}_minmax'] = (df[col] - min_val) / (max_val - min_val)
                
        logger.info(f"Normalized data using {method} method")
        
        return df
    
    def add_sma(self, df: pd.DataFrame, windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Add Simple Moving Average indicators.
        
        Args:
            df: DataFrame with stock data
            windows: List of window sizes
            
        Returns:
            DataFrame with SMA indicators
        """
        df = df.copy()
        
        if windows is None:
            windows = [10, 20, 30, 50, 100, 200]
            
        for window in windows:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            
        logger.info(f"Added SMA indicators for windows: {windows}")
        
        return df
    
    def add_ema(self, df: pd.DataFrame, windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Add Exponential Moving Average indicators.
        
        Args:
            df: DataFrame with stock data
            windows: List of window sizes
            
        Returns:
            DataFrame with EMA indicators
        """
        df = df.copy()
        
        if windows is None:
            windows = [12, 26]
            
        for window in windows:
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
            
        logger.info(f"Added EMA indicators for windows: {windows}")
        
        return df
    
    def add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add Relative Strength Index indicator.
        
        Args:
            df: DataFrame with stock data
            period: RSI period
            
        Returns:
            DataFrame with RSI indicator
        """
        df = df.copy()
        
        # Calculate RSI
        delta = df['close'].diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # FIX: Removed redundant rolling mean calculation, using only EWM (Wilder's Smoothing)
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        df[f'rsi_{period}'] = rsi
        
        logger.info(f"Added RSI indicator with period: {period}")
        
        return df
    
    def add_macd(
        self,
        df: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> pd.DataFrame:
        """
        Add MACD (Moving Average Convergence Divergence) indicator.
        
        Args:
            df: DataFrame with stock data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            DataFrame with MACD indicator
        """
        df = df.copy()
        
        # Calculate MACD
        ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_histogram'] = macd - signal
        
        logger.info(f"Added MACD indicator (fast={fast_period}, slow={slow_period}, signal={signal_period})")
        
        return df
    
    def add_bollinger_bands(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0
    ) -> pd.DataFrame:
        """
        Add Bollinger Bands indicator.
        
        Args:
            df: DataFrame with stock data
            period: Moving average period
            std_dev: Standard deviation multiplier
            
        Returns:
            DataFrame with Bollinger Bands
        """
        df = df.copy()
        
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        
        df['bb_upper'] = sma + (std_dev * std)
        df['bb_middle'] = sma
        df['bb_lower'] = sma - (std_dev * std)
        
        # Calculate bandwidth and %B
        df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        logger.info(f"Added Bollinger Bands (period={period}, std_dev={std_dev})")
        
        return df
    
    def add_atr(
        self,
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.DataFrame:
        """
        Add Average True Range (ATR) indicator.
        
        Args:
            df: DataFrame with stock data
            period: ATR period
            
        Returns:
            DataFrame with ATR indicator
        """
        df = df.copy()
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        df[f'atr_{period}'] = atr
        
        logger.info(f"Added ATR indicator with period: {period}")
        
        return df
    
    def add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add On-Balance Volume (OBV) indicator.
        
        Args:
            df: DataFrame with stock data
            
        Returns:
            DataFrame with OBV indicator
        """
        df = df.copy()
        
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv'] = obv
        
        # Add OBV SMA
        df['obv_sma'] = obv.rolling(window=20).mean()
        
        logger.info("Added OBV indicator")
        
        return df
    
    def add_all_indicators(
        self,
        df: pd.DataFrame,
        config: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Add all technical indicators to the data.
        
        Args:
            df: DataFrame with stock data
            config: Configuration dictionary
            
        Returns:
            DataFrame with all indicators
        """
        if config is None:
            config = {}
            
        df = df.copy()
        
        # Get indicator settings from config
        sma_windows = config.get('sma_windows', [10, 20, 30, 50, 100, 200])
        ema_windows = config.get('ema_windows', [12, 26])
        rsi_period = config.get('rsi_period', 14)
        macd_fast = config.get('macd_fast', 12)
        macd_slow = config.get('macd_slow', 26)
        macd_signal = config.get('macd_signal', 9)
        bb_period = config.get('bollinger_period', 20)
        bb_std = config.get('bollinger_std', 2)
        
        # Add indicators
        df = self.add_sma(df, sma_windows)
        df = self.add_ema(df, ema_windows)
        df = self.add_rsi(df, rsi_period)
        df = self.add_macd(df, macd_fast, macd_slow, macd_signal)
        df = self.add_bollinger_bands(df, bb_period, bb_std)
        df = self.add_atr(df)
        df = self.add_obv(df)
        
        # Add price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_5d'] = df['close'].pct_change(5)
        df['price_change_10d'] = df['close'].pct_change(10)
        df['price_change_20d'] = df['close'].pct_change(20)
        
        # Add volume change
        df['volume_change'] = df['volume'].pct_change()
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        # Drop NaN values created by indicators
        df = df.dropna()
        
        logger.info(f"Added all technical indicators: {len(df.columns)} total columns")
        
        return df
    
    def split_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Dict[str, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: DataFrame to split
            train_ratio: Ratio of training data
            validation_ratio: Ratio of validation data
            test_ratio: Ratio of test data
            
        Returns:
            Dictionary with train, validation, and test DataFrames
        """
        assert abs(train_ratio + validation_ratio + test_ratio - 1.0) < 0.001, \
            "Ratios must sum to 1"
            
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + validation_ratio))
        
        train = df.iloc[:train_end]
        validation = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]
        
        logger.info(f"Split data: train={len(train)}, validation={len(validation)}, test={len(test)}")
        
        return {
            'train': train,
            'validation': validation,
            'test': test
        }
    
    def create_sequences(
        self,
        data: np.ndarray,
        sequence_length: int = 60
    ) -> tuple:
        """
        Create sequences for LSTM/GRU models.
        
        Args:
            data: Input data array
            sequence_length: Length of sequences
            
        Returns:
            Tuple of (X, y) arrays
        """
        X, y = [], []
        
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length])
            
        return np.array(X), np.array(y)
    
    def prepare_for_ml(
        self,
        df: pd.DataFrame,
        target_column: str = 'close',
        feature_columns: Optional[List[str]] = None,
        sequence_length: int = 60
    ) -> Dict[str, np.ndarray]:
        """
        Prepare data for machine learning models.
        
        Args:
            df: DataFrame with features
            target_column: Target column to predict
            feature_columns: List of feature columns (if None, use all)
            sequence_length: Sequence length for LSTM/GRU
            
        Returns:
            Dictionary with prepared data arrays
        """
        df = df.copy()
        
        # Select features
        if feature_columns is None:
            # Use all numeric columns except target
            feature_columns = [col for col in df.columns 
                             if col != target_column and df[col].dtype in [np.float64, np.int64]]
        
        # Handle missing values
        df = df[feature_columns + [target_column]].dropna()
        
        # Scale features
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[feature_columns + [target_column]])
        
        # Create sequences
        X, y = self.create_sequences(scaled_data, sequence_length)
        
        # Split target
        target_idx = len(feature_columns)
        y = y[:, target_idx]
        
        # Train/test split
        split_idx = int(len(X) * 0.8)
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"Prepared data for ML: X_train={X_train.shape}, X_test={X_test.shape}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'scaler': scaler,
            'feature_columns': feature_columns
        }
