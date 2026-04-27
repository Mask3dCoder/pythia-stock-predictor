"""
Pytest configuration and shared fixtures for Stock Prediction CLI tests.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_stock_data():
    """Generate sample stock data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    
    # Generate realistic price data
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 2)
    prices = np.maximum(prices, 10)  # Ensure positive prices
    
    df = pd.DataFrame({
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    return df


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'data': {
            'yahoo': {
                'default_years': 5,
                'interval': '1d'
            }
        },
        'models': {
            'arima': {'order': [5, 1, 0]},
            'lstm': {
                'sequence_length': 60,
                'lstm_units': [50, 50],
                'dropout': 0.2,
                'epochs': 10,
                'batch_size': 32
            },
            'ensemble': {
                'weights': {'arima': 0.3, 'lstm': 0.4, 'gru': 0.3}
            }
        },
        'indicators': {
            'sma_windows': [10, 20, 30],
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9
        },
        'sentiment': {
            'enabled': False,
            'method': 'vader'
        }
    }
