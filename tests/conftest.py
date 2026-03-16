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
def sample_stock_data_with_zeros():
    """Generate sample stock data including zero/near-zero prices for MAPE testing."""
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    
    # Include some zero/near-zero prices
    prices = np.array([0.0, 0.01, 0.05, 100.0, 101.0, 99.0, 0.0, 100.5])
    prices = np.tile(prices, len(dates) // len(prices) + 1)[:len(dates)]
    
    df = pd.DataFrame({
        'close': prices,
    }, index=dates[:len(prices)])
    
    return df


@pytest.fixture
def mock_yfinance_download(monkeypatch):
    """Mock yfinance download to return sample data."""
    def mock_download(*args, **kwargs):
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'Open': np.random.uniform(90, 110, 100),
            'High': np.random.uniform(95, 115, 100),
            'Low': np.random.uniform(85, 105, 100),
            'Close': np.random.uniform(90, 110, 100),
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
    
    monkeypatch.setattr('yfinance.download', mock_download)


@pytest.fixture
def mock_arima_model():
    """Mock ARIMA model for testing."""
    from unittest.mock import Mock
    
    mock_model = Mock()
    mock_results = Mock()
    mock_results.forecast.return_value = np.array([100.0, 101.0, 102.0])
    mock_results.get_forecast.return_value = Mock(
        predicted_mean=np.array([100.0, 101.0, 102.0]),
        conf_int=Mock(return_value=pd.DataFrame({
            'lower': [99.0, 100.0, 101.0],
            'upper': [101.0, 102.0, 103.0]
        }))
    )
    mock_results.aic = 1000.0
    mock_results.summary.return_value = "ARIMA Summary"
    
    return mock_model, mock_results


@pytest.fixture
def mock_lstm_model():
    """Mock LSTM model for testing."""
    from unittest.mock import Mock, MagicMock
    
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[100.0], [101.0], [102.0]])
    mock_model.evaluate.return_value = 0.01
    mock_model.summary.return_value = "LSTM Summary"
    
    return mock_model


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
