"""
Stock Prediction CLI

A real-time stock market prediction system using machine learning.
"""

__version__ = "1.0.0"
__author__ = "Stock Prediction Team"

from .data.collector import StockDataCollector
from .data.preprocessor import DataPreprocessor
from .sentiment.analyzer import SentimentAnalyzer
from .models.arima_model import ARIMAModel
from .models.lstm_model import LSTMModel, GRUModel
from .models.ensemble_model import EnsembleModel
from .models.predictor import StockPredictor, create_predictor
from .visualization.dashboard import StockDashboard
from .visualization.plots import StockPlotter

__all__ = [
    'StockDataCollector',
    'DataPreprocessor',
    'SentimentAnalyzer',
    'ARIMAModel',
    'LSTMModel',
    'GRUModel',
    'EnsembleModel',
    'StockPredictor',
    'create_predictor',
    'StockDashboard',
    'StockPlotter'
]
