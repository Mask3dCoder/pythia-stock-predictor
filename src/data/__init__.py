"""
Data Collection Module for Stock Prediction CLI

This module handles data collection from various sources:
- Yahoo Finance (historical and real-time)
- Alpha Vantage (historical and real-time)
- News/Sentiment data
"""

from .collector import StockDataCollector
from .preprocessor import DataPreprocessor

__all__ = ['StockDataCollector', 'DataPreprocessor']
