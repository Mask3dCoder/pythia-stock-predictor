"""
Core Infrastructure Module

Provides foundational components for the Stock Prediction CLI including
custom exceptions, configuration management, base classes, and utilities.
"""

from .exceptions import (
    PredictionError,
    DataCollectionError,
    ModelTrainingError,
    ValidationError,
    ConfigurationError,
    APIError,
    format_exception,
    retryable_error,
)
from .config import (
    AppConfig,
    load_config,
    validate_config,
    get_config_value,
    DataConfig,
    ModelsConfig,
    IndicatorsConfig,
    SentimentConfig,
    PreprocessingConfig,
    BacktestConfig,
    APIConfig,
    DashboardConfig,
    LoggingConfig,
)
from .base import (
    BaseModel,
    BaseDataCollector,
    BasePreprocessor,
    BaseSentimentAnalyzer,
)

__all__ = [
    # Exceptions
    'PredictionError',
    'DataCollectionError',
    'ModelTrainingError',
    'ValidationError',
    'ConfigurationError',
    'APIError',
    'format_exception',
    'retryable_error',
    # Config
    'AppConfig',
    'load_config',
    'validate_config',
    'get_config_value',
    'DataConfig',
    'ModelsConfig',
    'IndicatorsConfig',
    'SentimentConfig',
    'PreprocessingConfig',
    'BacktestConfig',
    'APIConfig',
    'DashboardConfig',
    'LoggingConfig',
    # Base classes
    'BaseModel',
    'BaseDataCollector',
    'BasePreprocessor',
    'BaseSentimentAnalyzer',
]
