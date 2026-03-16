"""
Configuration Management Module

Enhanced configuration handling with Pydantic validation, environment variable support,
and comprehensive schema definitions for the Stock Prediction CLI.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from datetime import timedelta
from enum import Enum

import yaml
from pydantic import BaseModel, Field, validator, field_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


# Enums for type safety
class NormalizationMethod(str, Enum):
    """Data normalization methods."""
    LOG_RETURNS = "log_returns"
    ZSCORE = "zscore"
    MINMAX = "minmax"
    NONE = "none"


class ModelType(str, Enum):
    """Available model types."""
    ARIMA = "arima"
    LSTM = "lstm"
    GRU = "gru"
    ENSEMBLE = "ensemble"


class SentimentMethod(str, Enum):
    """Sentiment analysis methods."""
    VADER = "vader"
    TEXTBLOB = "textblob"
    COMBINED = "combined"


# Pydantic Models for Configuration

class YahooConfig(BaseModel):
    """Yahoo Finance configuration."""
    default_years: int = Field(default=5, ge=1, le=50)
    interval: str = Field(default="1d")
    auto_adjust: bool = True
    back_adjust: bool = False
    repair: bool = False
    keepna: bool = False
    actions: bool = False
    group_by: str = "ticker"
    threads: bool = True
    progress: bool = True


class AlphaVantageConfig(BaseModel):
    """Alpha Vantage API configuration."""
    api_key: str = Field(default="")
    base_url: str = "https://www.alphavantage.co/query"
    timeout: int = Field(default=30, ge=1, le=120)
    
    @field_validator('api_key', mode='before')
    @classmethod
    def get_api_key_from_env(cls, v: str) -> str:
        """Get API key from environment variable if not provided."""
        return os.environ.get('ALPHA_VANTAGE_API_KEY', v)


class RealtimeConfig(BaseModel):
    """Real-time data configuration."""
    enabled: bool = False
    update_interval: int = Field(default=60, ge=5)
    websocket: bool = False


class DataConfig(BaseModel):
    """Data collection configuration."""
    yahoo: YahooConfig = Field(default_factory=YahooConfig)
    alpha_vantage: AlphaVantageConfig = Field(default_factory=AlphaVantageConfig)
    realtime: RealtimeConfig = Field(default_factory=RealtimeConfig)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_backoff: float = Field(default=2.0, ge=1.0, le=10.0)
    
    @property
    def api_key(self) -> str:
        """Get Alpha Vantage API key."""
        return self.alpha_vantage.api_key


class ARIMAConfig(BaseModel):
    """ARIMA model configuration."""
    order: List[int] = Field(default=[5, 1, 0])
    seasonal_order: List[int] = Field(default=[0, 0, 0, 0])
    auto_fit: bool = False
    max_p: int = Field(default=5, ge=0, le=10)
    max_d: int = Field(default=2, ge=0, le=3)
    max_q: int = Field(default=5, ge=0, le=10)
    
    @validator('order')
    def validate_order(cls, v):
        if len(v) != 3:
            raise ValueError("ARIMA order must have 3 values (p, d, q)")
        return v


class LSTMConfig(BaseModel):
    """LSTM model configuration."""
    sequence_length: int = Field(default=60, ge=10, le=500)
    lstm_units: List[int] = Field(default=[50, 50])
    dropout: float = Field(default=0.2, ge=0.0, le=0.5)
    epochs: int = Field(default=50, ge=1, le=500)
    batch_size: int = Field(default=32, ge=1, le=512)
    learning_rate: float = Field(default=0.001, ge=0.00001, le=0.1)
    bidirectional: bool = False
    attention: bool = False
    early_stopping_patience: int = Field(default=10, ge=1, le=50)
    use_multifeature: bool = True
    
    @validator('lstm_units')
    def validate_units(cls, v):
        if not v or len(v) == 0:
            raise ValueError("lstm_units cannot be empty")
        return v


class GRUConfig(BaseModel):
    """GRU model configuration (inherits from LSTMConfig)."""
    sequence_length: int = Field(default=60, ge=10, le=500)
    gru_units: List[int] = Field(default=[50, 50])
    dropout: float = Field(default=0.2, ge=0.0, le=0.5)
    epochs: int = Field(default=50, ge=1, le=500)
    batch_size: int = Field(default=32, ge=1, le=512)
    learning_rate: float = Field(default=0.001, ge=0.00001, le=0.1)
    bidirectional: bool = False
    attention: bool = False
    early_stopping_patience: int = Field(default=10, ge=1, le=50)
    use_multifeature: bool = True
    
    @validator('gru_units')
    def validate_units(cls, v):
        if not v or len(v) == 0:
            raise ValueError("gru_units cannot be empty")
        return v


class EnsembleConfig(BaseModel):
    """Ensemble model configuration."""
    weights: Dict[str, float] = Field(
        default={'arima': 0.3, 'lstm': 0.4, 'gru': 0.3}
    )
    use_stacking: bool = False
    stacking_meta_learner: str = "ridge"
    dynamic_weights: bool = False
    recalculate_interval: int = Field(default=5, ge=1)
    
    @validator('weights')
    def validate_weights(cls, v):
        total = sum(v.values())
        if not 0.99 <= total <= 1.01:
            # Normalize weights
            total = sum(v.values())
            v = {k: val/total for k, val in v.items()}
        return v


class ModelsConfig(BaseModel):
    """Models configuration."""
    arima: ARIMAConfig = Field(default_factory=ARIMAConfig)
    lstm: LSTMConfig = Field(default_factory=LSTMConfig)
    gru: GRUConfig = Field(default_factory=GRUConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    
    def get_model_config(self, model_type: str) -> Union[ARIMAConfig, LSTMConfig, GRUConfig, EnsembleConfig]:
        """Get configuration for a specific model type."""
        return getattr(self, model_type.lower())


class IndicatorsConfig(BaseModel):
    """Technical indicators configuration."""
    sma_windows: List[int] = Field(default=[10, 20, 30, 50, 100, 200])
    ema_windows: List[int] = Field(default=[12, 26])
    rsi_period: int = Field(default=14, ge=2, le=100)
    macd_fast: int = Field(default=12, ge=2, le=100)
    macd_slow: int = Field(default=26, ge=2, le=200)
    macd_signal: int = Field(default=9, ge=2, le=50)
    bollinger_period: int = Field(default=20, ge=5, le=100)
    bollinger_std: float = Field(default=2.0, ge=0.5, le=5.0)
    volume_sma_period: int = Field(default=20, ge=5, le=100)
    # New indicators
    stochastic_period: int = Field(default=14, ge=5, le=50)
    williams_period: int = Field(default=14, ge=5, le=50)
    cci_period: int = Field(default=20, ge=5, le=50)
    adx_period: int = Field(default=14, ge=5, le=50)


class SentimentConfig(BaseModel):
    """Sentiment analysis configuration."""
    enabled: bool = False
    method: SentimentMethod = Field(default=SentimentMethod.VADER)
    vader_lexicon_file: str = "vader_lexicon.txt"
    textblob_threshold: float = Field(default=0.0, ge=-1.0, le=1.0)
    news_enabled: bool = False
    news_sources: List[str] = Field(default=["reuters", "bloomberg", "cnbc"])
    news_keywords: List[str] = Field(default_factory=list)
    max_articles_per_day: int = Field(default=50, ge=1, le=200)


class PreprocessingConfig(BaseModel):
    """Preprocessing configuration."""
    normalization: NormalizationMethod = Field(default=NormalizationMethod.LOG_RETURNS)
    train_test_split: float = Field(default=0.8, ge=0.5, le=0.95)
    validation_split: float = Field(default=0.1, ge=0.0, le=0.3)
    shuffle: bool = False
    outlier_threshold: float = Field(default=3.0, ge=1.0, le=10.0)
    remove_outliers: bool = False


class BacktestConfig(BaseModel):
    """Backtesting configuration."""
    enabled: bool = False
    initial_capital: float = Field(default=100000, ge=1000)
    commission: float = Field(default=0.001, ge=0.0, le=0.1)
    slippage: float = Field(default=0.0005, ge=0.0, le=0.01)


class APIConfig(BaseModel):
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1024, le=65535)
    debug: bool = False
    reload: bool = False
    workers: int = Field(default=1, ge=1, le=16)
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    rate_limit_per_minute: int = Field(default=60, ge=1)
    api_key: str = Field(default="")
    
    @field_validator('api_key', mode='before')
    @classmethod
    def get_api_key_from_env(cls, v: str) -> str:
        """Get API key from environment variable if not provided."""
        return os.environ.get('API_KEY', v)


class DashboardConfig(BaseModel):
    """Dashboard configuration."""
    theme: str = "dark"
    refresh_interval: int = Field(default=60, ge=10)
    charts: List[str] = Field(
        default_factory=lambda: ["candlestick", "sentiment", "prediction", "technical_indicators"]
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/app.log"
    max_bytes: int = Field(default=10485760, ge=1048576)  # 10MB
    backup_count: int = Field(default=5, ge=1, le=20)
    console: bool = True


# Main Application Configuration

class AppConfig(BaseModel):
    """
    Main application configuration.
    
    This is the root configuration object that combines all
    sub-configurations into a single, validated structure.
    """
    data: DataConfig = Field(default_factory=DataConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    indicators: IndicatorsConfig = Field(default_factory=IndicatorsConfig)
    sentiment: SentimentConfig = Field(default_factory=SentimentConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Metadata
    version: str = "2.0.0"
    environment: str = "development"
    
    class Config:
        """Pydantic config."""
        validate_assignment = True
        extra = "forbid"


# Configuration Loading Functions

def load_config(config_path: str = 'config.yaml') -> AppConfig:
    """
    Load configuration from YAML file with validation.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Validated AppConfig instance
        
    Raises:
        ConfigurationError: If config file is invalid
    """
    from .exceptions import ConfigurationError
    
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        return AppConfig()
    
    try:
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f) or {}
        
        return AppConfig(**config_data)
        
    except yaml.YAMLError as e:
        raise ConfigurationError(
            f"Invalid YAML in config file: {e}",
            error_code='C002',
            config_path=config_path
        )
    except Exception as e:
        raise ConfigurationError(
            f"Error loading configuration: {e}",
            error_code='C001',
            config_path=config_path
        )


def validate_config(config: Dict[str, Any]) -> AppConfig:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Validated AppConfig instance
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    from .exceptions import ConfigurationError
    
    try:
        return AppConfig(**config)
    except Exception as e:
        raise ConfigurationError(
            f"Configuration validation failed: {e}",
            error_code='C005'
        )


def get_config_value(config: AppConfig, path: str, default: Any = None) -> Any:
    """
    Get configuration value using dot notation.
    
    Args:
        config: AppConfig instance
        path: Dot-separated path (e.g., 'models.arima.order')
        default: Default value if path not found
        
    Returns:
        Configuration value or default
    """
    parts = path.split('.')
    current = config
    
    for part in parts:
        if hasattr(current, part):
            current = getattr(current, part)
        else:
            return default
    
    return current


