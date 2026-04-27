"""
Base Classes Module

Provides abstract base classes for models, data collectors, and preprocessors
to ensure consistent interfaces across the application.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all prediction models.
    
    All models must implement fit, predict, evaluate, save, and load methods.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config or {}
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, data: pd.Series, *args, **kwargs) -> 'BaseModel':
        """
        Fit the model to training data.
        
        Args:
            data: Training data
            
        Returns:
            Self
        """
        pass
    
    @abstractmethod
    def predict(self, steps: int, *args, **kwargs) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            steps: Number of steps to predict
            
        Returns:
            Array of predictions
        """
        pass
    
    @abstractmethod
    def predict_with_confidence(
        self, 
        steps: int, 
        alpha: float = 0.05,
        *args, 
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions with confidence intervals.
        
        Args:
            steps: Number of steps to predict
            alpha: Significance level for confidence intervals
            
        Returns:
            Dictionary with predictions, lower_bound, upper_bound
        """
        pass
    
    @abstractmethod
    def evaluate(self, test_data: pd.Series) -> Dict[str, Any]:
        """
        Evaluate model on test data.
        
        Args:
            test_data: Test time series
            
        Returns:
            Dictionary with evaluation metrics
        """
        pass
    
    @abstractmethod
    def save_model(self, path: Path) -> None:
        """
        Save model to file.
        
        Args:
            path: Path to save model
        """
        pass
    
    @abstractmethod
    def load_model(self, path: Path) -> 'BaseModel':
        """
        Load model from file.
        
        Args:
            path: Path to model file
            
        Returns:
            Self
        """
        pass
    
    def get_model_summary(self) -> str:
        """
        Get model summary.
        
        Returns:
            Model summary string
        """
        return f"{self.__class__.__name__} model"
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dictionary of parameters
        """
        return self.config.copy()
    
    def set_params(self, **params) -> 'BaseModel':
        """
        Set model parameters.
        
        Args:
            **params: Parameters to set
            
        Returns:
            Self
        """
        self.config.update(params)
        return self


class BaseDataCollector(ABC):
    """
    Abstract base class for data collectors.
    
    All data collectors must implement download and save methods.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data collector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
    
    @abstractmethod
    def download(
        self,
        symbol: str,
        *args,
        **kwargs
    ) -> pd.DataFrame:
        """
        Download data for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            **kwargs: Additional download parameters
            
        Returns:
            DataFrame with downloaded data
        """
        pass
    
    @abstractmethod
    def save_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        data_type: str = "historical"
    ) -> Path:
        """
        Save data to file.
        
        Args:
            df: DataFrame to save
            symbol: Stock symbol
            data_type: Type of data
            
        Returns:
            Path to saved file
        """
        pass
    
    @abstractmethod
    def load_data(
        self,
        symbol: str,
        data_type: str = "historical"
    ) -> Optional[pd.DataFrame]:
        """
        Load saved data.
        
        Args:
            symbol: Stock symbol
            data_type: Type of data
            
        Returns:
            DataFrame if found, None otherwise
        """
        pass
    
    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get company information for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with company information
        """
        return {'symbol': symbol, 'error': 'Not implemented'}


class BasePreprocessor(ABC):
    """
    Abstract base class for data preprocessors.
    
    All preprocessors must implement cleaning, transformation, and splitting methods.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
    
    @abstractmethod
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess data.
        
        Args:
            df: Raw data
            
        Returns:
            Cleaned DataFrame
        """
        pass
    
    @abstractmethod
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to data.
        
        Args:
            df: Data with price data
            
        Returns:
            DataFrame with indicators
        """
        pass
    
    @abstractmethod
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
        pass
    
    def normalize_data(
        self,
        df: pd.DataFrame,
        method: str = "log_returns",
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Normalize data.
        
        Args:
            df: DataFrame to normalize
            method: Normalization method
            columns: Columns to normalize
            
        Returns:
            Normalized DataFrame
        """
        return df
    
    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        std_threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Remove outliers from data.
        
        Args:
            df: DataFrame
            columns: Columns to check
            std_threshold: Standard deviation threshold
            
        Returns:
            DataFrame with outliers removed
        """
        return df


class BaseSentimentAnalyzer(ABC):
    """
    Abstract base class for sentiment analyzers.
    
    All sentiment analyzers must implement text analysis methods.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the sentiment analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
    
    @abstractmethod
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        pass
    
    @abstractmethod
    def analyze_batch(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            DataFrame with sentiment results
        """
        pass
    
    def get_sentiment_summary(self, results: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics of sentiment analysis.
        
        Args:
            results: DataFrame with sentiment results
            
        Returns:
            Dictionary with summary statistics
        """
        if 'sentiment' not in results.columns:
            return {}
        
        return {
            'positive_count': len(results[results['sentiment'] == 'positive']),
            'negative_count': len(results[results['sentiment'] == 'negative']),
            'neutral_count': len(results[results['sentiment'] == 'neutral']),
        }


# Utility classes

# ModelRegistry is in src.models.registry as the canonical implementation
