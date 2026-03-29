"""
Base Model Interface

Abstract base class for all prediction models in the stock prediction system.
Provides a consistent interface for training, prediction, and model management.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all prediction models.
    
    All models must implement:
    - fit(): Train the model on historical data
    - predict(): Generate predictions
    - save(): Save model to disk
    - load(): Load model from disk
    
    Optional methods:
    - predict_with_uncertainty(): Generate predictions with confidence intervals
    - get_feature_importance(): Return feature importance scores
    - get_config(): Return model configuration
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base model.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config or {}
        self.is_fitted = False
        self.scaler = None
        self.feature_names = []
        self.training_history = {}
        
    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs
    ) -> 'BaseModel':
        """
        Train the model on historical data.
        
        Args:
            X: Feature DataFrame
            y: Target series
            validation_data: Optional tuple of (X_val, y_val)
            **kwargs: Additional training parameters
            
        Returns:
            Self
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Generate predictions for input features.
        
        Args:
            X: Feature DataFrame
            **kwargs: Additional prediction parameters
            
        Returns:
            Array of predictions
        """
        pass
    
    def predict_with_uncertainty(
        self,
        X: pd.DataFrame,
        n_samples: int = 100,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Generate predictions with uncertainty estimates.
        
        Args:
            X: Feature DataFrame
            n_samples: Number of Monte Carlo samples
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with predictions, lower_bound, upper_bound
        """
        predictions = self.predict(X, **kwargs)
        
        return {
            'predictions': predictions,
            'lower_bound': predictions,
            'upper_bound': predictions,
            'std': np.zeros_like(predictions)
        }
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save the model
        """
        pass
    
    @abstractmethod
    def load(self, path: Path) -> 'BaseModel':
        """
        Load model from disk.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Self
        """
        pass
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with feature names and importance scores, or None
        """
        return None
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration.
        
        Returns:
            Dictionary with model configuration
        """
        return self.config.copy()
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """
        Get training history.
        
        Returns:
            Dictionary with training metrics
        """
        return self.training_history
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            X: Feature DataFrame
            y: True target values
            metrics: List of metrics to compute
            
        Returns:
            Dictionary with evaluation metrics
        """
        if metrics is None:
            metrics = ['mae', 'rmse', 'mape', 'r2']
            
        predictions = self.predict(X)
        
        results = {}
        
        if 'mae' in metrics:
            from sklearn.metrics import mean_absolute_error
            results['mae'] = mean_absolute_error(y, predictions)
            
        if 'rmse' in metrics:
            from sklearn.metrics import mean_squared_error
            results['rmse'] = np.sqrt(mean_squared_error(y, predictions))
            
        if 'mape' in metrics:
            mask = y != 0
            if mask.any():
                results['mape'] = np.mean(np.abs((y[mask] - predictions[mask]) / y[mask])) * 100
            else:
                results['mape'] = 0.0
                
        if 'r2' in metrics:
            from sklearn.metrics import r2_score
            results['r2'] = r2_score(y, predictions)
            
        if 'direction_accuracy' in metrics:
            actual_direction = np.sign(np.diff(y.values))
            pred_direction = np.sign(np.diff(predictions))
            results['direction_accuracy'] = np.mean(actual_direction == pred_direction)
            
        return results
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fitted={self.is_fitted})"


class TimeSeriesModel(BaseModel):
    """
    Base class for time series prediction models.
    
    Extends BaseModel with time series specific functionality.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.sequence_length = self.config.get('sequence_length', 60)
        self.forecast_horizon = self.config.get('forecast_horizon', 1)
        
    def create_sequences(
        self,
        data: np.ndarray,
        sequence_length: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series modeling.
        
        Args:
            data: Input data array
            sequence_length: Length of sequences
            
        Returns:
            Tuple of (X, y) sequences
        """
        seq_len = sequence_length or self.sequence_length
        
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len])
            
        return np.array(X), np.array(y)
    
    def inverse_transform_predictions(
        self,
        predictions: np.ndarray
    ) -> np.ndarray:
        """
        Inverse transform scaled predictions back to original scale.
        
        Args:
            predictions: Scaled predictions
            
        Returns:
            Original scale predictions
        """
        if self.scaler is None:
            return predictions
            
        if predictions.ndim == 1:
            return self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        return self.scaler.inverse_transform(predictions)


class EnsembleModelBase(BaseModel):
    """
    Base class for ensemble models.
    
    Provides common functionality for combining multiple models.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.models: Dict[str, BaseModel] = {}
        self.weights: Dict[str, float] = {}
        
    def add_model(self, name: str, model: BaseModel, weight: Optional[float] = None) -> None:
        """
        Add a model to the ensemble.
        
        Args:
            name: Model identifier
            model: Model instance
            weight: Model weight (optional)
        """
        self.models[name] = model
        
        if weight is not None:
            self.weights[name] = weight
        elif name not in self.weights:
            self.weights[name] = 1.0 / len(self.models)
            
    def _normalize_weights(self) -> None:
        """Normalize model weights to sum to 1."""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
            
    def get_model(self, name: str) -> Optional[BaseModel]:
        """Get a model by name."""
        return self.models.get(name)
    
    def list_models(self) -> List[str]:
        """List all model names in the ensemble."""
        return list(self.models.keys())
