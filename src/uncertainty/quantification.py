"""
Uncertainty Quantification

Implements:
- Monte Carlo dropout for uncertainty estimation
- Deep ensembles
- Conformal prediction intervals
"""

import logging
from typing import Optional, Dict, List, Tuple, Any
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class UncertaintyEstimator(ABC):
    """Abstract base class for uncertainty estimators."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'UncertaintyEstimator':
        """Fit the uncertainty estimator."""
        pass
    
    @abstractmethod
    def predict_with_uncertainty(
        self,
        X: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Predict with uncertainty intervals."""
        pass


class MonteCarloDropout(UncertaintyEstimator):
    """
    Monte Carlo Dropout for uncertainty estimation.
    
    Uses dropout at inference time to create multiple predictions
    and estimates uncertainty from the variance.
    """
    
    def __init__(
        self,
        model: Any,
        n_samples: int = 100,
        dropout_rate: float = 0.1
    ):
        """
        Initialize MC Dropout estimator.
        
        Args:
            model: Keras/TensorFlow model with dropout layers
            n_samples: Number of forward passes
            dropout_rate: Dropout rate for inference
        """
        self.model = model
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MonteCarloDropout':
        """Fit is handled by the base model."""
        return self
    
    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        batch_size: int = 32
    ) -> Dict[str, np.ndarray]:
        """
        Generate predictions with uncertainty.
        
        Args:
            X: Input features
            batch_size: Batch size for inference
            
        Returns:
            Dictionary with predictions, lower_bound, upper_bound, std
        """
        try:
            import tensorflow as tf
            tf.keras.backend.set_learning_mode(True)
        except ImportError:
            pass
            
        # Run multiple forward passes
        predictions = []
        
        for _ in range(self.n_samples):
            try:
                pred = self.model.predict(X, verbose=0, batch_size=batch_size)
                if isinstance(pred, list):
                    pred = pred[0]
                predictions.append(pred.flatten())
            except Exception as e:
                logger.warning(f"MC Dropout forward pass failed: {e}")
                continue
                
        if not predictions:
            raise ValueError("All MC Dropout forward passes failed")
            
        predictions = np.array(predictions)
        
        # Calculate statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # 95% confidence interval
        lower_bound = mean_pred - 1.96 * std_pred
        upper_bound = mean_pred + 1.96 * std_pred
        
        # Epistemic vs Aleatoric uncertainty
        # Epistemic: uncertainty due to model limitations (reduces with more data)
        # Aleatoric: inherent noise in the data
        
        return {
            'predictions': mean_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'std': std_pred,
            'samples': predictions
        }
    
    def get_uncertainty_decomposition(
        self,
        X: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Decompose uncertainty into epistemic and aleatoric.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary with uncertainty components
        """
        # This is a simplified version
        # For proper decomposition, you'd need multiple models
        
        result = self.predict_with_uncertainty(X)
        
        return {
            'epistemic': result['std'] * 0.5,  # Placeholder
            'aleatoric': result['std'] * 0.5,  # Placeholder
            'total': result['std']
        }


class DeepEnsemble(UncertaintyEstimator):
    """
    Deep Ensemble for uncertainty estimation.
    
    Trains multiple models with different initializations and
    combines their predictions to estimate uncertainty.
    """
    
    def __init__(
        self,
        model_factory,
        n_models: int = 5,
        train_params: Optional[Dict] = None
    ):
        """
        Initialize Deep Ensemble.
        
        Args:
            model_factory: Function that creates a new model instance
            n_models: Number of models in ensemble
            train_params: Parameters for model training
        """
        self.model_factory = model_factory
        self.n_models = n_models
        self.train_params = train_params or {}
        self.models = []
        
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2
    ) -> 'DeepEnsemble':
        """
        Train ensemble of models.
        
        Args:
            X: Training features
            y: Training targets
            validation_split: Validation split ratio
            
        Returns:
            Self
        """
        for i in range(self.n_models):
            logger.info(f"Training model {i+1}/{self.n_models}")
            
            # Create new model instance
            model = self.model_factory()
            
            # Train model
            try:
                model.fit(X, y, validation_split=validation_split, verbose=0)
                self.models.append(model)
            except Exception as e:
                logger.warning(f"Failed to train model {i+1}: {e}")
                
        logger.info(f"Trained {len(self.models)} models in ensemble")
        
        return self
    
    def predict_with_uncertainty(
        self,
        X: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Generate predictions with uncertainty.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary with predictions, bounds, std
        """
        predictions = []
        
        for model in self.models:
            try:
                pred = model.predict(X)
                predictions.append(pred.flatten())
            except Exception as e:
                logger.warning(f"Prediction failed: {e}")
                continue
                
        if not predictions:
            raise ValueError("No models produced valid predictions")
            
        predictions = np.array(predictions)
        
        # Calculate statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Confidence intervals
        lower_bound = np.percentile(predictions, 2.5, axis=0)
        upper_bound = np.percentile(predictions, 97.5, axis=0)
        
        return {
            'predictions': mean_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'std': std_pred,
            'samples': predictions
        }
    
    def get_model_diversity(self) -> float:
        """
        Calculate model diversity (average pairwise disagreement).
        
        Returns:
            Diversity score
        """
        if len(self.models) < 2:
            return 0.0
            
        # This is a simplified diversity measure
        # Would need actual predictions to calculate properly
        return 1.0 / len(self.models)


class ConformalPrediction:
    """
    Conformal prediction for distribution-free uncertainty.
    
    Provides valid prediction intervals without distributional assumptions.
    """
    
    def __init__(
        self,
        calibration_size: float = 0.2,
        confidence_level: float = 0.95
    ):
        """
        Initialize conformal predictor.
        
        Args:
            calibration_size: Fraction of data for calibration
            confidence_level: Desired confidence level
        """
        self.calibration_size = calibration_size
        self.confidence_level = confidence_level
        self.quantile = None
        
    def fit(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray
    ) -> 'ConformalPrediction':
        """
        Fit conformal predictor.
        
        Args:
            model: Trained model with predict method
            X: Features
            y: Targets
            
        Returns:
            Self
        """
        # Split into training and calibration
        n = len(X)
        n_cal = int(n * self.calibration_size)
        
        X_train = X[:-n_cal]
        y_train = y[:-n_cal]
        X_cal = X[-n_cal:]
        y_cal = y[-n_cal:]
        
        # Retrain model on training set
        try:
            model.fit(X_train, y_train, verbose=0)
        except:
            pass
            
        # Get predictions on calibration set
        predictions = model.predict(X_cal).flatten()
        
        # Calculate conformity scores (absolute errors)
        errors = np.abs(y_cal - predictions)
        
        # Calculate quantile
        alpha = 1 - self.confidence_level
        self.quantile = np.percentile(errors, (1 - alpha) * 100)
        
        return self
    
    def predict_intervals(
        self,
        model: Any,
        X: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Predict with confidence intervals.
        
        Args:
            model: Trained model
            X: Features
            
        Returns:
            Dictionary with predictions and intervals
        """
        predictions = model.predict(X).flatten()
        
        return {
            'predictions': predictions,
            'lower_bound': predictions - self.quantile,
            'upper_bound': predictions + self.quantile,
            'interval_width': 2 * self.quantile
        }


class EnsembleUncertaintyCombiner:
    """
    Combines multiple uncertainty estimation methods.
    
    Aggregates:
    - Monte Carlo Dropout
    - Deep Ensembles
    - Conformal Prediction
    """
    
    def __init__(self):
        self.estimators = []
        
    def add_estimator(
        self,
        name: str,
        estimator: UncertaintyEstimator
    ) -> 'EnsembleUncertaintyCombiner':
        """Add an uncertainty estimator."""
        self.estimators.append((name, estimator))
        return self
    
    def fit_estimators(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> 'EnsembleUncertaintyCombiner':
        """Fit all estimators."""
        for name, estimator in self.estimators:
            logger.info(f"Fitting {name}...")
            estimator.fit(X, y)
        return self
    
    def predict_with_uncertainty(
        self,
        X: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Combine predictions from all estimators."""
        all_predictions = []
        all_lower = []
        all_upper = []
        all_std = []
        
        for name, estimator in self.estimators:
            try:
                result = estimator.predict_with_uncertainty(X)
                all_predictions.append(result['predictions'])
                all_lower.append(result['lower_bound'])
                all_upper.append(result['upper_bound'])
                all_std.append(result['std'])
            except Exception as e:
                logger.warning(f"{name} prediction failed: {e}")
                
        if not all_predictions:
            raise ValueError("No estimators produced valid predictions")
            
        # Combine predictions
        predictions = np.mean(all_predictions, axis=0)
        
        # Use ensemble of intervals
        lower_bound = np.mean(all_lower, axis=0)
        upper_bound = np.mean(all_upper, axis=0)
        
        # Combine uncertainties
        std = np.sqrt(np.mean([s**2 for s in all_std], axis=0))
        
        return {
            'predictions': predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'std': std,
            'individual_predictions': all_predictions
        }
    
    def get_uncertainty_summary(
        self,
        X: np.ndarray
    ) -> pd.DataFrame:
        """
        Get summary of uncertainty across estimators.
        
        Args:
            X: Input features
            
        Returns:
            DataFrame with uncertainty metrics
        """
        results = []
        
        for name, estimator in self.estimators:
            try:
                result = estimator.predict_with_uncertainty(X)
                results.append({
                    'estimator': name,
                    'mean_prediction': np.mean(result['predictions']),
                    'mean_std': np.mean(result['std']),
                    'mean_interval_width': np.mean(
                        result['upper_bound'] - result['lower_bound']
                    )
                })
            except Exception:
                continue
                
        return pd.DataFrame(results)
