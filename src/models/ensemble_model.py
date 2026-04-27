"""
Ensemble Model for Stock Price Prediction

Combines multiple models (ARIMA, LSTM, GRU) for more robust predictions.
"""

import logging
from typing import Optional, Dict, List, Any
from pathlib import Path

import pandas as pd
import numpy as np

from src.core.base import BaseModel

logger = logging.getLogger(__name__)


class EnsembleModel(BaseModel):
    """Ensemble model combining multiple forecasting models."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Ensemble model.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # Model weights
        self.weights = self.config.get('weights', {
            'arima': 0.33,
            'lstm': 0.34,
            'gru': 0.33
        })

        self.models = {}

        # Store last sequence for LSTM/GRU predictions
        self.last_sequence = None
        self.sequence_length = self.config.get('sequence_length', 60)
        
    def add_model(self, name: str, model: Any, weight: Optional[float] = None) -> 'EnsembleModel':
        """
        Add a model to the ensemble.
        
        Args:
            name: Model name (arima, lstm, gru)
            model: Model instance
            weight: Model weight (optional)
            
        Returns:
            Self
        """
        self.models[name] = model
        
        if weight is not None:
            self.weights[name] = weight
            
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        logger.info(f"Added model '{name}' with weight {self.weights.get(name, 0):.2f}")
        
        return self
    
    def fit(self, data: pd.Series, model_configs: Optional[Dict] = None) -> 'EnsembleModel':
        """
        Train all models in the ensemble.
        
        Args:
            data: Time series data
            model_configs: Configuration for each model
            
        Returns:
            Self
        """
        from .arima_model import ARIMAModel
        from .lstm_model import LSTMModel, GRUModel
        
        if model_configs is None:
            model_configs = {}
            
        # Train ARIMA if not already added
        if 'arima' not in self.models:
            arima_config = model_configs.get('arima', {})
            self.models['arima'] = ARIMAModel(arima_config)
            
        if not hasattr(self.models['arima'], 'results') or self.models['arima'].results is None:
            logger.info("Training ARIMA model...")
            self.models['arima'].fit(data)
            
        # Train LSTM if not already added
        if 'lstm' not in self.models:
            lstm_config = model_configs.get('lstm', {})
            self.models['lstm'] = LSTMModel(lstm_config)
            
        try:
            if self.models['lstm'].model is None:
                logger.info("Training LSTM model...")
                self.models['lstm'].fit(data, verbose=0)
        except Exception as e:
            logger.warning(f"LSTM training failed: {e}")
            
        # Train GRU if not already added
        if 'gru' not in self.models:
            gru_config = model_configs.get('gru', {})
            self.models['gru'] = GRUModel(gru_config)
            
        try:
            if self.models['gru'].model is None:
                logger.info("Training GRU model...")
                self.models['gru'].fit(data, verbose=0)
        except Exception as e:
            logger.warning(f"GRU training failed: {e}")
        
        # FIX: Critical bug - capture and store the last valid sequence for predictions
        # The last sequence is needed for LSTM/GRU predict_multiple() to work correctly
        # The sequence is stored as a pandas Series so LSTM/GRU can scale it using their internal MinMaxScaler
        if len(data) >= self.sequence_length:
            self.last_sequence = data.iloc[-self.sequence_length:].reset_index(drop=True)
            logger.info(f"Stored last sequence of length {self.sequence_length} for predictions")
        else:
            logger.warning(
                f"Data length {len(data)} < sequence_length {self.sequence_length}. "
                f"Padding to ensure prediction stability."
            )
            padding = pd.Series([data.iloc[0]] * (self.sequence_length - len(data)))
            self.last_sequence = pd.concat([padding, data]).reset_index(drop=True)
        
        self.is_fitted = True
        
        logger.info("Ensemble model training complete")
        
        return self
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Make ensemble predictions.

        Args:
            steps: Number of steps to predict

        Returns:
            Array of predictions
        """
        result = self._compute_ensemble(steps)
        return result['predictions']

    def _compute_ensemble(self, steps: int = 1) -> Dict:
        """
        Compute ensemble predictions with per-model details.

        Args:
            steps: Number of steps to predict

        Returns:
            Dictionary with 'predictions', 'details', 'weights'
        """
        if not self.is_fitted:
            raise ValueError("Models not fitted. Call fit() first.")

        predictions = {}
        weights_used = {}

        # ARIMA prediction
        if 'arima' in self.models and self.models['arima'].results is not None:
            try:
                predictions['arima'] = self.models['arima'].predict(steps)
                weights_used['arima'] = self.weights.get('arima', 0)
            except Exception as e:
                logger.warning(f"ARIMA prediction failed: {e}")

        # LSTM prediction
        if 'lstm' in self.models and self.models['lstm'].model is not None:
            try:
                predictions['lstm'] = self.models['lstm'].predict_multiple(
                    self.last_sequence, steps
                )
                weights_used['lstm'] = self.weights.get('lstm', 0)
            except Exception as e:
                logger.warning(f"LSTM prediction failed: {e}")

        # GRU prediction
        if 'gru' in self.models and self.models['gru'].model is not None:
            try:
                predictions['gru'] = self.models['gru'].predict_multiple(
                    self.last_sequence, steps
                )
                weights_used['gru'] = self.weights.get('gru', 0)
            except Exception as e:
                logger.warning(f"GRU prediction failed: {e}")

        # Normalize weights for available models
        total_weight = sum(weights_used.values())
        if total_weight > 0:
            weights_used = {k: v / total_weight for k, v in weights_used.items()}
        else:
            error_msg = (
                "No models available for prediction. All models failed during prediction. "
                f"Attempted models: arima={bool(self.models.get('arima') and self.models['arima'].results is not None)}, "
                f"lstm={bool(self.models.get('lstm') and self.models['lstm'].model is not None)}, "
                f"gru={bool(self.models.get('gru') and self.models['gru'].model is not None)}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Calculate weighted average
        ensemble_predictions = np.zeros(steps)
        for model_name, preds in predictions.items():
            ensemble_predictions += weights_used[model_name] * preds

        return {
            'predictions': ensemble_predictions,
            'details': predictions,
            'weights': weights_used
        }
    
    def predict_with_confidence(self, steps: int = 1) -> Dict:
        """
        Make predictions with confidence intervals.
        
        Args:
            steps: Number of steps to predict
            
        Returns:
            Dictionary with predictions and confidence intervals
        """
        # Get individual predictions
        prediction_result = self._compute_ensemble(steps)

        predictions = prediction_result['predictions']
        details = prediction_result.get('details', {})
        
        # Calculate prediction variance
        if len(details) > 1:
            all_preds = np.array([preds[:steps] for preds in details.values()])
            
            lower_bound = np.min(all_preds, axis=0)
            upper_bound = np.max(all_preds, axis=0)
        else:
            # Use ARIMA confidence if available
            if 'arima' in self.models:
                try:
                    arima_result = self.models['arima'].predict_with_confidence(steps)
                    lower_bound = arima_result['lower_bound']
                    upper_bound = arima_result['upper_bound']
                except Exception:
                    # Use percentage bounds
                    lower_bound = predictions * 0.95
                    upper_bound = predictions * 1.05
            else:
                lower_bound = predictions * 0.95
                upper_bound = predictions * 1.05
                
        return {
            'predictions': predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence': 0.95,
            'details': details
        }
    
    def evaluate(self, test_data: pd.Series) -> Dict:
        """
        Evaluate ensemble on test data.
        
        Args:
            test_data: Test time series
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self._compute_ensemble(len(test_data))['predictions']
        
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        y_true = test_data.values[:len(predictions)]
        
        mae = mean_absolute_error(y_true, predictions)
        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        
        if np.std(y_true) > 0:
            r2 = r2_score(y_true, predictions)
            # FIX: Safe division to prevent division by zero errors in MAPE calculation
            mape = np.mean(np.abs((y_true - predictions) / np.maximum(y_true, 1e-10))) * 100
        else:
            r2 = 0
            mape = 0
            
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'predictions': predictions,
            'actual': y_true
        }
    
    def get_model_summaries(self) -> Dict:
        """Get summaries from all models."""
        summaries = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'get_model_summary'):
                summaries[name] = model.get_model_summary()
            elif hasattr(model, 'summary'):
                summaries[name] = model.summary()
                
        return summaries
    
    def save_model(self, path: Path) -> None:
        """Save ensemble model to file. Satisfies BaseModel ABC."""
        self.save_models(path)

    def load_model(self, path: Path) -> 'EnsembleModel':
        """Load ensemble model from file. Satisfies BaseModel ABC."""
        return self.load_models(path)

    def save_models(self, path: Path) -> None:
        """Save all models."""
        import joblib
        
        path.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = path / f"{name}_model"
            
            try:
                if hasattr(model, 'save_model'):
                    model.save_model(model_path)
            except Exception as e:
                logger.warning(f"Could not save {name} model: {e}")
                
        # Save weights
        joblib.dump({'weights': self.weights}, path / "weights.joblib")
        
        logger.info(f"Models saved to {path}")
    
    def load_models(self, path: Path) -> 'EnsembleModel':
        """Load all models."""
        import joblib
        
        # Load weights
        weights = joblib.load(path / "weights.joblib")
        self.weights = weights['weights']
        
        # Load individual models
        from .arima_model import ARIMAModel
        from .lstm_model import LSTMModel, GRUModel
        
        model_types = {
            'arima': ARIMAModel,
            'lstm': LSTMModel,
            'gru': GRUModel
        }
        
        for name, model_class in model_types.items():
            model_path = path / f"{name}_model"
            
            if model_path.exists():
                model = model_class()
                try:
                    model.load_model(model_path)
                    self.models[name] = model
                except Exception as e:
                    logger.warning(f"Could not load {name} model: {e}")
                    
        self.is_fitted = True
        
        logger.info(f"Models loaded from {path}")
        
        return self
