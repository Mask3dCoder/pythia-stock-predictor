"""
Bayesian Hyperparameter Optimization

Implements hyperparameter optimization using Optuna for:
- Neural network architectures (CNN-LSTM, etc.)
- Traditional ML models
- Feature selection
"""

import logging
from typing import Optional, Dict, List, Callable, Any
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import Optuna
OPTUNA_AVAILABLE = False
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    logger.warning("Optuna not installed. Run: pip install optuna")


class HyperparameterOptimizer:
    """
    Bayesian hyperparameter optimization using Optuna.
    
    Supports:
    - Neural network architecture search
    - Traditional model tuning
    - Feature selection
    - Custom objectives
    """
    
    def __init__(
        self,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        direction: str = 'minimize'
    ):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            n_jobs: Number of parallel jobs
            direction: 'minimize' or 'maximize'
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required. Install with: pip install optuna")
            
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.direction = direction
        
        self.study = None
        self.best_params = None
        self.best_value = None
        
    def optimize_cnn_lstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        verbose: int = 0
    ) -> Dict[str, Any]:
        """
        Optimize CNN-LSTM hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Maximum epochs per trial
            verbose: Verbosity level
            
        Returns:
            Dictionary with best params and results
        """
        if X_val is None:
            split = int(len(X_train) * 0.8)
            X_val = X_train[split:]
            y_val = y_train[split:]
            X_train = X_train[:split]
            y_train = y_train[:split]
            
        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
            params = {
                # Conv parameters
                'conv_filters_0': trial.suggest_int('conv_filters_0', 16, 128),
                'conv_filters_1': trial.suggest_int('conv_filters_1', 16, 128),
                'kernel_size': trial.suggest_int('kernel_size', 2, 5),
                
                # LSTM parameters
                'lstm_units_0': trial.suggest_int('lstm_units_0', 32, 128),
                'lstm_units_1': trial.suggest_int('lstm_units_1', 16, 64),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                
                # Attention parameters
                'attention_units': trial.suggest_int('attention_units', 32, 128),
                'num_attention_heads': trial.suggest_int('num_attention_heads', 2, 8),
                
                # Training parameters
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            }
            
            # Build and train model
            from src.models.temporal.cnn_lstm import CNNLSTMModel
            
            model_config = {
                'sequence_length': X_train.shape[1],
                'n_features': X_train.shape[2] if X_train.ndim > 2 else 1,
                'conv_filters': [params['conv_filters_0'], params['conv_filters_1']],
                'kernel_size': params['kernel_size'],
                'lstm_units': [params['lstm_units_0'], params['lstm_units_1']],
                'dropout': params['dropout'],
                'attention_units': params['attention_units'],
                'num_attention_heads': params['num_attention_heads'],
                'learning_rate': params['learning_rate'],
                'batch_size': params['batch_size'],
                'epochs': epochs,
                'early_stopping_patience': 10
            }
            
            try:
                model = CNNLSTMModel(model_config)
                model.fit(X_train, y_train, validation_split=0.2, verbose=0)
                
                # Evaluate
                predictions = model.predict(X_val)
                
                # Use validation loss (MSE)
                val_loss = np.mean((y_val - predictions) ** 2)
                
                return val_loss
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return float('inf')
                
        # Run optimization
        sampler = TPESampler(seed=42)
        self.study = optuna.create_study(
            direction=self.direction,
            sampler=sampler
        )
        
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=verbose > 0
        )
        
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        
        return {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'study': self.study
        }
    
    def optimize_lstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50
    ) -> Dict[str, Any]:
        """
        Optimize LSTM hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Maximum epochs per trial
            
        Returns:
            Dictionary with best params
        """
        if X_val is None:
            split = int(len(X_train) * 0.8)
            X_val = X_train[split:]
            y_val = y_train[split:]
            X_train = X_train[:split]
            y_train = y_train[:split]
            
        def objective(trial: optuna.Trial) -> float:
            params = {
                'sequence_length': X_train.shape[1],
                'lstm_units': [
                    trial.suggest_int('lstm_units_0', 16, 128),
                    trial.suggest_int('lstm_units_1', 16, 64)
                ],
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                'epochs': epochs
            }
            
            from src.models.lstm_model import LSTMModel
            
            try:
                model = LSTMModel(params)
                model.fit(pd.Series(y_train.flatten()), verbose=0)
                
                predictions = model.model.predict(X_val, verbose=0).flatten()
                val_loss = np.mean((y_val - predictions) ** 2)
                
                return val_loss
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return float('inf')
                
        sampler = TPESampler(seed=42)
        self.study = optuna.create_study(
            direction=self.direction,
            sampler=sampler
        )
        
        self.study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        
        return {
            'best_params': self.best_params,
            'best_value': self.best_value
        }
    
    def optimize_arima(
        self,
        y_train: np.ndarray,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5
    ) -> Dict[str, Any]:
        """
        Optimize ARIMA hyperparameters.
        
        Args:
            y_train: Training time series
            max_p: Maximum p value
            max_d: Maximum d value
            max_q: Maximum q value
            
        Returns:
            Best ARIMA parameters
        """
        def objective(trial: optuna.Trial) -> float:
            p = trial.suggest_int('p', 0, max_p)
            d = trial.suggest_int('d', 0, max_d)
            q = trial.suggest_int('q', 0, max_q)
            
            try:
                from statsmodels.tsa.arima.model import ARIMA
                
                model = ARIMA(y_train, order=(p, d, q))
                results = model.fit()
                
                # Use AIC as objective (lower is better)
                return results.aic
                
            except Exception:
                return float('inf')
                
        sampler = TPESampler(seed=42)
        self.study = optuna.create_study(
            direction='minimize',
            sampler=sampler
        )
        
        self.study.optimize(objective, n_trials=self.n_trials)
        
        best_order = (
            self.study.best_params['p'],
            self.study.best_params['d'],
            self.study.best_params['q']
        )
        
        return {
            'best_order': best_order,
            'best_aic': self.study.best_value,
            'best_params': self.study.best_params
        }
    
    def optimize_feature_selection(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str],
        model_type: str = 'lstm'
    ) -> Dict[str, Any]:
        """
        Optimize feature selection.
        
        Args:
            X_train: Training features
            y_train: Training targets
            feature_names: List of feature names
            model_type: Type of model to use
            
        Returns:
            Selected features and scores
        """
        n_features = X_train.shape[1] if X_train.ndim > 1 else 1
        
        def objective(trial: optuna.Trial) -> float:
            # Select features to use
            selected = [
                trial.suggest_int(f'feature_{i}', 0, 1) 
                for i in range(n_features)
            ]
            
            if sum(selected) == 0:
                return float('inf')
                
            # Create subset
            if X_train.ndim > 1:
                feature_indices = [i for i, s in enumerate(selected) if s]
                X_subset = X_train[:, :, feature_indices] if X_train.ndim > 2 else X_train[:, feature_indices]
            else:
                X_subset = X_train
                
            # Train and evaluate
            try:
                from src.models.lstm_model import LSTMModel
                
                model = LSTMModel({
                    'sequence_length': X_subset.shape[1],
                    'epochs': 20
                })
                model.fit(pd.Series(y_train.flatten()), verbose=0)
                
                return model.history.history['val_loss'][-1]
                
            except Exception:
                return float('inf')
                
        sampler = TPESampler(seed=42)
        self.study = optuna.create_study(
            direction='minimize',
            sampler=sampler
        )
        
        self.study.optimize(objective, n_trials=min(self.n_trials, 2**n_features))
        
        # Get selected features
        selected_features = [
            feature_names[i] 
            for i, v in enumerate(self.study.best_params.values()) 
            if v == 1
        ]
        
        return {
            'selected_features': selected_features,
            'best_params': self.study.best_params,
            'best_value': self.study.best_value
        }
    
    def get_study_results(self) -> pd.DataFrame:
        """Get optimization results as DataFrame."""
        if self.study is None:
            return pd.DataFrame()
            
        return self.study.trials_dataframe()


class AutoMLPipeline:
    """
    Automated ML pipeline that tries multiple models
    and selects the best one.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.models = {}
        self.results = {}
        
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> 'AutoMLPipeline':
        """
        Fit multiple models and select best.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Self
        """
        from src.models.lstm_model import LSTMModel
        from src.models.arima_model import ARIMAModel
        
        models_to_try = ['lstm', 'gru', 'arima']
        
        for model_type in models_to_try:
            logger.info(f"Training {model_type}...")
            
            try:
                if model_type == 'lstm':
                    model = LSTMModel(self.config.get('lstm', {}))
                    model.fit(pd.Series(y_train.flatten()), verbose=0)

                elif model_type == 'gru':
                    model = LSTMModel(self.config.get('gru', {}))
                    model.fit(pd.Series(y_train.flatten()), verbose=0)
                    
                elif model_type == 'arima':
                    model = ARIMAModel(self.config.get('arima', {}))
                    model.fit(pd.Series(y_train))
                    
                self.models[model_type] = model
                
                if X_val is not None:
                    predictions = model.predict(X_val)
                    loss = np.mean((y_val - predictions) ** 2)
                    self.results[model_type] = loss
                    
            except Exception as e:
                logger.warning(f"Failed to train {model_type}: {e}")
                
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Input features
            
        Returns:
            Ensemble predictions
        """
        if not self.models:
            raise ValueError("No models fitted")
            
        predictions = []
        
        for model_type, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions.append(pred)
            except Exception:
                continue
                
        if not predictions:
            raise ValueError("All models failed")
            
        # Average predictions
        return np.mean(predictions, axis=0)
    
    def get_best_model(self) -> Optional[Any]:
        """Get the best performing model."""
        if not self.results:
            return None
            
        best_type = min(self.results, key=self.results.get)
        return self.models.get(best_type)
