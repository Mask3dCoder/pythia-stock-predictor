"""
Validation Framework

Implements robust time series validation schemes:
- Walk-forward validation (expanding window)
- Purged K-fold cross-validation
- Custom financial loss functions
- Early stopping based on financial metrics
"""

import logging
from typing import Optional, Dict, List, Tuple, Callable, Any
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


class BaseValidator(ABC):
    """Abstract base class for validators."""
    
    @abstractmethod
    def split(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test splits."""
        pass
    
    @abstractmethod
    def evaluate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate model using the validation scheme."""
        pass


class WalkForwardValidator(BaseValidator):
    """
    Walk-forward validation for time series.
    
    Simulates real-world forecasting by:
    1. Training on historical data
    2. Making predictions for a future window
    3. Rolling the window forward
    4. Repeating until all data is used
    
    This prevents lookahead bias and provides realistic performance estimates.
    """
    
    def __init__(
        self,
        train_size: int = 252,
        test_size: int = 21,
        step_size: Optional[int] = None,
        min_train_size: int = 60
    ):
        """
        Initialize walk-forward validator.
        
        Args:
            train_size: Initial training window size (in samples)
            test_size: Forecast/test horizon size
            step_size: Step size between windows (None = test_size)
            min_train_size: Minimum training size
        """
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size or test_size
        self.min_train_size = min_train_size
        
    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> List[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
        """
        Generate walk-forward splits.
        
        Args:
            X: Feature array
            y: Target array (optional)
            
        Returns:
            List of ((X_train, y_train), (X_test, y_test)) tuples
        """
        n_samples = len(X)
        
        splits = []
        
        # Start position
        start = self.min_train_size
        
        while start + self.test_size <= n_samples:
            # Training window
            train_end = start
            train_start = max(0, train_end - self.train_size)
            
            # Test window
            test_end = min(start + self.test_size, n_samples)
            test_start = start
            
            # Get data
            X_train = X[train_start:train_end]
            X_test = X[test_start:test_end]
            
            if y is not None:
                y_train = y[train_start:train_end]
                y_test = y[test_start:test_end]
                splits.append(((X_train, y_train), (X_test, y_test)))
            else:
                splits.append(((X_train, None), (X_test, None)))
            
            # Move window forward
            start += self.step_size
            
        logger.info(f"Created {len(splits)} walk-forward splits")
        
        return splits
    
    def evaluate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        metrics: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate model using walk-forward validation.
        
        Args:
            model: Model with fit() and predict() methods
            X: Feature array
            y: Target array
            metrics: List of metrics to compute
            verbose: Print progress
            
        Returns:
            Dictionary with evaluation results
        """
        splits = self.split(X, y)
        
        if not splits:
            raise ValueError("No valid splits generated")
            
        all_predictions = []
        all_actuals = []
        fold_results = []
        
        for i, ((X_train, y_train), (X_test, y_test)) in enumerate(splits):
            if verbose:
                logger.info(f"Fold {i+1}/{len(splits)}: train={len(X_train)}, test={len(X_test)}")
                
            # Train model
            try:
                model.fit(X_train, y_train)
                
                # Predict
                predictions = model.predict(X_test)
                
                all_predictions.extend(predictions)
                all_actuals.extend(y_test)
                
                # Compute fold metrics
                fold_metrics = self._compute_metrics(y_test, predictions, metrics)
                fold_results.append(fold_metrics)
                
            except Exception as e:
                logger.warning(f"Error in fold {i+1}: {e}")
                continue
                
        # Aggregate results
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)
        
        results = {
            'predictions': all_predictions,
            'actuals': all_actuals,
            'n_folds': len(fold_results),
            'metrics': self._aggregate_metrics(fold_results)
        }
        
        return results
    
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Compute metrics for a single fold."""
        if metrics is None:
            metrics = ['mae', 'rmse', 'mape', 'r2', 'directional_accuracy']
            
        results = {}
        
        if 'mae' in metrics:
            results['mae'] = np.mean(np.abs(y_true - y_pred))
            
        if 'rmse' in metrics:
            results['rmse'] = np.sqrt(np.mean((y_true - y_pred) ** 2))
            
        if 'mape' in metrics:
            mask = y_true != 0
            if mask.any():
                results['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            else:
                results['mape'] = 0
                
        if 'r2' in metrics:
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            results['r2'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
        if 'directional_accuracy' in metrics:
            if len(y_true) > 1:
                actual_dir = np.sign(np.diff(y_true))
                pred_dir = np.sign(np.diff(y_pred))
                results['directional_accuracy'] = np.mean(actual_dir == pred_dir)
            else:
                results['directional_accuracy'] = 0
                
        return results
    
    def _aggregate_metrics(
        self,
        fold_results: List[Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics across folds."""
        if not fold_results:
            return {}
            
        # Get all metric names
        metric_names = fold_results[0].keys()
        
        aggregated = {}
        
        for metric in metric_names:
            values = [r[metric] for r in fold_results if metric in r]
            
            aggregated[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
            
        return aggregated


class PurgedKFoldValidator(BaseValidator):
    """
    Purged K-Fold cross-validation for time series.
    
    Key features:
    - Prevents information leakage between train and test sets
    - "Purges" observations that could leak information
    - Supports embargo periods for additional protection
    
    This is crucial for financial data where temporal dependencies
    can cause lookahead bias.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        purge_pct: float = 0.01,
        embargo_pct: float = 0.0,
        gap_between_splits: int = 0
    ):
        """
        Initialize purged K-fold validator.
        
        Args:
            n_splits: Number of folds
            purge_pct: Percentage of training data to purge after each test set
            embargo_pct: Additional embargo period as fraction of test size
            gap_between_splits: Gap between test sets (in samples)
        """
        self.n_splits = n_splits
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct
        self.gap_between_splits = gap_between_splits
        
    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> List[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
        """
        Generate purged K-fold splits.
        
        Args:
            X: Feature array
            y: Target array (optional)
            
        Returns:
            List of ((X_train, y_train), (X_test, y_test)) tuples
        """
        n_samples = len(X)
        
        # Calculate fold sizes
        fold_size = n_samples // self.n_splits
        test_size = int(fold_size * 0.2)  # 20% for testing
        purge_size = int(fold_size * self.purge_pct)
        embargo_size = int(test_size * self.embargo_pct)
        
        splits = []
        
        for i in range(self.n_splits):
            # Calculate test indices
            test_start = i * fold_size + self.gap_between_splits
            test_end = min(test_start + test_size, n_samples)
            
            # Calculate training indices
            train_end = test_start - purge_size  # Purge zone
            train_start = 0
            
            # Apply embargo
            train_end = max(0, train_end - embargo_size)
            
            if train_start >= train_end or test_start >= test_end:
                continue
                
            # Get data
            X_train = X[train_start:train_end]
            X_test = X[test_start:test_end]
            
            if y is not None:
                y_train = y[train_start:train_end]
                y_test = y[test_start:test_end]
                splits.append(((X_train, y_train), (X_test, y_test)))
            else:
                splits.append(((X_train, None), (X_test, None)))
                
        logger.info(f"Created {len(splits)} purged K-fold splits")
        
        return splits
    
    def evaluate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        metrics: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Evaluate model using purged K-fold."""
        splits = self.split(X, y)
        
        fold_results = []
        
        for i, ((X_train, y_train), (X_test, y_test)) in enumerate(splits):
            if verbose:
                logger.info(f"Fold {i+1}/{len(splits)}")
                
            try:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                
                fold_metrics = self._compute_metrics(y_test, predictions, metrics)
                fold_results.append(fold_metrics)
                
            except Exception as e:
                logger.warning(f"Error in fold {i+1}: {e}")
                continue
                
        return self._aggregate_results(fold_results)
    
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Compute metrics for a fold."""
        if metrics is None:
            metrics = ['mae', 'rmse', 'r2']
            
        results = {}
        
        if 'mae' in metrics:
            results['mae'] = np.mean(np.abs(y_true - y_pred))
            
        if 'rmse' in metrics:
            results['rmse'] = np.sqrt(np.mean((y_true - y_pred) ** 2))
            
        if 'r2' in metrics:
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            results['r2'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
        return results
    
    def _aggregate_results(
        self,
        fold_results: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Aggregate results across folds."""
        if not fold_results:
            return {}
            
        metric_names = fold_results[0].keys()
        
        aggregated = {
            'n_folds': len(fold_results),
            'metrics': {}
        }
        
        for metric in metric_names:
            values = [r[metric] for r in fold_results]
            aggregated['metrics'][metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
            
        return aggregated


class FinancialLossFunctions:
    """
    Custom loss functions for financial applications.
    
    These losses optimize for trading-relevant metrics rather than
    simple error minimization.
    """
    
    @staticmethod
    def sharpe_ratio_loss(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        risk_free_rate: float = 0.0
    ) -> float:
        """
        Negative Sharpe ratio as loss function.
        
        Args:
            y_true: Actual returns
            y_pred: Predicted returns
            risk_free_rate: Risk-free rate
            
        Returns:
            Negative Sharpe ratio (to minimize)
        """
        returns = y_pred - y_true  # Prediction error
        excess_returns = returns - risk_free_rate
        
        if np.std(excess_returns) == 0:
            return 0.0
            
        sharpe = np.mean(excess_returns) / np.std(excess_returns)
        
        return -sharpe  # Negative for minimization
    
    @staticmethod
    def sortino_ratio_loss(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        risk_free_rate: float = 0.0
    ) -> float:
        """
        Negative Sortino ratio as loss function.
        
        Args:
            y_true: Actual returns
            y_pred: Predicted returns
            risk_free_rate: Risk-free rate
            
        Returns:
            Negative Sortino ratio
        """
        returns = y_pred - y_true
        excess_returns = returns - risk_free_rate
        
        # Downside deviation (only negative returns)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0
            
        sortino = np.mean(excess_returns) / np.std(downside_returns)
        
        return -sortino
    
    @staticmethod
    def max_drawdown_loss(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Maximum drawdown as loss function.
        
        Args:
            y_true: Actual returns
            y_pred: Predicted returns
            
        Returns:
            Maximum drawdown (to minimize)
        """
        # Calculate cumulative returns
        cumulative = np.cumprod(1 + y_pred - y_true)
        
        # Running maximum
        running_max = np.maximum.accumulate(cumulative)
        
        # Drawdown
        drawdown = (cumulative - running_max) / running_max
        
        # Max drawdown
        max_dd = np.min(drawdown)
        
        return -max_dd  # Negative to minimize (we want to maximize)
    
    @staticmethod
    def calmar_ratio_loss(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calmar ratio as loss function.
        
        Args:
            y_true: Actual returns
            y_pred: Predicted returns
            
        Returns:
            Negative Calmar ratio
        """
        # Annualized return
        total_return = np.sum(y_pred - y_true)
        annualized_return = total_return / len(y_true) * 252
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + y_pred - y_true)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_dd = np.abs(np.min(drawdown))
        
        if max_dd == 0:
            return 0.0
            
        calmar = annualized_return / max_dd
        
        return -calmar
    
    @staticmethod
    def directional_accuracy_loss(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Directional accuracy loss (1 - accuracy).
        
        Args:
            y_true: Actual returns
            y_pred: Predicted returns
            
        Returns:
            1 - directional accuracy
        """
        if len(y_true) <= 1:
            return 0.0
            
        actual_dir = np.sign(np.diff(y_true))
        pred_dir = np.sign(np.diff(y_pred))
        
        accuracy = np.mean(actual_dir == pred_dir)
        
        return 1 - accuracy
    
    @staticmethod
    def create_combined_loss(
        weights: Optional[Dict[str, float]] = None
    ) -> Callable:
        """
        Create a combined loss function.
        
        Args:
            weights: Dictionary of loss weights
            
        Returns:
            Combined loss function
        """
        if weights is None:
            weights = {
                'mse': 0.3,
                'sharpe': 0.3,
                'directional': 0.4
            }
            
        def combined_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            loss = 0.0
            
            if 'mse' in weights:
                mse = np.mean((y_true - y_pred) ** 2)
                loss += weights['mse'] * mse
                
            if 'sharpe' in weights:
                sharpe_loss = FinancialLossFunctions.sharpe_ratio_loss(y_true, y_pred)
                loss += weights['sharpe'] * sharpe_loss
                
            if 'directional' in weights:
                dir_loss = FinancialLossFunctions.directional_accuracy_loss(y_true, y_pred)
                loss += weights['directional'] * dir_loss
                
            return loss
            
        return combined_loss


class FinancialEarlyStopping:
    """
    Early stopping based on financial metrics.
    
    Monitors metrics like Sharpe ratio and max drawdown
    to prevent overfitting to historical patterns.
    """
    
    def __init__(
        self,
        metric: str = 'sharpe_ratio',
        min_delta: float = 0.0,
        patience: int = 10,
        mode: str = 'max'
    ):
        """
        Initialize early stopping.
        
        Args:
            metric: Metric to monitor
            min_delta: Minimum change to qualify as improvement
            patience: Number of epochs without improvement
            mode: 'max' or 'min'
        """
        self.metric = metric
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        
        self.best_value = None
        self.wait = 0
        self.stopped_epoch = 0
        
    def __call__(
        self,
        epoch: int,
        val_metrics: Dict[str, float]
    ) -> bool:
        """
        Check if training should stop.
        
        Args:
            epoch: Current epoch
            val_metrics: Validation metrics
            
        Returns:
            True if training should stop
        """
        if self.metric not in val_metrics:
            return False
            
        current = val_metrics[self.metric]
        
        if self.best_value is None:
            self.best_value = current
            return False
            
        # Check for improvement
        if self.mode == 'max':
            improved = current > self.best_value + self.min_delta
        else:
            improved = current < self.best_value - self.min_delta
            
        if improved:
            self.best_value = current
            self.wait = 0
        else:
            self.wait += 1
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                return True
                
        return False
    
    def get_best_value(self) -> float:
        """Get best observed value."""
        return self.best_value
    
    def reset(self) -> None:
        """Reset the early stopping state."""
        self.best_value = None
        self.wait = 0
        self.stopped_epoch = 0
