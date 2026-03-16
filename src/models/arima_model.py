"""
ARIMA Model for Stock Price Prediction

Implements ARIMA and SARIMA models for time-series forecasting.
"""

import logging
from typing import Optional, Dict, Tuple
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ARIMAModel:
    """ARIMA model for stock price prediction."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize ARIMA model.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # ARIMA order parameters
        self.order = self.config.get('order', [5, 1, 0])
        self.seasonal_order = self.config.get('seasonal_order', [0, 0, 0, 0])
        
        # Auto-fit configuration
        self.auto_fit = self.config.get('auto_fit', False)
        self.max_p = self.config.get('max_p', 5)
        self.max_d = self.config.get('max_d', 2)
        self.max_q = self.config.get('max_q', 5)
        
        self.model = None
        self.results = None
        self.scaler = None
        self._is_stationary = None
    
    def check_stationarity(self, data: pd.Series) -> Tuple[bool, float]:
        """
        Check if time series is stationary using Augmented Dickey-Fuller test.
        
        Args:
            data: Time series data
            
        Returns:
            Tuple of (is_stationary, p_value)
        """
        try:
            from statsmodels.tsa.stattools import adfuller
            
            result = adfuller(data.dropna(), autolag='AIC')
            p_value = result[1]
            is_stationary = p_value < 0.05
            
            logger.info(f"ADF test: p-value={p_value:.4f}, stationary={is_stationary}")
            return is_stationary, p_value
            
        except Exception as e:
            logger.warning(f"Could not perform ADF test: {e}")
            return False, 1.0
    
    def auto_select_parameters(self, data: pd.Series) -> Tuple[Tuple, float]:
        """
        Automatically select best ARIMA parameters using AIC.
        
        Args:
            data: Time series data
            
        Returns:
            Tuple of (best_order, best_aic)
        """
        from statsmodels.tsa.arima.model import ARIMA
        import warnings
        
        warnings.filterwarnings('ignore')
        
        best_aic = float('inf')
        best_order = tuple(self.order)
        
        logger.info(f"Starting auto-ARIMA parameter search (p<={self.max_p}, d<={self.max_d}, q<={self.max_q})...")
        
        for p in range(self.max_p + 1):
            for d in range(self.max_d + 1):
                for q in range(self.max_q + 1):
                    try:
                        model = ARIMA(data.values, order=(p, d, q))
                        results = model.fit()
                        
                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_order = (p, d, q)
                            logger.debug(f"New best: order={best_order}, AIC={best_aic:.2f}")
                            
                    except Exception:
                        continue
        
        logger.info(f"Best ARIMA order: {best_order} with AIC: {best_aic:.2f}")
        return best_order, best_aic
        
    def fit(self, data: pd.Series, order: Optional[Tuple] = None) -> 'ARIMAModel':
        """
        Fit ARIMA model to data.
        
        Args:
            data: Time series data
            order: ARIMA order (p, d, q)
            
        Returns:
            Self
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            # Check stationarity and apply differencing if needed
            is_stationary, p_value = self.check_stationarity(data)
            self._is_stationary = is_stationary
            
            # Use auto-fit if enabled or if no order provided
            if self.auto_fit or order is None:
                order, aic = self.auto_select_parameters(data)
                logger.info(f"Auto-selected ARIMA order: {order}")
            elif order is None:
                order = tuple(self.order)
            
            logger.info(f"Fitting ARIMA model with order={order}")
            
            # Scale data for better results
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
            scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
            
            # Fit model
            self.model = ARIMA(scaled_data, order=order)
            self.results = self.model.fit()
            
            # Store the order used
            self.order = list(order)
            
            logger.info(f"ARIMA model fitted successfully")
            
            return self
            
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {str(e)}")
            raise
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            steps: Number of steps to predict
            
        Returns:
            Array of predictions
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        # Make forecast
        forecast = self.results.forecast(steps=steps)
        
        # Inverse transform predictions
        if self.scaler is not None:
            forecast = self.scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()
            
        return forecast
    
    def predict_with_confidence(self, steps: int = 1, alpha: float = 0.05) -> Dict:
        """
        Make predictions with confidence intervals.
        
        Args:
            steps: Number of steps to predict
            alpha: Significance level for confidence intervals
            
        Returns:
            Dictionary with predictions and confidence intervals
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        # Get forecast with confidence intervals
        result = self.results.get_forecast(steps=steps)
        conf_int = result.conf_int(alpha=alpha)
        
        # Handle both DataFrame and ndarray
        if hasattr(conf_int, 'iloc'):
            lower = conf_int.iloc[:, 0].values
            upper = conf_int.iloc[:, 1].values
        else:
            # It's a numpy array
            lower = conf_int[:, 0]
            upper = conf_int[:, 1]
            
        predictions = result.predicted_mean
        
        # Handle both Series and ndarray
        if hasattr(predictions, 'values'):
            predictions_arr = predictions.values
        else:
            predictions_arr = np.array(predictions)
            
        # Inverse transform
        if self.scaler is not None:
            predictions_arr = self.scaler.inverse_transform(
                predictions_arr.reshape(-1, 1)
            ).flatten()
            
            lower = self.scaler.inverse_transform(
                lower.reshape(-1, 1)
            ).flatten()
            
            upper = self.scaler.inverse_transform(
                upper.reshape(-1, 1)
            ).flatten()
            
        return {
            'predictions': predictions_arr,
            'lower_bound': lower,
            'upper_bound': upper,
            'confidence': 1 - alpha
        }
    
    def auto_fit(self, data: pd.Series, max_p: int = 5, max_d: int = 2, max_q: int = 5) -> 'ARIMAModel':
        """
        Automatically find best ARIMA parameters.
        
        Args:
            data: Time series data
            max_p: Maximum p value to try
            max_d: Maximum d value to try
            max_q: Maximum q value to try
            
        Returns:
            Self with best parameters
        """
        from statsmodels.tsa.arima.model import ARIMA
        from sklearn.preprocessing import MinMaxScaler
        
        try:
            import warnings
            warnings.filterwarnings('ignore')
            
            # Scale data
            self.scaler = MinMaxScaler()
            scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
            
            # Grid search for best parameters
            best_aic = float('inf')
            best_order = None
            
            logger.info("Starting auto-ARIMA parameter search...")
            
            for p in range(max_p + 1):
                for d in range(max_d + 1):
                    for q in range(max_q + 1):
                        try:
                            model = ARIMA(scaled_data, order=(p, d, q))
                            results = model.fit()
                            
                            if results.aic < best_aic:
                                best_aic = results.aic
                                best_order = (p, d, q)
                                
                        except Exception:
                            continue
            
            if best_order is None:
                logger.warning("Could not find optimal parameters, using defaults")
                best_order = tuple(self.order)
                
            logger.info(f"Best ARIMA order: {best_order} with AIC: {best_aic:.2f}")
            
            # Fit with best parameters
            self.order = list(best_order)
            self.model = ARIMA(scaled_data, order=best_order)
            self.results = self.model.fit()
            
            return self
            
        except Exception as e:
            logger.error(f"Error in auto-ARIMA: {str(e)}")
            # Fall back to simple fit
            return self.fit(data)
    
    def evaluate(self, test_data: pd.Series) -> Dict:
        """
        Evaluate model on test data.
        
        Args:
            test_data: Test time series
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        # Make predictions
        predictions = self.predict(len(test_data))
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        y_true = test_data.values[:len(predictions)]
        
        mae = mean_absolute_error(y_true, predictions)
        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        
        # Handle edge cases
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
    
    def save_model(self, path: Path) -> None:
        """Save model to file."""
        import joblib
        joblib.dump({
            'model': self.results,
            'scaler': self.scaler,
            'order': self.order,
            'seasonal_order': self.seasonal_order
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Path) -> 'ARIMAModel':
        """Load model from file."""
        import joblib
        data = joblib.load(path)
        self.results = data['model']
        self.scaler = data['scaler']
        self.order = data['order']
        self.seasonal_order = data['seasonal_order']
        logger.info(f"Model loaded from {path}")
        return self
    
    def get_model_summary(self) -> str:
        """Get model summary."""
        if self.results is None:
            return "Model not fitted"
        return str(self.results.summary())
