"""
LSTM Model for Stock Price Prediction

Implements LSTM (Long Short-Term Memory) neural networks for time-series forecasting.
"""

import logging
from typing import Optional, Dict, Tuple
from pathlib import Path
import os

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# TensorFlow imports for callbacks
TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TF_AVAILABLE = True
except ImportError:
    pass


class LSTMModel:
    """LSTM model for stock price prediction."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize LSTM model.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Model parameters
        self.sequence_length = self.config.get('sequence_length', 60)
        self.lstm_units = self.config.get('lstm_units', [50, 50])
        self.dropout = self.config.get('dropout', 0.2)
        self.epochs = self.config.get('epochs', 50)
        self.batch_size = self.config.get('batch_size', 32)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        
        self.model = None
        self.scaler = None
        self.history = None
        
    def _build_model(self, input_shape: Tuple) -> 'LSTMModel':
        """
        Build LSTM model architecture.
        
        Args:
            input_shape: Shape of input data
            
        Returns:
            Self
        """
        try:
            # Import TensorFlow
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
            
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
        except (ImportError, ModuleNotFoundError) as e:
            logger.error(f"TensorFlow not available: {e}")
            raise ImportError(
                "TensorFlow is not available or not compatible with your Python version. "
                "Please install TensorFlow: pip install tensorflow "
                "Note: TensorFlow may not be compatible with Python 3.13. "
                "Consider using Python 3.9-3.11 or use ARIMA model instead."
            )
            
            # Build model
            self.model = Sequential()
            
            # First LSTM layer
            self.model.add(LSTM(
                units=self.lstm_units[0],
                return_sequences=True,
                input_shape=input_shape
            ))
            self.model.add(Dropout(self.dropout))
            
            # Additional LSTM layers
            for units in self.lstm_units[1:]:
                self.model.add(LSTM(units=units, return_sequences=True))
                self.model.add(Dropout(self.dropout))
                
            # Final LSTM layer (not returning sequences)
            self.model.add(LSTM(units=self.lstm_units[-1], return_sequences=False))
            self.model.add(Dropout(self.dropout))
            
            # Output layer
            self.model.add(Dense(units=1))
            
            # Compile model
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss='mean_squared_error'
            )
            
            logger.info(f"LSTM model built with layers: {self.lstm_units}")
            
            return self
            
        except ImportError as e:
            logger.error("TensorFlow not available. Please install tensorflow.")
            raise
    
    def _prepare_data(
        self,
        data: pd.Series,
        train_ratio: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training.
        
        Args:
            data: Time series data
            train_ratio: Ratio of training data
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        from sklearn.preprocessing import MinMaxScaler
        
        # Scale data
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
        
        # Create sequences
        X, y = [], []
        
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:(i + self.sequence_length)])
            y.append(scaled_data[i + self.sequence_length])
            
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        split_idx = int(len(X) * train_ratio)
        
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_test = X[split_idx:]
        y_test = y[split_idx:]
        
        logger.info(f"Prepared data: X_train={X_train.shape}, X_test={X_test.shape}")
        
        return X_train, y_train, X_test, y_test
    
    def fit(
        self,
        data: pd.Series,
        validation_split: float = 0.1,
        verbose: int = 1
    ) -> 'LSTMModel':
        """
        Train LSTM model.
        
        Args:
            data: Time series data
            validation_split: Ratio of validation data
            verbose: Verbosity level
            
        Returns:
            Self
        """
        # Prepare data
        X_train, y_train, X_test, y_test = self._prepare_data(data)
        
        # Build model if not already built
        if self.model is None:
            self._build_model(input_shape=(X_train.shape[1], 1))
            
        # Train model
        logger.info(f"Training LSTM model for {self.epochs} epochs...")
        
        # FIX: Add EarlyStopping and ModelCheckpoint for better training
        callbacks = []
        if TF_AVAILABLE:
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
            # Optional: Save best model during training
            # checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            verbose=verbose,
            callbacks=callbacks  # FIX: Added early stopping
        )
        
        # Evaluate on test set
        test_loss = self.model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Test loss: {test_loss:.6f}")
        
        return self
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            data: Input data (sequences)
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        # Make predictions
        predictions = self.model.predict(data, verbose=0)
        
        # Inverse transform
        if self.scaler is not None:
            predictions = self.scaler.inverse_transform(predictions)
            
        return predictions.flatten()
    
    def predict_next(self, last_sequence: np.ndarray) -> float:
        """
        Predict next value given last sequence.
        
        Args:
            last_sequence: Last N values
            
        Returns:
            Predicted next value
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        # Scale input
        scaled = self.scaler.transform(last_sequence.reshape(-1, 1))
        
        # Reshape for LSTM input
        X = scaled.reshape(1, -1, 1)
        
        # Predict
        prediction = self.model.predict(X, verbose=0)
        
        # Inverse transform
        prediction = self.scaler.inverse_transform(prediction)
        
        return float(prediction[0, 0])
    
    def predict_multiple(self, data: pd.Series, steps: int = 1) -> np.ndarray:
        """
        Predict multiple future steps.
        
        Args:
            data: Historical data
            steps: Number of steps to predict
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        # Get last sequence
        scaled_data = self.scaler.transform(data.values.reshape(-1, 1))
        
        # Start with last sequence
        current_sequence = scaled_data[-self.sequence_length:].flatten()
        
        predictions = []
        
        for _ in range(steps):
            # Reshape for LSTM
            X = current_sequence.reshape(1, -1, 1)
            
            # Predict
            pred = self.model.predict(X, verbose=0)
            
            predictions.append(pred[0, 0])
            
            # Update sequence
            current_sequence = np.append(current_sequence[1:], pred[0, 0])
            
        # Inverse transform
        predictions = self.scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()
        
        return predictions
    
    def evaluate(self, data: pd.Series) -> Dict:
        """
        Evaluate model on test data.
        
        Args:
            data: Full time series data
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        # Prepare test data
        X_train, y_train, X_test, y_test = self._prepare_data(data)
        
        # Make predictions
        predictions = self.predict(X_test)
        
        # Inverse transform actual values
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(y_test_actual, predictions)
        rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
        
        if np.std(y_test_actual) > 0:
            r2 = r2_score(y_test_actual, predictions)
            # FIX: Safe division to prevent division by zero errors in MAPE calculation
            mape = np.mean(np.abs((y_test_actual - predictions) / np.maximum(y_test_actual, 1e-10))) * 100
        else:
            r2 = 0
            mape = 0
            
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'predictions': predictions,
            'actual': y_test_actual
        }
    
    def save_model(self, path: Path) -> None:
        """Save model to file."""
        if self.model is None:
            raise ValueError("No model to save")
            
        self.model.save(path)
        
        # Save scaler and config separately
        import joblib
        config_path = path.with_suffix('.config')
        joblib.dump({
            'scaler': self.scaler,
            'sequence_length': self.sequence_length,
            'lstm_units': self.lstm_units,
            'config': self.config
        }, config_path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Path) -> 'LSTMModel':
        """Load model from file."""
        import joblib
        
        # Load config
        config_path = path.with_suffix('.config')
        config_data = joblib.load(config_path)
        
        self.scaler = config_data['scaler']
        self.sequence_length = config_data['sequence_length']
        self.lstm_units = config_data['lstm_units']
        self.config = config_data['config']
        
        # Load model
        import tensorflow as tf
        self.model = tf.keras.models.load_model(path)
        
        logger.info(f"Model loaded from {path}")
        return self
    
    def get_training_history(self) -> Optional[Dict]:
        """Get training history."""
        if self.history is None:
            return None
        return self.history.history
    
    def summary(self) -> str:
        """Get model summary."""
        if self.model is None:
            return "Model not built"
        return str(self.model.summary())


class GRUModel(LSTMModel):
    """GRU model for stock price prediction (extends LSTMModel)."""
    
    def _build_model(self, input_shape: Tuple) -> 'GRUModel':
        """
        Build GRU model architecture.
        
        Args:
            input_shape: Shape of input data
            
        Returns:
            Self
        """
        try:
            # Import TensorFlow
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import GRU, Dense, Dropout
            
            # Build model
            self.model = Sequential()
            
            # First GRU layer
            self.model.add(GRU(
                units=self.lstm_units[0],  # Using lstm_units for consistency
                return_sequences=True,
                input_shape=input_shape
            ))
            self.model.add(Dropout(self.dropout))
            
            # Additional GRU layers
            for units in self.lstm_units[1:]:
                self.model.add(GRU(units=units, return_sequences=True))
                self.model.add(Dropout(self.dropout))
                
            # Final GRU layer
            self.model.add(GRU(units=self.lstm_units[-1], return_sequences=False))
            self.model.add(Dropout(self.dropout))
            
            # Output layer
            self.model.add(Dense(units=1))
            
            # Compile model
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss='mean_squared_error'
            )
            
            logger.info(f"GRU model built with layers: {self.lstm_units}")
            
            return self
            
        except ImportError as e:
            logger.error("TensorFlow not available.")
            raise
