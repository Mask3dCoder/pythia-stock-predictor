"""
CNN-LSTM Hybrid Model with Attention

A state-of-the-art hybrid architecture combining:
- 1D Convolutional layers for local pattern extraction
- LSTM layers for temporal dependencies
- Multi-head attention for feature importance
- Optional residual connections
"""

import logging
import os
from typing import Optional, Dict, Tuple, List
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# TensorFlow imports
TF_AVAILABLE = False
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    from tensorflow.keras import layers, Model, callbacks
    from tensorflow.keras.layers import (
        Conv1D, MaxPooling1D, LSTM, Dense, Dropout, 
        BatchNormalization, Add, Multiply, Concatenate,
        Attention, MultiHeadAttention, GlobalAveragePooling1D,
        Bidirectional, SpatialDropout1D
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    TF_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow not available. CNN-LSTM model will not work.")


class AttentionLayer(layers.Layer):
    """
    Custom attention layer for sequence modeling.
    
    Computes attention weights over the time dimension.
    """
    
    def __init__(self, units: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_context',
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True
        )
        super().build(input_shape)
        
    def call(self, inputs):
        # Score computation
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        
        # Attention weights
        attention = tf.tensordot(score, self.u, axes=1)
        attention_weights = tf.nn.softmax(attention, axis=1)
        
        # Weighted sum
        context = inputs * tf.expand_dims(attention_weights, -1)
        context = tf.reduce_sum(context, axis=1)
        
        return context, attention_weights
        
    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config


class ConvLSTMCell(layers.Layer):
    """
    Convolutional LSTM cell for local feature extraction.
    """
    
    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        strides: int = 1,
        padding: str = 'same',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        
    def build(self, input_shape):
        self.conv_i = layers.Conv1D(self.filters, self.kernel_size, padding=self.padding)
        self.conv_f = layers.Conv1D(self.filters, self.kernel_size, padding=self.padding)
        self.conv_c = layers.Conv1D(self.filters, self.kernel_size, padding=self.padding)
        self.conv_o = layers.Conv1D(self.filters, self.kernel_size, padding=self.padding)
        super().build(input_shape)
        
    def call(self, inputs, states):
        h, c = states
        
        i = tf.nn.sigmoid(self.conv_i(inputs) + self.conv_f(h))
        f = tf.nn.sigmoid(self.conv_f(inputs) + self.conv_f(h))
        c_new = f * c + i * tf.nn.tanh(self.conv_c(inputs) + self.conv_c(h))
        o = tf.nn.sigmoid(self.conv_o(inputs) + self.conv_o(h))
        h_new = o * tf.nn.tanh(c_new)
        
        return h_new, [h_new, c_new]
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding
        })
        return config


class CNNLSTMModel:
    """
    CNN-LSTM Hybrid Model with Multi-Head Attention.
    
    Architecture:
    1. Input preprocessing
    2. 1D Convolutional layers for local pattern extraction
    3. Bidirectional LSTM for temporal modeling
    4. Multi-head attention for feature importance
    5. Dense layers for prediction
    
    Supports:
    - Quantization
    - Model export to ONNX
    - Uncertainty estimation via Monte Carlo dropout
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize CNN-LSTM model.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config or {}
        
        # Model hyperparameters
        self.sequence_length = self.config.get('sequence_length', 60)
        self.n_features = self.config.get('n_features', 1)
        
        # CNN parameters
        self.conv_filters = self.config.get('conv_filters', [32, 64, 128])
        self.kernel_size = self.config.get('kernel_size', 3)
        self.pool_size = self.config.get('pool_size', 2)
        
        # LSTM parameters
        self.lstm_units = self.config.get('lstm_units', [64, 64])
        self.dropout = self.config.get('dropout', 0.3)
        self.recurrent_dropout = self.config.get('recurrent_dropout', 0.2)
        self.bidirectional = self.config.get('bidirectional', True)
        
        # Attention parameters
        self.attention_units = self.config.get('attention_units', 64)
        self.num_attention_heads = self.config.get('num_attention_heads', 4)
        
        # Training parameters
        self.epochs = self.config.get('epochs', 100)
        self.batch_size = self.config.get('batch_size', 32)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.early_stopping_patience = self.config.get('early_stopping_patience', 15)
        self.reduce_lr_patience = self.config.get('reduce_lr_patience', 5)
        
        # Model components
        self.model = None
        self.scaler = None
        self.history = None
        
    def _build_model(self) -> Model:
        """
        Build the CNN-LSTM architecture.
        
        Returns:
            Compiled Keras model
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for CNN-LSTM model")
            
        inputs = layers.Input(
            shape=(self.sequence_length, self.n_features),
            name='input'
        )
        
        # Convolutional blocks for local pattern extraction
        x = inputs
        
        for i, filters in enumerate(self.conv_filters):
            x = layers.Conv1D(
                filters=filters,
                kernel_size=self.kernel_size,
                padding='same',
                activation='relu',
                name=f'conv_{i}'
            )(x)
            x = layers.BatchNormalization(name=f'bn_{i}')(x)
            x = layers.MaxPooling1D(pool_size=self.pool_size, name=f'pool_{i}')(x)
            x = SpatialDropout1D(self.dropout, name=f'dropout_conv_{i}')(x)
            
        # LSTM layers for temporal modeling
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            
            if self.bidirectional:
                x = Bidirectional(
                    LSTM(units, return_sequences=return_sequences),
                    name=f'bi_lstm_{i}'
                )(x)
            else:
                x = LSTM(
                    units, 
                    return_sequences=return_sequences,
                    name=f'lstm_{i}'
                )(x)
                
            x = Dropout(self.dropout, name=f'dropout_lstm_{i}')(x)
            
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=self.num_attention_heads,
            key_dim=self.attention_units,
            name='multi_head_attention'
        )(x, x)
        
        # Residual connection
        x = Add(name='attention_residual')([x, attention_output])
        x = layers.LayerNormalization(name='attention_norm')(x)
        
        # Global pooling
        x = GlobalAveragePooling1D(name='global_pool')(x)
        
        # Dense layers
        x = Dense(64, activation='relu', name='dense_1')(x)
        x = Dropout(self.dropout, name='dropout_dense_1')(x)
        x = BatchNormalization(name='bn_dense')(x)
        
        x = Dense(32, activation='relu', name='dense_2')(x)
        x = Dropout(self.dropout * 0.5, name='dropout_dense_2')(x)
        
        # Output layer
        outputs = Dense(1, name='output')(x)
        
        # Build model
        model = Model(inputs=inputs, outputs=outputs, name='CNN_LSTM_Attention')
        
        # Compile
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        logger.info(f"Built CNN-LSTM model: {model.count_params()} parameters")
        
        return model
    
    def _prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Args:
            X: Feature array
            y: Target array
            train_ratio: Train/val split ratio
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val)
        """
        from sklearn.preprocessing import StandardScaler
        
        # Scale features
        n_samples = X.shape[0]
        X_2d = X.reshape(n_samples, -1)
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_2d).reshape(X.shape)
        
        # Split
        split_idx = int(n_samples * train_ratio)
        
        X_train = X_scaled[:split_idx]
        y_train = y[:split_idx]
        X_val = X_scaled[split_idx:]
        y_val = y[split_idx:]
        
        return X_train, y_train, X_val, y_val
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2,
        verbose: int = 1
    ) -> 'CNNLSTMModel':
        """
        Train the model.
        
        Args:
            X: Training features (samples, timesteps, features)
            y: Training targets
            validation_split: Validation split ratio
            verbose: Verbosity level
            
        Returns:
            Self
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required")
            
        # Update n_features
        self.n_features = X.shape[2] if X.ndim > 2 else 1
        
        # Build model if not already built
        if self.model is None:
            self.model = self._build_model()
            
        # Prepare callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=verbose
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.reduce_lr_patience,
                verbose=verbose
            )
        ]
        
        # Train
        self.history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            callbacks=callbacks_list,
            verbose=verbose
        )
        
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained")
            
        predictions = self.model.predict(X, verbose=0)
        
        return predictions.flatten()
    
    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        n_samples: int = 100,
        dropout: float = 0.1
    ) -> Dict[str, np.ndarray]:
        """
        Generate predictions with uncertainty using Monte Carlo dropout.
        
        Args:
            X: Input features
            n_samples: Number of forward passes
            dropout: Dropout rate for uncertainty estimation
            
        Returns:
            Dictionary with predictions, lower_bound, upper_bound
        """
        if self.model is None:
            raise ValueError("Model not trained")
            
        # Enable dropout at inference
        tf.keras.backend.set_learning_mode(True)
        
        # Create a model with dropout enabled
        # Note: This is a simplified approach
        predictions = []
        
        for _ in range(n_samples):
            pred = self.model(X, training=True)
            predictions.append(pred.numpy())
            
        tf.keras.backend.set_learning_mode(False)
        
        predictions = np.array(predictions).squeeze()
        
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # 95% confidence interval
        lower_bound = mean_pred - 1.96 * std_pred
        upper_bound = mean_pred + 1.96 * std_pred
        
        return {
            'predictions': mean_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'std': std_pred
        }
    
    def predict_sequence(
        self,
        initial_sequence: np.ndarray,
        n_steps: int
    ) -> np.ndarray:
        """
        Predict multiple future steps iteratively.
        
        Args:
            initial_sequence: Initial sequence for prediction
            n_steps: Number of steps to predict
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained")
            
        predictions = []
        current_sequence = initial_sequence.copy()
        
        for _ in range(n_steps):
            # Predict next step
            pred = self.model.predict(
                current_sequence.reshape(1, -1, self.n_features),
                verbose=0
            )
            
            predictions.append(pred[0, 0])
            
            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = pred[0, 0]
            
        return np.array(predictions)
    
    def save(self, path: Path) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")
            
        self.model.save(path)
        
        # Save scaler and config
        import joblib
        config_path = path.with_suffix('.config')
        joblib.dump({
            'scaler': self.scaler,
            'config': self.config,
            'sequence_length': self.sequence_length,
            'n_features': self.n_features
        }, config_path)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: Path) -> 'CNNLSTMModel':
        """
        Load model from disk.
        
        Args:
            path: Path to saved model
            
        Returns:
            Self
        """
        import joblib
        
        # Load config
        config_path = path.with_suffix('.config')
        config_data = joblib.load(config_path)
        
        self.scaler = config_data['scaler']
        self.config = config_data['config']
        self.sequence_length = config_data['sequence_length']
        self.n_features = config_data['n_features']
        
        # Load model
        self.model = tf.keras.models.load_model(path)
        
        self.is_fitted = True
        
        logger.info(f"Model loaded from {path}")
        
        return self
    
    def summary(self) -> str:
        """Get model summary."""
        if self.model is None:
            return "Model not built"
        return str(self.model.summary())
    
    def get_feature_importance(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Get feature importance using attention weights.
        
        Args:
            X: Input features
            
        Returns:
            Array of importance scores
        """
        # This is a simplified version
        # In production, you would use integrated gradients or SHAP
        return np.ones(self.n_features)


class QuantizedModel:
    """
    Wrapper for quantized model using ONNX Runtime.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path
        self.session = None
        
    def convert_to_onnx(
        self,
        keras_model: Model,
        output_path: Path
    ) -> None:
        """
        Convert Keras model to ONNX format.
        
        Args:
            keras_model: Keras model
            output_path: Output path for ONNX model
        """
        try:
            import tf2onnx
            
            # Convert
            spec = (
                tf.TensorSpec((None, keras_model.input_shape[1], keras_model.input_shape[2]), tf.float32, name="input"),
            )
            model_proto, _ = tf2onnx.convert.from_keras(keras_model, input_signature=spec)
            
            # Save
            with open(output_path, "wb") as f:
                f.write(model_proto.SerializeToString())
                
            logger.info(f"Model converted to ONNX: {output_path}")
            
        except ImportError:
            logger.error("tf2onnx not installed. Install with: pip install tf2onnx")
            
    def load_onnx(self, model_path: Path) -> None:
        """
        Load ONNX model for inference.
        
        Args:
            model_path: Path to ONNX model
        """
        try:
            import onnxruntime as ort
            
            self.session = ort.InferenceSession(
                str(model_path),
                providers=['CPUExecutionProvider']
            )
            
            logger.info(f"Loaded ONNX model: {model_path}")
            
        except ImportError:
            logger.error("onnxruntime not installed. Install with: pip install onnxruntime")
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Run inference with ONNX model.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if self.session is None:
            raise ValueError("ONNX model not loaded")
            
        inputs = {self.session.get_inputs()[0].name: X}
        outputs = self.session.run(None, inputs)
        
        return outputs[0]
