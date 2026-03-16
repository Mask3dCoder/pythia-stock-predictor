"""
Tests for Ensemble Model - Critical Bug Fix Verification

This test verifies that the ensemble model's predict() method uses actual data
instead of dummy zeros (the critical bug that was fixed).
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch


class TestEnsembleModelFix:
    """Test suite for Ensemble Model critical bug fix."""
    
    def test_last_sequence_is_stored_after_fit(self):
        """Verify that fit() stores the last sequence for predictions."""
        # Import after patches to avoid loading issues
        from src.models.ensemble_model import EnsembleModel
        
        # Create ensemble with sequence_length=10 for easier testing
        config = {
            'sequence_length': 10,
            'weights': {'arima': 0.33, 'lstm': 0.34, 'gru': 0.33}
        }
        ensemble = EnsembleModel(config)
        
        # Create sample training data (more than sequence_length)
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        data = pd.Series(np.random.randn(50).cumsum() + 100, index=dates)
        
        # Mock the ARIMA model
        mock_arima = Mock()
        mock_arima.results = Mock()
        mock_arima.predict.return_value = np.array([101.0, 102.0])
        mock_arima.predict_with_confidence.return_value = {
            'predictions': np.array([101.0, 102.0]),
            'lower_bound': np.array([100.0, 101.0]),
            'upper_bound': np.array([102.0, 103.0])
        }
        
        # Mock LSTM/GRU models
        mock_lstm = MagicMock()
        mock_lstm.model = MagicMock()
        mock_lstm.predict_multiple.return_value = np.array([100.5, 101.5])
        
        mock_gru = MagicMock()
        mock_gru.model = MagicMock()
        mock_gru.predict_multiple.return_value = np.array([100.0, 101.0])
        
        # Add models to ensemble
        ensemble.models['arima'] = mock_arima
        ensemble.models['lstm'] = mock_lstm
        ensemble.models['gru'] = mock_gru
        
        # Call fit to store last_sequence
        ensemble.fit(data)
        
        # Assert: last_sequence should be populated
        assert ensemble.last_sequence is not None, "last_sequence should be populated after fit()"
        
        # Assert: last_sequence should have correct length
        assert len(ensemble.last_sequence) == 10, f"Expected length 10, got {len(ensemble.last_sequence)}"
        
        # Assert: last_sequence should match the tail of data
        expected_tail = data.iloc[-10:].reset_index(drop=True)
        pd.testing.assert_series_equal(
            ensemble.last_sequence.reset_index(drop=True),
            expected_tail.reset_index(drop=True),
            check_names=False
        )
    
    def test_predict_uses_actual_sequence_not_zeros(self):
        """Verify predict() calls LSTM/GRU with actual sequence, not zeros."""
        from src.models.ensemble_model import EnsembleModel
        
        # Create ensemble
        config = {
            'sequence_length': 5,
            'weights': {'arima': 0.33, 'lstm': 0.34, 'gru': 0.33}
        }
        ensemble = EnsembleModel(config)
        
        # Create and store last_sequence
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        data = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], index=dates)
        ensemble.last_sequence = data.iloc[-5:].reset_index(drop=True)
        
        # Mock the models
        mock_arima = Mock()
        mock_arima.results = Mock()
        mock_arima.predict.return_value = np.array([110.0])
        
        mock_lstm = MagicMock()
        mock_lstm.model = MagicMock()
        mock_lstm.predict_multiple = Mock(return_value=np.array([110.5]))
        
        mock_gru = MagicMock()
        mock_gru.model = MagicMock()
        mock_gru.predict_multiple = Mock(return_value=np.array([110.0]))
        
        ensemble.models['arima'] = mock_arima
        ensemble.models['lstm'] = mock_lstm
        ensemble.models['gru'] = mock_gru
        ensemble.is_fitted = True
        
        # Call predict
        result = ensemble.predict(steps=1)
        
        # Assert: LSTM predict_multiple was called with actual last_sequence
        mock_lstm.predict_multiple.assert_called_once()
        call_args = mock_lstm.predict_multiple.call_args[0]
        
        # Verify it was NOT called with zeros
        assert not np.allclose(call_args[0], 0), "LSTM should NOT be called with zeros!"
        
        # Verify it was called with the stored last_sequence
        expected_sequence = ensemble.last_sequence
        assert np.allclose(call_args[0], expected_sequence), \
            f"LSTM should be called with last_sequence {expected_sequence.values}, got {call_args[0]}"
        
        # Assert: GRU predict_multiple was also called correctly
        mock_gru.predict_multiple.assert_called_once()
        gru_call_args = mock_gru.predict_multiple.call_args[0]
        assert not np.allclose(gru_call_args[0], 0), "GRU should NOT be called with zeros!"
    
    def test_ensemble_with_insufficient_data(self):
        """Test ensemble handles data shorter than sequence_length."""
        from src.models.ensemble_model import EnsembleModel
        
        config = {'sequence_length': 10}
        ensemble = EnsembleModel(config)
        
        # Data shorter than sequence_length
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        data = pd.Series([100, 101, 102, 103, 104], index=dates)
        
        # Mock ARIMA
        mock_arima = Mock()
        mock_arima.results = Mock()
        mock_arima.predict.return_value = np.array([105.0])
        ensemble.models['arima'] = mock_arima
        
        # Fit should handle this gracefully
        ensemble.fit(data)
        
        # last_sequence should be None when data is insufficient
        assert ensemble.last_sequence is None or len(ensemble.last_sequence) < 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
