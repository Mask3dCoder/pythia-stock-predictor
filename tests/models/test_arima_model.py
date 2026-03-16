"""
Tests for ARIMA Model - MAPE Safe Division

This test verifies that the MAPE calculation handles zero/near-zero prices correctly
without raising Division by Zero errors.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch


class TestARPIMAMAPESafeDivision:
    """Test suite for ARIMA MAPE safe division fix."""
    
    def test_mape_with_zero_values(self):
        """Verify MAPE handles zero values without division by zero."""
        from src.models.arima_model import ARIMAModel
        
        # Create ARIMA model
        model = ARIMAModel({'order': [5, 1, 0]})
        
        # Mock the model results
        mock_results = Mock()
        mock_results.forecast.return_value = np.array([100.0, 101.0, 99.0])
        model.results = mock_results
        model.scaler = None  # No scaler for simplicity
        
        # Test data with zeros
        test_data = pd.Series([0.0, 0.0, 100.0, 101.0, 99.0])
        
        # This should NOT raise a DivisionByZero error
        try:
            metrics = model.evaluate(test_data)
        except ZeroDivisionError as e:
            pytest.fail(f"MAPE calculation raised ZeroDivisionError: {e}")
        
        # Verify metrics are valid
        assert 'mape' in metrics, "MAPE should be in metrics"
        assert metrics['mape'] is not None, "MAPE should not be None"
        assert not np.isnan(metrics['mape']), "MAPE should not be NaN"
        assert metrics['mape'] >= 0, "MAPE should be non-negative"
    
    def test_mape_with_near_zero_values(self):
        """Verify MAPE handles near-zero values correctly."""
        from src.models.arima_model import ARIMAModel
        
        model = ARIMAModel({'order': [5, 1, 0]})
        
        # Mock results
        mock_results = Mock()
        mock_results.forecast.return_value = np.array([1e-10, 1e-10, 100.0])
        model.results = mock_results
        model.scaler = None
        
        # Test data with near-zero values
        test_data = pd.Series([1e-10, 1e-10, 100.0])
        
        # Should handle gracefully
        try:
            metrics = model.evaluate(test_data)
        except ZeroDivisionError:
            pytest.fail("Should not raise ZeroDivisionError for near-zero values")
        
        # Should return finite value
        assert np.isfinite(metrics['mape']), "MAPE should be finite"
    
    def test_mape_with_all_zeros(self):
        """Verify MAPE handles all-zero test data."""
        from src.models.arima_model import ARIMAModel
        
        model = ARIMAModel({'order': [5, 1, 0]})
        
        mock_results = Mock()
        mock_results.forecast.return_value = np.array([0.0, 0.0, 0.0])
        model.results = mock_results
        model.scaler = None
        
        # All zeros
        test_data = pd.Series([0.0, 0.0, 0.0])
        
        # Should not crash
        try:
            metrics = model.evaluate(test_data)
        except ZeroDivisionError:
            pytest.fail("Should handle all-zero data gracefully")
        
        # MAPE may be undefined or very high - but shouldn't crash
        assert 'mape' in metrics
    
    def test_mape_with_normal_values(self):
        """Verify MAPE still works correctly for normal non-zero values."""
        from src.models.arima_model import ARIMAModel
        
        model = ARIMAModel({'order': [5, 1, 0]})
        
        mock_results = Mock()
        # Predictions close to actual should give low MAPE
        mock_results.forecast.return_value = np.array([100.5, 101.0, 99.5])
        model.results = mock_results
        model.scaler = None
        
        test_data = pd.Series([100.0, 100.0, 100.0])
        
        metrics = model.evaluate(test_data)
        
        # Should be low MAPE since predictions are close
        assert metrics['mape'] < 2.0, f"MAPE should be low, got {metrics['mape']}%"
    
    def test_mape_safe_division_formula(self):
        """Verify the safe division formula is applied correctly."""
        # Test the formula directly
        y_true = np.array([0.0, 0.0, 100.0, 101.0, 99.0])
        y_pred = np.array([100.0, 101.0, 99.0, 100.0, 98.0])[:len(y_true)]
        
        # Original (unsafe) formula would fail
        # Fixed formula with safe division
        safe_denominator = np.maximum(y_true, 1e-10)
        mape = np.mean(np.abs((y_true - y_pred) / safe_denominator)) * 100
        
        assert np.isfinite(mape), "MAPE should be finite with safe division"
        assert mape >= 0, "MAPE should be non-negative"


class TestARIMAModel:
    """Additional tests for ARIMA model functionality."""
    
    def test_predict_with_confidence_intervals(self):
        """Verify prediction with confidence intervals works."""
        from src.models.arima_model import ARIMAModel
        
        model = ARIMAModel({'order': [5, 1, 0]})
        
        # Create mock results
        mock_conf_int = pd.DataFrame({
            'lower': [95.0, 94.0, 93.0],
            'upper': [105.0, 106.0, 107.0]
        })
        
        mock_results = Mock()
        mock_results.get_forecast.return_value = Mock(
            predicted_mean=pd.Series([100.0, 101.0, 102.0]),
            conf_int=lambda alpha: mock_conf_int
        )
        
        model.results = mock_results
        model.scaler = None
        
        result = model.predict_with_confidence(steps=3)
        
        assert 'predictions' in result
        assert 'lower_bound' in result
        assert 'upper_bound' in result
        assert 'confidence' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
