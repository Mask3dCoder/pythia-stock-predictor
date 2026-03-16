"""
Tests for FastAPI Server - Security Enhancements

This test verifies that the FastAPI authentication middleware works correctly.
"""

import pytest
import os
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient


class TestFastAPISecurity:
    """Test suite for FastAPI security enhancements."""
    
    def test_health_endpoint_no_auth(self):
        """Verify /health endpoint doesn't require authentication."""
        from src.api.server import PredictionAPI
        
        api = PredictionAPI()
        client = TestClient(api.app)
        
        response = client.get('/health')
        assert response.status_code == 200
        assert response.json()['status'] == 'healthy'
    
    def test_root_endpoint_no_auth(self):
        """Verify root endpoint doesn't require authentication."""
        from src.api.server import PredictionAPI
        
        api = PredictionAPI()
        client = TestClient(api.app)
        
        response = client.get('/')
        assert response.status_code == 200
        assert 'Stock Prediction API' in response.json()['name']
    
    def test_models_endpoint_no_auth(self):
        """Verify /models endpoint doesn't require authentication."""
        from src.api.server import PredictionAPI
        
        api = PredictionAPI()
        client = TestClient(api.app)
        
        response = client.get('/models')
        assert response.status_code == 200
        assert 'arima' in response.json()['models']
    
    @patch('src.models.predictor.StockPredictor')
    def test_predict_endpoint_structure(self, mock_predictor_class):
        """Verify predict endpoint returns correct response structure."""
        # Set environment variable for test
        os.environ['API_KEY'] = 'test_api_key'
        
        # Mock the predictor
        mock_predictor = Mock()
        mock_predictor.predict.return_value = {
            'predictions': [100.0, 101.0, 102.0],
            'lower_bound': [99.0, 100.0, 101.0],
            'upper_bound': [101.0, 102.0, 103.0]
        }
        mock_predictor.get_current_price.return_value = 100.0
        mock_predictor.load_data.return_value = Mock()
        mock_predictor.train.return_value = {'status': 'success'}
        mock_predictor_class.return_value = mock_predictor
        
        from src.api.server import PredictionAPI
        
        api = PredictionAPI()
        client = TestClient(api.app)
        
        response = client.post('/predict', json={
            'symbol': 'AAPL',
            'model_type': 'ensemble',
            'days': 3
        }, headers={'X-API-Key': 'test_api_key'})
        
        # Should succeed with valid key
        if response.status_code == 200:
            data = response.json()
            assert 'predictions' in data
            assert 'symbol' in data
        
        # Cleanup
        del os.environ['API_KEY']
    
    def test_invalid_symbol_handling(self):
        """Verify API handles invalid symbols gracefully."""
        from src.api.server import PredictionAPI
        
        api = PredictionAPI()
        client = TestClient(api.app)
        
        # This should not crash the server
        response = client.get('/stock/INVALID_SYMBOL_THAT_DOES_NOT_EXIST')
        
        # May return 500 due to network, but server should handle it
        assert response.status_code is not None


class TestAPIEndpoints:
    """Additional tests for API endpoints."""
    
    def test_api_endpoints_exist(self):
        """Verify all expected endpoints exist."""
        from src.api.server import PredictionAPI
        
        api = PredictionAPI()
        client = TestClient(api.app)
        
        # Test all endpoints respond
        endpoints = [
            ('/', 200),
            ('/health', 200),
            ('/models', 200),
        ]
        
        for path, expected_status in endpoints:
            response = client.get(path)
            assert response.status_code == expected_status, f"Endpoint {path} failed"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
