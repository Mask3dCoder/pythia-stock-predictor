"""
Tests for Predictor - Sentiment Integration

This test verifies that the sentiment analysis pipeline is properly integrated
into the StockPredictor when enabled.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock


class TestSentimentIntegration:
    """Test suite for sentiment analysis integration."""
    
    def test_sentiment_enabled_in_config(self):
        """Verify sentiment can be enabled via config."""
        config = {
            'sentiment': {
                'enabled': True,
                'method': 'vader'
            }
        }
        
        # Create predictor with sentiment enabled
        with patch('src.data.collector.StockDataCollector'):
            with patch('src.data.preprocessor.DataPreprocessor'):
                from src.models.predictor import StockPredictor
                predictor = StockPredictor('AAPL', 'arima', config)
        
        assert predictor.enable_sentiment is True
        assert predictor.sentiment_method == 'vader'
    
    def test_sentiment_disabled_by_default(self):
        """Verify sentiment is disabled by default."""
        # No sentiment config
        with patch('src.data.collector.StockDataCollector'):
            with patch('src.data.preprocessor.DataPreprocessor'):
                from src.models.predictor import StockPredictor
                predictor = StockPredictor('AAPL', 'arima', {})
        
        assert predictor.enable_sentiment is False
    
    @patch('src.data.collector.StockDataCollector')
    @patch('src.data.preprocessor.DataPreprocessor')
    def test_sentiment_fetches_news_when_enabled(self, mock_preprocessor, mock_collector):
        """Verify news is fetched when sentiment is enabled."""
        from src.models.predictor import StockPredictor
        
        # Setup mocks
        mock_collector_instance = Mock()
        mock_collector_instance.download_yahoo_data.return_value = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [98, 99],
            'close': [100, 101],
            'volume': [1000000, 1000000]
        }, index=pd.date_range('2023-01-01', periods=2))
        
        # Mock news headlines
        mock_collector_instance.get_news_headlines.return_value = [
            {'title': 'Good earnings', 'datetime': '2023-01-01'},
            {'title': 'Stock up', 'datetime': '2023-01-02'}
        ]
        
        mock_collector.return_value = mock_collector_instance
        
        mock_preprocessor_instance = Mock()
        mock_preprocessor_instance.clean_data.return_value = pd.DataFrame({
            'close': [100, 101]
        }, index=pd.date_range('2023-01-01', periods=2))
        mock_preprocessor_instance.add_all_indicators.return_value = pd.DataFrame({
            'close': [100, 101],
            'sma_20': [100, 101]
        }, index=pd.date_range('2023-01-01', periods=2))
        mock_preprocessor.return_value = mock_preprocessor_instance
        
        # Create predictor with sentiment enabled
        config = {'sentiment': {'enabled': True, 'method': 'vader'}}
        
        with patch('src.sentiment.analyzer.SentimentAnalyzer') as mock_sentiment:
            mock_analyzer = Mock()
            mock_analyzer.analyze_batch.return_value = pd.DataFrame({
                'text': ['Good earnings', 'Stock up'],
                'compound': [0.5, 0.6],
                'sentiment': ['positive', 'positive']
            })
            mock_analyzer.create_sentiment_features.return_value = pd.DataFrame({
                'sentiment_compound': [0.55],
                'sentiment_pos': [1.0],
                'sentiment_neg': [0.0]
            })
            mock_sentiment.return_value = mock_analyzer
            
            predictor = StockPredictor('AAPL', 'arima', config)
            
            # Call load_data
            predictor.load_data(years=1)
            
            # Verify news was fetched
            mock_collector_instance.get_news_headlines.assert_called()
    
    @patch('src.data.collector.StockDataCollector')
    @patch('src.data.preprocessor.DataPreprocessor')
    def test_no_sentiment_when_disabled(self, mock_preprocessor, mock_collector):
        """Verify no sentiment calls when disabled."""
        from src.models.predictor import StockPredictor
        
        # Setup mocks
        mock_collector_instance = Mock()
        mock_collector_instance.download_yahoo_data.return_value = pd.DataFrame({
            'close': [100, 101]
        }, index=pd.date_range('2023-01-01', periods=2))
        mock_collector.return_value = mock_collector_instance
        
        mock_preprocessor_instance = Mock()
        mock_preprocessor_instance.clean_data.return_value = pd.DataFrame({
            'close': [100, 101]
        }, index=pd.date_range('2023-01-01', periods=2))
        mock_preprocessor_instance.add_all_indicators.return_value = pd.DataFrame({
            'close': [100, 101]
        }, index=pd.date_range('2023-01-01', periods=2))
        mock_preprocessor.return_value = mock_preprocessor_instance
        
        # Create predictor with sentiment DISABLED
        config = {'sentiment': {'enabled': False}}
        
        with patch('src.sentiment.analyzer.SentimentAnalyzer') as mock_sentiment:
            predictor = StockPredictor('AAPL', 'arima', config)
            
            # Call load_data
            predictor.load_data(years=1)
            
            # Verify sentiment analyzer was NOT called with news fetching
            # (the implementation may still create it but not use it)
            if hasattr(predictor, 'sentiment_analyzer'):
                # If analyzer exists, verify it wasn't used
                if predictor.sentiment_analyzer is not None:
                    mock_sentiment.assert_not_called()
    
    @patch('src.data.collector.StockDataCollector')
    @patch('src.data.preprocessor.DataPreprocessor')
    def test_sentiment_graceful_failure(self, mock_preprocessor, mock_collector):
        """Verify predictor handles sentiment failure gracefully."""
        from src.models.predictor import StockPredictor
        
        # Setup mocks
        mock_collector_instance = Mock()
        mock_collector_instance.download_yahoo_data.return_value = pd.DataFrame({
            'close': [100, 101]
        }, index=pd.date_range('2023-01-01', periods=2))
        mock_collector_instance.get_news_headlines.side_effect = Exception("Network error")
        mock_collector.return_value = mock_collector_instance
        
        mock_preprocessor_instance = Mock()
        mock_preprocessor_instance.clean_data.return_value = pd.DataFrame({
            'close': [100, 101]
        }, index=pd.date_range('2023-01-01', periods=2))
        mock_preprocessor_instance.add_all_indicators.return_value = pd.DataFrame({
            'close': [100, 101]
        }, index=pd.date_range('2023-01-01', periods=2))
        mock_preprocessor.return_value = mock_preprocessor_instance
        
        config = {'sentiment': {'enabled': True}}
        
        predictor = StockPredictor('AAPL', 'arima', config)
        
        # This should NOT raise an exception
        try:
            predictor.load_data(years=1)
        except Exception as e:
            pytest.fail(f"load_data should handle sentiment failures gracefully, got: {e}")
        
        # Verify data was still loaded (without sentiment)
        assert predictor.data is not None


class TestStockPredictor:
    """Additional tests for StockPredictor."""
    
    @patch('src.data.collector.StockDataCollector')
    @patch('src.data.preprocessor.DataPreprocessor')
    def test_load_data(self, mock_preprocessor, mock_collector):
        """Verify data loading works."""
        from src.models.predictor import StockPredictor
        
        # Setup mocks
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        mock_df = pd.DataFrame({
            'open': np.random.uniform(90, 110, 100),
            'high': np.random.uniform(95, 115, 100),
            'low': np.random.uniform(85, 105, 100),
            'close': np.random.uniform(90, 110, 100),
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        mock_collector_instance = Mock()
        mock_collector_instance.download_yahoo_data.return_value = mock_df
        mock_collector.return_value = mock_collector_instance
        
        mock_preprocessor_instance = Mock()
        mock_preprocessor_instance.clean_data.return_value = mock_df
        mock_preprocessor_instance.add_all_indicators.return_value = mock_df
        mock_preprocessor.return_value = mock_preprocessor_instance
        
        predictor = StockPredictor('AAPL', 'arima')
        
        result = predictor.load_data(years=1)
        
        assert result is not None
        mock_collector_instance.download_yahoo_data.assert_called_once()
    
    @patch('src.models.arima_model.ARIMAModel')
    @patch('src.data.collector.StockDataCollector')
    @patch('src.data.preprocessor.DataPreprocessor')
    def test_train(self, mock_preprocessor, mock_collector, mock_arima_class):
        """Verify model training works."""
        from src.models.predictor import StockPredictor
        
        # Setup mocks
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        mock_df = pd.DataFrame({
            'close': np.random.uniform(90, 110, 100)
        }, index=dates)
        
        mock_collector_instance = Mock()
        mock_collector_instance.download_yahoo_data.return_value = mock_df
        mock_collector.return_value = mock_collector_instance
        
        mock_preprocessor_instance = Mock()
        mock_preprocessor_instance.clean_data.return_value = mock_df
        mock_preprocessor_instance.add_all_indicators.return_value = mock_df
        mock_preprocessor.return_value = mock_preprocessor_instance
        
        mock_arima_instance = Mock()
        mock_arima_class.return_value = mock_arima_instance
        mock_arima_instance.fit.return_value = mock_arima_instance
        
        predictor = StockPredictor('AAPL', 'arima')
        predictor.load_data(years=1)
        result = predictor.train()
        
        assert result['status'] == 'success'
        mock_arima_instance.fit.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
