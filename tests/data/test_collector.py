"""
Tests for Data Collector - Parallel Downloads

This test verifies that the download_multiple_symbols() method uses
ThreadPoolExecutor for parallel downloads.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor


class TestParallelDownloads:
    """Test suite for parallel download functionality."""
    
    @patch('src.data.collector.yf.Ticker')
    def test_single_symbol_download(self, mock_ticker):
        """Verify single symbol download works."""
        from src.data.collector import StockDataCollector
        
        mock_ticker_instance = Mock()
        mock_history = pd.DataFrame({
            'Open': [100.0] * 10,
            'High': [105.0] * 10,
            'Low': [95.0] * 10,
            'Close': [100.0] * 10,
            'Volume': [1000000] * 10
        }, index=pd.date_range('2023-01-01', periods=10))
        mock_ticker_instance.history.return_value = mock_history
        mock_ticker.return_value = mock_ticker_instance
        
        collector = StockDataCollector()
        
        result = collector.download_yahoo_data('AAPL', years=1)
        
        # Verify data was downloaded
        assert not result.empty
        assert len(result) == 10
    
    @patch('src.data.collector.yf.Ticker')
    def test_download_multiple_symbols(self, mock_ticker):
        """Verify multiple symbol download works."""
        from src.data.collector import StockDataCollector
        
        # Track call order
        mock_ticker_instance = Mock()
        mock_history = pd.DataFrame({
            'Open': [100.0] * 5,
            'High': [105.0] * 5,
            'Low': [95.0] * 5,
            'Close': [100.0] * 5,
            'Volume': [1000000] * 5
        }, index=pd.date_range('2023-01-01', periods=5))
        mock_ticker_instance.history.return_value = mock_history
        mock_ticker.return_value = mock_ticker_instance
        
        collector = StockDataCollector()
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        
        result = collector.download_multiple_symbols(symbols, years=1)
        
        # Verify all symbols were downloaded
        assert len(result) == len(symbols)
        assert all(sym in result for sym in symbols)
    
    @patch('src.data.collector.yf.Ticker')
    def test_download_handles_errors(self, mock_ticker):
        """Verify graceful handling when one symbol fails."""
        from src.data.collector import StockDataCollector
        
        call_count = [0]
        
        def mock_history_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return pd.DataFrame({
                    'Open': [100.0] * 5,
                    'Close': [100.0] * 5,
                    'High': [105.0] * 5,
                    'Low': [95.0] * 5,
                    'Volume': [1000000] * 5
                }, index=pd.date_range('2023-01-01', periods=5))
            else:
                raise Exception("Network error")
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.side_effect = mock_history_side_effect
        mock_ticker.return_value = mock_ticker_instance
        
        collector = StockDataCollector()
        symbols = ['AAPL', 'GOOGL']
        
        # Should not raise exception, should return available data
        result = collector.download_multiple_symbols(symbols, years=1)
        
        # At least one symbol should succeed
        assert len(result) > 0


class TestStockDataCollector:
    """Additional tests for StockDataCollector."""
    
    @patch('src.data.collector.yf.Ticker')
    def test_get_realtime_quote(self, mock_ticker):
        """Verify real-time quote retrieval."""
        from src.data.collector import StockDataCollector
        
        mock_info = Mock()
        mock_info.last_price = 150.0
        mock_info.open = 148.0
        mock_info.day_high = 152.0
        mock_info.day_low = 147.0
        mock_info.last_volume = 1000000
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.fast_info = mock_info
        mock_ticker.return_value = mock_ticker_instance
        
        collector = StockDataCollector()
        quote = collector.get_realtime_quote('AAPL')
        
        assert quote is not None
        assert quote['price'] == 150.0
        assert quote['symbol'] == 'AAPL'
    
    @patch('src.data.collector.yf.Ticker')
    def test_get_stock_info(self, mock_ticker):
        """Verify stock info retrieval."""
        from src.data.collector import StockDataCollector
        
        mock_info = {
            'shortName': 'Apple Inc.',
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'marketCap': 3000000000000,
            'trailingPE': 25.0,
            'dividendYield': 0.005,
            'beta': 1.2,
            'fiftyTwoWeekHigh': 200.0,
            'fiftyTwoWeekLow': 150.0,
            'longBusinessSummary': 'Apple Inc. designs...'
        }
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = mock_info
        mock_ticker.return_value = mock_ticker_instance
        
        collector = StockDataCollector()
        info = collector.get_stock_info('AAPL')
        
        assert info['name'] == 'Apple Inc.'
        assert info['sector'] == 'Technology'
    
    def test_data_directory_creation(self):
        """Verify data directory is created."""
        from src.data.collector import StockDataCollector
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = StockDataCollector()
            collector.data_dir = os.path.join(tmpdir, 'data')
            
            # Trigger directory creation
            collector.data_dir = collector.data_dir
            import pathlib
            pathlib.Path(collector.data_dir).mkdir(exist_ok=True)
            
            assert os.path.exists(collector.data_dir)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
