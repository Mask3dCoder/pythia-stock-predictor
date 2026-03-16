"""
Stock Data Collector Module

Collects historical and real-time stock data from Yahoo Finance,
Alpha Vantage, and other data sources.
"""

import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Union
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class StockDataCollector:
    """Collects stock data from various sources."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Stock Data Collector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
    def download_yahoo_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        years: int = 5,
        interval: str = "1d",
        auto_adjust: bool = True
    ) -> pd.DataFrame:
        """
        Download historical stock data from Yahoo Finance.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            years: Number of years of historical data (used if start_date not provided)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 1h, 1d, 1wk, 1mo)
            auto_adjust: Adjust for splits and dividends
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Calculate dates if not provided
            if end_date is None:
                end_date = datetime.now()
            else:
                end_date = pd.to_datetime(end_date)
                
            if start_date is None:
                start_date = end_date - timedelta(days=years * 365)
            else:
                start_date = pd.to_datetime(start_date)
            
            logger.info(f"Downloading {symbol} data from {start_date.date()} to {end_date.date()}")
            
            # Download data
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=auto_adjust
            )
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Clean up column names
            df.columns = [col.lower() for col in df.columns]
            
            # Add symbol column if multiple tickers
            df['symbol'] = symbol
            
            logger.info(f"Downloaded {len(df)} rows for {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading {symbol} data: {str(e)}")
            return pd.DataFrame()
    
    def download_multiple_symbols(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        years: int = 5,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Download data for multiple symbols.
        
        Args:
            symbols: List of stock ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            years: Number of years of historical data
            interval: Data interval
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        # FIX: Parallelized download using ThreadPoolExecutor for faster batch processing
        result = {}
        
        def download_single(symbol: str) -> tuple:
            """Helper function to download data for a single symbol."""
            try:
                df = self.download_yahoo_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    years=years,
                    interval=interval
                )
                if not df.empty:
                    return (symbol, df)
            except Exception as e:
                logger.error(f"Error downloading {symbol}: {str(e)}")
            return (symbol, None)
        
        # Use ThreadPoolExecutor for parallel downloads
        with ThreadPoolExecutor(max_workers=min(10, len(symbols))) as executor:
            # Submit all download tasks
            future_to_symbol = {
                executor.submit(download_single, symbol): symbol 
                for symbol in symbols
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_symbol):
                symbol, df = future.result()
                if df is not None:
                    result[symbol] = df
        
        return result
    
    def get_realtime_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get real-time quote for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with quote data
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            
            return {
                'symbol': symbol,
                'price': info.last_price,
                'open': info.open,
                'high': info.day_high,
                'low': info.day_low,
                'volume': info.last_volume,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting realtime quote for {symbol}: {str(e)}")
            return None
    
    def get_market_status(self) -> Dict:
        """
        Get current market status (open/closed).
        
        Returns:
            Dictionary with market status
        """
        try:
            ticker = yf.Ticker("^GSPC")  # S&P 500
            info = ticker.fast_info
            
            now = datetime.now()
            hour = now.hour
            
            # US market hours: 9:30 AM - 4:00 PM ET
            market_open = hour >= 9 and hour < 16
            
            return {
                'is_open': market_open,
                'timestamp': now,
                'next_open': None,
                'next_close': None
            }
            
        except Exception as e:
            logger.error(f"Error getting market status: {str(e)}")
            return {'is_open': False, 'timestamp': datetime.now()}
    
    def download_alpha_vantage(
        self,
        symbol: str,
        api_key: Optional[str] = None,
        function: str = "TIME_SERIES_DAILY",
        output_size: str = "compact"
    ) -> pd.DataFrame:
        """
        Download data from Alpha Vantage API.
        
        Args:
            symbol: Stock ticker symbol
            api_key: Alpha Vantage API key (optional if ALPHA_VANTAGE_API_KEY env var is set)
            function: API function (TIME_SERIES_DAILY, TIME_SERIES_INTRADAY, etc.)
            output_size: 'compact' (100 days) or 'full' (20+ years)
            
        Returns:
            DataFrame with stock data
        """
        # SECURITY FIX: Read API key from environment variable for security
        # Priority: 1) Environment variable ALPHA_VANTAGE_API_KEY
        #           2) Config file (self.config)
        #           3) Raise error if neither available
        api_key = os.environ.get('ALPHA_VANTAGE_API_KEY', 
                                  api_key or 
                                  self.config.get('alpha_vantage', {}).get('api_key', ''))
        
        if not api_key:
            raise ValueError(
                "Alpha Vantage API key not found. Set ALPHA_VANTAGE_API_KEY environment variable "
                "or add 'api_key' to the alpha_vantage section in config.yaml"
            )
        try:
            base_url = "https://www.alphavantage.co/query"
            params = {
                'function': function,
                'symbol': symbol,
                'outputsize': output_size,
                'apikey': api_key
            }
            
            response = requests.get(base_url, params=params)
            data = response.json()
            
            # Extract time series data
            time_series_key = None
            for key in data.keys():
                if 'Time Series' in key:
                    time_series_key = key
                    break
                    
            if time_series_key is None:
                logger.warning(f"No time series data for {symbol}")
                return pd.DataFrame()
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Rename columns
            column_mapping = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            }
            df = df.rename(columns=column_mapping)
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
                
            df['symbol'] = symbol
            
            logger.info(f"Downloaded {len(df)} rows from Alpha Vantage for {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading from Alpha Vantage: {str(e)}")
            return pd.DataFrame()
    
    def save_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        data_type: str = "historical"
    ) -> Path:
        """
        Save data to CSV file.
        
        Args:
            df: DataFrame to save
            symbol: Stock symbol
            data_type: Type of data (historical, realtime, etc.)
            
        Returns:
            Path to saved file
        """
        filename = f"{symbol}_{data_type}_{datetime.now().strftime('%Y%m%d')}.csv"
        filepath = self.data_dir / filename
        
        df.to_csv(filepath)
        logger.info(f"Saved data to {filepath}")
        
        return filepath
    
    def load_data(
        self,
        symbol: str,
        data_type: str = "historical"
    ) -> Optional[pd.DataFrame]:
        """
        Load saved data from CSV file.
        
        Args:
            symbol: Stock symbol
            data_type: Type of data
            
        Returns:
            DataFrame if found, None otherwise
        """
        pattern = f"{symbol}_{data_type}_*.csv"
        
        files = list(self.data_dir.glob(pattern))
        
        if not files:
            logger.warning(f"No saved data found for {symbol}")
            return None
            
        # Get most recent file
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        
        df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
        logger.info(f"Loaded data from {latest_file}")
        
        return df
    
    def get_stock_info(self, symbol: str) -> Dict:
        """
        Get company information for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with company information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'name': info.get('shortName', info.get('longName', 'N/A')),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0),
                'description': info.get('longBusinessSummary', '')
            }
            
        except Exception as e:
            logger.error(f"Error getting stock info for {symbol}: {str(e)}")
            return {'symbol': symbol, 'error': str(e)}

    def get_news_headlines(
        self,
        symbol: str,
        max_news: int = 10
    ) -> List[Dict]:
        """
        Fetch recent news headlines for a stock symbol.
        
        Args:
            symbol: Stock ticker symbol
            max_news: Maximum number of news items to fetch
            
        Returns:
            List of dictionaries with headline, date, and source
        """
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            if not news:
                logger.info(f"No news available for {symbol}")
                return []
            
            headlines = []
            for item in news[:max_news]:
                try:
                    headline = {
                        'title': item.get('title', ''),
                        'publisher': item.get('publisher', 'Unknown'),
                        'link': item.get('link', ''),
                        'timestamp': item.get('providerPublishTime', None)
                    }
                    
                    # Convert timestamp to datetime if available
                    if headline['timestamp']:
                        try:
                            headline['datetime'] = datetime.fromtimestamp(headline['timestamp'])
                        except (ValueError, OSError):
                            headline['datetime'] = None
                    else:
                        headline['datetime'] = None
                    
                    headlines.append(headline)
                    
                except Exception as e:
                    logger.warning(f"Error parsing news item: {str(e)}")
                    continue
            
            logger.info(f"Fetched {len(headlines)} news headlines for {symbol}")
            return headlines
            
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {str(e)}")
            return []
