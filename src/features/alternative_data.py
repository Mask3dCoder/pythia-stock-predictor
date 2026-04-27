"""
Alternative Data Sources

Integration with external data sources:
- FRED (Federal Reserve Economic Data)
- Market indices and ETFs
- News and sentiment APIs
"""

import os
import logging
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import requests

logger = logging.getLogger(__name__)


class FredDataFetcher:
    """
    Fetches macroeconomic data from FRED (Federal Reserve Economic Data).
    
    Common indicators:
    - GDP: Gross Domestic Product
    - UNRATE: Unemployment Rate
    - FEDFUNDS: Federal Funds Rate
    - CPIAUCSL: Consumer Price Index
    - DGS10: 10-Year Treasury Rate
    - VIX: CBOE Volatility Index
    - M2SL: M2 Money Supply
    """
    
    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
    
    # Common FRED series IDs
    SERIES_MAP = {
        'gdp': 'GDP',
        'unemployment': 'UNRATE',
        'fed_funds_rate': 'FEDFUNDS',
        'cpi': 'CPIAUCSL',
        'treasury_10y': 'DGS10',
        'treasury_2y': 'DGS2',
        'vix': 'VIXC',
        'm2': 'M2SL',
        'sp500': 'SP500',
        'industrial_production': 'INDPRO',
        'consumer_sentiment': 'UMCSENT',
        'housing_starts': 'HOUST',
        'retail_sales': 'RETAIL',
        'balance_of_trade': 'BOPGSTB'
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED data fetcher.
        
        Args:
            api_key: FRED API key (optional, uses FRED_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get('FRED_API_KEY')
        self.cache: Dict[str, pd.DataFrame] = {}
        
    def fetch_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: str = 'm'
    ) -> pd.DataFrame:
        """
        Fetch a FRED series.
        
        Args:
            series_id: FRED series identifier
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            frequency: Data frequency (d, w, m, q, a)
            
        Returns:
            DataFrame with observations
        """
        if not self.api_key:
            logger.warning("FRED API key not set. Using cached/sample data.")
            return self._get_sample_data(series_id)
            
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json',
            'frequency': frequency
        }
        
        if start_date:
            params['observation_start'] = start_date
        if end_date:
            params['observation_end'] = end_date
            
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            observations = data['observations']
            
            df = pd.DataFrame(observations)
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.set_index('date').drop('realtime_start', axis=1)
            
            return df
            
        except requests.RequestException as e:
            logger.error(f"Error fetching FRED data for {series_id}: {e}")
            return self._get_sample_data(series_id)
    
    def fetch_multiple(
        self,
        series_ids: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch multiple FRED series and merge.
        
        Args:
            series_ids: List of FRED series identifiers
            start_date: Start date
            end_date: End date
            
        Returns:
            Merged DataFrame
        """
        dfs = []
        
        for series_id in series_ids:
            df = self.fetch_series(series_id, start_date, end_date)
            if not df.empty:
                df = df.rename(columns={'value': series_id})
                dfs.append(df)
                
        if not dfs:
            return pd.DataFrame()
            
        # Merge all series
        result = dfs[0]
        for df in dfs[1:]:
            result = result.join(df, how='outer')
            
        return result
    
    def _get_sample_data(self, series_id: str) -> pd.DataFrame:
        """
        Generate sample data when API is unavailable.
        
        Args:
            series_id: Series identifier
            
        Returns:
            Sample DataFrame
        """
        logger.info(f"Generating sample data for {series_id}")
        
        # Generate realistic sample data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='M')
        
        np.random.seed(hash(series_id) % 10000)
        
        if series_id == 'VIXC':
            values = np.random.exponential(20, len(dates))
        elif series_id in ['GDP', 'M2SL']:
            base = 1000 if series_id == 'GDP' else 5000
            values = base + np.cumsum(np.random.randn(len(dates)) * 50)
        else:
            values = np.random.randn(len(dates)) * 10 + 50
            
        return pd.DataFrame({'value': values}, index=dates)


class MarketIndexFetcher:
    """
    Fetches market index and ETF data.
    
    Useful for:
    - Market direction (SPY, QQQ, IWM)
    - Sector rotation (XLF, XLK, XLE, etc.)
    - Bond markets (TLT, AGG)
    - Volatility (VXX, UVXY)
    """
    
    INDEX_SYMBOLS = {
        'sp500': 'SPY',
        'nasdaq': 'QQQ',
        'russell2000': 'IWM',
        'dow': 'DIA',
        'treasury_20y': 'TLT',
        'treasury_7_10': 'IEF',
        'bond_aggregate': 'AGG',
        'gold': 'GLD',
        'oil': 'USO',
        'dollar': 'UUP',
        'volatility': 'VXX'
    }
    
    SECTOR_ETFS = {
        'technology': 'XLK',
        'financial': 'XLF',
        'healthcare': 'XLV',
        'energy': 'XLE',
        'consumer_discretionary': 'XLY',
        'consumer_staples': 'XLP',
        'industrial': 'XLI',
        'materials': 'XLB',
        'utilities': 'XLU',
        'real_estate': 'XLRE',
        'communication': 'XLC'
    }
    
    def __init__(self):
        """Initialize market index fetcher."""
        self.cache: Dict[str, pd.DataFrame] = {}
        
    def fetch_index(
        self,
        symbol: str,
        years: int = 5
    ) -> pd.DataFrame:
        """
        Fetch index/ETF data.
        
        Args:
            symbol: Ticker symbol
            years: Number of years of data
            
        Returns:
            DataFrame with OHLCV data
        """
        if symbol in self.cache:
            return self.cache[symbol]
            
        try:
            import yfinance as yf
            
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365)
            
            df = ticker.history(start=start_date, end=end_date)
            
            if not df.empty:
                df.columns = [col.lower() for col in df.columns]
                self.cache[symbol] = df
                
            return df
            
        except Exception as e:
            logger.error(f"Error fetching index {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_sector_returns(
        self,
        years: int = 1
    ) -> pd.DataFrame:
        """
        Fetch sector ETF returns.
        
        Args:
            years: Number of years
            
        Returns:
            DataFrame with daily returns for each sector
        """
        returns_dict = {}
        
        for sector, symbol in self.SECTOR_ETFS.items():
            df = self.fetch_index(symbol, years)
            if not df.empty:
                returns_dict[sector] = df['close'].pct_change()
                
        if not returns_dict:
            return pd.DataFrame()
            
        return pd.DataFrame(returns_dict)
    
    def calculate_market_breadth(
        self,
        symbols: List[str],
        years: int = 1
    ) -> pd.DataFrame:
        """
        Calculate market breadth indicators.
        
        Args:
            symbols: List of stock symbols
            years: Years of data
            
        Returns:
            DataFrame with breadth indicators
        """
        import yfinance as yf
        
        prices = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=datetime.now() - timedelta(days=years*365))
                if not df.empty:
                    prices[symbol] = df['Close']
            except Exception:
                continue
                
        if not prices:
            return pd.DataFrame()
            
        price_df = pd.DataFrame(prices)
        
        # Calculate breadth metrics
        above_ma50 = (price_df > price_df.rolling(50).mean()).sum(axis=1)
        above_ma200 = (price_df > price_df.rolling(200).mean()).sum(axis=1)
        
        return pd.DataFrame({
            'above_ma50': above_ma50,
            'above_ma200': above_ma200,
            'total_stocks': len(symbols),
            'breadth_50': above_ma50 / len(symbols),
            'breadth_200': above_ma200 / len(symbols)
        })


class AlternativeDataIntegrator:
    """
    Integrates all alternative data sources into a unified pipeline.
    """
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        
        # Initialize data fetchers
        self.fred = FredDataFetcher()
        self.market = MarketIndexFetcher()
        
    def get_macro_features(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get macroeconomic features.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with macro features
        """
        series_ids = [
            'GDP', 'UNRATE', 'FEDFUNDS', 'CPIAUCSL',
            'DGS10', 'VIXC', 'M2SL', 'UMCSENT'
        ]
        
        df = self.fred.fetch_multiple(series_ids, start_date, end_date)
        
        # Calculate derived features
        if not df.empty:
            # Real interest rate
            if 'DGS10' in df.columns and 'CPIAUCSL' in df.columns:
                df['real_rate_10y'] = df['DGS10'] - df['CPIAUCSL'].pct_change(12) * 100
                
            # Yield curve
            if 'DGS10' in df.columns and 'FEDFUNDS' in df.columns:
                df['yield_curve'] = df['DGS10'] - df['FEDFUNDS']
                
        return df
    
    def get_market_features(
        self,
        symbols: Optional[List[str]] = None,
        years: int = 1
    ) -> pd.DataFrame:
        """
        Get market features from indices and ETFs.
        
        Args:
            symbols: List of symbols (uses defaults if None)
            years: Years of data
            
        Returns:
            DataFrame with market features
        """
        if symbols is None:
            symbols = list(MarketIndexFetcher.INDEX_SYMBOLS.values())
            
        features = {}
        
        for symbol in symbols:
            df = self.market.fetch_index(symbol, years)
            if not df.empty:
                returns = df['close'].pct_change()
                volatility = returns.rolling(20).std()
                
                features[f'{symbol}_return'] = returns
                features[f'{symbol}_volatility'] = volatility
                
        return pd.DataFrame(features)
    
    def get_sector_rotation_signals(
        self,
        years: int = 1
    ) -> pd.DataFrame:
        """
        Get sector rotation signals.
        
        Args:
            years: Years of data
            
        Returns:
            DataFrame with sector returns and relative strength
        """
        returns = self.market.fetch_sector_returns(years)
        
        if returns.empty:
            return pd.DataFrame()
            
        # Calculate relative strength vs SPY
        spy = self.market.fetch_index('SPY', years)
        if spy.empty:
            return returns.to_frame('sector_return')
            
        spy_returns = spy['close'].pct_change()
        
        relative_strength = returns.sub(spy_returns, axis=0)
        
        # Get top and bottom sectors
        result = pd.DataFrame({
            'top_sector': relative_strength.idxmax(axis=1),
            'bottom_sector': relative_strength.idxmin(axis=1),
            'sector_momentum': relative_strength.mean()
        })
        
        return result
    
    def integrate_all(
        self,
        stock_data: pd.DataFrame,
        stock_symbol: str,
        years: int = 2
    ) -> pd.DataFrame:
        """
        Integrate all alternative data with stock data.
        
        Args:
            stock_data: Stock OHLCV DataFrame
            stock_symbol: Stock ticker
            years: Years of alternative data
            
        Returns:
            Stock DataFrame with integrated features
        """
        df = stock_data.copy()
        
        # Get macro features
        macro = self.get_macro_features(
            start_date=df.index.min().strftime('%Y-%m-%d'),
            end_date=df.index.max().strftime('%Y-%m-%d')
        )
        
        if not macro.empty:
            # Resample to daily and forward fill
            macro_daily = macro.resample('D').last().ffill()
            df = df.join(macro_daily, how='left')
            
        # Get market features
        market = self.get_market_features(years=years)
        
        if not market.empty:
            market_daily = market.resample('D').last().ffill()
            df = df.join(market_daily, how='left')
            
        # Fill missing values
        df = df.ffill().bfill()
        
        logger.info(f"Integrated alternative data: {len(df.columns)} total features")
        
        return df
