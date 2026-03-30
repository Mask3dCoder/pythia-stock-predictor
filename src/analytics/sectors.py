"""
Sector Analysis Module

Provides sector and industry analysis including:
- Sector performance comparison
- Industry group analysis
- Relative strength analysis
- Sector rotation insights
- GICS sector classification
"""

import logging
from typing import Optional, Dict, List
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import numpy as np

from src.core.exceptions import DataCollectionError

logger = logging.getLogger(__name__)

SECTOR_ETFS = {
    'Technology': 'XLK',
    'Healthcare': 'XLV',
    'Financials': 'XLF',
    'Consumer Discretionary': 'XLY',
    'Consumer Staples': 'XLP',
    'Energy': 'XLE',
    'Industrials': 'XLI',
    'Materials': 'XLB',
    'Real Estate': 'XLRE',
    'Utilities': 'XLU',
    'Communication Services': 'XLC',
}

SECTOR_SYMBOLS = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AVGO', 'ORCL', 'IBM', 'ADBE', 'CRM', 'INTC'],
    'Healthcare': ['UNH', 'JNJ', 'PFE', 'MRK', 'LLY', 'ABBV', 'BMY', 'AMGN', 'GILD', 'VRTX'],
    'Financials': ['JPM', 'V', 'MA', 'BLK', 'GS', 'MS', 'BAC', 'WFC', 'C', 'AXP'],
    'Consumer Discretionary': ['AMZN', 'META', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TGT', 'TJX'],
    'Consumer Staples': ['PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO', 'CL', 'KMB', 'GIS'],
    'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'MPC', 'PSX', 'VLO', 'SLB', 'HAL', 'OXY'],
    'Industrials': ['CAT', 'BA', 'DE', 'UNP', 'HON', 'GE', 'MMM', 'RTX', 'UPS', 'FDX'],
    'Materials': ['LIN', 'APD', 'ECL', 'SHW', 'NEM', 'FCX', 'NUE', 'DOW', 'DD', 'PPG'],
    'Real Estate': ['PLD', 'AMT', 'EQIX', 'SPG', 'CCI', 'PSA', 'O', 'WELL', 'DLR', 'AVB'],
    'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'XEL', 'ED', 'WEC', 'AWK'],
    'Communication Services': ['GOOGL', 'META', 'DIS', 'NFLX', 'CMCSA', 'T', 'VZ', 'TMUS', 'CHTR', 'EA'],
}


class SectorAnalyzer:
    """Analyzes sectors and industries."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    def get_sector_performance(self, period: str = "1mo") -> Dict:
        """
        Get performance of all sectors.

        Args:
            period: Time period (1mo, 3mo, 6mo, 1y, ytd)

        Returns:
            Dictionary with sector performance data
        """
        results = {}

        for sector, etf in SECTOR_ETFS.items():
            try:
                ticker = yf.Ticker(etf)
                hist = ticker.history(period=period)

                if not hist.empty:
                    start_price = hist['Close'].iloc[0]
                    end_price = hist['Close'].iloc[-1]
                    change = ((end_price - start_price) / start_price) * 100

                    results[sector] = {
                        'etf': etf,
                        'start_price': start_price,
                        'end_price': end_price,
                        'change_pct': change,
                        'volume': hist['Volume'].mean(),
                    }
            except Exception as e:
                logger.warning(f"Error fetching {sector}: {e}")

        sorted_sectors = sorted(results.items(), key=lambda x: x[1].get('change_pct', 0), reverse=True)

        return {
            'sectors': results,
            'top_performers': [s[0] for s in sorted_sectors[:3]],
            'bottom_performers': [s[0] for s in sorted_sectors[-3:]],
            'period': period,
        }

    def get_sector_momentum(self, periods: List[str] = None) -> Dict:
        """
        Get momentum across multiple timeframes.

        Args:
            periods: List of periods to analyze

        Returns:
            Dictionary with momentum data
        """
        if periods is None:
            periods = ['5d', '1mo', '3mo', '6mo', '1y']

        momentum = {}

        for sector, etf in SECTOR_ETFS.items():
            sector_momentum = {}
            for period in periods:
                try:
                    ticker = yf.Ticker(etf)
                    hist = ticker.history(period=period)

                    if not hist.empty:
                        start_price = hist['Close'].iloc[0]
                        end_price = hist['Close'].iloc[-1]
                        change = ((end_price - start_price) / start_price) * 100
                        sector_momentum[period] = change
                except Exception:
                    pass

            momentum[sector] = sector_momentum

        return momentum

    def get_sector_comparison(self, symbol: str) -> Dict:
        """
        Compare a stock's performance to its sector.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with comparison data
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            sector = info.get('sector')
            if not sector:
                return {'error': 'Sector not found'}

            etf = SECTOR_ETFS.get(sector)
            if not etf:
                return {'error': 'No ETF found for sector'}

            periods = ['1mo', '3mo', '6mo', '1y']
            stock_perf = {}
            sector_perf = {}

            stock_ticker = yf.Ticker(symbol)
            sector_ticker = yf.Ticker(etf)

            for period in periods:
                stock_hist = stock_ticker.history(period=period)
                sector_hist = sector_ticker.history(period=period)

                if not stock_hist.empty and not sector_hist.empty:
                    stock_change = ((stock_hist['Close'].iloc[-1] - stock_hist['Close'].iloc[0]) / stock_hist['Close'].iloc[0]) * 100
                    sector_change = ((sector_hist['Close'].iloc[-1] - sector_hist['Close'].iloc[0]) / sector_hist['Close'].iloc[0]) * 100

                    stock_perf[period] = stock_change
                    sector_perf[period] = sector_change

            relative_strength = {}
            for period in stock_perf:
                relative_strength[period] = stock_perf[period] - sector_perf.get(period, 0)

            return {
                'symbol': symbol,
                'sector': sector,
                'stock_performance': stock_perf,
                'sector_performance': sector_perf,
                'relative_strength': relative_strength,
                'outperforming': sum(1 for rs in relative_strength.values() if rs > 0),
            }

        except Exception as e:
            logger.error(f"Error in sector comparison: {e}")
            return {'error': str(e)}

    def get_top_sector_stocks(self, sector: str, limit: int = 10) -> List[Dict]:
        """
        Get top performing stocks in a sector.

        Args:
            sector: Sector name
            limit: Number of stocks to return

        Returns:
            List of top performing stocks
        """
        symbols = SECTOR_SYMBOLS.get(sector, [])

        if not symbols:
            return []

        stocks = []

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='1mo')

                if not hist.empty:
                    info = ticker.info
                    start_price = hist['Close'].iloc[0]
                    end_price = hist['Close'].iloc[-1]
                    change = ((end_price - start_price) / start_price) * 100

                    stocks.append({
                        'symbol': symbol,
                        'name': info.get('shortName', symbol),
                        'price': end_price,
                        'change_pct': change,
                        'volume': hist['Volume'].mean(),
                        'market_cap': info.get('marketCap'),
                    })
            except Exception:
                pass

        return sorted(stocks, key=lambda x: x.get('change_pct', 0), reverse=True)[:limit]

    def get_sector_correlation(self) -> Dict:
        """Get correlation between sectors."""
        prices = {}

        for sector, etf in SECTOR_ETFS.items():
            try:
                ticker = yf.Ticker(etf)
                hist = ticker.history(period='1y')
                if not hist.empty:
                    prices[sector] = hist['Close']
            except Exception:
                pass

        if not prices:
            return {'error': 'Could not fetch sector data'}

        df = pd.DataFrame(prices)
        corr = df.corr()

        return {
            'correlation': corr.to_dict(),
            'most_similar': self._find_most_similar(corr),
            'least_similar': self._find_least_similar(corr),
        }

    def _find_most_similar(self, corr: pd.DataFrame) -> List[tuple]:
        """Find most correlated sector pairs."""
        pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

        return sorted(pairs, key=lambda x: x[2], reverse=True)[:3]

    def _find_least_similar(self, corr: pd.DataFrame) -> List[tuple]:
        """Find least correlated sector pairs."""
        pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

        return sorted(pairs, key=lambda x: x[2])[:3]

    def get_sector_rotation_signal(self) -> Dict:
        """Generate sector rotation signal based on relative strength."""
        momentum = self.get_sector_momentum(['1m', '3m', '6m'])

        scores = {}
        for sector, periods in momentum.items():
            score = 0
            if periods.get('1m', 0) > 0:
                score += 1
            if periods.get('3m', 0) > periods.get('1m', 0):
                score += 1
            if periods.get('6m', 0) > 0:
                score += 1

            weights = {'1m': 0.2, '3m': 0.3, '6m': 0.5}
            weighted_score = sum(periods.get(p, 0) * weights[p] for p in weights)

            scores[sector] = {
                'score': score,
                'weighted_momentum': weighted_score,
            }

        sorted_sectors = sorted(scores.items(), key=lambda x: x[1]['weighted_momentum'], reverse=True)

        return {
            'leader': sorted_sectors[0][0] if sorted_sectors else None,
            'laggard': sorted_sectors[-1][0] if sorted_sectors else None,
            'rankings': [s[0] for s in sorted_sectors],
            'scores': scores,
            'signal': 'risk_on' if scores.get(sorted_sectors[0][0], {}).get('weighted_momentum', 0) > 0 else 'risk_off',
        }


def get_sector_performance(period: str = "1mo") -> Dict:
    """Convenience function to get sector performance."""
    analyzer = SectorAnalyzer()
    return analyzer.get_sector_performance(period)


def compare_to_sector(symbol: str) -> Dict:
    """Convenience function to compare stock to sector."""
    analyzer = SectorAnalyzer()
    return analyzer.get_sector_comparison(symbol)
