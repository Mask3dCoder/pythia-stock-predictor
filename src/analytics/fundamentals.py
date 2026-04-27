"""
Fundamental Analysis Module

Provides comprehensive fundamental analysis of stocks including:
- Financial ratios (P/E, P/B, EV/EBITDA, etc.)
- Income statement metrics
- Balance sheet metrics
- Cash flow metrics
- Growth rates
- Dividend analysis
- Piotroski F-Score
- Altman Z-Score
"""

import logging
from typing import Optional, Dict, List
from datetime import datetime

import yfinance as yf
import pandas as pd
import numpy as np

from src.core.exceptions import DataCollectionError

logger = logging.getLogger(__name__)


SECTOR_INDUSTRY_MAP = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AVGO', 'ORCL', 'IBM', 'ADBE', 'CRM', 'INTC', 'AMD', 'QCOM', 'TXN', 'AMAT', 'MU', 'ADI', 'LRCX', 'KLAC', 'SNPS', 'CDNS'],
    'Healthcare': ['UNH', 'JNJ', 'PFE', 'MRK', 'LLY', 'ABBV', 'BMY', 'AMGN', 'GILD', 'VRTX', 'REGN', 'ISRG', 'MDT', 'SYK', 'BSX', 'ZTS', 'CI', 'CVS', 'HUM', 'MCK'],
    'Financials': ['JPM', 'V', 'MA', 'BLK', 'GS', 'MS', 'BAC', 'WFC', 'C', 'AXP', 'SCHW', 'USB', 'PNC', 'TFC', 'COF', 'AIG', 'MET', 'PRU', 'AFL', 'TRV'],
    'Consumer Discretionary': ['AMZN', 'META', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TGT', 'TJX', 'BKNG', 'CMG', 'ORLY', 'AZO', 'ROST', 'BBY', 'DLTR', 'DRI', 'YUM', 'DPZ'],
    'Consumer Staples': ['PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO', 'CL', 'KMB', 'GIS', 'K', 'HSY', 'MDLZ', 'KHC', 'STZ', 'DG', 'OLN', 'LW', 'CAG', 'SYY'],
    'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'MPC', 'PSX', 'VLO', 'SLB', 'HAL', 'OXY', 'PXD', 'FANG', 'DVN', 'HES', 'KMI', 'WMB', 'OKE', 'TRGP', 'LNG', 'COG'],
    'Industrials': ['CAT', 'BA', 'DE', 'UNP', 'HON', 'GE', 'MMM', 'RTX', 'UPS', 'FDX', 'LMT', 'GD', 'NOC', 'ITW', 'EMR', 'ETN', 'PH', 'ROK', 'CMI', 'FAST'],
    'Materials': ['LIN', 'APD', 'ECL', 'SHW', 'NEM', 'FCX', 'NUE', 'DOW', 'DD', 'PPG', 'ECL', 'VMC', 'MLM', 'AME', 'RS', 'IFF', 'ALB', 'FMC', 'CE', 'MTCH'],
    'Real Estate': ['PLD', 'AMT', 'EQIX', 'SPG', 'CCI', 'PSA', 'O', 'WELL', 'DLR', 'AVB', 'EQR', 'VTR', 'MAA', 'ESS', 'KIM', 'UDR', 'REG', 'FRT', 'SLG', 'BXP'],
    'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'XEL', 'ED', 'WEC', 'AWK', 'CMS', 'AEE', 'NI', 'ES', 'PEG', 'DTE', 'PPL', 'ETR', 'KE', 'LNT'],
    'Communication Services': ['GOOGL', 'META', 'DIS', 'NFLX', 'CMCSA', 'T', 'VZ', 'TMUS', 'CHTR', 'EA', 'TTWO', 'ATVI', 'NWSA', 'OMC', 'IPG', 'DISCA', 'VIAC', 'LUMN', 'FOX', 'NWSA'],
}


class FundamentalAnalyzer:
    """Provides comprehensive fundamental analysis."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    def get_full_analysis(self, symbol: str) -> Dict:
        """
        Get complete fundamental analysis for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with comprehensive analysis
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                'ratios': self._analyze_ratios(info),
                'income': self._analyze_income(info),
                'balance': self._analyze_balance(info),
                'cashflow': self._analyze_cashflow(info),
                'growth': self._analyze_growth(info),
                'dividends': self._analyze_dividends(info),
                'scores': self._calculate_scores(symbol, info),
                'recommendation': self._get_recommendation(info),
            }

        except Exception as e:
            logger.error(f"Error analyzing fundamentals for {symbol}: {e}")
            return {'error': str(e)}

    def _analyze_ratios(self, info: Dict) -> Dict:
        """Analyze key financial ratios."""
        return {
            'pe_ratio': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'peg_ratio': info.get('pegRatio'),
            'price_to_book': info.get('priceToBook'),
            'price_to_sales': info.get('priceToSalesTrailing12Months'),
            'enterprise_to_revenue': info.get('enterpriseToRevenue'),
            'enterprise_to_ebitda': info.get('enterpriseToEbitda'),
            'profit_margin': info.get('profitMargins'),
            'operating_margin': info.get('operatingMargins'),
            'ebitda_margin': info.get('ebitdaMargins'),
            'return_on_assets': info.get('returnOnAssets'),
            'return_on_equity': info.get('returnOnEquity'),
            'debt_to_equity': info.get('debtToEquity'),
            'current_ratio': info.get('currentRatio'),
            'quick_ratio': info.get('quickRatio'),
            'interest_coverage': info.get('interestCoverage'),
        }

    def _analyze_income(self, info: Dict) -> Dict:
        """Analyze income statement metrics."""
        return {
            'revenue': info.get('totalRevenue'),
            'revenue_per_share': info.get('revenuePerShare'),
            'quarterly_revenue_growth': info.get('quarterlyRevenueGrowth'),
            'gross_profit': info.get('grossProfit'),
            'gross_margin': info.get('grossMargins'),
            'ebitda': info.get('ebitda'),
            'net_income': info.get('netIncomeToCommon'),
            'earnings_per_share': info.get('trailingEps'),
            'forward_eps': info.get('forwardEps'),
            'quarterly_earnings_growth': info.get('quarterlyEarningsGrowth'),
        }

    def _analyze_balance(self, info: Dict) -> Dict:
        """Analyze balance sheet metrics."""
        return {
            'total_cash': info.get('totalCash'),
            'cash_per_share': info.get('cashPerShare'),
            'total_debt': info.get('totalDebt'),
            'totalAssets': info.get('totalAssets'),
            'intangible_assets': info.get('intangibleAssets'),
            'treasury_stock': info.get('treasuryStock'),
            'retained_earnings': info.get('retainedEarnings'),
            'shareholder_equity': info.get('stockholdersEquity'),
            'book_value': info.get('bookValue'),
            'operating_cashflow': info.get('operatingCashflow'),
            'free_cashflow': info.get('freeCashflow'),
        }

    def _analyze_cashflow(self, info: Dict) -> Dict:
        """Analyze cash flow metrics."""
        return {
            'operating_cashflow': info.get('operatingCashflow'),
            'free_cashflow': info.get('freeCashflow'),
            'cashflow_from_operations': info.get('cashflowFromOperations'),
            'cashflow_from_investing': info.get('cashflowFromInvesting'),
            'cashflow_from_financing': info.get('cashflowFromFinancing'),
            'capex': info.get('capitalExpenditures'),
        }

    def _analyze_growth(self, info: Dict) -> Dict:
        """Analyze growth metrics."""
        return {
            'revenue_growth': info.get('revenueGrowth'),
            'earnings_growth': info.get('earningsGrowth'),
            'earnings_quarterly_growth': info.get('quarterlyEarningsGrowth'),
            'revenue_quarterly_growth': info.get('quarterlyRevenueGrowth'),
            'eps_growth': info.get('EPSGrowth'),
            'employees_growth': info.get('lastAnnualRevenue'),
        }

    def _analyze_dividends(self, info: Dict) -> Dict:
        """Analyze dividend metrics."""
        dividend_yield = info.get('dividendYield')
        if dividend_yield:
            dividend_yield = dividend_yield * 100

        return {
            'dividend_yield': dividend_yield,
            'dividend_rate': info.get('dividendRate'),
            'dividend_per_share': info.get('dividendPerShare'),
            'payout_ratio': info.get('payoutRatio'),
            'ex_dividend_date': info.get('exDividendDate'),
            'dividend_date': info.get('dividendDate'),
            'trailing_annual_dividend': info.get('trailingAnnualDividend'),
            'trailing_annual_dividend_yield': info.get('trailingAnnualDividendYield'),
        }

    def _calculate_scores(self, symbol: str, info: Dict) -> Dict:
        """Calculate scoring metrics."""
        scores = {}

        piotroski = 0

        if info.get('returnOnEquity', 0) > 0:
            piotroski += 1
        if info.get('returnOnAssets', 0) > 0:
            piotroski += 1
        if info.get('operatingCashflow', 0) > 0:
            piotroski += 1
        if info.get('operatingCashflow', 0) > info.get('netIncomeToCommon', 0):
            piotroski += 1

        debt_ratio = info.get('debtToEquity', 0)
        if debt_ratio and debt_ratio < 50:
            piotroski += 1
        elif debt_ratio and debt_ratio < 100:
            piotroski += 0.5

        current_ratio = info.get('currentRatio', 0)
        if current_ratio and current_ratio > 1.5:
            piotroski += 1
        elif current_ratio and current_ratio > 1:
            piotroski += 0.5

        if info.get('grossMargins', 0) > 0:
            piotroski += 1

        scores['piotroski_f_score'] = min(9, int(piotroski))

        altman_z = self._calculate_altman_z(info)
        scores['altman_z_score'] = altman_z

        if altman_z > 2.99:
            scores['altman_zone'] = 'Safe'
        elif altman_z > 1.81:
            scores['altman_zone'] = 'Grey'
        else:
            scores['altman_zone'] = 'Distress'

        return scores

    def _calculate_altman_z(self, info: Dict) -> Optional[float]:
        """Calculate Altman Z-Score."""
        try:
            x1 = (info.get('totalCash', 0) - info.get('totalDebt', 0)) / info.get('totalAssets', 1)
            x2 = info.get('retainedEarnings', 0) / info.get('totalAssets', 1)
            x3 = info.get('ebitda', 0) / info.get('totalAssets', 1)
            x4 = info.get('marketCap', 0) / info.get('totalLiabilities', 1)
            x5 = info.get('totalRevenue', 0) / info.get('totalAssets', 1)

            z_score = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
            return round(z_score, 2)
        except Exception:
            return None

    def _get_recommendation(self, info: Dict) -> Dict:
        """Get investment recommendation based on fundamentals."""
        score = 0
        max_score = 10

        pe = info.get('trailingPE')
        if pe and 10 < pe < 25:
            score += 2
        elif pe and pe < 10:
            score += 1

        if info.get('returnOnEquity', 0) > 0.15:
            score += 2
        elif info.get('returnOnEquity', 0) > 0.10:
            score += 1

        if info.get('debtToEquity', 999) < 50:
            score += 2
        elif info.get('debtToEquity', 999) < 100:
            score += 1

        if info.get('dividendYield', 0) > 0.03:
            score += 1

        if info.get('earningsGrowth', 0) > 0.1:
            score += 1

        if info.get('profitMargins', 0) > 0.2:
            score += 1

        if score >= 7:
            recommendation = 'Strong Buy'
        elif score >= 5:
            recommendation = 'Buy'
        elif score >= 3:
            recommendation = 'Hold'
        elif score >= 1:
            recommendation = 'Sell'
        else:
            recommendation = 'Strong Sell'

        return {
            'score': score,
            'max_score': max_score,
            'rating': recommendation,
        }

    def compare_sectors(self, symbol: str) -> Dict:
        """Compare a stock's metrics to its sector averages."""
        ticker = yf.Ticker(symbol)
        info = ticker.info

        sector = info.get('sector')
        industry = info.get('industry')

        if not sector:
            return {'error': 'Sector information not available'}

        sector_symbols = SECTOR_INDUSTRY_MAP.get(sector, [])

        if not sector_symbols:
            return {'sector': sector, 'industry': industry, 'error': 'No sector comparison available'}

        sector_metrics = []
        for sym in sector_symbols[:10]:
            try:
                s_ticker = yf.Ticker(sym)
                s_info = s_ticker.info
                sector_metrics.append({
                    'symbol': sym,
                    'pe': s_info.get('trailingPE'),
                    'roe': s_info.get('returnOnEquity'),
                    'profit_margin': s_info.get('profitMargins'),
                    'debt_to_equity': s_info.get('debtToEquity'),
                })
            except Exception:
                pass

        if not sector_metrics:
            return {'error': 'Could not fetch sector data'}

        pe_values = [m['pe'] for m in sector_metrics if m['pe']]
        roe_values = [m['roe'] for m in sector_metrics if m['roe']]
        pm_values = [m['profit_margin'] for m in sector_metrics if m['profit_margin']]

        return {
            'sector': sector,
            'industry': industry,
            'symbol_pe': info.get('trailingPE'),
            'sector_avg_pe': np.mean(pe_values) if pe_values else None,
            'symbol_roe': info.get('returnOnEquity'),
            'sector_avg_roe': np.mean(roe_values) if roe_values else None,
            'symbol_profit_margin': info.get('profitMargins'),
            'sector_avg_profit_margin': np.mean(pm_values) if pm_values else None,
        }


def get_fundamentals(symbol: str) -> Dict:
    """Convenience function to get fundamentals."""
    analyzer = FundamentalAnalyzer()
    return analyzer.get_full_analysis(symbol)


def compare_to_sector(symbol: str) -> Dict:
    """Convenience function to compare to sector."""
    analyzer = FundamentalAnalyzer()
    return analyzer.compare_sectors(symbol)
