"""
ANALYZE Command

AI-powered analysis of a symbol.
Usage: AI <symbol> or ANALYZE <symbol>
"""

import asyncio
from typing import List
from datetime import datetime

import yfinance as yf
import pandas as pd
import numpy as np
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from src.tui.commands.base import AnalyticsCommandHandler
from src.tui.app import CommandResult, PythiaTerminal, CommandCategory


class AnalyzeCommand(AnalyticsCommandHandler):
    """AI-powered analysis of a stock."""
    
    @property
    def name(self) -> str:
        return "ANALYZE"
    
    @property
    def aliases(self) -> List[str]:
        return ["AI", "ANALYZE", "ANALYSIS"]
    
    @property
    def description(self) -> str:
        return "AI-powered analysis of a stock"
    
    @property
    def usage(self) -> str:
        return "AI <symbol>  or  ANALYZE <symbol>"
    
    @property
    def min_args(self) -> int:
        return 1
    
    async def execute(self, terminal: PythiaTerminal, args: List[str]) -> CommandResult:
        symbol = args[0].upper()
        
        try:
            # Fetch data
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1y")
            
            if hist.empty:
                return CommandResult(False, f"No data found for {symbol}")
            
            # Calculate various metrics
            current_price = hist['Close'].iloc[-1]
            prices = hist['Close']
            
            # Technical signals
            signals = self._analyze_technical(prices)
            
            # Fundamental score
            fundamental = self._analyze_fundamental(info)
            
            # Sentiment
            sentiment = self._analyze_sentiment(info)
            
            # Overall score
            overall = (signals['technical'] + fundamental['score'] + sentiment['score']) / 3
            
            # Create analysis table
            table = Table(show_header=False)
            table.add_column("Analysis", style="cyan")
            table.add_column("Value", style="green")
            
            # Technical
            table.add_row("Trend", signals['trend'])
            table.add_row("Momentum", signals['momentum'])
            table.add_row("Volatility", signals['volatility'])
            
            # Fundamental
            table.add_row("P/E Ratio", f"{fundamental['pe_ratio']}")
            table.add_row("Profit Margin", f"{fundamental['profit_margin']}%")
            table.add_row("Growth", f"{fundamental['growth']}%")
            
            # Sentiment
            table.add_row("Analyst Rating", sentiment['rating'])
            table.add_row("News Sentiment", sentiment['news'])
            
            # Overall
            if overall >= 70:
                overall_str = "[bold green]BULLISH[/bold green]"
            elif overall >= 50:
                overall_str = "[bold yellow]NEUTRAL[/bold yellow]"
            else:
                overall_str = "[bold red]BEARISH[/bold red]"
            
            table.add_row("Overall", overall_str)
            
            # Summary
            summary = self._generate_summary(symbol, current_price, signals, fundamental, sentiment, overall)
            
            panel = Panel(
                f"[bold cyan]{summary}[/bold cyan]\n\n{table}",
                title=f"🤖 {symbol} AI Analysis",
                subtitle=f"Price: ${current_price:.2f} | Score: {overall:.0f}/100"
            )
            
            return CommandResult(
                success=True,
                message="",
                panel=panel
            )
            
        except Exception as e:
            return CommandResult(False, f"Error analyzing: {str(e)}")
    
    def _analyze_technical(self, prices: pd.Series) -> dict:
        """Analyze technical indicators."""
        # Trend
        sma_20 = prices.rolling(20).mean().iloc[-1]
        sma_50 = prices.rolling(50).mean().iloc[-1]
        
        if sma_20 > sma_50:
            trend = "🟢 Uptrend"
        elif sma_20 < sma_50:
            trend = "🔴 Downtrend"
        else:
            trend = "🟡 Sideways"
        
        # Momentum (RSI-like)
        returns = prices.pct_change()
        momentum = returns.tail(14).mean() * 100
        
        if momentum > 0.5:
            momentum_str = "🟢 Strong Positive"
        elif momentum > 0:
            momentum_str = "🟡 Weak Positive"
        elif momentum > -0.5:
            momentum_str = "🟡 Weak Negative"
        else:
            momentum_str = "🔴 Strong Negative"
        
        # Volatility
        vol = returns.std() * np.sqrt(252) * 100
        
        if vol < 15:
            volatility = "🟢 Low"
        elif vol < 30:
            volatility = "🟡 Medium"
        else:
            volatility = "🔴 High"
        
        # Technical score
        technical_score = 50
        if "Uptrend" in trend:
            technical_score += 20
        if "Positive" in momentum_str:
            technical_score += 15
        if "Low" in volatility:
            technical_score += 15
        
        return {
            'trend': trend,
            'momentum': momentum_str,
            'volatility': volatility,
            'score': min(100, technical_score)
        }
    
    def _analyze_fundamental(self, info: dict) -> dict:
        """Analyze fundamental data."""
        pe_ratio = info.get('trailingPE', 0)
        if pe_ratio:
            if pe_ratio < 15:
                pe_ratio_str = f"{pe_ratio:.1f} (🟢 Cheap)"
            elif pe_ratio < 25:
                pe_ratio_str = f"{pe_ratio:.1f} (🟡 Fair)"
            else:
                pe_ratio_str = f"{pe_ratio:.1f} (🔴 Expensive)"
        else:
            pe_ratio_str = "N/A"
        
        profit_margin = info.get('profitMargins', 0)
        if profit_margin:
            profit_margin_str = f"{profit_margin * 100:.1f}%"
        else:
            profit_margin_str = "N/A"
        
        growth = info.get('earningsGrowth', 0)
        if growth:
            growth_str = f"{growth * 100:.1f}%"
        else:
            growth_str = "N/A"
        
        # Fundamental score
        score = 50
        if profit_margin and profit_margin > 0.2:
            score += 20
        if growth and growth > 0.1:
            score += 20
        if pe_ratio and pe_ratio < 20:
            score += 10
        
        return {
            'pe_ratio': pe_ratio_str,
            'profit_margin': profit_margin_str,
            'growth': growth_str,
            'score': min(100, score)
        }
    
    def _analyze_sentiment(self, info: dict) -> dict:
        """Analyze sentiment data."""
        # Get analyst recommendations
        target_mean = info.get('targetMeanPrice', 0)
        current = info.get('currentPrice', 0)
        
        if target_mean and current:
            upside = (target_mean - current) / current * 100
            if upside > 20:
                rating = "🟢 Strong Buy"
            elif upside > 10:
                rating = "🟢 Buy"
            elif upside > -10:
                rating = "🟡 Hold"
            elif upside > -20:
                rating = "🔴 Sell"
            else:
                rating = "🔴 Strong Sell"
        else:
            rating = "N/A"
        
        # News sentiment (placeholder)
        news = "🟡 Neutral"
        
        # Sentiment score
        score = 50
        if "Buy" in rating:
            score += 30
        elif "Hold" in rating:
            score += 10
        
        return {
            'rating': rating,
            'news': news,
            'score': min(100, score)
        }
    
    def _generate_summary(self, symbol: str, price: float, signals: dict, 
                         fundamental: dict, sentiment: dict, overall: float) -> str:
        """Generate AI summary."""
        # Determine bias
        if overall >= 70:
            bias = "BULLISH"
            emoji = "📈"
        elif overall >= 50:
            bias = "NEUTRAL"
            emoji = "➡️"
        else:
            bias = "BEARISH"
            emoji = "📉"
        
        summary = f"""
{emoji} {symbol} Analysis Summary

The stock shows a {signals['trend'].replace('🟢 ', '').replace('🔴 ', '').replace('🟡 ', '')} 
technical trend with {signals['momentum'].replace('🟢 ', '').replace('🔴 ', '').replace('🟡 ', '')} momentum.

Fundamentally, it has a P/E ratio of {fundamental['pe_ratio'].split(' ')[0]} 
with {fundamental['profit_margin']} profit margin.

Analyst consensus is {sentiment['rating'].replace('🟢 ', '').replace('🔴 ', '').replace('🟡 ', '')}.

Overall: {bias} ({overall:.0f}/100)
"""
        return summary.strip()
