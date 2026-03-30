"""
NEWS Command

Display latest news for a symbol.
Usage: NW <symbol> or NEWS <symbol>
"""

import asyncio
from typing import List
from datetime import datetime

import yfinance as yf
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
import requests

from src.tui.commands.base import DataCommandHandler
from src.tui.app import CommandResult, PythiaTerminal, CommandCategory


class NewsCommand(DataCommandHandler):
    """Display latest news for a stock."""
    
    @property
    def name(self) -> str:
        return "NEWS"
    
    @property
    def aliases(self) -> List[str]:
        return ["NW", "NEWS", "N"]
    
    @property
    def description(self) -> str:
        return "Display latest news with sentiment"
    
    @property
    def usage(self) -> str:
        return "NW <symbol>  or  NEWS <symbol>"
    
    @property
    def min_args(self) -> int:
        return 1
    
    async def execute(self, terminal: PythiaTerminal, args: List[str]) -> CommandResult:
        symbol = args[0].upper()
        
        try:
            # Get news from Yahoo Finance
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            if not news:
                return CommandResult(False, f"No news found for {symbol}")
            
            # Create news table
            table = Table(title=f"📰 {symbol} Latest News")
            table.add_column("#", style="cyan", width=3)
            table.add_column("Title", style="white")
            table.add_column("Source", style="yellow")
            table.add_column("Time", style="dim")
            
            # Add news items
            for i, item in enumerate(news[:10], 1):
                title = item.get('title', 'No title')
                if len(title) > 60:
                    title = title[:57] + "..."
                
                source = item.get('source', 'Unknown')
                time_str = item.get('providerPublishTime', '')
                if time_str:
                    try:
                        from datetime import datetime
                        dt = datetime.fromtimestamp(time_str)
                        time_str = dt.strftime('%H:%M')
                    except:
                        time_str = ""
                
                table.add_row(str(i), title, source, time_str)
            
            # Analyze sentiment
            sentiment = self._analyze_sentiment(news[:5])
            
            panel = Panel(
                f"[bold]{sentiment}[/bold]\n\n{table}",
                title=f"📰 {symbol} News Feed"
            )
            
            return CommandResult(
                success=True,
                message="",
                panel=panel
            )
            
        except Exception as e:
            return CommandResult(False, f"Error fetching news: {str(e)}")
    
    def _analyze_sentiment(self, news: list) -> str:
        """Quick sentiment analysis."""
        # Simple keyword-based sentiment
        positive_words = ['rise', 'gain', 'surge', 'bullish', 'upgrade', 'beat', 'growth', 'profit']
        negative_words = ['fall', 'drop', 'bearish', 'downgrade', 'miss', 'loss', 'fear']
        
        pos_count = 0
        neg_count = 0
        
        for item in news:
            title = item.get('title', '').lower()
            for word in positive_words:
                if word in title:
                    pos_count += 1
            for word in negative_words:
                if word in title:
                    neg_count += 1
        
        total = pos_count + neg_count
        if total == 0:
            return "🟡 Overall: Neutral sentiment"
        elif pos_count > neg_count:
            pct = pos_count / total * 100
            return f"🟢 Overall: Bullish sentiment ({pct:.0f}% positive)"
        else:
            pct = neg_count / total * 100
            return f"🔴 Overall: Bearish sentiment ({pct:.0f}% negative)"
