"""
SEARCH Command

Search for securities by name or symbol.
Usage: FIND <query> or SEARCH <query>
"""

from typing import List
import asyncio
from datetime import datetime

import yfinance as yf
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.tui.commands.base import DataCommandHandler
from src.tui.app import CommandResult, PythiaTerminal


class SearchCommand(DataCommandHandler):
    """Search for securities by name or ticker."""
    
    @property
    def name(self) -> str:
        return "FIND"
    
    @property
    def aliases(self) -> List[str]:
        return ["FIND", "F", "SEARCH", "S"]
    
    @property
    def description(self) -> str:
        return "Search for securities by name or symbol"
    
    @property
    def usage(self) -> str:
        return "FIND <query>  or  SEARCH <query>"
    
    @property
    def requires_symbol(self) -> bool:
        return False
    
    @property
    def requires_data(self) -> bool:
        return False
    
    @property
    def min_args(self) -> int:
        return 1
    
    async def execute(self, terminal: PythiaTerminal, args: List[str]) -> CommandResult:
        query = ' '.join(args).upper()
        return await self._search(terminal, query)
    
    async def _search(self, terminal: PythiaTerminal, query: str) -> CommandResult:
        """Search for securities."""
        # Common stock tickers to search from
        common_tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM',
            'V', 'UNH', 'JNJ', 'WMT', 'PG', 'MA', 'HD', 'CVX', 'LLY', 'ABBV',
            'MRK', 'PFE', 'KO', 'PEP', 'COST', 'AVGO', 'TMO', 'MCD', 'DIS',
            'CSCO', 'ACN', 'ABT', 'DHR', 'NKE', 'ADBE', 'CRM', 'TXN', 'NEE',
            'PM', 'UPS', 'MS', 'QCOM', 'RTX', 'LOW', 'HON', 'INTC', 'ORCL',
            'IBM', 'GS', 'CAT', 'BA', 'GE', 'AMD', 'INTU', 'AMAT', 'SBUX',
            'BLK', 'DE', 'MMM', 'SPGI', 'ADP', 'MDLZ', 'GILD', 'BKNG', 'ISRG',
            'TGT', 'SYK', 'VRTX', 'ZTS', 'REGN', 'PLD', 'MU', 'ADI', 'LRCX',
            'CVS', 'CI', 'CME', 'CB', 'SCHW', 'BDX', 'SO', 'DUK', 'NOC',
            'MO', 'SPG', 'PNC', 'CL', 'ETN', 'ITW', 'APD', 'ICE', 'EOG',
            'WM', 'TFC', 'AMT', 'AON', 'FI', 'EMR', 'SHW', 'GM', 'F',
            'AIG', 'MMC', 'FCX', 'USB', 'T', 'VZ', 'AXP', 'CSX', 'NSC',
            'HUM', 'BSX', 'EL', 'MCO', 'SNPS', 'CDNS', 'PANW', 'NOW', 'DDOG',
            'NET', 'TEAM', 'SQ', 'SHOP', 'COIN', 'UBER', 'ABNB', 'RBLX',
            'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD',
            'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X',
            '^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX',
            'SPY', 'QQQ', 'IWM', 'DIA', 'TLT', 'GLD', 'SLV', 'UNG',
        ]
        
        results = []
        query_upper = query.upper()
        
        # Check for exact or partial match
        for ticker in common_tickers:
            if query_upper in ticker:
                results.append(ticker)
                if len(results) >= 20:
                    break
        
        # Also try to get info on the query itself if it looks like a ticker
        if query_upper not in results and len(query_upper) <= 5:
            try:
                ticker = yf.Ticker(query_upper)
                info = ticker.info
                if info.get('shortName') or info.get('longName'):
                    results.insert(0, query_upper)
            except Exception:
                pass
        
        if not results:
            return CommandResult(
                False,
                f"No results found for '{query}'. Try: AAPL, MSFT, SPY, BTC-USD, etc."
            )
        
        # Fetch details for top results
        table = Table(title=f"🔍 Search Results for '{query}'")
        table.add_column("Symbol", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Type", style="yellow")
        table.add_column("Price", style="magenta", justify="right")
        
        for symbol in results[:10]:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                name = info.get('shortName') or info.get('longName') or symbol
                price = info.get('currentPrice') or info.get('previousClose')
                
                # Determine type
                if '-' in symbol or symbol.startswith('^'):
                    s_type = "Index/Crypto"
                elif len(symbol) == 3 and symbol.endswith('=X'):
                    s_type = "Forex"
                else:
                    s_type = info.get('quoteType', 'Stock')
                
                price_str = f"${price:.2f}" if price else "N/A"
                
                table.add_row(symbol, name[:40], s_type, price_str)
                
            except Exception:
                table.add_row(symbol, "N/A", "Unknown", "N/A")
        
        panel = Panel(
            f"[dim]Found {len(results)} matches. Showing top 10.[/dim]",
            title="[bold cyan]Search Results[/bold cyan]"
        )
        
        return CommandResult(success=True, message="", panel=panel, table=table)
