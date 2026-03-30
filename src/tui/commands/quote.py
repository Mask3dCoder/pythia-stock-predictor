"""
QUOTE Command

Display quote and basic info for a symbol.
Usage: Q <symbol> or QUOTE <symbol>
"""

import asyncio
from typing import List
from datetime import datetime

import yfinance as yf
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from src.tui.commands.base import DataCommandHandler
from src.tui.app import CommandResult, PythiaTerminal, CommandCategory


class QuoteCommand(DataCommandHandler):
    """Display quote and basic market data."""
    
    @property
    def name(self) -> str:
        return "QUOTE"
    
    @property
    def aliases(self) -> List[str]:
        return ["Q", "QUOTE", "QOTE"]
    
    @property
    def description(self) -> str:
        return "Display quote and basic market information"
    
    @property
    def usage(self) -> str:
        return "Q <symbol>  or  QUOTE <symbol>"
    
    @property
    def min_args(self) -> int:
        return 1
    
    async def execute(self, terminal: PythiaTerminal, args: List[str]) -> CommandResult:
        symbol = args[0].upper()
        
        try:
            # Fetch data
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Create quote table
            table = Table(title=f"📊 {symbol} Quote", show_header=False)
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="green")
            
            # Price info
            current_price = info.get('currentPrice', info.get('previousClose', 'N/A'))
            previous_close = info.get('previousClose', 'N/A')
            open_price = info.get('open', 'N/A')
            day_high = info.get('dayHigh', 'N/A')
            day_low = info.get('dayLow', 'N/A')
            volume = info.get('volume', 'N/A')
            market_cap = info.get('marketCap', 'N/A')
            
            # Calculate change
            if current_price != 'N/A' and previous_close != 'N/A':
                change = current_price - previous_close
                change_pct = (change / previous_close) * 100
                change_str = f"{change:+.2f} ({change_pct:+.2f}%)"
                change_style = "green" if change >= 0 else "red"
            else:
                change_str = "N/A"
                change_style = "white"
            
            # Add rows
            table.add_row("Price", f"${current_price}")
            table.add_row("Change", f"[{change_style}]{change_str}[/{change_style}]")
            table.add_row("Open", f"${open_price}")
            table.add_row("High", f"${day_high}")
            table.add_row("Low", f"${day_low}")
            table.add_row("Volume", f"{volume:,}" if volume != 'N/A' else "N/A")
            
            if market_cap != 'N/A':
                if market_cap > 1e12:
                    market_cap_str = f"${market_cap/1e12:.2f}T"
                elif market_cap > 1e9:
                    market_cap_str = f"${market_cap/1e9:.2f}B"
                else:
                    market_cap_str = f"${market_cap/1e6:.2f}M"
                table.add_row("Market Cap", market_cap_str)
            
            # Additional info
            table.add_row("52W High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
            table.add_row("52W Low", f"${info.get('fiftyTwoWeekLow', 'N/A')}")
            table.add_row("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
            table.add_row("EPS", f"${info.get('trailingEps', 'N/A')}")
            
            # Company info
            company_name = info.get('shortName', info.get('longName', symbol))
            
            # Create panel with company name
            panel = Panel(
                table,
                title=f"[bold cyan]{company_name}[/bold cyan]",
                subtitle=f"Data as of {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            # Update current symbol
            terminal.current_symbol = symbol
            
            return CommandResult(
                success=True,
                message="",
                panel=panel
            )
            
        except Exception as e:
            return CommandResult(False, f"Error fetching quote: {str(e)}")
