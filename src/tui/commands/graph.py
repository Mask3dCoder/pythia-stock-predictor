"""
GRAPH Command

Display price chart for a symbol.
Usage: GP <symbol> or GRAPH <symbol>
"""

import asyncio
from typing import List
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import numpy as np
from rich.table import Table
from rich.panel import Panel
from rich.console import Console
from rich.style import Style

from src.tui.commands.base import DataCommandHandler
from src.tui.app import CommandResult, PythiaTerminal, CommandCategory


class GraphCommand(DataCommandHandler):
    """Display price chart (ASCII art)."""
    
    @property
    def name(self) -> str:
        return "GRAPH"
    
    @property
    def aliases(self) -> List[str]:
        return ["GP", "GRAPH", "CHART", "C"]
    
    @property
    def description(self) -> str:
        return "Display price chart (candlestick/line)"
    
    @property
    def usage(self) -> str:
        return "GP <symbol> [days]  or  GRAPH <symbol> [days]"
    
    @property
    def min_args(self) -> int:
        return 1
    
    async def execute(self, terminal: PythiaTerminal, args: List[str]) -> CommandResult:
        symbol = args[0].upper()
        days = int(args[1]) if len(args) > 1 else 30
        
        try:
            # Fetch data
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                return CommandResult(False, f"No data found for {symbol}")
            
            # Calculate price change
            first_price = df['Close'].iloc[0]
            last_price = df['Close'].iloc[-1]
            change = last_price - first_price
            change_pct = (change / first_price) * 100
            
            # Create mini chart (ASCII)
            prices = df['Close'].values
            chart = self._create_ascii_chart(prices, width=50, height=10)
            
            # Create summary table
            table = Table(show_header=False)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Period", f"{days} days")
            table.add_row("Start Price", f"${first_price:.2f}")
            table.add_row("End Price", f"${last_price:.2f}")
            
            change_color = "green" if change >= 0 else "red"
            table.add_row(
                "Change", 
                f"[{change_color}]{change:+.2f} ({change_pct:+.2f}%)[/{change_color}]"
            )
            table.add_row("High", f"${df['High'].max():.2f}")
            table.add_row("Low", f"${df['Low'].min():.2f}")
            table.add_row("Volume", f"{df['Volume'].sum():,}")
            
            # Create panel
            panel = Panel(
                f"[bold cyan]{chart}[/bold cyan]\n\n{table}",
                title=f"📈 {symbol} - {days} Day Chart",
                subtitle=f"Last updated: {datetime.now().strftime('%H:%M:%S')}"
            )
            
            return CommandResult(
                success=True,
                message="",
                panel=panel
            )
            
        except Exception as e:
            return CommandResult(False, f"Error generating chart: {str(e)}")
    
    def _create_ascii_chart(self, prices: np.ndarray, width: int = 50, height: int = 10) -> str:
        """Create ASCII art chart from prices."""
        if len(prices) == 0:
            return "No data"
        
        # Normalize prices to 0-height range
        min_price = np.min(prices)
        max_price = np.max(prices)
        
        if max_price == min_price:
            return "Flat"
        
        # Resample to width
        step = max(1, len(prices) // width)
        sampled = []
        for i in range(0, len(prices), step):
            sampled.append(np.mean(prices[i:i+step]))
        
        # Normalize to height
        norm = (sampled - min_price) / (max_price - min_price)
        scaled = (norm * (height - 1)).astype(int)
        
        # Create chart
        lines = []
        for h in range(height - 1, -1, -1):
            line = ""
            for val in scaled:
                if val == h:
                    line += "●"
                elif val > h:
                    line += "│"
                else:
                    line += " "
            lines.append(line)
        
        # Add price labels
        result = "\n".join(lines)
        result += f"\n${min_price:.2f}" + " " * (width - 15) + f"${max_price:.2f}"
        
        return result
