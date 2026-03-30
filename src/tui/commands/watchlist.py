"""
WATCHLIST Command

Display and manage watchlist.
Usage: WL or WATCHLIST
"""

from typing import List
from datetime import datetime

import yfinance as yf
from rich.table import Table
from rich.panel import Panel

from src.tui.commands.base import PortfolioCommandHandler
from src.tui.app import CommandResult, PythiaTerminal


class WatchlistCommand(PortfolioCommandHandler):
    """Manage watchlist of securities to track."""
    
    @property
    def name(self) -> str:
        return "WATCHLIST"
    
    @property
    def aliases(self) -> List[str]:
        return ["WL", "WATCHLIST", "WATCH"]
    
    @property
    def description(self) -> str:
        return "Manage watchlist of securities to track"
    
    @property
    def usage(self) -> str:
        return "WL            - Show watchlist\nWL ADD <sym> - Add to watchlist\nWL RM <sym> - Remove from watchlist"
    
    @property
    def requires_symbol(self) -> bool:
        return False
    
    @property
    def requires_data(self) -> bool:
        return False
    
    @property
    def min_args(self) -> int:
        return 0
    
    async def execute(self, terminal: PythiaTerminal, args: List[str]) -> CommandResult:
        if not args:
            return await self._list_watchlist(terminal)
        
        action = args[0].upper()
        
        if action in ["ADD", "A"]:
            return self._add_to_watchlist(terminal, args[1:])
        elif action in ["REMOVE", "RM", "DEL", "R"]:
            return self._remove_from_watchlist(terminal, args[1:])
        elif action in ["CLEAR"]:
            return self._clear_watchlist(terminal)
        elif action in ["REFRESH"]:
            return await self._list_watchlist(terminal)
        else:
            # Try treating as symbol
            return await self._list_watchlist(terminal)
    
    async def _list_watchlist(self, terminal: PythiaTerminal) -> CommandResult:
        """List all symbols in watchlist."""
        watchlist = terminal.watchlist
        
        if not watchlist:
            panel = Panel(
                "[yellow]Watchlist is empty.[/yellow]\n\n"
                "Commands:\n"
                "  WL ADD <symbol> - Add to watchlist\n"
                "  WL RM <symbol> - Remove from watchlist\n"
                "  WL CLEAR - Clear watchlist",
                title="[bold cyan]Watchlist[/bold cyan]"
            )
            return CommandResult(success=True, message="", panel=panel)
        
        # Fetch prices
        if watchlist:
            try:
                tickers = yf.Tickers(' '.join(watchlist))
            except Exception:
                tickers = {}
        else:
            tickers = {}
        
        table = Table(title="👁️ Watchlist")
        table.add_column("Symbol", style="cyan")
        table.add_column("Price", style="green", justify="right")
        table.add_column("Change", style="magenta", justify="right")
        table.add_column("Change %", style="magenta", justify="right")
        table.add_column("Volume", style="yellow", justify="right")
        
        for symbol in watchlist:
            try:
                if symbol in tickers:
                    info = tickers.symbols[symbol].info
                    price = info.get('currentPrice') or info.get('previousClose')
                    prev_close = info.get('previousClose')
                    
                    if price and prev_close:
                        change = price - prev_close
                        change_pct = (change / prev_close) * 100
                        change_str = f"{change:+.2f}"
                        change_pct_str = f"{change_pct:+.2f}%"
                        style = "green" if change >= 0 else "red"
                    else:
                        change_str = "N/A"
                        change_pct_str = "N/A"
                        style = "white"
                    
                    volume = info.get('volume', 0)
                    volume_str = f"{volume:,}" if volume else "N/A"
                    
                    price_str = f"${price:.2f}" if price else "N/A"
                    
                    table.add_row(
                        symbol,
                        price_str,
                        f"[{style}]{change_str}[/{style}]",
                        f"[{style}]{change_pct_str}[/{style}]",
                        volume_str
                    )
                else:
                    table.add_row(symbol, "N/A", "N/A", "N/A", "N/A")
            except Exception:
                table.add_row(symbol, "Error", "N/A", "N/A", "N/A")
        
        panel = Panel(
            f"[dim]{len(watchlist)} symbols in watchlist[/dim]",
            title="[bold cyan]Watchlist[/bold cyan]"
        )
        
        return CommandResult(success=True, message="", panel=panel, table=table)
    
    def _add_to_watchlist(self, terminal: PythiaTerminal, args: List[str]) -> CommandResult:
        """Add a symbol to watchlist."""
        if not args:
            return CommandResult(False, "Usage: WL ADD <symbol>")
        
        symbol = args[0].upper()
        
        if symbol in terminal.watchlist:
            return CommandResult(False, f"{symbol} already in watchlist")
        
        terminal.watchlist.append(symbol)
        return CommandResult(True, f"Added {symbol} to watchlist")
    
    def _remove_from_watchlist(self, terminal: PythiaTerminal, args: List[str]) -> CommandResult:
        """Remove a symbol from watchlist."""
        if not args:
            return CommandResult(False, "Usage: WL RM <symbol>")
        
        symbol = args[0].upper()
        
        if symbol in terminal.watchlist:
            terminal.watchlist.remove(symbol)
            return CommandResult(True, f"Removed {symbol} from watchlist")
        else:
            return CommandResult(False, f"{symbol} not in watchlist")
    
    def _clear_watchlist(self, terminal: PythiaTerminal) -> CommandResult:
        """Clear the watchlist."""
        terminal.watchlist = []
        return CommandResult(True, "Watchlist cleared")
