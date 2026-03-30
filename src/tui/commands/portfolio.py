"""
PORTFOLIO Command

Display and manage portfolio positions.
Usage: PORT or PORTFOLIO
"""

from typing import List, Dict
from datetime import datetime

import yfinance as yf
from rich.table import Table
from rich.panel import Panel

from src.tui.commands.base import PortfolioCommandHandler
from src.tui.app import CommandResult, PythiaTerminal


class PortfolioCommand(PortfolioCommandHandler):
    """Display portfolio positions and performance."""
    
    @property
    def name(self) -> str:
        return "PORTFOLIO"
    
    @property
    def aliases(self) -> List[str]:
        return ["PORT", "PORTFOLIO", "P"]
    
    @property
    def description(self) -> str:
        return "Display portfolio positions and P&L"
    
    @property
    def usage(self) -> str:
        return "PORT  or  PORTFOLIO"
    
    @property
    def requires_symbol(self) -> bool:
        return False
    
    @property
    def min_args(self) -> int:
        return 0
    
    async def execute(self, terminal: PythiaTerminal, args: List[str]) -> CommandResult:
        # Get action from args (add, remove, list)
        action = args[0].upper() if args else "LIST"
        
        if action in ["ADD", "A"]:
            return await self._add_position(terminal, args[1:])
        elif action in ["REMOVE", "RM", "DEL"]:
            return await self._remove_position(terminal, args[1:])
        elif action in ["CLEAR"]:
            return self._clear_portfolio(terminal)
        else:
            return await self._list_portfolio(terminal)
    
    async def _list_portfolio(self, terminal: PythiaTerminal) -> CommandResult:
        """List all positions in portfolio."""
        portfolio = terminal.portfolio
        
        if not portfolio:
            panel = Panel(
                "[yellow]No positions in portfolio.[/yellow]\n\n"
                "Commands:\n"
                "  PORT ADD <symbol> <shares> <cost> - Add position\n"
                "  PORT RM <symbol> - Remove position\n"
                "  PORT CLEAR - Clear all positions",
                title="[bold cyan]Portfolio[/bold cyan]"
            )
            return CommandResult(success=True, message="", panel=panel)
        
        # Fetch current prices
        symbols = list(portfolio.keys())
        try:
            tickers = yf.Tickers(' '.join(symbols))
        except Exception:
            tickers = {}
        
        # Build table
        table = Table(title="📊 Portfolio Positions")
        table.add_column("Symbol", style="cyan")
        table.add_column("Shares", style="yellow", justify="right")
        table.add_column("Avg Cost", style="blue", justify="right")
        table.add_column("Current", style="green", justify="right")
        table.add_column("Value", style="green", justify="right")
        table.add_column("P&L", style="magenta", justify="right")
        table.add_column("P&L%", style="magenta", justify="right")
        
        total_value = 0
        total_cost = 0
        
        for symbol, position in portfolio.items():
            shares = position.get('shares', 0)
            cost = position.get('cost', 0)
            avg_cost = cost / shares if shares > 0 else 0
            
            # Get current price
            current_price = avg_cost
            try:
                if symbol in tickers:
                    info = tickers.symbols[symbol].info
                    current_price = info.get('currentPrice', avg_cost)
            except Exception:
                pass
            
            value = shares * current_price
            pnl = value - cost
            pnl_pct = (pnl / cost * 100) if cost > 0 else 0
            
            total_value += value
            total_cost += cost
            
            pnl_style = "green" if pnl >= 0 else "red"
            
            table.add_row(
                symbol,
                str(shares),
                f"${avg_cost:.2f}",
                f"${current_price:.2f}",
                f"${value:.2f}",
                f"[{pnl_style}]{pnl:+.2f}[/{pnl_style}]",
                f"[{pnl_style}]{pnl_pct:+.2f}%[/{pnl_style}]"
            )
        
        # Total row
        total_pnl = total_value - total_cost
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
        total_style = "green" if total_pnl >= 0 else "red"
        
        table.add_row(
            "[bold]TOTAL[/bold]",
            "",
            "",
            "",
            f"[bold]${total_value:.2f}[/bold]",
            f"[bold][{total_style}]{total_pnl:+.2f}[/{total_style}][/bold]",
            f"[bold][{total_style}]{total_pnl_pct:+.2f}%[/{total_style}][/bold]"
        )
        
        panel = Panel(
            f"[bold]Total Value:[/bold] ${total_value:.2f}\n"
            f"[bold]Total Cost:[/bold] ${total_cost:.2f}",
            title="[bold cyan]Portfolio Summary[/bold cyan]"
        )
        
        return CommandResult(success=True, message="", panel=panel, table=table)
    
    async def _add_position(self, terminal: PythiaTerminal, args: List[str]) -> CommandResult:
        """Add a position to portfolio."""
        if len(args) < 3:
            return CommandResult(
                False,
                "Usage: PORT ADD <symbol> <shares> <cost_per_share>"
            )
        
        symbol = args[0].upper()
        try:
            shares = float(args[1])
            cost_per_share = float(args[2])
        except ValueError:
            return CommandResult(False, "Invalid shares or cost value")
        
        total_cost = shares * cost_per_share
        
        if symbol in terminal.portfolio:
            # Update existing position
            existing = terminal.portfolio[symbol]
            old_shares = existing['shares']
            old_cost = existing['cost']
            
            new_shares = old_shares + shares
            new_cost = old_cost + total_cost
            
            terminal.portfolio[symbol] = {
                'shares': new_shares,
                'cost': new_cost,
                'added_at': datetime.now().isoformat()
            }
        else:
            # Add new position
            terminal.portfolio[symbol] = {
                'shares': shares,
                'cost': total_cost,
                'added_at': datetime.now().isoformat()
            }
        
        return CommandResult(
            True,
            f"Added {shares} shares of {symbol} at ${cost_per_share:.2f}"
        )
    
    async def _remove_position(self, terminal: PythiaTerminal, args: List[str]) -> CommandResult:
        """Remove a position from portfolio."""
        if not args:
            return CommandResult(False, "Usage: PORT RM <symbol>")
        
        symbol = args[0].upper()
        
        if symbol in terminal.portfolio:
            del terminal.portfolio[symbol]
            return CommandResult(True, f"Removed {symbol} from portfolio")
        else:
            return CommandResult(False, f"{symbol} not in portfolio")
    
    def _clear_portfolio(self, terminal: PythiaTerminal) -> CommandResult:
        """Clear all positions."""
        terminal.portfolio = {}
        return CommandResult(True, "Portfolio cleared")
