"""
OPTIONS Command

Display options chain for a symbol.
Usage: OPT <symbol> or OPTIONS <symbol>
"""

from typing import List
from datetime import datetime

import yfinance as yf
from rich.table import Table
from rich.panel import Panel

from src.tui.commands.base import DataCommandHandler
from src.tui.app import CommandResult, PythiaTerminal


class OptionsCommand(DataCommandHandler):
    """Display options chain for a symbol."""
    
    @property
    def name(self) -> str:
        return "OPTIONS"
    
    @property
    def aliases(self) -> List[str]:
        return ["OPT", "OPTIONS", "OPTS"]
    
    @property
    def description(self) -> str:
        return "Display options chain and pricing"
    
    @property
    def usage(self) -> str:
        return "OPT <symbol>  or  OPTIONS <symbol>"
    
    @property
    def min_args(self) -> int:
        return 1
    
    async def execute(self, terminal: PythiaTerminal, args: List[str]) -> CommandResult:
        symbol = args[0].upper()
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get current price
            info = ticker.info
            current_price = info.get('currentPrice', info.get('previousClose', None))
            
            if current_price is None:
                return CommandResult(False, f"Could not fetch price for {symbol}")
            
            # Get options dates
            try:
                expiration_dates = ticker.options
            except Exception:
                expiration_dates = []
            
            if not expiration_dates:
                return CommandResult(
                    False, 
                    f"No options data available for {symbol}"
                )
            
            # Get nearest expiration (within 30 days)
            from datetime import timedelta
            target_date = datetime.now() + timedelta(days=30)
            nearest_date = None
            for date_str in expiration_dates:
                exp_date = datetime.strptime(date_str, '%Y-%m-%d')
                if exp_date <= target_date:
                    nearest_date = date_str
            
            if not nearest_date:
                nearest_date = expiration_dates[0]
            
            # Get options chain
            try:
                opt = ticker.option_chain(nearest_date)
                calls = opt.calls
                puts = opt.puts
            except Exception:
                return CommandResult(
                    False, 
                    f"Could not fetch options chain for {symbol}"
                )
            
            # Create calls table
            calls_table = Table(title=f"📞 {symbol} Calls - {nearest_date}")
            calls_table.add_column("Strike", style="cyan", justify="right")
            calls_table.add_column("Last", style="green", justify="right")
            calls_table.add_column("Bid", style="yellow", justify="right")
            calls_table.add_column("Ask", style="yellow", justify="right")
            calls_table.add_column("Vol", style="magenta", justify="right")
            calls_table.add_column("OI", style="blue", justify="right")
            calls_table.add_column("IV%", style="red", justify="right")
            
            # Add top calls (in the money and near the money)
            atm_strike = round(current_price / 5) * 5
            relevant_calls = calls[
                (calls['strike'] >= atm_strike - 15) & 
                (calls['strike'] <= atm_strike + 15)
            ].head(10)
            
            for _, row in relevant_calls.iterrows():
                calls_table.add_row(
                    f"${row['strike']:.2f}",
                    f"${row['lastPrice']:.2f}" if row['lastPrice'] else "N/A",
                    f"${row['bid']:.2f}" if row['bid'] else "N/A",
                    f"${row['ask']:.2f}" if row['ask'] else "N/A",
                    f"{int(row['volume']):,}" if row['volume'] else "N/A",
                    f"{int(row['openInterest']):,}" if row['openInterest'] else "N/A",
                    f"{row['impliedVolatility']*100:.1f}%" if row['impliedVolatility'] else "N/A"
                )
            
            # Create puts table
            puts_table = Table(title=f"📉 {symbol} Puts - {nearest_date}")
            puts_table.add_column("Strike", style="cyan", justify="right")
            puts_table.add_column("Last", style="green", justify="right")
            puts_table.add_column("Bid", style="yellow", justify="right")
            puts_table.add_column("Ask", style="yellow", justify="right")
            puts_table.add_column("Vol", style="magenta", justify="right")
            puts_table.add_column("OI", style="blue", justify="right")
            puts_table.add_column("IV%", style="red", justify="right")
            
            relevant_puts = puts[
                (puts['strike'] >= atm_strike - 15) & 
                (puts['strike'] <= atm_strike + 15)
            ].head(10)
            
            for _, row in relevant_puts.iterrows():
                puts_table.add_row(
                    f"${row['strike']:.2f}",
                    f"${row['lastPrice']:.2f}" if row['lastPrice'] else "N/A",
                    f"${row['bid']:.2f}" if row['bid'] else "N/A",
                    f"${row['ask']:.2f}" if row['ask'] else "N/A",
                    f"{int(row['volume']):,}" if row['volume'] else "N/A",
                    f"{int(row['openInterest']):,}" if row['openInterest'] else "N/A",
                    f"{row['impliedVolatility']*100:.1f}%" if row['impliedVolatility'] else "N/A"
                )
            
            # Summary panel
            summary = Table(show_header=False)
            summary.add_column("Field", style="cyan")
            summary.add_column("Value", style="green")
            summary.add_row("Symbol", symbol)
            summary.add_row("Current Price", f"${current_price:.2f}")
            summary.add_row("Expiration", nearest_date)
            summary.add_row("Available Expirations", str(len(expiration_dates)))
            
            panel = Panel(
                summary,
                title=f"[bold cyan]{symbol} Options[/bold cyan]",
                subtitle=f"Data as of {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            
            # Update current symbol
            terminal.current_symbol = symbol
            
            return CommandResult(
                success=True,
                message="",
                panel=panel,
                table=calls_table
            )
            
        except Exception as e:
            return CommandResult(False, f"Error fetching options: {str(e)}")
