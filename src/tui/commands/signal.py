"""
SIGNAL Command

Display trading signals for a symbol.
Usage: SIGNAL <symbol>
"""

from typing import List

from rich.table import Table
from rich.panel import Panel

from src.tui.commands.base import MLCommandHandler
from src.tui.app import CommandResult, PythiaTerminal
from src.ml.signals import SignalGenerator


class SignalCommand(MLCommandHandler):
    """Display trading signals for a stock."""
    
    @property
    def name(self) -> str:
        return "SIGNAL"
    
    @property
    def aliases(self) -> List[str]:
        return ["SIGNAL", "SIG", "SIGNALS", "BUY", "SELL"]
    
    @property
    def description(self) -> str:
        return "Display multi-factor trading signals"
    
    @property
    def usage(self) -> str:
        return "SIGNAL <symbol>"
    
    @property
    def min_args(self) -> int:
        return 1
    
    async def execute(self, terminal: PythiaTerminal, args: List[str]) -> CommandResult:
        symbol = args[0].upper()
        return await self._show_signals(terminal, symbol)
    
    async def _show_signals(self, terminal: PythiaTerminal, symbol: str) -> CommandResult:
        """Show trading signals analysis."""
        try:
            generator = SignalGenerator()
            signals = generator.generate_signals(symbol)
            
            if 'error' in signals:
                return CommandResult(False, f"Error: {signals['error']}")
            
            current_price = signals.get('current_price')
            combined = signals.get('combined', {})
            recommendation = signals.get('recommendation', {})
            
            table = Table(title=f"📈 {symbol} Trading Signals")
            table.add_column("Indicator", style="cyan")
            table.add_column("Signal", style="yellow")
            table.add_column("Score", style="green", justify="right")
            
            ma = signals.get('moving_averages', {})
            if ma:
                score = ma.get('score', 0)
                signal_str = "🟢" if score > 0 else "🔴" if score < 0 else "⚪"
                table.add_row("Moving Averages", signal_str, f"{score:+.2f}")
            
            rsi = signals.get('rsi', {})
            if rsi:
                rsi_val = rsi.get('rsi', 50)
                signal_str = "🔴" if rsi_val > 70 else "🟢" if rsi_val < 30 else "⚪"
                table.add_row(f"RSI ({rsi_val:.1f})", signal_str, f"{rsi.get('score', 0):+.2f}")
            
            macd = signals.get('macd', {})
            if macd:
                score = macd.get('score', 0)
                signal_str = "🟢" if score > 0 else "🔴" if score < 0 else "⚪"
                table.add_row("MACD", signal_str, f"{score:+.2f}")
            
            bb = signals.get('bollinger', {})
            if bb:
                score = bb.get('score', 0)
                signal_str = "🟢" if score > 0 else "🔴" if score < 0 else "⚪"
                position = bb.get('position', 0.5)
                table.add_row(f"Bollinger ({position:.0%})", signal_str, f"{score:+.2f}")
            
            momentum = signals.get('momentum', {})
            if momentum:
                score = momentum.get('score', 0)
                signal_str = "🟢" if score > 0 else "🔴" if score < 0 else "⚪"
                table.add_row("Momentum", signal_str, f"{score:+.2f}")
            
            summary = Table(show_header=False)
            summary.add_column("Field", style="cyan")
            summary.add_column("Value", style="green")
            
            summary.add_row("Current Price", f"${current_price:.2f}")
            summary.add_row("Combined Score", f"{combined.get('score', 0):+.2f}")
            summary.add_row("Normalized", f"{combined.get('normalized', 0):.1f}/100")
            
            rating = recommendation.get('rating', 'N/A')
            action = recommendation.get('action', 'N/A')
            
            if "Buy" in rating:
                rating_style = "bold green"
            elif "Sell" in rating:
                rating_style = "bold red"
            else:
                rating_style = "bold yellow"
            
            summary.add_row("Recommendation", f"[{rating_style}]{rating}[/{rating_style}]")
            summary.add_row("Action", action)
            summary.add_row("Confidence", f"{recommendation.get('confidence', 0):.0f}%")
            
            panel = Panel(summary, title=f"[bold cyan]{symbol} Signal Summary[/bold cyan]")
            
            terminal.current_symbol = symbol
            
            return CommandResult(success=True, message="", panel=panel, table=table)
            
        except Exception as e:
            return CommandResult(False, f"Error generating signals: {str(e)}")
