"""
PATTERN Command

Display pattern recognition for a symbol.
Usage: PATTERN <symbol>
"""

from typing import List

from rich.table import Table
from rich.panel import Panel

from src.tui.commands.base import MLCommandHandler
from src.tui.app import CommandResult, PythiaTerminal
from src.ml.patterns import PatternRecognizer


class PatternCommand(MLCommandHandler):
    """Display pattern recognition for a stock."""
    
    @property
    def name(self) -> str:
        return "PATTERN"
    
    @property
    def aliases(self) -> List[str]:
        return ["PATTERN", "PAT", "PATTERNS"]
    
    @property
    def description(self) -> str:
        return "Display candlestick and chart pattern recognition"
    
    @property
    def usage(self) -> str:
        return "PATTERN <symbol>"
    
    @property
    def min_args(self) -> int:
        return 1
    
    async def execute(self, terminal: PythiaTerminal, args: List[str]) -> CommandResult:
        symbol = args[0].upper()
        return await self._show_patterns(terminal, symbol)
    
    async def _show_patterns(self, terminal: PythiaTerminal, symbol: str) -> CommandResult:
        """Show pattern recognition analysis."""
        try:
            recognizer = PatternRecognizer()
            patterns = recognizer.get_all_patterns(symbol)
            
            if 'error' in patterns:
                return CommandResult(False, f"Error: {patterns['error']}")
            
            candlestick = patterns.get('candlestick', [])
            chart = patterns.get('chart', [])
            recent = patterns.get('recent', {})
            
            table = Table(title=f"🕯️ {symbol} Pattern Recognition")
            table.add_column("Pattern", style="cyan")
            table.add_column("Type", style="yellow")
            table.add_column("Strength", style="green", justify="right")
            table.add_column("Date", style="blue")
            
            for p in candlestick[:8]:
                pattern_type = "🟢" if p.get('bullish') else "🔴" if p.get('bullish') is False else "⚪"
                table.add_row(
                    p.get('pattern', 'N/A'),
                    pattern_type,
                    f"{p.get('strength', 0):.2f}",
                    p.get('date', 'N/A')
                )
            
            if chart:
                table.add_row("", "", "", "")
                table.add_row("[bold]Chart Patterns[/bold]", "", "", "")
                for p in chart:
                    direction = "🟢" if p.get('direction') == 'bullish' else "🔴"
                    table.add_row(
                        p.get('pattern', 'N/A'),
                        direction,
                        f"{p.get('confidence', 0):.0%}",
                        p.get('type', 'N/A')
                    )
            
            summary = Table(show_header=False)
            summary.add_column("Field", style="cyan")
            summary.add_column("Value", style="green")
            
            summary.add_row("Total Patterns", str(recent.get('total_candlesticks', 0)))
            summary.add_row("Bullish", str(recent.get('bullish', 0)))
            summary.add_row("Bearish", str(recent.get('bearish', 0)))
            summary.add_row("Overall Bias", f"[bold]{recent.get('overall_bias', 'N/A').upper()}[/bold]")
            
            recognizer2 = PatternRecognizer()
            ticker = terminal
            import yfinance as yf
            ticker2 = yf.Ticker(symbol)
            hist = ticker2.history(period='3mo')
            if not hist.empty:
                sr = recognizer2.detect_support_resistance(hist)
                summary.add_row("Support", f"${sr.get('nearest_support', 'N/A'):.2f}" if sr.get('nearest_support') else "N/A")
                summary.add_row("Resistance", f"${sr.get('nearest_resistance', 'N/A'):.2f}" if sr.get('nearest_resistance') else "N/A")
            
            panel = Panel(summary, title=f"[bold cyan]{symbol} Summary[/bold cyan]")
            
            terminal.current_symbol = symbol
            
            return CommandResult(success=True, message="", panel=panel, table=table)
            
        except Exception as e:
            return CommandResult(False, f"Error analyzing patterns: {str(e)}")
