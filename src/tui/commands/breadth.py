"""
BREADTH Command

Display market breadth indicators.
Usage: BREADTH or BREADTH
"""

from typing import List

from rich.table import Table
from rich.panel import Panel

from src.tui.commands.base import AnalyticsCommandHandler
from src.tui.app import CommandResult, PythiaTerminal
from src.analytics.breadth import MarketBreadthAnalyzer


class BreadthCommand(AnalyticsCommandHandler):
    """Display market breadth indicators."""
    
    @property
    def name(self) -> str:
        return "BREADTH"
    
    @property
    def aliases(self) -> List[str]:
        return ["BREADTH", "BR", "MARKET", "BREAD"]
    
    @property
    def description(self) -> str:
        return "Display market breadth and internals"
    
    @property
    def usage(self) -> str:
        return "BREADTH"
    
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
        return await self._show_breadth()
    
    async def _show_breadth(self) -> CommandResult:
        """Show comprehensive market breadth."""
        try:
            analyzer = MarketBreadthAnalyzer()
            score = analyzer.get_market_score()
            
            table = Table(title="📊 Market Breadth Indicators")
            table.add_column("Indicator", style="cyan")
            table.add_column("Value", style="green", justify="right")
            table.add_column("Signal", style="yellow")
            
            ad = analyzer.get_ad_ratio()
            if 'current' in ad:
                signal = ad.get('interpretation', 'N/A')
                signal_style = "green" if "overbought" not in signal else "red" if "oversold" not in signal else "yellow"
                table.add_row("A/D Ratio", f"{ad['current']:.2f}", f"[{signal_style}]{signal}[/{signal_style}]")
            
            hl = analyzer.get_new_highs_lows()
            if 'ratio' in hl:
                signal = hl.get('interpretation', 'N/A')
                signal_style = "green" if "bullish" in signal else "red" if "bearish" in signal else "yellow"
                table.add_row("Highs/Lows", f"{hl.get('highs', 0)}/{hl.get('lows', 0)}", f"[{signal_style}]{signal}[/{signal_style}]")
            
            pc = analyzer.get_put_call_ratio()
            if 'ratio' in pc:
                signal = pc.get('interpretation', 'N/A')
                signal_style = "green" if "bullish" in signal else "red" if "bearish" in signal else "yellow"
                table.add_row("Put/Call", f"{pc['ratio']:.2f}", f"[{signal_style}]{signal}[/{signal_style}]")
            
            vix = analyzer.get_vix_analysis()
            if 'current' in vix:
                signal = vix.get('fear_level', 'N/A')
                signal_style = "green" if "complacency" in signal else "red" if "fear" in signal else "yellow"
                table.add_row("VIX", f"{vix['current']:.2f}", f"[{signal_style}]{signal}[/{signal_style}]")
            
            ma = analyzer.get_percent_above_ma()
            if 'above_ma20_pct' in ma:
                signal = ma.get('interpretation', 'N/A')
                signal_style = "green" if "bullish" in signal else "red" if "bearish" in signal else "yellow"
                table.add_row("Above MA20", f"{ma['above_ma20_pct']:.1f}%", f"[{signal_style}]{signal}[/{signal_style}]")
            
            trin = analyzer.get_arms_index()
            if 'value' in trin:
                signal = trin.get('interpretation', 'N/A')
                signal_style = "green" if signal == "oversold" else "red" if signal == "overbought" else "yellow"
                table.add_row("TRIN", f"{trin['value']:.2f}", f"[{signal_style}]{signal}[/{signal_style}]")
            
            summary = Table(show_header=False)
            summary.add_column("Field", style="cyan")
            summary.add_column("Value", style="green")
            
            rating = score.get('rating', 'N/A')
            rating_style = "bold green" if "Strong Buy" in rating else "bold red" if "Sell" in rating else "yellow"
            
            summary.add_row("Market Score", f"{score.get('score', 0)}/{score.get('max_score', 10)}")
            summary.add_row("Rating", f"[{rating_style}]{rating}[/{rating_style}]")
            
            panel = Panel(
                summary,
                title="[bold cyan]Market Breadth Summary[/bold cyan]"
            )
            
            return CommandResult(success=True, message="", panel=panel, table=table)
            
        except Exception as e:
            return CommandResult(False, f"Error fetching breadth data: {str(e)}")
