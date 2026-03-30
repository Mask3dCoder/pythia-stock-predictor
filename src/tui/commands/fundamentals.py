"""
FUND Command

Display fundamental analysis for a symbol.
Usage: FUND <symbol> or FUNDAMENTALS <symbol>
"""

from typing import List

from rich.table import Table
from rich.panel import Panel

from src.tui.commands.base import AnalyticsCommandHandler
from src.tui.app import CommandResult, PythiaTerminal
from src.analytics.fundamentals import FundamentalAnalyzer


class FundCommand(AnalyticsCommandHandler):
    """Display fundamental analysis for a stock."""
    
    @property
    def name(self) -> str:
        return "FUNDAMENTALS"
    
    @property
    def aliases(self) -> List[str]:
        return ["FUND", "FUNDAMENTALS", "FUNDAMENTAL"]
    
    @property
    def description(self) -> str:
        return "Display fundamental analysis and financial ratios"
    
    @property
    def usage(self) -> str:
        return "FUND <symbol>  or  FUNDAMENTALS <symbol>"
    
    @property
    def min_args(self) -> int:
        return 1
    
    async def execute(self, terminal: PythiaTerminal, args: List[str]) -> CommandResult:
        symbol = args[0].upper()
        return await self._show_fundamentals(terminal, symbol)
    
    async def _show_fundamentals(self, terminal: PythiaTerminal, symbol: str) -> CommandResult:
        """Show comprehensive fundamental analysis."""
        try:
            analyzer = FundamentalAnalyzer()
            analysis = analyzer.get_full_analysis(symbol)
            
            if 'error' in analysis:
                return CommandResult(False, f"Error: {analysis['error']}")
            
            ratios = analysis.get('ratios', {})
            income = analysis.get('income', {})
            growth = analysis.get('growth', {})
            dividends = analysis.get('dividends', {})
            scores = analysis.get('scores', {})
            recommendation = analysis.get('recommendation', {})
            
            table = Table(title=f"📊 {symbol} Fundamental Analysis", show_header=False)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            def format_value(val):
                if val is None:
                    return "N/A"
                if isinstance(val, float):
                    if abs(val) > 1:
                        return f"{val:.2f}"
                    else:
                        return f"{val:.2%}"
                if isinstance(val, int):
                    return f"{val:,}"
                return str(val)
            
            table.add_row(
                "P/E Ratio", format_value(ratios.get('pe_ratio')),
                "P/B Ratio", format_value(ratios.get('price_to_book'))
            )
            table.add_row(
                "P/E Forward", format_value(ratios.get('forward_pe')),
                "PEG Ratio", format_value(ratios.get('peg_ratio'))
            )
            table.add_row(
                "Profit Margin", format_value(ratios.get('profit_margin')),
                "ROE", format_value(ratios.get('return_on_equity'))
            )
            table.add_row(
                "ROA", format_value(ratios.get('return_on_assets')),
                "Debt/Equity", format_value(ratios.get('debt_to_equity'))
            )
            table.add_row(
                "Current Ratio", format_value(ratios.get('current_ratio')),
                "Quick Ratio", format_value(ratios.get('quick_ratio'))
            )
            table.add_row(
                "Revenue", format_value(income.get('revenue')),
                "Net Income", format_value(income.get('net_income'))
            )
            table.add_row(
                "EPS", format_value(income.get('earnings_per_share')),
                "Revenue Growth", format_value(growth.get('revenue_growth'))
            )
            table.add_row(
                "Earnings Growth", format_value(growth.get('earnings_growth')),
                "Dividend Yield", format_value(dividends.get('dividend_yield'))
            )
            
            summary = Table(show_header=False)
            summary.add_column("Field", style="cyan")
            summary.add_column("Value", style="green")
            
            summary.add_row("Piotroski F-Score", f"{scores.get('piotroski_f_score', 'N/A')}/9")
            summary.add_row("Altman Z-Score", f"{scores.get('altman_z_score', 'N/A')} ({scores.get('altman_zone', 'N/A')})")
            summary.add_row("Recommendation", f"[bold green]{recommendation.get('rating', 'N/A')}[/bold green]")
            summary.add_row("Score", f"{recommendation.get('score', 'N/A')}/{recommendation.get('max_score', 'N/A')}")
            
            panel = Panel(
                summary,
                title=f"[bold cyan]{symbol} Summary[/bold cyan]"
            )
            
            terminal.current_symbol = symbol
            
            return CommandResult(success=True, message="", panel=panel, table=table)
            
        except Exception as e:
            return CommandResult(False, f"Error analyzing fundamentals: {str(e)}")
