"""
SECTOR Command

Display sector analysis and performance.
Usage: SECTOR or SECTOR <symbol>
"""

from typing import List

from rich.table import Table
from rich.panel import Panel

from src.tui.commands.base import AnalyticsCommandHandler
from src.tui.app import CommandResult, PythiaTerminal
from src.analytics.sectors import SectorAnalyzer


class SectorCommand(AnalyticsCommandHandler):
    """Display sector analysis and performance."""

    @property
    def name(self) -> str:
        return "SECTOR"

    @property
    def aliases(self) -> List[str]:
        return ["SECTOR", "SECT", "SECTORS"]

    @property
    def description(self) -> str:
        return "Display sector performance and analysis"

    @property
    def usage(self) -> str:
        return "SECTOR        - All sectors\nSECTOR <sym> - Compare to sector"

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
            return await self._show_all_sectors()
        else:
            return await self._compare_to_sector(args[0].upper())

    async def _show_all_sectors(self) -> CommandResult:
        """Show all sector performance."""
        try:
            analyzer = SectorAnalyzer()
            perf = analyzer.get_sector_performance("1mo")

            if "error" in perf:
                return CommandResult(False, f"Error: {perf['error']}")

            table = Table(title="📈 Sector Performance (1 Month)")
            table.add_column("Rank", style="cyan", justify="right")
            table.add_column("Sector", style="yellow")
            table.add_column("Change %", style="green", justify="right")
            table.add_column("ETF", style="blue")

            sectors = perf.get("sectors", {})
            sorted_sectors = sorted(
                sectors.items(), key=lambda x: x[1].get("change_pct", 0), reverse=True
            )

            for i, (sector, data) in enumerate(sorted_sectors, 1):
                change = data.get("change_pct", 0)
                style = "green" if change >= 0 else "red"
                table.add_row(
                    str(i),
                    sector,
                    f"[{style}]{change:+.2f}%[/{style}]",
                    data.get("etf", ""),
                )

            summary = Table(show_header=False)
            summary.add_column("Field", style="cyan")
            summary.add_column("Value", style="green")

            top = perf.get("top_performers", [])
            bottom = perf.get("bottom_performers", [])

            summary.add_row("Top Performer", top[0] if top else "N/A")
            summary.add_row("Bottom Performer", bottom[0] if bottom else "N/A")

            panel = Panel(summary, title="[bold cyan]Sector Summary[/bold cyan]")

            return CommandResult(success=True, message="", panel=panel, table=table)

        except Exception as e:
            return CommandResult(False, f"Error fetching sector data: {str(e)}")

    async def _compare_to_sector(self, symbol: str) -> CommandResult:
        """Compare stock to its sector."""
        try:
            analyzer = SectorAnalyzer()
            comparison = analyzer.get_sector_comparison(symbol)

            if "error" in comparison:
                return CommandResult(False, f"Error: {comparison['error']}")

            sector = comparison.get("sector", "N/A")
            stock_perf = comparison.get("stock_performance", {})
            sector_perf = comparison.get("sector_performance", {})
            relative = comparison.get("relative_strength", {})

            table = Table(title=f"📊 {symbol} vs {sector}")
            table.add_column("Period", style="cyan")
            table.add_column("Stock %", style="green", justify="right")
            table.add_column("Sector %", style="yellow", justify="right")
            table.add_column("Relative", style="magenta", justify="right")

            for period in ["1m", "3m", "6m", "1y"]:
                stock_chg = stock_perf.get(period, 0)
                sector_chg = sector_perf.get(period, 0)
                rel = relative.get(period, 0)

                stock_style = "green" if stock_chg >= 0 else "red"
                sector_style = "green" if sector_chg >= 0 else "red"
                rel_style = "green" if rel >= 0 else "red"

                table.add_row(
                    period,
                    f"[{stock_style}]{stock_chg:+.2f}%[/{stock_style}]",
                    f"[{sector_style}]{sector_chg:+.2f}%[/{sector_style}]",
                    f"[{rel_style}]{rel:+.2f}%[/{rel_style}]",
                )

            outperforming = comparison.get("outperforming", 0)

            summary = Table(show_header=False)
            summary.add_column("Field", style="cyan")
            summary.add_column("Value", style="green")

            summary.add_row("Sector", sector)
            summary.add_row("Outperforming", f"{outperforming}/4 periods")

            panel = Panel(
                summary, title=f"[bold cyan]{symbol} Sector Comparison[/bold cyan]"
            )

            return CommandResult(success=True, message="", panel=panel, table=table)

        except Exception as e:
            return CommandResult(False, f"Error comparing to sector: {str(e)}")
