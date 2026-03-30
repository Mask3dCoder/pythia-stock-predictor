"""
FOREX Command

Display forex data and exchange rates.
Usage: FX <pair> or FOREX <pair>
"""

from typing import List
from datetime import datetime

from rich.table import Table
from rich.panel import Panel

from src.tui.commands.base import DataCommandHandler
from src.tui.app import CommandResult, PythiaTerminal
from src.data.forex import ForexDataCollector


class ForexCommand(DataCommandHandler):
    """Display forex data and exchange rates."""

    @property
    def name(self) -> str:
        return "FOREX"

    @property
    def aliases(self) -> List[str]:
        return ["FX", "FOREX", "FX", "CURRENCY"]

    @property
    def description(self) -> str:
        return "Display forex exchange rates"

    @property
    def usage(self) -> str:
        return "FX        - Show all major rates\nFX <pair> - Show specific pair"

    @property
    def min_args(self) -> int:
        return 0

    @property
    def requires_symbol(self) -> bool:
        return False

    async def execute(self, terminal: PythiaTerminal, args: List[str]) -> CommandResult:
        if not args:
            return await self._show_all_rates()

        action = args[0].upper()

        if action in ["ALL", "RATES", "LIST"]:
            return await self._show_all_rates()
        elif action in ["CONVERT", "CONV"]:
            return await self._convert(args[1:])
        elif action in ["GOLD", "XAU"]:
            return await self._show_gold()
        elif action in ["SILVER", "XAG"]:
            return await self._show_silver()
        else:
            return await self._show_quote(args)

    async def _show_quote(self, args: List[str]) -> CommandResult:
        """Show quote for a forex pair."""
        pair = args[0].upper().replace("/", "").replace("-", "")

        try:
            collector = ForexDataCollector()
            data = collector.get_quote(pair)

            if "error" in data:
                return CommandResult(False, data.get("error", "Error fetching quote"))

            current = data.get("current")
            change = (
                (
                    (current - data.get("previous_close", current))
                    / data.get("previous_close", current)
                    * 100
                )
                if current and data.get("previous_close")
                else 0
            )

            table = Table(title=f"💱 {data['pair']}", show_header=False)
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="green")

            change_style = "green" if change >= 0 else "red"

            table.add_row("Current", f"{current:.5f}" if current else "N/A")
            table.add_row(
                "Bid", f"{data.get('bid', 'N/A'):.5f}" if data.get("bid") else "N/A"
            )
            table.add_row(
                "Ask", f"{data.get('ask', 'N/A'):.5f}" if data.get("ask") else "N/A"
            )

            if data.get("spread"):
                table.add_row("Spread", f"{data['spread']:.5f}")

            table.add_row(
                "Day Change", f"[{change_style}]{change:+.2f}%[/{change_style}]"
            )
            table.add_row(
                "High", f"{data.get('high', 'N/A'):.5f}" if data.get("high") else "N/A"
            )
            table.add_row(
                "Low", f"{data.get('low', 'N/A'):.5f}" if data.get("low") else "N/A"
            )

            panel = Panel(
                table,
                title=f"[bold cyan]Forex Quote[/bold cyan]",
                subtitle=f"Updated: {datetime.now().strftime('%H:%M:%S')}",
            )

            return CommandResult(success=True, message="", panel=panel)

        except Exception as e:
            return CommandResult(False, f"Error fetching forex data: {str(e)}")

    async def _show_all_rates(self) -> CommandResult:
        """Show all major forex rates."""
        major_pairs = [
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "USDCHF",
            "AUDUSD",
            "USDCAD",
            "NZDUSD",
            "EURGBP",
            "EURJPY",
            "GBPJPY",
        ]

        try:
            collector = ForexDataCollector()
            quotes = collector.get_multiple_quotes(major_pairs)

            table = Table(title="Major Forex Rates")
            table.add_column("Pair", style="cyan")
            table.add_column("Rate", style="green", justify="right")
            table.add_column("Bid", style="yellow", justify="right")
            table.add_column("Ask", style="yellow", justify="right")
            table.add_column("Spread", style="magenta", justify="right")

            for pair in major_pairs:
                data = quotes.get(pair, {})
                current = data.get("current")
                bid = data.get("bid")
                ask = data.get("ask")

                if current:
                    spread = (ask - bid) if (bid and ask) else 0
                    table.add_row(
                        pair,
                        f"{current:.5f}",
                        f"{bid:.5f}" if bid else "N/A",
                        f"{ask:.5f}" if ask else "N/A",
                        f"{spread:.5f}",
                    )

            return CommandResult(success=True, message="", table=table)

        except Exception as e:
            return CommandResult(False, f"Error fetching forex rates: {str(e)}")

    async def _convert(self, args: List[str]) -> CommandResult:
        """Convert between currencies."""
        if len(args) < 3:
            return CommandResult(False, "Usage: FX CONVERT <amount> <from> <to>")

        try:
            amount = float(args[0])
            from_curr = args[1].upper()
            to_curr = args[2].upper()
        except ValueError:
            return CommandResult(
                False, "Invalid amount. Usage: FX CONVERT <amount> <from> <to>"
            )

        try:
            collector = ForexDataCollector()
            result = collector.convert(amount, from_curr, to_curr)

            if result is None:
                return CommandResult(
                    False, f"Could not convert {from_curr} to {to_curr}"
                )

            table = Table(show_header=False)
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("From", f"{amount:,.2f} {from_curr}")
            table.add_row("To", f"{result:,.2f} {to_curr}")
            table.add_row("Rate", f"{result / amount:.5f}")

            panel = Panel(table, title=f"[bold cyan]Currency Conversion[/bold cyan]")

            return CommandResult(success=True, message="", panel=panel)

        except Exception as e:
            return CommandResult(False, f"Error converting: {str(e)}")

    async def _show_gold(self) -> CommandResult:
        """Show gold price."""
        try:
            collector = ForexDataCollector()
            data = collector.get_gold_price()

            if "error" in data:
                return CommandResult(
                    False, data.get("error", "Error fetching gold price")
                )

            table = Table(title="Gold (XAU/USD)", show_header=False)
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Price", f"${data.get('current', 0):,.2f}")
            table.add_row("Bid", f"${data.get('bid', 0):,.2f}")
            table.add_row("Ask", f"${data.get('ask', 0):,.2f}")
            table.add_row("High", f"${data.get('high', 0):,.2f}")
            table.add_row("Low", f"${data.get('low', 0):,.2f}")

            panel = Panel(table, title="[bold yellow]Gold Price[/bold yellow]")

            return CommandResult(success=True, message="", panel=panel)

        except Exception as e:
            return CommandResult(False, f"Error fetching gold: {str(e)}")

    async def _show_silver(self) -> CommandResult:
        """Show silver price."""
        try:
            collector = ForexDataCollector()
            data = collector.get_silver_price()

            if "error" in data:
                return CommandResult(
                    False, data.get("error", "Error fetching silver price")
                )

            table = Table(title="Silver (XAG/USD)", show_header=False)
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Price", f"${data.get('current', 0):,.2f}")
            table.add_row("Bid", f"${data.get('bid', 0):,.2f}")
            table.add_row("Ask", f"${data.get('ask', 0):,.2f}")
            table.add_row("High", f"${data.get('high', 0):,.2f}")
            table.add_row("Low", f"${data.get('low', 0):,.2f}")

            panel = Panel(table, title="[bold]Silver Price[/bold]")

            return CommandResult(success=True, message="", panel=panel)

        except Exception as e:
            return CommandResult(False, f"Error fetching silver: {str(e)}")
