"""
CRYPTO Command

Display cryptocurrency data and prices.
Usage: CR <symbol> or CRYPTO <symbol>
"""

from typing import List
from datetime import datetime

from rich.table import Table
from rich.panel import Panel

from src.tui.commands.base import DataCommandHandler
from src.tui.app import CommandResult, PythiaTerminal
from src.data.crypto import CryptoDataCollector


class CryptoCommand(DataCommandHandler):
    """Display cryptocurrency data and prices."""

    @property
    def name(self) -> str:
        return "CRYPTO"

    @property
    def aliases(self) -> List[str]:
        return ["CR", "CRYPTO", "COIN", "BTC", "ETH"]

    @property
    def description(self) -> str:
        return "Display cryptocurrency prices and data"

    @property
    def usage(self) -> str:
        return "CR <symbol>  or  CRYPTO <symbol>"

    @property
    def min_args(self) -> int:
        return 1

    async def execute(self, terminal: PythiaTerminal, args: List[str]) -> CommandResult:
        action = args[0].upper()

        if action in ["TOP", "LIST", "MARKETS"]:
            return await self._show_top_coins(args[1:])
        elif action in ["INFO"]:
            return await self._show_coin_info(args[1:])
        else:
            return await self._show_quote(args)

    async def _show_quote(self, args: List[str]) -> CommandResult:
        """Show quote for a cryptocurrency."""
        symbol = args[0].upper()

        try:
            collector = CryptoDataCollector()
            coin_id = collector._get_coin_id(symbol)

            if not coin_id:
                return CommandResult(False, f"Unknown cryptocurrency: {symbol}")

            data = collector.get_market_data(coin_id)

            if not data or not data.get("current_price"):
                return CommandResult(False, f"No data available for {symbol}")

            table = Table(
                title=f"₿ {data['name']} ({data['symbol']})", show_header=False
            )
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="green")

            price = data["current_price"]
            change_24h = data.get("price_change_percentage_24h", 0) or 0
            if isinstance(change_24h, dict):
                change_24h = change_24h.get("usd", 0)
            change_style = "green" if change_24h >= 0 else "red"

            change_7d = data.get("price_change_percentage_7d", 0)
            if isinstance(change_7d, dict):
                change_7d = change_7d.get("usd", 0)

            change_30d = data.get("price_change_percentage_30d", 0)
            if isinstance(change_30d, dict):
                change_30d = change_30d.get("usd", 0)

            table.add_row("Price", f"${price:,.2f}")
            table.add_row(
                "24h Change", f"[{change_style}]{change_24h:+.2f}%[/{change_style}]"
            )
            table.add_row(
                "7d Change", f"[{change_style}]{change_7d:+.2f}%[/{change_style}]"
            )
            table.add_row(
                "30d Change", f"[{change_style}]{change_30d:+.2f}%[/{change_style}]"
            )
            table.add_row("Market Cap", f"${data.get('market_cap', 0):,.0f}")
            table.add_row("Volume 24h", f"${data.get('total_volume', 0):,.0f}")
            table.add_row("Rank", f"#{data.get('market_cap_rank', 'N/A')}")
            table.add_row("High 24h", f"${data.get('high_24h', 0):,.2f}")
            table.add_row("Low 24h", f"${data.get('low_24h', 0):,.2f}")

            if data.get("ath"):
                table.add_row("All-Time High", f"${data['ath']:,.2f}")
                table.add_row(
                    "ATH Change",
                    f"[red]{data.get('ath_change_percentage', 0):.2f}%[/red]",
                )

            panel = Panel(
                table,
                title=f"[bold cyan]{data['symbol']}/USD[/bold cyan]",
                subtitle=f"Updated: {datetime.now().strftime('%H:%M:%S')}",
            )

            return CommandResult(success=True, message="", panel=panel)

        except Exception as e:
            return CommandResult(False, f"Error fetching crypto data: {str(e)}")

    async def _show_top_coins(self, args: List[str]) -> CommandResult:
        """Show top cryptocurrencies by market cap."""
        limit = int(args[0]) if args and args[0].isdigit() else 10

        try:
            collector = CryptoDataCollector()
            coins = collector.get_top_coins(limit=limit)

            table = Table(title=f"Top {limit} Cryptocurrencies")
            table.add_column("#", style="cyan", justify="right")
            table.add_column("Coin", style="yellow")
            table.add_column("Price", style="green", justify="right")
            table.add_column("24h %", style="magenta", justify="right")
            table.add_column("Market Cap", style="blue", justify="right")
            table.add_column("Volume", style="cyan", justify="right")

            for coin in coins:
                change = coin.get("price_change_percentage_24h", 0)
                style = "green" if change >= 0 else "red"

                table.add_row(
                    str(coin.get("market_cap_rank", "-")),
                    coin.get("symbol", ""),
                    f"${coin.get('current_price', 0):,.2f}",
                    f"[{style}]{change:+.2f}%[/{style}]",
                    f"${coin.get('market_cap', 0) / 1e9:.1f}B"
                    if coin.get("market_cap")
                    else "N/A",
                    f"${coin.get('total_volume', 0) / 1e9:.1f}B"
                    if coin.get("total_volume")
                    else "N/A",
                )

            return CommandResult(success=True, message="", table=table)

        except Exception as e:
            return CommandResult(False, f"Error fetching top coins: {str(e)}")

    async def _show_coin_info(self, args: List[str]) -> CommandResult:
        """Show detailed coin information."""
        if not args:
            return CommandResult(False, "Usage: CR INFO <symbol>")

        symbol = args[0].upper()

        try:
            collector = CryptoDataCollector()
            info = collector.get_coin_info(symbol)

            if not info:
                return CommandResult(False, f"No info found for {symbol}")

            table = Table(show_header=False)
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Name", info.get("name", "N/A"))
            table.add_row("Symbol", info.get("symbol", "N/A"))
            table.add_row("Rank", f"#{info.get('market_cap_rank', 'N/A')}")
            table.add_row("Community Score", str(info.get("community_score", "N/A")))

            if info.get("description"):
                desc = (
                    info["description"][:200] + "..."
                    if len(info["description"]) > 200
                    else info["description"]
                )
                table.add_row("Description", desc)

            panel = Panel(
                table, title=f"[bold cyan]{info.get('name')} Info[/bold cyan]"
            )

            return CommandResult(success=True, message="", panel=panel)

        except Exception as e:
            return CommandResult(False, f"Error fetching coin info: {str(e)}")
