"""
Pythia Terminal User Interface (TUI)

A Bloomberg Terminal-like command interface for stock market analysis.
"""

import sys
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

try:
    import readline
except ImportError:
    readline = None

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.style import Style
from rich.text import Text
from rich.prompt import Prompt
from rich.theme import Theme

# Custom theme for Bloomberg-like colors
CUSTOM_THEME = Theme(
    {
        "repr.str": "cyan",
        "repr.number": "green",
        "repr.bool": "yellow",
        "prompt": "bold cyan",
        "good": "bold green",
        "bad": "bold red",
        "warning": "bold yellow",
        "info": "bold blue",
    }
)

console = Console(theme=CUSTOM_THEME)


class CommandCategory(Enum):
    """Command categories for organization."""

    DATA = "Data"
    ANALYTICS = "Analytics"
    ML = "Machine Learning"
    PORTFOLIO = "Portfolio"
    SYSTEM = "System"


@dataclass
class Command:
    """Represents a terminal command."""

    name: str
    aliases: List[str]
    description: str
    usage: str
    category: CommandCategory
    handler: Callable
    min_args: int = 0
    max_args: int = 10
    requires_symbol: bool = False
    requires_data: bool = True


@dataclass
class CommandResult:
    """Result of command execution."""

    success: bool
    message: str
    data: Any = None
    table: Optional[Table] = None
    panel: Optional[Panel] = None


class PythiaTerminal:
    """
    Main terminal interface for Pythia.

    Provides a Bloomberg Terminal-like experience with commands:
    - Q <symbol> : Quote
    - GP <symbol> : Graph/Chart
    - AI <symbol> : Analyze/AI
    - IN <indicator> <symbol> : Indicators
    - OPT <symbol> : Options
    - NW <symbol> : News
    - PORT : Portfolio
    - RSK : Risk
    - FIND <term> : Search
    - SET : Settings
    - HELP : Help
    """

    def __init__(self):
        self.console = Console()
        self.commands: Dict[str, Command] = {}
        self.watchlist: List[str] = []
        self.portfolio: Dict[str, Dict] = {}
        self.settings: Dict[str, Any] = {
            "theme": "dark",
            "auto_refresh": True,
            "refresh_interval": 30,
            "cache_enabled": True,
            "max_results": 50,
        }
        self.history: List[str] = []
        self.current_symbol: Optional[str] = None
        self.running = True

        # Initialize commands
        self._register_commands()

    def _register_commands(self):
        """Register all available commands."""
        from src.tui.commands.quote import QuoteCommand
        from src.tui.commands.graph import GraphCommand
        from src.tui.commands.analyze import AnalyzeCommand
        from src.tui.commands.technical import TechnicalCommand
        from src.tui.commands.news import NewsCommand
        from src.tui.commands.options import OptionsCommand
        from src.tui.commands.portfolio import PortfolioCommand
        from src.tui.commands.risk import RiskCommand
        from src.tui.commands.search import SearchCommand
        from src.tui.commands.watchlist import WatchlistCommand
        from src.tui.commands.predict import PredictCommand
        from src.tui.commands.backtest import BacktestCommand
        from src.tui.commands.crypto import CryptoCommand
        from src.tui.commands.forex import ForexCommand
        from src.tui.commands.fundamentals import FundCommand
        from src.tui.commands.sector import SectorCommand
        from src.tui.commands.breadth import BreadthCommand
        from src.tui.commands.pattern import PatternCommand
        from src.tui.commands.anomaly import AnomalyCommand
        from src.tui.commands.signal import SignalCommand

        # Register all commands
        cmd_handlers = [
            QuoteCommand(),
            GraphCommand(),
            AnalyzeCommand(),
            TechnicalCommand(),
            NewsCommand(),
            OptionsCommand(),
            PortfolioCommand(),
            RiskCommand(),
            SearchCommand(),
            WatchlistCommand(),
            PredictCommand(),
            BacktestCommand(),
            CryptoCommand(),
            ForexCommand(),
            FundCommand(),
            SectorCommand(),
            BreadthCommand(),
            PatternCommand(),
            AnomalyCommand(),
            SignalCommand(),
        ]

        for handler in cmd_handlers:
            cmd = handler.get_command()
            for alias in cmd.aliases:
                self.commands[alias.lower()] = cmd

    def get_command(self, name: str) -> Optional[Command]:
        """Get command by name or alias."""
        return self.commands.get(name.lower())

    def parse_input(self, input_str: str) -> tuple:
        """Parse input string into command and arguments."""
        parts = input_str.strip().split()
        if not parts:
            return None, []

        cmd_name = parts[0].upper()
        args = parts[1:] if len(parts) > 1 else []

        return cmd_name, args

    async def execute(self, input_str: str) -> CommandResult:
        """Execute a command."""
        # Add to history
        if input_str.strip():
            self.history.append(input_str)

        # Parse input
        cmd_name, args = self.parse_input(input_str)

        if not cmd_name:
            return CommandResult(True, "")

        # Get command
        command = self.get_command(cmd_name)

        if not command:
            # Try fuzzy matching
            suggestions = self._suggest_commands(cmd_name)
            if suggestions:
                return CommandResult(
                    False,
                    f"Unknown command: {cmd_name}. Did you mean: {', '.join(suggestions)}?",
                )
            return CommandResult(False, f"Unknown command: {cmd_name}")

        # Validate arguments
        if len(args) < command.min_args:
            return CommandResult(False, f"Missing arguments. Usage: {command.usage}")

        if len(args) > command.max_args:
            return CommandResult(False, f"Too many arguments. Usage: {command.usage}")

        # Check symbol requirement
        symbol = args[0] if args else None
        if command.requires_symbol and not symbol:
            return CommandResult(
                False, f"This command requires a symbol. Usage: {command.usage}"
            )

        # Execute command
        try:
            result = await command.handler(self, args)
            return result
        except Exception as e:
            return CommandResult(False, f"Error: {str(e)}")

    def _suggest_commands(self, cmd_name: str) -> List[str]:
        """Suggest similar commands."""
        suggestions = []
        for name in self.commands.keys():
            if name.startswith(cmd_name[:2].lower()):
                suggestions.append(name)
        return suggestions[:3]

    async def run_interactive(self):
        """Run the terminal in interactive mode."""
        self.print_welcome()

        while self.running:
            try:
                # Get input with prompt
                prompt = self._get_prompt()
                user_input = Prompt.ask(prompt)

                if not user_input.strip():
                    continue

                # Handle special commands
                if user_input.upper() in ["EXIT", "QUIT", "X"]:
                    self.running = False
                    self.console.print("[bold yellow]Goodbye![/bold yellow]")
                    break

                # Execute command
                result = await self.execute(user_input)

                # Display result
                self._display_result(result)

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Use EXIT to quit[/yellow]")
            except EOFError:
                break

    def _get_prompt(self) -> str:
        """Get the prompt string."""
        if self.current_symbol:
            return f"[bold cyan]Pythia:{self.current_symbol}>[/bold cyan] "
        return "[bold cyan]Pythia>[/bold cyan] "

    def _display_result(self, result: CommandResult):
        """Display command result."""
        if result.panel:
            self.console.print(result.panel)
        elif result.table:
            self.console.print(result.table)
        elif result.message:
            if result.success:
                self.console.print(result.message)
            else:
                self.console.print(f"[bold red]{result.message}[/bold red]")

    def print_welcome(self):
        """Print welcome message."""
        welcome = """
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║   █████╗ ██████╗ ██████╗     ███████╗██╗ ██████╗ ███╗   ██╗██╗      ║
║  ██╔══██╗██╔══██╗██╔══██╗    ██╔════╝██║██╔════╝ ████╗  ██║██║      ║
║  ███████║██████╔╝██║  ██║    ███████╗██║██║  ███╗██╔██╗ ██║██║      ║
║  ██╔══██║██╔═══╝ ██║  ██║    ╚════██║██║██║   ██║██║╚██╗██║██║      ║
║  ██║  ██║██║     ██████╔╝    ███████║██║╚██████╔╝██║ ╚████║███████╗║
║  ╚═╝  ╚═╝╚═╝     ╚═════╝     ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝║
║                                                                       ║
║                    TERMINAL - v3.0.0                                   ║
║              Bloomberg Terminal for Everyone                           ║
╚═══════════════════════════════════════════════════════════════════════╝

[bold]Available Commands:[/bold]
  [cyan]Q <sym>[/cyan]     Quote           - Show quote & basic info
  [cyan]GP <sym>[/cyan]    Graph           - Show price chart
  [cyan]AI <sym>[/cyan]    Analyze         - AI-powered analysis
  [cyan]IN <ind> <sym>[/cyan]  Indicators    - Technical indicators
  [cyan]OPT <sym>[/cyan]    Options         - Options chain
  [cyan]NW <sym>[/cyan]    News            - Latest news
  [cyan]PORT[/cyan]         Portfolio       - Portfolio view
  [cyan]RSK[/cyan]         Risk            - Risk metrics
  [cyan]FIND <term>[/cyan] Search          - Search securities
  [cyan]WL[/cyan]          Watchlist       - Manage watchlist
  [cyan]PRED <sym>[/cyan]  Predict         - AI prediction
  [cyan]BT <sym>[/cyan]   Backtest        - Run backtest
  [cyan]SET[/cyan]         Settings        - Configure
  [cyan]HELP[/cyan]        Help            - Show help
  [cyan]EXIT[/cyan]         Exit            - Quit

[dim]Type HELP <command> for detailed help[/dim]
        """
        self.console.print(welcome)

    def print_help(self, command: Optional[str] = None):
        """Print help for a command or all commands."""
        if command:
            cmd = self.get_command(command)
            if cmd:
                self.console.print(
                    Panel(
                        f"[bold]{cmd.name}[/bold]\n"
                        f"[cyan]Aliases:[/cyan] {', '.join(cmd.aliases)}\n"
                        f"[cyan]Usage:[/cyan] {cmd.usage}\n"
                        f"[cyan]Description:[/cyan] {cmd.description}",
                        title="Help",
                    )
                )
            else:
                self.console.print(f"[red]Unknown command: {command}[/red]")
        else:
            # Print all commands by category
            table = Table(title="Available Commands")
            table.add_column("Command", style="cyan")
            table.add_column("Aliases", style="yellow")
            table.add_column("Description")

            for cmd in self.commands.values():
                table.add_row(cmd.name, ", ".join(cmd.aliases), cmd.description)

            self.console.print(table)


def create_terminal() -> PythiaTerminal:
    """Create and return a new terminal instance."""
    return PythiaTerminal()


async def run_terminal():
    """Run the terminal."""
    terminal = create_terminal()
    await terminal.run_interactive()


def main():
    """Main entry point for TUI."""
    try:
        asyncio.run(run_terminal())
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
