"""
Pythia CLI Components

Reusable Rich-based UI widgets for consistent terminal output.
"""

import sys
import os
from typing import Optional, List, Dict, Any
from datetime import datetime

# Windows: reconfigure for UTF-8 before Rich imports
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.environ.setdefault("TERM", "xterm-256color")

from rich.console import Console, Group
from rich.table import Table, box
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TaskProgressColumn,
)
from rich.layout import Layout
from rich.columns import Columns
from rich.text import Text
from rich.rule import Rule
from rich import box as rich_box

from .theme import (
    PYTHIA_THEME,
    PANEL_STYLES,
    TABLE_STYLE,
    RULE_STYLES,
    ICONS,
    PANEL_PADDING,
    MAX_TABLE_WIDTH,
    MAX_PANEL_WIDTH,
    style_price,
    style_change,
    format_change,
)

console = Console(
    theme=PYTHIA_THEME,
    force_terminal=(sys.platform == "win32"),
    highlight=False,
    color_system="standard" if sys.platform == "win32" else "auto",
)


# ── Headers & Separators ─────────────────────────────────────────────────────

def _safe_rule(style_name: str = "section") -> Rule:
    """Create an ASCII-safe Rule that works on Windows cp1252 terminals."""
    try:
        return Rule(style=RULE_STYLES.get(style_name, RULE_STYLES["section"]))
    except Exception:
        return Rule(style=RULE_STYLES.get(style_name, RULE_STYLES["section"]), characters="-")


def print_command_header(title: str, subtitle: Optional[str] = None) -> None:
    """Print a styled command header with rule separators."""
    console.print()
    try:
        console.print(Rule(style=RULE_STYLES["accent"]))
    except Exception:
        console.print("-" * 60)
    console.print(f"  [brand]{title}[/]")
    if subtitle:
        console.print(f"  [text.secondary]{subtitle}[/]")
    try:
        console.print(Rule(style=RULE_STYLES["accent"]))
    except Exception:
        console.print("-" * 60)
    console.print()


def print_section_header(title: str) -> None:
    """Print a section header within a command."""
    console.print(f"\n[heading]{ICONS['diamond']} {title}[/heading]")


def print_separator(style: str = "section") -> None:
    """Print a thin rule separator."""
    try:
        console.print(Rule(style=RULE_STYLES.get(style, RULE_STYLES["section"])))
    except Exception:
        console.print("-" * 60)


# ── Status Messages ──────────────────────────────────────────────────────────

def status_ok(message: str) -> None:
    """Print success status."""
    console.print(f"  [success]{ICONS['success']}[/] {message}")


def status_fail(message: str) -> None:
    """Print failure status."""
    console.print(f"  [error]{ICONS['error']}[/] {message}", file=sys.stderr)


def status_warn(message: str) -> None:
    """Print warning status."""
    console.print(f"  [warning]{ICONS['warning']}[/] {message}")


def status_info(message: str) -> None:
    """Print info status."""
    console.print(f"  [info]{ICONS['info']}[/] {message}")


# ── Panels ────────────────────────────────────────────────────────────────────

def info_panel(content: str, title: str = "") -> Panel:
    """Create an info-styled panel."""
    return Panel(
        content,
        title=title,
        border_style=PANEL_STYLES["info"],
        padding=PANEL_PADDING,
        width=MAX_PANEL_WIDTH,
    )


def success_panel(content: str, title: str = "") -> Panel:
    """Create a success-styled panel."""
    return Panel(
        content,
        title=title,
        border_style=PANEL_STYLES["success"],
        padding=PANEL_PADDING,
        width=MAX_PANEL_WIDTH,
    )


def warning_panel(content: str, title: str = "") -> Panel:
    """Create a warning-styled panel."""
    return Panel(
        content,
        title=title,
        border_style=PANEL_STYLES["warning"],
        padding=PANEL_PADDING,
        width=MAX_PANEL_WIDTH,
    )


def error_panel(content: str, title: str = "") -> Panel:
    """Create an error-styled panel."""
    return Panel(
        content,
        title=title,
        border_style=PANEL_STYLES["error"],
        padding=PANEL_PADDING,
        width=MAX_PANEL_WIDTH,
    )


def result_panel(content: str, title: str = "") -> Panel:
    """Create a branded result panel."""
    return Panel(
        content,
        title=title,
        border_style=PANEL_STYLES["brand"],
        padding=PANEL_PADDING,
        width=MAX_PANEL_WIDTH,
    )


# ── Price Display ─────────────────────────────────────────────────────────────

def price_card(
    current_price: float,
    predicted_price: float,
    change_pct: Optional[float] = None,
    title: str = "Price Summary",
) -> Panel:
    """Create a price summary card with current, predicted, and change."""
    if change_pct is None and current_price != 0:
        change_pct = ((predicted_price - current_price) / current_price) * 100
    elif change_pct is None:
        change_pct = 0.0

    change_style = style_change(change_pct)
    change_arrow = "↑" if change_pct > 0 else "↓" if change_pct < 0 else "→"
    change_str = format_change(change_pct)

    content = "\n".join(
        [
            f"  [text.secondary]Current Price:[/]      [value]${current_price:,.2f}[/]",
            f"  [text.secondary]Predicted Price:[/]    [value]${predicted_price:,.2f}[/]",
            f"  [text.secondary]Change:[/]             [{change_style}]{change_arrow} {change_str}[/]",
        ]
    )

    return Panel(
        content,
        title=f"[{PANEL_STYLES['brand']}]{title}[/]",
        border_style=PANEL_STYLES["brand"],
        padding=(1, 2),
    )


# ── Metric Display ────────────────────────────────────────────────────────────

def metrics_grid(metrics: Dict[str, Any], title: str = "Metrics") -> Table:
    """Create a compact key-value metrics table."""
    table = Table(
        title=title,
        title_style="heading",
        box=box.ROUNDED,
        border_style=PANEL_STYLES["default"],
        show_header=False,
        expand=False,
        padding=(0, 2),
    )
    table.add_column("Metric", style="text.secondary", width=20)
    table.add_column("Value", style="value", justify="right")

    for key, value in metrics.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:,.4f}")
        elif isinstance(value, int):
            table.add_row(key, f"{value:,}")
        else:
            table.add_row(key, str(value))

    return table


def metric_cards(metrics: Dict[str, Any], columns: int = 3) -> None:
    """Display metrics as compact cards in columns."""
    cards = []
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted = f"{value:,.4f}"
        elif isinstance(value, int):
            formatted = f"{value:,}"
        else:
            formatted = str(value)

        card = Panel(
            f"[value]{formatted}[/]\n[text.muted]{key}[/]",
            border_style=PANEL_STYLES["default"],
            padding=(0, 2),
            expand=True,
        )
        cards.append(card)

    console.print(Columns(cards, equal=True))


# ── Tables ────────────────────────────────────────────────────────────────────

def styled_table(
    columns: List[str],
    rows: List[List[Any]],
    title: str = "",
    highlight_col: int = -1,
) -> Table:
    """Create a consistently styled data table."""
    table = Table(
        title=title,
        title_style="heading",
        box=box.ROUNDED,
        border_style=PANEL_STYLES["default"],
        show_header=True,
        header_style="subheading",
        pad_edge=True,
    )

    for i, col_name in enumerate(columns):
        justify = "right" if i > 0 else "left"
        style_name = "text.primary" if i > 0 else "symbol"
        table.add_column(col_name, style=style_name, justify=justify)

    for row in rows:
        styled_row = []
        for i, cell in enumerate(row):
            if i == highlight_col and isinstance(cell, str):
                styled_row.append(f"[highlight]{cell}[/]")
            else:
                styled_row.append(str(cell))
        table.add_row(*styled_row)

    return table


# ── Prediction Table ──────────────────────────────────────────────────────────

def prediction_table(
    predictions: List[float],
    lower_bound: Optional[List[float]] = None,
    upper_bound: Optional[List[float]] = None,
    title: str = "Price Predictions",
) -> Table:
    """Create a styled prediction results table with confidence bounds."""
    table = Table(
        title=title,
        title_style="heading",
        box=box.ROUNDED,
        border_style=PANEL_STYLES["brand"],
        show_header=True,
        header_style="subheading",
        pad_edge=True,
    )
    table.add_column("Day", style="text.muted", justify="center", width=6)
    table.add_column("Forecast", style="value", justify="right")
    if lower_bound is not None and upper_bound is not None:
        table.add_column("Low", style="warning", justify="right")
        table.add_column("High", style="success", justify="right")
        table.add_column("Range", style="text.muted", justify="right")

    for i, pred in enumerate(predictions[:10], 1):
        row = [str(i), f"${pred:,.2f}"]
        if lower_bound is not None and upper_bound is not None:
            low = lower_bound[i - 1]
            high = upper_bound[i - 1]
            row.append(f"${low:,.2f}")
            row.append(f"${high:,.2f}")
            row.append(f"±${(high - low) / 2:,.2f}")
        table.add_row(*row)

    return table


# ── Progress ──────────────────────────────────────────────────────────────────

def create_progress_bar(description: str = "Processing") -> Progress:
    """Create a styled progress bar with spinner."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[text.primary]{task.description}"),
        BarColumn(bar_width=40, style=PANEL_STYLES["default"]),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        expand=False,
    )


class LiveProgress:
    """Context manager for progress display with status messages."""

    def __init__(self, description: str = "Working"):
        self.progress = create_progress_bar(description)
        self.task = None

    def __enter__(self):
        self.progress.start()
        return self

    def __exit__(self, *args):
        self.progress.stop()

    def update(self, message: str, completed: float = None):
        """Update progress status."""
        if self.task is None:
            self.task = self.progress.add_task(message, total=100)
        elif completed is not None:
            self.progress.update(self.task, completed=completed,
                                 description=message)


# ── Welcome Screen ────────────────────────────────────────────────────────────

def print_welcome_screen() -> None:
    """Print the full welcome/branding screen."""
    from .theme import LOGO

    # Plain print for cross-platform logo safety
    print(LOGO)

    console.print("  [text.secondary]Intelligent stock prediction powered by ML[/]")
    console.print()
    console.print("  [text.muted]Type [bold]--help[/] for commands or [bold]--interactive[/] for guided mode[/]")
    console.print()


# ── Quick Start Tips ──────────────────────────────────────────────────────────

QUICK_START_TIPS = [
    ("collect", "Download stock data", "pythia collect --symbol AAPL --years 5"),
    ("train", "Train a prediction model", "pythia train --symbol AAPL --model ensemble"),
    ("predict", "Generate predictions", "pythia predict --symbol AAPL --days 7"),
    ("dashboard", "Launch web dashboard", "pythia dashboard"),
    ("evaluate", "Evaluate model accuracy", "pythia evaluate --symbol AAPL"),
    ("backtest", "Test trading strategies", "pythia backtest --symbol AAPL --capital 100000"),
]


def print_quick_start() -> None:
    """Print a quick-start guide for new users."""
    console.print()
    console.print(Rule("Quick Start", style=RULE_STYLES["accent"]))
    console.print()

    for cmd, desc, example in QUICK_START_TIPS:
        console.print(f"  [brand]{cmd:<12}[/] [text.secondary]{desc}[/]")
        console.print(f"  [text.muted]{'':>12}{example}[/]")

    console.print()
    console.print(Rule(style=RULE_STYLES["accent"]))
    console.print()


# ── Help Display ──────────────────────────────────────────────────────────────

def print_help_header() -> None:
    """Print the main help screen header."""
    print_welcome_screen()
    print_quick_start()


# ── Error Display ─────────────────────────────────────────────────────────────

def print_error_box(title: str, message: str, suggestion: Optional[str] = None) -> None:
    """Print a styled error box with optional suggestion."""
    from rich.markup import escape

    lines = [f"[error]{escape(str(message))}[/]"]
    if suggestion:
        lines.append("")
        lines.append(f"[text.secondary]{escape(str(suggestion))}[/]")

    console.print(
        Panel(
            "\n".join(lines),
            title=f"[error]{ICONS['error']} {escape(str(title))}[/]",
            border_style=PANEL_STYLES["error"],
            padding=(1, 2),
        )
    )


# ── Confirmation ──────────────────────────────────────────────────────────────

def confirm_action(message: str, default: bool = False) -> bool:
    """Ask for user confirmation using Rich prompt if available."""
    try:
        from rich.prompt import Confirm
        return Confirm.ask(message, default=default, console=console)
    except ImportError:
        response = input(f"{message} (y/n): ").strip().lower()
        return response in ("y", "yes")
