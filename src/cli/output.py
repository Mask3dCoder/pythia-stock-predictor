"""
Pythia CLI Output Module

Rich-powered console output with centralized theming.
Provides backward-compatible wrappers and enhanced display functions.
"""

import sys
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

# Platform-specific terminal setup — must happen BEFORE Rich imports
if sys.platform == "win32":
    # Reconfigure stdout/stderr to use UTF-8 to prevent cp1252 encoding errors
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    # Let Rich use VT100 escape codes directly (bypass legacy Windows renderer)
    if os.environ.get("TERM") is None:
        os.environ["TERM"] = "xterm-256color"

from rich.console import Console
from rich.table import Table, box
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rich.text import Text
from rich import box as rich_box

from .theme import (
    PYTHIA_THEME,
    PANEL_STYLES,
    ICONS,
    style_price,
    style_change,
    format_change,
)

# ── Global Console ────────────────────────────────────────────────────────────

# Use force_terminal=True to bypass Windows legacy renderer entirely
console = Console(
    theme=PYTHIA_THEME,
    force_terminal=(sys.platform == "win32"),
    highlight=False,
    color_system="standard" if sys.platform == "win32" else "auto",
)


# ── Backward-Compatible Status Functions ─────────────────────────────────────

def print_success(message: str) -> None:
    """Print success message."""
    console.print(f"  [success]{ICONS['success']}[/] {message}")


def print_error(message: str) -> None:
    """Print error message to stderr."""
    console.print(f"  [error]{ICONS['error']}[/] {message}")


def print_warning(message: str) -> None:
    """Print warning message."""
    console.print(f"  [warning]{ICONS['warning']}[/] {message}")


def print_info(message: str) -> None:
    """Print info message."""
    console.print(f"  [info]{ICONS['info']}[/] {message}")


def print_header(message: str) -> None:
    """Print a styled section header."""
    console.print(f"\n[heading]{message}[/heading]")


def print_welcome() -> None:
    """Print the Pythia welcome banner."""
    from .components import print_welcome_screen

    print_welcome_screen()


# ── Rich Table Creators ──────────────────────────────────────────────────────


def create_predictions_table(
    predictions: List[float],
    lower_bound: Optional[List[float]] = None,
    upper_bound: Optional[List[float]] = None,
    title: str = "Predictions",
) -> Table:
    """Create a styled predictions table with confidence bounds."""
    table = Table(
        title=title,
        title_style="heading",
        box=box.ROUNDED,
        border_style=PANEL_STYLES["brand"],
        show_header=True,
        header_style="subheading",
    )
    table.add_column("Day", style="text.muted", justify="center")
    table.add_column("Forecast", style="value", justify="right")
    if lower_bound and upper_bound:
        table.add_column("Lower", style="warning", justify="right")
        table.add_column("Upper", style="success", justify="right")

    for i, pred in enumerate(predictions[:10], 1):
        row = [str(i), f"${pred:,.2f}"]
        if lower_bound and upper_bound:
            row.append(f"${lower_bound[i - 1]:,.2f}")
            row.append(f"${upper_bound[i - 1]:,.2f}")
        table.add_row(*row)

    return table


def create_metrics_table(metrics: Dict[str, float], title: str = "Metrics") -> Table:
    """Create a styled metrics table."""
    table = Table(
        title=title,
        title_style="heading",
        box=box.ROUNDED,
        border_style=PANEL_STYLES["default"],
        show_header=True,
        header_style="subheading",
    )
    table.add_column("Metric", style="text.secondary")
    table.add_column("Value", style="value", justify="right")

    for name, value in metrics.items():
        if isinstance(value, float):
            table.add_row(name, f"{value:.4f}")
        else:
            table.add_row(name, str(value))

    return table


def create_data_summary_table(data: Dict[str, Any], title: str = "Summary") -> Table:
    """Create a key-value data summary table."""
    table = Table(
        title=title,
        title_style="heading",
        box=box.ROUNDED,
        border_style=PANEL_STYLES["default"],
        show_header=False,
    )
    table.add_column("Key", style="text.secondary", width=20)
    table.add_column("Value", style="value")

    for key, value in data.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:,.2f}")
        elif isinstance(value, datetime):
            table.add_row(key, value.strftime("%Y-%m-%d %H:%M"))
        else:
            table.add_row(key, str(value))

    return table


def create_sentiment_table(results) -> Table:
    """Create a styled sentiment analysis table."""
    table = Table(
        title="Sentiment Analysis",
        title_style="heading",
        box=box.ROUNDED,
        border_style=PANEL_STYLES["info"],
        show_header=True,
        header_style="subheading",
    )
    table.add_column("Text", style="text.primary", max_width=50)
    table.add_column("Score", style="value", justify="right")
    table.add_column("Sentiment", justify="center")

    for _, row in results.iterrows():
        text = row.get("text", "")[:50]
        if len(str(row.get("text", ""))) > 50:
            text += "..."
        score = row.get("compound", row.get("polarity", 0))
        sentiment = row.get("sentiment", "neutral")

        if sentiment == "positive":
            sentiment_style = "success"
        elif sentiment == "negative":
            sentiment_style = "error"
        else:
            sentiment_style = "text.muted"

        table.add_row(text, f"{score:.4f}", f"[{sentiment_style}]{sentiment}[/]")

    return table


def create_progress() -> Progress:
    """Create a styled progress bar with spinner."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[text.primary]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        expand=False,
    )


def print_panel(content: str, title: str = "", style: str = "brand") -> None:
    """Print content in a styled panel."""
    border = PANEL_STYLES.get(style, PANEL_STYLES["default"])
    console.print(
        Panel(content, title=title, border_style=border, padding=(1, 2))
    )


# ── Enhanced Display Functions ───────────────────────────────────────────────


def display_price_summary(
    current: float,
    predicted: float,
    change_pct: Optional[float] = None,
    title: str = "Price Summary",
) -> None:
    """Display a formatted price summary card."""
    from .components import price_card

    console.print(price_card(current, predicted, change_pct, title))


def display_results(
    title: str, data: Dict[str, Any], style: str = "brand"
) -> None:
    """Display results in a panel with a table."""
    table = create_data_summary_table(data)
    panel = Panel(table, title=title, border_style=PANEL_STYLES.get(style, PANEL_STYLES["default"]))
    console.print(panel)


def display_error(message: str, suggestion: Optional[str] = None) -> None:
    """Display a formatted error with optional suggestion."""
    from .components import print_error_box

    print_error_box(message, suggestion)


def display_welcome_full() -> None:
    """Display full welcome with quick-start guide."""
    from .components import print_welcome_screen, print_quick_start

    print_welcome_screen()
    print_quick_start()
