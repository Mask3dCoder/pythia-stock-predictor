"""
CLI Output Module

Provides console output for the CLI.
"""

import sys
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

# Disable Rich formatting for Windows compatibility
os.environ['TERM'] = 'dumb'

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel

console = Console(force_terminal=False)


def print_success(message: str) -> None:
    """Print success message."""
    print(f"[+] {message}")


def print_error(message: str) -> None:
    """Print error message."""
    print(f"[!] {message}", file=sys.stderr)


def print_warning(message: str) -> None:
    """Print warning message."""
    print(f"[*] {message}")


def print_info(message: str) -> None:
    """Print info message."""
    print(f"[i] {message}")


def print_header(message: str) -> None:
    """Print header message."""
    print(f"\n=== {message} ===\n")


def print_welcome() -> None:
    """Print welcome message."""
    welcome = """
=========================================
  PYTHIA STOCK PREDICTOR v3.0
=========================================
    """
    print(welcome)


def create_predictions_table(
    predictions: List[float],
    lower_bound: Optional[List[float]] = None,
    upper_bound: Optional[List[float]] = None,
    title: str = "Predictions"
) -> Table:
    """Create a table of predictions."""
    table = Table(title=title, show_header=True)
    table.add_column("Day")
    table.add_column("Prediction")
    if lower_bound and upper_bound:
        table.add_column("Lower")
        table.add_column("Upper")
    
    for i, pred in enumerate(predictions[:10], 1):
        row = [str(i), f"${pred:.2f}"]
        if lower_bound and upper_bound:
            row.append(f"${lower_bound[i-1]:.2f}")
            row.append(f"${upper_bound[i-1]:.2f}")
        table.add_row(*row)
    
    return table


def create_metrics_table(metrics: Dict[str, float], title: str = "Metrics") -> Table:
    """Create a table of metrics."""
    table = Table(title=title, show_header=True)
    table.add_column("Metric")
    table.add_column("Value")
    
    for name, value in metrics.items():
        if isinstance(value, float):
            table.add_row(name, f"{value:.4f}")
        else:
            table.add_row(str(name), str(value))
    
    return table


def create_data_summary_table(data: Dict[str, Any], title: str = "Data Summary") -> Table:
    """Create a table of data summary."""
    table = Table(title=title, show_header=False)
    table.add_column("Key")
    table.add_column("Value")
    
    for key, value in data.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.2f}")
        elif isinstance(value, datetime):
            table.add_row(key, value.strftime("%Y-%m-%d"))
        else:
            table.add_row(str(key), str(value))
    
    return table


def create_sentiment_table(results) -> Table:
    """Create a table of sentiment results."""
    table = Table(title="Sentiment Analysis", show_header=True)
    table.add_column("Text")
    table.add_column("Score")
    table.add_column("Sentiment")
    
    for _, row in results.iterrows():
        text = row.get('text', '')[:50] + '...' if len(str(row.get('text', ''))) > 50 else row.get('text', '')
        score = row.get('compound', row.get('polarity', 0))
        sentiment = row.get('sentiment', 'neutral')
        
        table.add_row(text, f"{score:.4f}", sentiment)
    
    return table


def create_progress() -> Progress:
    """Create a progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    )


def print_panel(content: str, title: str = "", style: str = "blue") -> None:
    """Print content in a panel."""
    console.print(Panel(content, title=title))
