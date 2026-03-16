"""
CLI Output Module - Rich Console and Theme Management

Provides beautiful colored output for the Stock Prediction CLI using rich.
"""

from rich.console import Console
from rich.theme import Theme
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich import box
from typing import Optional, List, Dict, Any

# Custom theme for the CLI - Professional dark theme palette
_custom_theme = Theme({
    # Status colors
    "success": "bold bright_green",
    "error": "bold bright_red",
    "warning": "bold bright_yellow",
    "info": "bright_cyan",
    
    # Title and headers
    "title": "bold bright_magenta",
    "subtitle": "italic bright_blue",
    "header": "bold white",
    
    # Content styling
    "dim": "dim white",
    "highlight": "bold bright_white",
    "accent": "bright_blue",
    
    # Data styling
    "positive": "green",
    "negative": "red",
    "neutral": "yellow",
    
    # Symbol indicators
    "up": "▲",
    "down": "▼",
})

# Global console instance
console = Console(theme=_custom_theme)


# ============== Output Functions ==============

def print_success(message: str) -> None:
    """Print a success message in green."""
    console.print(f"[success]OK {message}")


def print_error(message: str) -> None:
    """Print an error message in red."""
    console.print(f"[error]X {message}")


def print_warning(message: str) -> None:
    """Print a warning message in yellow."""
    console.print(f"[warning]! {message}")


def print_info(message: str) -> None:
    """Print an info message in cyan."""
    console.print(f"[info]i {message}")


def print_header(title: str, width: int = 60) -> None:
    """Print a centered header."""
    console.print()
    console.print(f"[title]{title.center(width)}[/title]")
    console.print(f"[dim]{'=' * width}[/dim]")
    console.print()


def print_subheader(title: str) -> None:
    """Print a subheader."""
    console.print(f"[subtitle]▸ {title}")


def print_price_change(price: float, change: float, change_pct: float) -> None:
    """Print price with change, colored based on positive/negative."""
    if change > 0:
        color = "[positive]"
        arrow = "▲"
    elif change < 0:
        color = "[negative]"
        arrow = "▼"
    else:
        color = "[neutral]"
        arrow = "•"
    
    console.print(f"{color}${price:.2f} {arrow} {change:+.2f} ({change_pct:+.2f}%)")


def print_version() -> None:
    """Print version information."""
    from src import __version__ as version
    console.print(f"[dim]Pythia Stock Predictor v{version}[/dim]")


# ============== Panel Functions ==============

def print_panel(content: str, title: str = "", style: str = "cyan") -> None:
    """Print content in a rich panel."""
    panel = Panel(
        content,
        title=title,
        border_style=style,
        box=box.ROUNDED,
        expand=False
    )
    console.print(panel)


def print_welcome() -> None:
    """Print welcome message."""
    welcome_text = """
[bold magenta]╔═══════════════════════════════════════════════════════════╗
║         📈 Stock Market Prediction CLI v1.0              ║
╠═══════════════════════════════════════════════════════════╣
║  Predict stock prices using machine learning models       ║
║  ARIMA, LSTM, GRU, and Ensemble models supported         ║
╚═══════════════════════════════════════════════════════════╝[/bold magenta]
    """
    console.print(welcome_text)


# ============== Table Functions ==============

def create_predictions_table(
    predictions: List[float],
    lower_bound: Optional[List[float]] = None,
    upper_bound: Optional[List[float]] = None,
    title: str = "Predictions"
) -> Table:
    """Create a formatted table for predictions."""
    table = Table(
        title=title,
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan"
    )
    
    table.add_column("Day", style="cyan", justify="center")
    table.add_column("Prediction", style="green", justify="right")
    
    if lower_bound is not None and upper_bound is not None:
        table.add_column("Lower Bound", style="yellow", justify="right")
        table.add_column("Upper Bound", style="yellow", justify="right")
    
    for i, pred in enumerate(predictions):
        row = [
            f"Day {i + 1}",
            f"${pred:.2f}"
        ]
        
        if lower_bound is not None and upper_bound is not None:
            row.extend([
                f"${lower_bound[i]:.2f}",
                f"${upper_bound[i]:.2f}"
            ])
        
        table.add_row(*row)
    
    return table


def create_metrics_table(metrics: Dict[str, Any], title: str = "Model Evaluation Metrics") -> Table:
    """Create a formatted table for model metrics."""
    table = Table(
        title=title,
        box=box.ROUNDED,
        show_header=False,
        header_style="bold cyan"
    )
    
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    
    # Format metric values
    metric_formatters = {
        'mae': lambda v: f"${v:.4f}",
        'rmse': lambda v: f"${v:.4f}",
        'mape': lambda v: f"{v:.2f}%",
        'r2': lambda v: f"{v:.4f}",
    }
    
    for key, value in metrics.items():
        if key in ['predictions', 'actual']:
            continue  # Skip array values
        
        formatted_value = metric_formatters.get(key, lambda v: str(v))(value)
        table.add_row(key.upper(), formatted_value)
    
    return table


def create_data_summary_table(data_info: Dict[str, Any]) -> Table:
    """Create a summary table for collected data."""
    table = Table(
        title="Data Summary",
        box=box.ROUNDED,
        show_header=False
    )
    
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in data_info.items():
        if isinstance(value, (int, float)):
            if 'date' in key.lower():
                formatted = str(value)
            elif 'price' in key.lower() or 'close' in key.lower():
                formatted = f"${value:.2f}"
            else:
                formatted = f"{value:,}"
        else:
            formatted = str(value)
        
        table.add_row(key.replace('_', ' ').title(), formatted)
    
    return table


def create_sentiment_table(results: Any) -> Table:
    """Create a table for sentiment analysis results."""
    table = Table(
        title="Sentiment Analysis Results",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan"
    )
    
    table.add_column("Text", style="white", max_width=50)
    table.add_column("Sentiment", style="cyan", justify="center")
    table.add_column("Score", style="yellow", justify="right")
    
    for _, row in results.iterrows():
        text = row.get('text', '')[:47] + "..." if len(str(row.get('text', ''))) > 50 else row.get('text', '')
        
        # Color based on sentiment
        sentiment = row.get('sentiment', 'neutral')
        if sentiment == 'positive':
            style = "green"
        elif sentiment == 'negative':
            style = "red"
        else:
            style = "yellow"
        
        score = row.get('compound', row.get('polarity', 0))
        
        table.add_row(
            text,
            f"[{style}]{sentiment.upper()}[/{style}]",
            f"{score:.3f}"
        )
    
    return table


# ============== Progress Bar ==============

def create_progress() -> Progress:
    """Create a rich progress bar instance."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        console=console,
        expand=True
    )


# ============== Confirmation ==============

def confirm(message: str, default: bool = False) -> bool:
    """Ask for user confirmation."""
    from rich.prompt import Confirm
    return Confirm.ask(message, default=default)


# ============== Input ==============

def prompt(message: str, default: str = "") -> str:
    """Prompt user for input."""
    from rich.prompt import Prompt
    return Prompt.ask(message, default=default)


def prompt_choice(message: str, choices: List[str]) -> str:
    """Prompt user to choose from a list."""
    from rich.prompt import Prompt
    return Prompt.ask(message, choices=choices, default=choices[0])
