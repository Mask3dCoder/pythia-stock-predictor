"""
Pythia CLI Interactive Mode

Rich-powered interactive experience with guided workflows,
auto-complete, and polished prompts. Inspired by Claude Code's
clean terminal UX.
"""

import sys
from typing import Dict, Any, Optional, Callable

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table, box
from rich.text import Text
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.columns import Columns

from .theme import (
    PYTHIA_THEME,
    PANEL_STYLES,
    RULE_STYLES,
    ICONS,
    LOGO_SMALL,
    style_price,
    style_change,
    format_change,
)
from .output import console


# ── Interactive Console ──────────────────────────────────────────────────────

interactive_console = Console(
    theme=PYTHIA_THEME,
    force_terminal=(sys.platform == "win32"),
    highlight=False,
    color_system="standard" if sys.platform == "win32" else "auto",
)


# ── Main Menu ────────────────────────────────────────────────────────────────

MAIN_MENU_OPTIONS = [
    ("1", "Collect Data", "Download historical stock data", "collect"),
    ("2", "Train Model", "Train a prediction model", "train"),
    ("3", "Make Predictions", "Generate price forecasts", "predict"),
    ("4", "Full Workflow", "Complete prediction pipeline", "workflow"),
    ("5", "Evaluate Model", "Check model accuracy", "evaluate"),
    ("6", "Run Backtest", "Test trading strategies", "backtest"),
    ("7", "Analyze Sentiment", "Market sentiment analysis", "sentiment"),
    ("8", "Risk Analysis", "Position sizing & risk", "risk"),
    ("9", "Compare Models", "Benchmark multiple models", "compare"),
    ("", "", "", ""),
    ("d", "Launch Dashboard", "Interactive web dashboard", "dashboard"),
    ("u", "Uncertainty Analysis", "Confidence intervals", "uncertainty"),
    ("h", "Help & Tips", "Show quick reference", "help"),
    ("q", "Quit", "Exit Pythia CLI", "quit"),
]


def show_main_menu() -> str:
    """Display the main interactive menu and return selected action."""
    interactive_console.clear()
    print(LOGO_SMALL)
    interactive_console.print()

    table = Table(
        title="Main Menu",
        title_style="heading",
        box=box.ROUNDED,
        border_style=PANEL_STYLES["brand"],
        show_header=True,
        header_style="subheading",
        pad_edge=True,
    )
    table.add_column("", style="brand", width=4, justify="center")
    table.add_column("Action", style="text.primary")
    table.add_column("Description", style="text.secondary")

    for key, name, desc, _ in MAIN_MENU_OPTIONS:
        if key:
            table.add_row(key, name, desc)
        else:
            table.add_row("", "", "")

    interactive_console.print(table)
    interactive_console.print()

    return Prompt.ask(
        f"[brand]{ICONS['chevron']}[/] Select action",
        choices=[opt[0] for opt in MAIN_MENU_OPTIONS if opt[0]],
        default="1",
        console=interactive_console,
    )


# ── Prompt Helpers ────────────────────────────────────────────────────────────

def prompt_symbol(default: str = "AAPL") -> str:
    """Prompt for stock symbol with auto-uppercase."""
    return Prompt.ask(
        f"[info]{ICONS['info']}[/] Stock symbol",
        default=default,
        console=interactive_console,
    ).strip().upper()


def prompt_model(default: str = "ensemble") -> str:
    """Prompt for model type with choices."""
    return Prompt.ask(
        f"[info]{ICONS['info']}[/] Model type",
        choices=["arima", "lstm", "gru", "ensemble", "cnn_lstm"],
        default=default,
        console=interactive_console,
    ).strip().lower()


def prompt_years(default: int = 5) -> int:
    """Prompt for years of data."""
    return IntPrompt.ask(
        f"[info]{ICONS['info']}[/] Years of historical data",
        default=default,
        console=interactive_console,
    )


def prompt_days(default: int = 7) -> int:
    """Prompt for prediction days."""
    return IntPrompt.ask(
        f"[info]{ICONS['info']}[/] Days to predict",
        default=default,
        console=interactive_console,
    )


def prompt_capital(default: float = 100000) -> float:
    """Prompt for capital amount."""
    return FloatPrompt.ask(
        f"[info]{ICONS['info']}[/] Initial capital ($)",
        default=default,
        console=interactive_console,
    )


# ── Section Display ──────────────────────────────────────────────────────────

def print_step(step: int, total: int, description: str) -> None:
    """Print a workflow step indicator."""
    interactive_console.print()
    interactive_console.print(
        Panel(
            f"[brand]Step {step}/{total}[/]\n[text.primary]{description}[/]",
            border_style=PANEL_STYLES["brand"],
            padding=(1, 2),
        )
    )


def print_step_complete(description: str) -> None:
    """Print step completion."""
    interactive_console.print(f"  [success]{ICONS['success']}[/] {description}")


def print_step_error(description: str) -> None:
    """Print step failure."""
    interactive_console.print(f"  [error]{ICONS['error']}[/] {description}")


# ── Workflows ─────────────────────────────────────────────────────────────────

def run_collect_workflow(config: Dict) -> None:
    """Interactive data collection workflow."""
    interactive_console.clear()
    interactive_console.print(Rule("Collect Stock Data", style=RULE_STYLES["accent"]))
    interactive_console.print()

    symbol = prompt_symbol()
    years = prompt_years()

    from .output import print_header, print_info

    class Args:
        pass

    args = Args()
    args.symbol = symbol
    args.years = years
    args.source = "yfinance"
    args.interactive = True

    # Import and run
    from main import collect_data
    collect_data(args, config)


def run_train_workflow(config: Dict) -> None:
    """Interactive model training workflow."""
    interactive_console.clear()
    interactive_console.print(Rule("Train Model", style=RULE_STYLES["accent"]))
    interactive_console.print()

    symbol = prompt_symbol()
    model = prompt_model()
    years = prompt_years()
    save = Confirm.ask(
        f"[info]{ICONS['info']}[/] Save model after training?",
        default=True,
        console=interactive_console,
    )

    class Args:
        pass

    args = Args()
    args.symbol = symbol
    args.model = model
    args.years = years
    args.save = save
    args.interactive = True

    from main import train_model
    train_model(args, config)


def run_predict_workflow(config: Dict) -> None:
    """Interactive prediction workflow."""
    interactive_console.clear()
    interactive_console.print(Rule("Make Predictions", style=RULE_STYLES["accent"]))
    interactive_console.print()

    symbol = prompt_symbol()
    model = prompt_model()
    days = prompt_days()
    years = prompt_years()

    class Args:
        pass

    args = Args()
    args.symbol = symbol
    args.model = model
    args.days = days
    args.years = years
    args.interactive = True

    from main import make_predictions
    make_predictions(args, config)


def run_full_workflow(config: Dict) -> None:
    """Run the complete prediction pipeline interactively."""
    interactive_console.clear()
    interactive_console.print(
        Panel(
            "[brand]Complete Prediction Workflow[/]\n\n"
            "This will guide you through the full pipeline:\n"
            "  1. Collect historical data\n"
            "  2. Train a prediction model\n"
            "  3. Generate price forecasts",
            border_style=PANEL_STYLES["brand"],
            padding=(1, 2),
        )
    )
    interactive_console.print()

    symbol = prompt_symbol()
    model = prompt_model()
    years = prompt_years()
    days = prompt_days()

    total_steps = 3

    # Step 1: Collect
    print_step(1, total_steps, f"Downloading {years} years of {symbol} data")

    class Args1:
        pass

    args1 = Args1()
    args1.symbol = symbol
    args1.years = years
    args1.source = "yfinance"
    args1.interactive = True

    try:
        from main import collect_data
        collect_data(args1, config)
        print_step_complete(f"Data collected for {symbol}")
    except Exception as e:
        print_step_error(f"Data collection failed: {e}")
        if not Confirm.ask("Continue anyway?", default=False, console=interactive_console):
            return

    # Step 2: Train
    print_step(2, total_steps, f"Training {model.upper()} model")

    class Args2:
        pass

    args2 = Args2()
    args2.symbol = symbol
    args2.model = model
    args2.years = years
    args2.save = True
    args2.interactive = True

    try:
        from main import train_model
        train_model(args2, config)
        print_step_complete(f"{model.upper()} model trained")
    except Exception as e:
        print_step_error(f"Training failed: {e}")
        if not Confirm.ask("Continue anyway?", default=False, console=interactive_console):
            return

    # Step 3: Predict
    print_step(3, total_steps, f"Generating {days}-day forecast")

    class Args3:
        pass

    args3 = Args3()
    args3.symbol = symbol
    args3.model = model
    args3.days = days
    args3.years = years
    args3.interactive = True

    try:
        from main import make_predictions
        make_predictions(args3, config)
        print_step_complete("Predictions generated")
    except Exception as e:
        print_step_error(f"Prediction failed: {e}")

    # Summary
    interactive_console.print()
    interactive_console.print(
        Panel(
            "[success]Workflow complete![/]",
            border_style=PANEL_STYLES["success"],
            padding=(1, 2),
        )
    )


def run_evaluate_workflow(config: Dict) -> None:
    """Interactive model evaluation workflow."""
    interactive_console.clear()
    interactive_console.print(Rule("Evaluate Model", style=RULE_STYLES["accent"]))
    interactive_console.print()

    symbol = prompt_symbol()
    model = prompt_model()
    years = prompt_years()

    class Args:
        pass

    args = Args()
    args.symbol = symbol
    args.model = model
    args.years = years
    args.interactive = True

    from main import evaluate_model
    evaluate_model(args, config)


def run_backtest_workflow(config: Dict) -> None:
    """Interactive backtesting workflow."""
    interactive_console.clear()
    interactive_console.print(Rule("Run Backtest", style=RULE_STYLES["accent"]))
    interactive_console.print()

    symbol = prompt_symbol()
    model = prompt_model()
    years = prompt_years()
    capital = prompt_capital()
    commission = FloatPrompt.ask(
        f"[info]{ICONS['info']}[/] Commission rate (%)",
        default=0.1,
        console=interactive_console,
    )
    stop_loss = FloatPrompt.ask(
        f"[info]{ICONS['info']}[/] Stop loss (%)",
        default=5.0,
        console=interactive_console,
    )

    class Args:
        pass

    args = Args()
    args.symbol = symbol
    args.model = model
    args.years = years
    args.capital = capital
    args.commission = commission / 100
    args.slippage = 0.0005
    args.spread = 0.0002
    args.stop_loss = stop_loss / 100
    args.take_profit = 0.10
    args.allow_short = False
    args.interactive = True

    from main import run_backtest
    run_backtest(args, config)


def run_sentiment_workflow(config: Dict) -> None:
    """Interactive sentiment analysis workflow."""
    interactive_console.clear()
    interactive_console.print(Rule("Analyze Sentiment", style=RULE_STYLES["accent"]))
    interactive_console.print()

    text = Prompt.ask(
        f"[info]{ICONS['info']}[/] Enter text (or leave blank for samples)",
        default="",
        console=interactive_console,
    )

    class Args:
        pass

    args = Args()
    args.text = text if text.strip() else None
    args.method = "vader"
    args.interactive = True

    from main import analyze_sentiment
    analyze_sentiment(args, config)


def run_risk_workflow(config: Dict) -> None:
    """Interactive risk analysis workflow."""
    interactive_console.clear()
    interactive_console.print(Rule("Risk Analysis", style=RULE_STYLES["accent"]))
    interactive_console.print()

    symbol = prompt_symbol()
    capital = prompt_capital()
    max_pos = FloatPrompt.ask(
        f"[info]{ICONS['info']}[/] Max position size (%)",
        default=20.0,
        console=interactive_console,
    )
    use_kelly = Confirm.ask(
        f"[info]{ICONS['info']}[/] Use Kelly criterion?",
        default=False,
        console=interactive_console,
    )

    class Args:
        pass

    args = Args()
    args.symbol = symbol
    args.capital = capital
    args.max_position = max_pos
    args.use_kelly = use_kelly
    args.interactive = True

    from main import run_risk
    run_risk(args, config)


def run_compare_workflow(config: Dict) -> None:
    """Interactive model comparison workflow."""
    interactive_console.clear()
    interactive_console.print(Rule("Compare Models", style=RULE_STYLES["accent"]))
    interactive_console.print()

    symbol = prompt_symbol()
    years = prompt_years()
    models = Prompt.ask(
        f"[info]{ICONS['info']}[/] Models to compare (comma-separated)",
        default="arima,lstm,gru,ensemble",
        console=interactive_console,
    )

    class Args:
        pass

    args = Args()
    args.symbol = symbol
    args.models = models
    args.years = years
    args.interactive = True

    from main import compare_models
    compare_models(args, config)


def run_uncertainty_workflow(config: Dict) -> None:
    """Interactive uncertainty analysis workflow."""
    interactive_console.clear()
    interactive_console.print(Rule("Uncertainty Analysis", style=RULE_STYLES["accent"]))
    interactive_console.print()

    symbol = prompt_symbol()
    model = prompt_model()
    days = prompt_days()
    years = prompt_years()

    class Args:
        pass

    args = Args()
    args.symbol = symbol
    args.model = model
    args.days = days
    args.years = years
    args.interactive = True

    from main import run_uncertainty
    run_uncertainty(args, config)


def show_help() -> None:
    """Display quick reference help in interactive mode."""
    interactive_console.clear()
    print(LOGO_SMALL)
    interactive_console.print()

    from .components import print_quick_start
    print_quick_start()

    interactive_console.print()
    interactive_console.print(
        Panel(
            "[text.secondary]Commands also available directly:[/]\n"
            "  [brand]pythia collect[/] [text.muted]--symbol AAPL --years 5[/]\n"
            "  [brand]pythia train[/]   [text.muted]--symbol AAPL --model ensemble[/]\n"
            "  [brand]pythia predict[/] [text.muted]--symbol AAPL --days 7[/]\n"
            "  [brand]pythia dashboard[/] [text.muted]--symbol AAPL[/]\n"
            "  [brand]pythia backtest[/] [text.muted]--symbol AAPL --capital 100000[/]",
            title="CLI Quick Reference",
            border_style=PANEL_STYLES["info"],
            padding=(1, 2),
        )
    )

    input("\nPress Enter to return to menu...")


# ── Action Dispatcher ────────────────────────────────────────────────────────

ACTION_MAP: Dict[str, Callable] = {
    "1": run_collect_workflow,
    "2": run_train_workflow,
    "3": run_predict_workflow,
    "4": run_full_workflow,
    "5": run_evaluate_workflow,
    "6": run_backtest_workflow,
    "7": run_sentiment_workflow,
    "8": run_risk_workflow,
    "9": run_compare_workflow,
    "d": lambda c: None,  # Dashboard — launch via main
    "u": run_uncertainty_workflow,
    "h": lambda c: show_help(),
}


# ── Main Interactive Loop ────────────────────────────────────────────────────

def run_interactive(config: Dict) -> int:
    """Run the main interactive mode loop."""
    try:
        while True:
            action = show_main_menu()

            if action == "q":
                interactive_console.clear()
                interactive_console.print()
                interactive_console.print(
                    Panel(
                        "[text.secondary]Thank you for using Pythia Stock Predictor![/]",
                        border_style=PANEL_STYLES["brand"],
                        padding=(1, 2),
                    )
                )
                interactive_console.print()
                return 0

            if action == "d":
                # Launch dashboard in subprocess
                import subprocess
                import os

                symbol = prompt_symbol()
                interactive_console.print()
                interactive_console.print(
                    "[info]Launching dashboard...[/]\n"
                    "[text.muted]Open http://localhost:8501 in your browser[/]\n"
                    "[text.muted]Press Ctrl+C to stop the dashboard and return.[/]"
                )

                cmd = [sys.executable, "-m", "streamlit", "run",
                       "src/visualization/dashboard.py"]
                try:
                    subprocess.run(cmd, cwd=os.getcwd())
                except KeyboardInterrupt:
                    interactive_console.print("\n[warning]Dashboard stopped.[/]")
                continue

            handler = ACTION_MAP.get(action)
            if handler:
                try:
                    handler(config)
                except KeyboardInterrupt:
                    interactive_console.print("\n[warning]Operation cancelled.[/]")
                except Exception as e:
                    interactive_console.print(f"\n[error]Error: {e}[/]")

                # Ask to continue
                interactive_console.print()
                interactive_console.print(Rule(style=RULE_STYLES["section"]))
                if not Confirm.ask(
                    "Return to main menu?",
                    default=True,
                    console=interactive_console,
                ):
                    interactive_console.print("[text.secondary]Goodbye![/]")
                    return 0

    except KeyboardInterrupt:
        interactive_console.print("\n[warning]Exiting. Goodbye![/]")
        return 130
    except EOFError:
        return 0

    return 0
