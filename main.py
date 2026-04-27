"""
Pythia Stock Prediction CLI — Main Entry Point

A polished command-line interface for stock market prediction:
collect data, train models, generate forecasts, analyze sentiment,
run backtests, and more.
"""

import sys
import argparse
import logging
import textwrap
from pathlib import Path
from typing import List, Tuple, Any, Optional

import yaml
import numpy as np
from rich.table import Table

# CLI modules — themed output and interactive mode
from src.cli.output import (
    console,
    print_success,
    print_error,
    print_warning,
    print_info,
    print_header,
    print_welcome,
    create_predictions_table,
    create_metrics_table,
    create_data_summary_table,
    create_sentiment_table,
    create_progress,
    print_panel,
    display_welcome_full,
    display_error,
)
from src.cli.components import (
    print_command_header,
    print_section_header,
    price_card,
    metrics_grid,
    metric_cards,
    styled_table,
    prediction_table as styled_prediction_table,
    status_ok,
    status_fail,
    status_warn,
    status_info,
    result_panel,
    success_panel,
)
from src.cli.validators import validate_all


# ============== Global State ==============

# Verbose mode flag (set during argument parsing)
VERBOSE_MODE = False


# ============== Logging Setup ==============


def setup_logging(level: int = logging.INFO) -> None:
    """Setup logging configuration."""
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=level,
        force=True,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(logs_dir / "app.log"),
        ],
    )


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print_warning(f"Config file not found: {config_path}. Using defaults.")
        return {}
    except yaml.YAMLError as e:
        print_error(f"Invalid YAML in config file: {e}")
        return {}


# ============== Command Functions ==============


def collect_data(args: argparse.Namespace, config: dict) -> Any:
    """Collect historical stock data."""
    from src.data.collector import StockDataCollector
    from src.data.preprocessor import DataPreprocessor

    print_command_header("Collect Stock Data", f"Symbol: {args.symbol}  |  Years: {args.years}")

    collector = StockDataCollector(config.get("data", {}))
    preprocessor = DataPreprocessor(config.get("indicators", {}))

    with create_progress() as progress:
        task = progress.add_task(f"Downloading {args.years}y of {args.symbol} data...", total=100)
        data = collector.download_yahoo_data(symbol=args.symbol, years=args.years)
        progress.update(task, completed=50)

        if data.empty:
            status_fail(f"No data collected for {args.symbol}")
            return None

        progress.update(task, description="Preprocessing data...", completed=60)
        data = preprocessor.clean_data(data)
        data = preprocessor.add_all_indicators(data)
        progress.update(task, completed=85)

        filepath = collector.save_data(data, args.symbol, "historical")
        progress.update(task, completed=100)

    console.print()
    console.print(create_data_summary_table({
        "Records": len(data),
        "Date Range": f"{data.index[0].date()} → {data.index[-1].date()}",
        "Latest Close": f"${data['close'].iloc[-1]:,.2f}",
    }, title=f"{args.symbol} Data Summary"))
    status_ok(f"Saved to {filepath}")

    return data


def train_model(args: argparse.Namespace, config: dict) -> Any:
    """Train prediction model."""
    from src.models.predictor import StockPredictor

    print_command_header("Train Model", f"Symbol: {args.symbol}  |  Model: {args.model.upper()}  |  Years: {args.years}")

    predictor = StockPredictor(args.symbol, args.model, config)

    with create_progress() as progress:
        task = progress.add_task("Loading data...", total=100)
        predictor.load_data(years=args.years)
        progress.update(task, completed=30)

        progress.update(task, description=f"Training {args.model.upper()} model...", completed=40)
        result = predictor.train()
        progress.update(task, completed=100)

    console.print()
    console.print(metrics_grid({
        "Model Type": result["model_type"],
        "Data Points": result["data_points"],
    }, title="Training Results"))

    if args.save:
        model_path = predictor.save_model()
        status_ok(f"Model saved to {model_path}")

    return predictor


def make_predictions(args: argparse.Namespace, config: dict) -> Tuple[Any, Any]:
    """Make stock predictions."""
    from src.models.predictor import StockPredictor

    print_command_header("Generate Predictions", f"Symbol: {args.symbol}  |  Model: {args.model.upper()}  |  Horizon: {args.days} days")

    predictor = StockPredictor(args.symbol, args.model, config)

    with create_progress() as progress:
        task = progress.add_task("Loading and training...", total=100)
        predictor.load_data(years=args.years)
        progress.update(task, completed=35)

        progress.update(task, description="Training model...", completed=45)
        predictor.train()
        progress.update(task, completed=70)

        progress.update(task, description=f"Generating {args.days}-day forecast...", completed=80)
        predictions = predictor.predict(args.days)
        progress.update(task, completed=100)

    preds = predictions.get("predictions", [])
    lower_bound = predictions.get("lower_bound", preds)
    upper_bound = predictions.get("upper_bound", preds)

    pred_list = preds.tolist() if hasattr(preds, "tolist") else list(preds)
    low_list = lower_bound.tolist() if hasattr(lower_bound, "tolist") else list(lower_bound)
    high_list = upper_bound.tolist() if hasattr(upper_bound, "tolist") else list(upper_bound)

    console.print()
    console.print(styled_prediction_table(
        predictions=pred_list,
        lower_bound=low_list,
        upper_bound=high_list,
        title=f"{args.symbol} — {args.days}-Day Forecast",
    ))

    current = predictor.get_current_price()
    if current:
        pred = pred_list[0]
        change_pct = ((pred - current) / current) * 100
        console.print()
        console.print(price_card(current, pred, change_pct))

    return predictor, predictions


def run_dashboard(args: argparse.Namespace, config: dict) -> None:
    """Run Streamlit dashboard."""
    import subprocess
    import os

    print_header("Starting Dashboard")

    print_info("Starting Streamlit dashboard...")
    print_info("If the dashboard doesn't open automatically, run:")
    console.print("  [cyan]streamlit run src/visualization/dashboard.py[/cyan]")
    console.print(
        "\nOr open [link=http://localhost:8501]http://localhost:8501[/link] in your browser"
    )

    # Change to project directory and run streamlit
    cmd = [sys.executable, "-m", "streamlit", "run", "src/visualization/dashboard.py"]
    subprocess.run(cmd, cwd=os.getcwd())


def analyze_sentiment(args: argparse.Namespace, config: dict) -> None:
    """Analyze sentiment from news/text."""
    from src.sentiment.analyzer import SentimentAnalyzer

    print_command_header("Sentiment Analysis", f"Method: {args.method.upper()}")

    if hasattr(args, 'text') and args.text:
        texts = [args.text]
    else:
        texts = [
            "Stock market reaches all-time high amid positive economic data",
            "Company reports strong quarterly earnings, beating expectations",
            "Market faces uncertainty due to geopolitical concerns",
            "Analysts downgrade stock rating citing slowdown risks",
            "CEO announces strategic partnership, investors react positively",
        ]

    analyzer = SentimentAnalyzer(config.get("sentiment", {}))

    with create_progress() as progress:
        task = progress.add_task(f"Analyzing {len(texts)} texts...", total=len(texts))
        results = analyzer.analyze_batch(texts, method=args.method)
        progress.update(task, completed=100)

    console.print()
    console.print(create_sentiment_table(results))

    summary = analyzer.get_sentiment_summary(results)
    console.print()
    console.print(success_panel(
        f"{summary.get('positive_pct', 0):.1f}% Positive  |  "
        f"{summary.get('neutral_pct', 0):.1f}% Neutral  |  "
        f"{summary.get('negative_pct', 0):.1f}% Negative",
        title="Sentiment Breakdown",
    ))


def evaluate_model(args: argparse.Namespace, config: dict) -> None:
    """Evaluate trained model."""
    from src.models.predictor import StockPredictor

    print_command_header("Evaluate Model", f"Symbol: {args.symbol}  |  Model: {args.model.upper()}")

    predictor = StockPredictor(args.symbol, args.model, config)

    with create_progress() as progress:
        task = progress.add_task("Loading data...", total=100)
        predictor.load_data(years=args.years)
        progress.update(task, completed=30)
        progress.update(task, description="Training model...", completed=40)
        predictor.train()
        progress.update(task, completed=60)
        progress.update(task, description="Evaluating...", completed=70)
        metrics = predictor.evaluate()
        progress.update(task, completed=100)

    console.print()
    console.print(metrics_grid(metrics, title=f"{args.symbol} — {args.model.upper()} Performance"))
    console.print()


def run_backtest(args: argparse.Namespace, config: dict) -> None:
    """Run backtesting with transaction costs."""
    from src.models.predictor import StockPredictor
    from src.backtest.engine import BacktestEngine, TransactionCosts

    print_command_header("Run Backtest", f"Symbol: {args.symbol}  |  Model: {args.model.upper()}  |  Capital: ${args.capital:,.0f}")

    predictor = StockPredictor(args.symbol, args.model, config)

    costs = TransactionCosts(
        commission_pct=args.commission,
        slippage_pct=args.slippage,
        spread_pct=args.spread,
    )

    engine = BacktestEngine(
        initial_capital=args.capital,
        transaction_costs=costs,
        stop_loss_pct=args.stop_loss if args.stop_loss > 0 else None,
        take_profit_pct=args.take_profit if args.take_profit > 0 else None,
        allow_shorting=args.allow_short,
    )

    with create_progress() as progress:
        task = progress.add_task("Loading data...", total=100)
        predictor.load_data(years=args.years)
        progress.update(task, completed=20)

        progress.update(task, description="Training model...", completed=30)
        predictor.train()
        progress.update(task, completed=50)

        if predictor.data is None:
            status_fail("No data loaded for backtesting")
            return
        data = predictor.data.dropna()

        progress.update(task, description="Generating signals...", completed=60)
        predictions = predictor.predict(len(data))
        pred_values = predictions["predictions"]
        actual_prices = data["close"].values[-len(pred_values):]
        signals = np.sign(pred_values - actual_prices)

        progress.update(task, description="Running backtest simulation...", completed=75)
        prices = data["close"]
        results = engine.run(prices, signals)
        progress.update(task, completed=100)

    if not results:
        status_fail("Backtest produced no results")
        return

    console.print()
    total_ret = results.get('total_return', 0) * 100
    console.print(metrics_grid({
        "Total Return": f"{total_ret:+.2f}%",
        "Annual Return": f"{results.get('annual_return', 0) * 100:+.2f}%",
        "Sharpe Ratio": f"{results.get('sharpe_ratio', 0):.2f}",
        "Max Drawdown": f"{results.get('max_drawdown', 0) * 100:.2f}%",
        "Win Rate": f"{results.get('win_rate', 0) * 100:.1f}%",
        "Total Trades": results.get("total_trades", 0),
        "Final Capital": f"${results.get('final_capital', 0):,.2f}",
    }, title=f"Backtest Results — {args.symbol}"))
    console.print()


def run_optimize(args: argparse.Namespace, config: dict) -> None:
    """Run Bayesian hyperparameter optimization."""
    from src.optimization.hyperopt import HyperparameterOptimizer
    from src.data.collector import StockDataCollector

    print_header(f"Optimizing {args.model.upper()} Model for {args.symbol}")

    # Collect data
    print_info("Collecting data...")
    collector = StockDataCollector(config.get("data", {}))
    data = collector.download_yahoo_data(args.symbol, years=args.years)

    # Prepare features
    if args.model == "lstm":
        from src.data.preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor(config.get("indicators", {}))
        data = preprocessor.clean_data(data)
        data = preprocessor.add_all_indicators(data)

        # Create sequences
        close_prices = data["close"].ffill().values
        seq_len = args.sequence_length

        X, y = [], []
        for i in range(len(close_prices) - seq_len):
            X.append(close_prices[i : i + seq_len])
            y.append(close_prices[i + seq_len])

        X = np.array(X)
        y = np.array(y)

        # Run optimization
        optimizer = HyperparameterOptimizer(n_trials=args.trials)

        print_info(f"Running {args.trials} optimization trials...")
        results = optimizer.optimize_lstm(X, y, epochs=args.epochs)

        print_success("Optimization Complete!")

        best_info = {
            "Best Params": str(results.get("best_params", {})),
            "Best Value": f"{results.get('best_value', 0):.4f}",
        }

        table = create_data_summary_table(best_info, title="Optimization Results")
        console.print(table)
    else:
        print_warning("Optimization currently supports LSTM model only")


def run_validate(args: argparse.Namespace, config: dict) -> None:
    """Run walk-forward validation."""
    from src.models.predictor import StockPredictor
    from src.validation.validators import WalkForwardValidator

    print_header(f"Walk-Forward Validation for {args.symbol}")

    # Initialize predictor
    predictor = StockPredictor(args.symbol, args.model, config)

    # Load data
    print_info("Loading data...")
    predictor.load_data(years=args.years)

    # Get data for validation
    if predictor.data is None:
        print_error("No data loaded for validation")
        return
    data = predictor.data.dropna()
    close_prices = data["close"]

    # Create simple features (price sequences) for walk-forward
    seq_len = 60
    X, y = [], []
    for i in range(len(close_prices) - seq_len):
        X.append(close_prices.iloc[i : i + seq_len].values)
        y.append(close_prices.iloc[i + seq_len])

    X = np.array(X)
    y = np.array(y)

    # Run walk-forward validation using sklearn-compatible linear model
    print_info("Running walk-forward validation...")

    validator = WalkForwardValidator(
        train_size=args.train_size, test_size=args.test_size
    )

    from sklearn.linear_model import LinearRegression

    wf_model = LinearRegression()
    results = validator.evaluate(wf_model, X, y, verbose=False)

    # Also train and evaluate the specified model
    print_info(f"Training {args.model} model for model-specific metrics...")
    try:
        predictor.train()
        model_metrics = predictor.evaluate()
    except Exception as e:
        print_warning(f"Could not train {args.model} model: {e}")
        model_metrics = {}

    print_success("Validation Complete!")

    val_info = {
        "Walk-Forward Folds": results.get("n_folds", 0),
        "WF Avg MAE": f"{results['metrics'].get('mae', {}).get('mean', 0):.4f}",
        "WF Avg RMSE": f"{results['metrics'].get('rmse', {}).get('mean', 0):.4f}",
        "WF Avg R2": f"{results['metrics'].get('r2', {}).get('mean', 0):.4f}",
    }
    if model_metrics:
        val_info["Model MAE"] = f"{model_metrics.get('mae', 0):.4f}"
        val_info["Model RMSE"] = f"{model_metrics.get('rmse', 0):.4f}"
        val_info["Model R2"] = f"{model_metrics.get('r2', 0):.4f}"

    table = create_data_summary_table(val_info, title="Validation Results")
    console.print(table)


def run_batch(args: argparse.Namespace, config: dict) -> None:
    """Process multiple symbols."""
    from src.models.predictor import StockPredictor
    from src.data.collector import StockDataCollector

    symbols = [s.strip() for s in args.symbols.split(",")]
    print_header(f"Batch Processing: {', '.join(symbols)}")

    results = []

    for symbol in symbols:
        print_info(f"\nProcessing {symbol}...")

        try:
            predictor = StockPredictor(symbol, args.model, config)
            predictor.load_data(years=args.years)
            predictor.train()

            predictions = predictor.predict(args.days)

            results.append(
                {
                    "symbol": symbol,
                    "current_price": predictor.get_current_price(),
                    "prediction": predictions["predictions"][0],
                    "change": predictions["predictions"][0]
                    - predictor.get_current_price(),
                    "status": "success",
                }
            )
        except Exception as e:
            results.append({"symbol": symbol, "status": "failed", "error": str(e)})

    # Display results
    print_success("\nBatch Processing Complete!")

    # Create results table
    table = Table(title="Batch Results")
    table.add_column("Symbol", style="cyan")
    table.add_column("Current Price", style="white", justify="right")
    table.add_column("Prediction", style="green", justify="right")
    table.add_column("Change", style="yellow", justify="right")
    table.add_column("Status", style="magenta")

    for r in results:
        if r["status"] == "success":
            table.add_row(
                r["symbol"],
                f"${r['current_price']:.2f}",
                f"${r['prediction']:.2f}",
                f"{r['change']:+.2f}",
                "OK",
            )
        else:
            table.add_row(r["symbol"], "-", "-", "-", "FAILED")

    console.print(table)


def run_uncertainty(args: argparse.Namespace, config: dict) -> None:
    """Get predictions with uncertainty intervals."""
    from src.models.predictor import StockPredictor

    print_command_header("Uncertainty Analysis", f"Symbol: {args.symbol}  |  Model: {args.model.upper()}  |  Horizon: {args.days} days")

    predictor = StockPredictor(args.symbol, args.model, config)

    with create_progress() as progress:
        task = progress.add_task("Loading and training...", total=100)
        predictor.load_data(years=args.years)
        progress.update(task, completed=30)
        predictor.train()
        progress.update(task, completed=60)

        if predictor.data is None:
            status_fail("No data loaded")
            return

        progress.update(task, description="Generating confidence intervals...", completed=75)
        predictions = predictor.predict(args.days, include_confidence=True)
        progress.update(task, completed=100)

    pred_values = predictions["predictions"]
    lower = predictions.get("lower_bound", pred_values * 0.95)
    upper = predictions.get("upper_bound", pred_values * 1.05)

    console.print()
    console.print(styled_prediction_table(
        predictions=pred_values.tolist() if hasattr(pred_values, "tolist") else list(pred_values),
        lower_bound=lower.tolist() if hasattr(lower, "tolist") else list(lower),
        upper_bound=upper.tolist() if hasattr(upper, "tolist") else list(upper),
        title=f"{args.symbol} — Forecast with 95% Confidence",
    ))


def run_risk(args: argparse.Namespace, config: dict) -> None:
    """Calculate risk metrics and position sizing."""
    from src.backtest.risk_manager import RiskManager
    from src.data.collector import StockDataCollector

    print_command_header("Risk Analysis", f"Symbol: {args.symbol}  |  Capital: ${args.capital:,.0f}")

    collector = StockDataCollector(config.get("data", {}))
    data = collector.download_yahoo_data(args.symbol, years=1)

    returns = data["close"].pct_change().dropna()
    volatility = returns.std() * np.sqrt(252)
    current_price = data["close"].iloc[-1]

    risk_mgr = RiskManager(
        max_position_pct=args.max_position / 100, use_kelly=args.use_kelly
    )

    risk = risk_mgr.calculate_position_size(
        price=current_price,
        portfolio_value=args.capital,
        volatility=volatility,
        historical_returns=returns,
    )

    console.print()

    # Display key metrics as cards
    console.print(metrics_grid({
        "Current Price": f"${current_price:,.2f}",
        "Annual Volatility": f"{volatility * 100:.2f}%",
        "Position Size": f"{risk.position_size} shares",
        "Position Value": f"${risk.position_size * current_price:,.2f}",
        "Stop Loss": f"{risk.stop_loss * 100:.2f}%",
        "Take Profit": f"{risk.take_profit * 100:.2f}%",
        "Kelly Fraction": f"{risk.kelly_fraction * 100:.1f}%",
        "Confidence": f"{risk.confidence * 100:.1f}%",
    }, title=f"Risk Metrics — {args.symbol}"))

    console.print()


def compare_models(args: argparse.Namespace, config: dict) -> None:
    """Compare multiple models."""
    from src.models.predictor import StockPredictor

    print_command_header("Model Comparison", f"Symbol: {args.symbol}  |  Years: {args.years}")

    models = (
        args.models.split(",") if args.models else ["arima", "lstm", "gru", "ensemble"]
    )

    results = []
    for model_type in models:
        model_type = model_type.strip()
        status_info(f"Evaluating {model_type}...")

        try:
            predictor = StockPredictor(args.symbol, model_type, config)
            predictor.load_data(years=args.years)
            predictor.train()
            metrics = predictor.evaluate()
            results.append({
                "Model": model_type.upper(),
                "MAE": metrics.get("mae", 0),
                "RMSE": metrics.get("rmse", 0),
                "MAPE": f"{metrics.get('mape', 0):.2f}%",
                "R²": metrics.get("r2", 0),
            })
        except Exception as e:
            status_warn(f"Skipping {model_type}: {e}")

    if not results:
        status_fail("No models could be evaluated")
        return

    console.print()
    # Build rows for styled table
    columns = ["Model", "MAE", "RMSE", "MAPE", "R²"]
    rows = [[r[c] for c in columns] for r in results]
    console.print(styled_table(columns, rows, title=f"Model Comparison — {args.symbol}"))
    console.print()


# ============== Argument Parser Setup ==============


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with comprehensive CLI help."""

    # Main parser with raw description formatter
    parser = argparse.ArgumentParser(
        prog="stock-prediction",
        description=textwrap.dedent("""
            Stock Market Prediction CLI
            ============================
            
            A comprehensive command-line interface for stock market prediction.
            Collect data, train models, make predictions, and analyze market sentiment.
            
            Use this tool to:
              • Download and preprocess historical stock data
              • Train machine learning models (ARIMA, LSTM, GRU, Ensemble, CNN-LSTM)
              • Generate predictions for future stock prices
              • Evaluate model performance with various metrics
              • Analyze market sentiment from news and text
              • Visualize results in an interactive dashboard
              • Run backtesting with transaction costs
              • Optimize hyperparameters with Bayesian optimization
              • Validate with walk-forward cross-validation
              • Calculate risk and position sizing
              • Compare multiple models
        """).strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            ========================================
            Global Options
            ========================================
            
            These options work with any command:
            
              --verbose, -V    Enable verbose debug output
              --config, -c     Path to configuration file (default: config.yaml)
              --help, -h       Show this help message
            
            ========================================
            Examples
            ========================================
            
            # Show this help
            python main.py --help
            
            # Show help for a specific command
            python main.py collect --help
            python main.py train --help
            
            # Basic workflow
            python main.py collect --symbol AAPL --years 5
            python main.py train --symbol AAPL --model ensemble
            python main.py predict --symbol AAPL --days 7
            
            # Advanced options
            python main.py train --symbol AAPL --model lstm --save
            python main.py predict --symbol MSFT --model arima --days 30
            
            # Run dashboard
            python main.py dashboard --symbol AAPL --model ensemble
            
            # Analyze sentiment
            python main.py sentiment --method vader
            
            # Run backtesting
            python main.py backtest --symbol AAPL --capital 100000
            
            # Optimize hyperparameters
            python main.py optimize --symbol AAPL --trials 20
            
            # Validate with walk-forward
            python main.py validate --symbol AAPL
            
            # Batch processing
            python main.py batch --symbols AAPL,MSFT,GOOGL
            
            # Get uncertainty intervals
            python main.py uncertainty --symbol AAPL --days 7
            
            # Risk analysis
            python main.py risk --symbol AAPL --capital 100000
            
            # Compare models
            python main.py compare --symbol AAPL
            
            ========================================
            Related Commands
            ========================================
            
            data     - Collect and manage stock data (alias: collect)
            train    - Train prediction models
            predict  - Make stock price predictions
            evaluate - Evaluate model performance
            sentiment - Analyze market sentiment
            dashboard - Launch interactive dashboard
            backtest - Run backtesting with transaction costs
            optimize - Bayesian hyperparameter optimization
            validate - Walk-forward cross-validation
            batch    - Process multiple symbols
            uncertainty - Get prediction confidence intervals
            risk     - Calculate position sizing
            compare  - Compare multiple models
            
            For more information, visit the documentation.
        """).strip(),
    )

    # Note: argparse automatically adds -h/--help
    # We handle help display in main() function

    # Global options group
    global_group = parser.add_argument_group(
        "Global Options", "Options that work with any command"
    )
    global_group.add_argument(
        "-V",
        "--verbose",
        action="store_true",
        dest="verbose",
        help="Enable verbose/debug output for detailed logging",
    )
    global_group.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        metavar="FILE",
        help="Path to configuration file (default: config.yaml)",
    )
    global_group.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        dest="interactive",
        help="Run in interactive mode with guided prompts",
    )

    # Create subparsers
    subparsers = parser.add_subparsers(
        dest="command",
        title="Commands",
        description="Available commands for stock prediction",
        help="Run with COMMAND --help for detailed usage",
    )

    # ============== COLLECT Command ==============
    collect_parser = subparsers.add_parser(
        "collect",
        aliases=["data"],
        help="Collect historical stock data from Yahoo Finance",
        description=textwrap.dedent("""
            Collect Historical Stock Data
            =============================
            
            Downloads historical stock data from Yahoo Finance and saves it
            for later use in training and prediction.
        """).strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            ========================================
            Examples
            ========================================
            
            # Collect 5 years of Apple data
            python main.py collect --symbol AAPL --years 5
            
            # Use short alias
            python main.py data --symbol MSFT --years 3
            
            # Collect with custom years
            python main.py collect --symbol GOOGL --years 10
            
            Related Commands:
              train   - Train model with collected data
              predict - Make predictions with collected data
        """).strip(),
    )

    # Collect arguments
    collect_group = collect_parser.add_argument_group(
        "Required Arguments", "Must be provided"
    )
    collect_group.add_argument(
        "--symbol",
        "-s",
        type=str,
        required=True,
        metavar="SYMBOL",
        help="Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)",
    )

    collect_opt = collect_parser.add_argument_group(
        "Optional Arguments", "May be specified"
    )
    collect_opt.add_argument(
        "--years",
        "-y",
        type=int,
        default=5,
        metavar="N",
        help="Number of years of historical data to collect (default: 5)",
    )
    collect_parser.set_defaults(func=collect_data)

    # ============== TRAIN Command ==============
    train_parser = subparsers.add_parser(
        "train",
        aliases=["model"],
        help="Train a prediction model on historical data",
        description=textwrap.dedent("""
            Train Prediction Model
            ======================
            
            Train a machine learning model using historical stock data.
            Supports multiple model types for different prediction approaches.
        """).strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            ========================================
            Model Types
            ========================================
            
            arima     - AutoRegressive Integrated Moving Average
                       Statistical model for time series analysis
            lstm      - Long Short-Term Memory neural network
                       Deep learning model for sequential data
            gru       - Gated Recurrent Unit
                       Simplified deep learning model
            ensemble  - Combined model using multiple approaches
                       Best for balanced predictions
            
            ========================================
            Examples
            ========================================
            
            # Train ensemble model (default)
            python main.py train --symbol AAPL
            
            # Train ARIMA model
            python main.py train --symbol AAPL --model arima
            
            # Train LSTM with custom data range
            python main.py train --symbol MSFT --model lstm --years 3
            
            # Save trained model
            python main.py train --symbol AAPL --model ensemble --save
            
            Related Commands:
              predict - Use trained model for predictions
              evaluate - Evaluate trained model performance
        """).strip(),
    )

    # Train arguments
    train_group = train_parser.add_argument_group(
        "Required Arguments", "Must be provided"
    )
    train_group.add_argument(
        "--symbol",
        "-s",
        type=str,
        required=True,
        metavar="SYMBOL",
        help="Stock ticker symbol to train on",
    )

    train_opt = train_parser.add_argument_group(
        "Model Options", "Configure the model type and behavior"
    )
    train_opt.add_argument(
        "--model",
        "-m",
        type=str,
        default="ensemble",
        choices=["arima", "lstm", "gru", "cnn_lstm", "ensemble"],
        metavar="TYPE",
        help="Model type to use (default: ensemble)",
    )
    train_opt.add_argument(
        "--years",
        "-y",
        type=int,
        default=5,
        metavar="N",
        help="Years of historical data to use (default: 5)",
    )
    train_opt.add_argument(
        "--save", action="store_true", help="Save trained model to disk for later use"
    )
    train_parser.set_defaults(func=train_model)

    # ============== PREDICT Command ==============
    predict_parser = subparsers.add_parser(
        "predict",
        aliases=["forecast", "pred"],
        help="Generate stock price predictions",
        description=textwrap.dedent("""
            Make Stock Predictions
            =======================
            
            Generate predictions for future stock prices using trained models.
            Returns predictions with confidence intervals.
        """).strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            ========================================
            Examples
            ========================================
            
            # Predict next 7 days (default)
            python main.py predict --symbol AAPL
            
            # Predict 30 days ahead
            python main.py predict --symbol MSFT --days 30
            
            # Use specific model
            python main.py predict --symbol AAPL --model lstm --days 14
            
            # Short alias
            python main.py pred --symbol GOOGL --days 7
            
            Output:
              - Table with daily predictions
              - Confidence intervals (lower/upper bounds)
              - Current price and expected change
            
            Related Commands:
              train   - Train the model first
              evaluate - Check model accuracy
        """).strip(),
    )

    # Predict arguments
    predict_group = predict_parser.add_argument_group(
        "Required Arguments", "Must be provided"
    )
    predict_group.add_argument(
        "--symbol",
        "-s",
        type=str,
        required=True,
        metavar="SYMBOL",
        help="Stock ticker symbol to predict",
    )

    predict_opt = predict_parser.add_argument_group(
        "Prediction Options", "Configure prediction behavior"
    )
    predict_opt.add_argument(
        "--model",
        "-m",
        type=str,
        default="ensemble",
        choices=["arima", "lstm", "gru", "cnn_lstm", "ensemble"],
        metavar="TYPE",
        help="Model to use for prediction (default: ensemble)",
    )
    predict_opt.add_argument(
        "--years",
        "-y",
        type=int,
        default=5,
        metavar="N",
        help="Years of data to use for training (default: 5)",
    )
    predict_opt.add_argument(
        "--days",
        "-d",
        type=int,
        default=7,
        metavar="N",
        help="Number of days to predict ahead (default: 7)",
    )
    predict_parser.set_defaults(func=make_predictions)

    # ============== DASHBOARD Command ==============
    dashboard_parser = subparsers.add_parser(
        "dashboard",
        aliases=["ui", "web"],
        help="Launch interactive Streamlit dashboard",
        description=textwrap.dedent("""
            Launch Interactive Dashboard
            ============================
            
            Opens a web-based dashboard for interactive stock analysis
            and visualization using Streamlit.
        """).strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            ========================================
            Examples
            ========================================
            
            # Launch dashboard with default settings
            python main.py dashboard
            
            # Specify stock symbol
            python main.py dashboard --symbol AAPL
            
            # Use specific model
            python main.py dashboard --symbol MSFT --model lstm
            
            # Use alias
            python main.py ui --symbol GOOGL
            
            Note:
              Opens in browser at http://localhost:8501
              Close with Ctrl+C to stop the server
        """).strip(),
    )

    # Dashboard arguments
    dash_group = dashboard_parser.add_argument_group(
        "Dashboard Options", "Configure dashboard behavior"
    )
    dash_group.add_argument(
        "--symbol",
        "-s",
        type=str,
        default="AAPL",
        metavar="SYMBOL",
        help="Default stock symbol to display (default: AAPL)",
    )
    dash_group.add_argument(
        "--model",
        "-m",
        type=str,
        default="ensemble",
        choices=["arima", "lstm", "gru", "cnn_lstm", "ensemble"],
        metavar="TYPE",
        help="Default model type (default: ensemble)",
    )
    dash_group.add_argument(
        "--years",
        "-y",
        type=int,
        default=5,
        metavar="N",
        help="Years of data to load (default: 5)",
    )
    dashboard_parser.set_defaults(func=run_dashboard)

    # ============== SENTIMENT Command ==============
    sentiment_parser = subparsers.add_parser(
        "sentiment",
        aliases=["sent", "analyze"],
        help="Analyze market sentiment from text",
        description=textwrap.dedent("""
            Analyze Market Sentiment
            =========================
            
            Analyze the sentiment of news articles, social media posts,
            or custom text to gauge market mood.
        """).strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            ========================================
            Analysis Methods
            ========================================
            
            vader     - VADER (Valence Aware Dictionary and sEntiment Reasoner)
                       Best for social media and informal text
            textblob  - TextBlob sentiment analysis
                       Good for general-purpose text
            combined  - Use both methods and average results
                       Most robust for mixed content
            
            ========================================
            Examples
            ========================================
            
            # Analyze with default method
            python main.py sentiment
            
            # Use TextBlob method
            python main.py sentiment --method textblob
            
            # Use combined analysis
            python main.py sentiment --method combined
            
            # Short alias
            python main.py sent --method vader
            
            Output:
              - Sentiment scores for each text
              - Summary statistics (positive/neutral/negative %)
            
            Note:
              Currently uses sample texts. Use API integration for real news.
        """).strip(),
    )

    # Sentiment arguments
    sent_group = sentiment_parser.add_argument_group(
        "Sentiment Options", "Configure sentiment analysis"
    )
    sent_group.add_argument(
        "--method",
        "-m",
        type=str,
        default="vader",
        choices=["vader", "textblob", "combined"],
        metavar="METHOD",
        help="Sentiment analysis method (default: vader)",
    )
    sentiment_parser.set_defaults(func=analyze_sentiment)

    # ============== EVALUATE Command ==============
    eval_parser = subparsers.add_parser(
        "evaluate",
        aliases=["eval", "metrics", "test"],
        help="Evaluate model performance with metrics",
        description=textwrap.dedent("""
            Evaluate Model Performance
            ==========================
            
            Evaluate a trained model using various metrics including
            RMSE, MAE, MAPE, and R-squared.
        """).strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            ========================================
            Metrics Used
            ========================================
            
            RMSE   - Root Mean Square Error
                     Measures prediction accuracy (lower is better)
            MAE    - Mean Absolute Error
                     Average absolute difference (lower is better)
            MAPE   - Mean Absolute Percentage Error
                     Percentage error (lower is better)
            R²     - R-squared coefficient
                     How well model fits data (higher is better)
            
            ========================================
            Examples
            ========================================
            
            # Evaluate with default model
            python main.py evaluate --symbol AAPL
            
            # Evaluate specific model
            python main.py eval --symbol MSFT --model arima
            
            # Use custom data range
            python main.py evaluate --symbol GOOGL --years 3
            
            # Short alias
            python main.py metrics --symbol AAPL
            
            Related Commands:
              train   - Train model before evaluation
              predict - Make predictions after evaluation
        """).strip(),
    )

    # Evaluate arguments
    eval_group = eval_parser.add_argument_group(
        "Required Arguments", "Must be provided"
    )
    eval_group.add_argument(
        "--symbol",
        "-s",
        type=str,
        required=True,
        metavar="SYMBOL",
        help="Stock ticker symbol to evaluate",
    )

    eval_opt = eval_parser.add_argument_group(
        "Evaluation Options", "Configure evaluation behavior"
    )
    eval_opt.add_argument(
        "--model",
        "-m",
        type=str,
        default="ensemble",
        choices=["arima", "lstm", "gru", "cnn_lstm", "ensemble"],
        metavar="TYPE",
        help="Model type to evaluate (default: ensemble)",
    )
    eval_opt.add_argument(
        "--years",
        "-y",
        type=int,
        default=5,
        metavar="N",
        help="Years of data to use (default: 5)",
    )
    eval_parser.set_defaults(func=evaluate_model)

    # ============== BACKTEST Command ==============
    backtest_parser = subparsers.add_parser(
        "backtest",
        aliases=["bt"],
        help="Run backtesting with transaction costs",
        description=textwrap.dedent("""
            Run Backtesting
            ===============
            
            Backtest trading strategies with realistic transaction costs
            including commission, slippage, and spread.
        """).strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    backtest_group = backtest_parser.add_argument_group(
        "Required Arguments", "Must be provided"
    )
    backtest_group.add_argument(
        "--symbol", "-s", type=str, required=True, metavar="SYMBOL", help="Stock ticker"
    )

    backtest_opt = backtest_parser.add_argument_group(
        "Optional Arguments", "May be specified"
    )
    backtest_opt.add_argument(
        "--model",
        "-m",
        type=str,
        default="ensemble",
        choices=["arima", "lstm", "gru", "cnn_lstm", "ensemble"],
        help="Model type",
    )
    backtest_opt.add_argument(
        "--years", "-y", type=int, default=5, help="Years of data"
    )
    backtest_opt.add_argument(
        "--capital", type=float, default=100000, help="Initial capital"
    )
    backtest_opt.add_argument(
        "--commission", type=float, default=0.001, help="Commission rate"
    )
    backtest_opt.add_argument(
        "--slippage", type=float, default=0.0005, help="Slippage rate"
    )
    backtest_opt.add_argument(
        "--spread", type=float, default=0.0002, help="Spread rate"
    )
    backtest_opt.add_argument(
        "--stop-loss", type=float, default=0.05, help="Stop loss percent"
    )
    backtest_opt.add_argument(
        "--take-profit", type=float, default=0.10, help="Take profit percent"
    )
    backtest_opt.add_argument(
        "--allow-short", action="store_true", help="Allow short positions"
    )
    backtest_parser.set_defaults(func=run_backtest)

    # ============== OPTIMIZE Command ==============
    optimize_parser = subparsers.add_parser(
        "optimize",
        aliases=["opt", "hyperopt"],
        help="Bayesian hyperparameter optimization",
        description=textwrap.dedent("""
            Hyperparameter Optimization
            ==========================
            
            Optimize model hyperparameters using Bayesian optimization
            with Optuna.
        """).strip(),
    )

    opt_group = optimize_parser.add_argument_group(
        "Required Arguments", "Must be provided"
    )
    opt_group.add_argument("--symbol", "-s", type=str, required=True, metavar="SYMBOL")

    opt_opt = optimize_parser.add_argument_group(
        "Optional Arguments", "May be specified"
    )
    opt_opt.add_argument(
        "--model", "-m", type=str, default="lstm", choices=["lstm", "arima"]
    )
    opt_opt.add_argument("--years", "-y", type=int, default=3)
    opt_opt.add_argument(
        "--trials", "-t", type=int, default=20, help="Number of trials"
    )
    opt_opt.add_argument(
        "--epochs", "-e", type=int, default=20, help="Epochs per trial"
    )
    opt_opt.add_argument("--sequence-length", type=int, default=60)
    optimize_parser.set_defaults(func=run_optimize)

    # ============== VALIDATE Command ==============
    validate_parser = subparsers.add_parser(
        "validate",
        aliases=["val"],
        help="Walk-forward validation",
        description=textwrap.dedent("""
            Walk-Forward Validation
            =======================
            
            Validate models using walk-forward cross-validation
            to prevent lookahead bias.
        """).strip(),
    )

    val_group = validate_parser.add_argument_group(
        "Required Arguments", "Must be provided"
    )
    val_group.add_argument("--symbol", "-s", type=str, required=True, metavar="SYMBOL")

    val_opt = validate_parser.add_argument_group(
        "Optional Arguments", "May be specified"
    )
    val_opt.add_argument("--model", "-m", type=str, default="lstm")
    val_opt.add_argument("--years", "-y", type=int, default=5)
    val_opt.add_argument(
        "--train-size", type=int, default=252, help="Training window size"
    )
    val_opt.add_argument("--test-size", type=int, default=21, help="Test window size")
    validate_parser.set_defaults(func=run_validate)

    # ============== BATCH Command ==============
    batch_parser = subparsers.add_parser(
        "batch",
        aliases=["multi"],
        help="Process multiple symbols",
        description=textwrap.dedent("""
            Batch Processing
            =================
            
            Process multiple stock symbols in batch.
        """).strip(),
    )

    batch_group = batch_parser.add_argument_group(
        "Required Arguments", "Must be provided"
    )
    batch_group.add_argument(
        "--symbols", type=str, required=True, help="Comma-separated symbols"
    )

    batch_opt = batch_parser.add_argument_group(
        "Optional Arguments", "May be specified"
    )
    batch_opt.add_argument("--model", "-m", type=str, default="ensemble")
    batch_opt.add_argument("--years", "-y", type=int, default=3)
    batch_opt.add_argument("--days", "-d", type=int, default=7)
    batch_parser.set_defaults(func=run_batch)

    # ============== UNCERTAINTY Command ==============
    uncertainty_parser = subparsers.add_parser(
        "uncertainty",
        aliases=["conf", "interval"],
        help="Get predictions with confidence intervals",
        description=textwrap.dedent("""
            Prediction Uncertainty
            =====================
            
            Generate predictions with uncertainty/confidence intervals
            using Monte Carlo dropout or ensemble variance.
        """).strip(),
    )

    unc_group = uncertainty_parser.add_argument_group(
        "Required Arguments", "Must be provided"
    )
    unc_group.add_argument("--symbol", "-s", type=str, required=True, metavar="SYMBOL")

    unc_opt = uncertainty_parser.add_argument_group(
        "Optional Arguments", "May be specified"
    )
    unc_opt.add_argument("--model", "-m", type=str, default="ensemble")
    unc_opt.add_argument("--days", "-d", type=int, default=7)
    unc_opt.add_argument("--years", "-y", type=int, default=5)
    unc_opt.add_argument("--samples", "-n", type=int, default=100, help="MC samples")
    uncertainty_parser.set_defaults(func=run_uncertainty)

    # ============== RISK Command ==============
    risk_parser = subparsers.add_parser(
        "risk",
        aliases=["position", "size"],
        help="Calculate position sizing and risk metrics",
        description=textwrap.dedent("""
            Risk Management
            ===============
            
            Calculate optimal position size based on volatility,
            Kelly criterion, and risk parameters.
        """).strip(),
    )

    risk_group = risk_parser.add_argument_group(
        "Required Arguments", "Must be provided"
    )
    risk_group.add_argument("--symbol", "-s", type=str, required=True, metavar="SYMBOL")

    risk_opt = risk_parser.add_argument_group("Optional Arguments", "May be specified")
    risk_opt.add_argument(
        "--capital", type=float, default=100000, help="Portfolio value"
    )
    risk_opt.add_argument(
        "--max-position", type=float, default=20, help="Max position percent"
    )
    risk_opt.add_argument(
        "--use-kelly", action="store_true", help="Use Kelly criterion"
    )
    risk_parser.set_defaults(func=run_risk)

    # ============== COMPARE Command ==============
    compare_parser = subparsers.add_parser(
        "compare",
        aliases=["cmp", "benchmark"],
        help="Compare multiple models",
        description=textwrap.dedent("""
            Model Comparison
            ================
            
            Compare performance of multiple models side by side.
        """).strip(),
    )

    cmp_group = compare_parser.add_argument_group(
        "Required Arguments", "Must be provided"
    )
    cmp_group.add_argument("--symbol", "-s", type=str, required=True, metavar="SYMBOL")

    cmp_opt = compare_parser.add_argument_group(
        "Optional Arguments", "May be specified"
    )
    cmp_opt.add_argument(
        "--models",
        type=str,
        default="arima,lstm,gru,ensemble",
        help="Comma-separated models",
    )
    cmp_opt.add_argument("--years", "-y", type=int, default=3)
    compare_parser.set_defaults(func=compare_models)

    return parser


# ============== Main Entry Point ==============


def run_interactive(config: dict) -> int:
    """Run CLI in interactive mode — delegates to Rich-powered interactive module."""
    from src.cli.interactive import run_interactive as _run

    return _run(config)


def main() -> int:
    """Main entry point for Pythia Stock Prediction CLI."""
    global VERBOSE_MODE

    # Setup basic logging first
    setup_logging()

    # Create and parse arguments
    parser = create_parser()
    args = parser.parse_args()

    # Handle interactive mode — delegates to new Rich-powered module
    if args.interactive:
        config = load_config(args.config)
        return run_interactive(config)

    # Show welcome screen if no command given
    if args.command is None:
        display_welcome_full()
        return 0

    # Handle verbose mode
    VERBOSE_MODE = args.verbose
    if VERBOSE_MODE:
        logging.getLogger().setLevel(logging.DEBUG)
        status_info("Verbose mode enabled")

    # Load configuration
    config = load_config(args.config)

    # Validate inputs
    is_valid, errors = validate_all(args)
    if not is_valid:
        display_error(
            "Invalid arguments",
            "Check the following and try again:\n  " + "\n  ".join(errors),
        )
        return 2

    # Execute command with structured error handling
    try:
        if hasattr(args, "func"):
            args.func(args, config)

        console.print()
        status_ok("Command completed successfully")
        return 0

    except KeyboardInterrupt:
        console.print()
        status_warn("Operation cancelled by user")
        return 130

    except FileNotFoundError as e:
        display_error(
            "File not found",
            f"{e}\n\nCheck that the file exists and the path is correct.",
        )
        if VERBOSE_MODE:
            console.print_exception()
        return 3

    except ValueError as e:
        display_error(
            "Invalid value",
            f"{e}\n\nCheck your input and try again.",
        )
        if VERBOSE_MODE:
            console.print_exception()
        return 4

    except ImportError as e:
        display_error(
            "Missing dependency",
            f"{e}\n\nInstall required packages with:\n  pip install -r requirements.txt",
        )
        if VERBOSE_MODE:
            console.print_exception()
        return 5

    except Exception as e:
        display_error(
            "Unexpected error",
            f"{e}\n\nRun with --verbose for full traceback.",
        )
        if VERBOSE_MODE:
            console.print_exception()
        return 1


if __name__ == "__main__":
    sys.exit(main())
