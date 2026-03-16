"""
Stock Prediction CLI - Main Entry Point

Command-line interface for stock market prediction system.

Phase 1 UX Enhancements:
- Rich colored output with custom theme
- Progress bars for long-running operations
- Tables for structured data display
- Robust input validation
- Global exception handling with proper exit codes
- Verbose/debug mode support

Phase 2 CLI Enhancements (Current):
- Comprehensive help text for all commands
- Argument groups with clear separation
- Detailed usage examples and epilog sections
- Command aliases for quick access
- Nested subparsers for grouped functionality
- Informative metavars and proper formatting
"""

import sys
import argparse
import logging
import textwrap
from pathlib import Path
from typing import List, Tuple, Any, Optional

import yaml

# Import CLI modules
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
)
from src.cli.validators import validate_all


# ============== Global State ==============

# Verbose mode flag (set during argument parsing)
VERBOSE_MODE = False


# ============== Logging Setup ==============

def setup_logging(level: int = logging.INFO) -> None:
    """Setup logging configuration."""
    # Create logs directory if it doesn't exist
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(logs_dir / 'app.log')
        ]
    )


def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
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
    
    print_header(f"Collecting Data for {args.symbol}")
    
    # Initialize collector
    collector = StockDataCollector(config.get('data', {}))
    preprocessor = DataPreprocessor(config.get('indicators', {}))
    
    # Download data with progress
    print_info(f"Downloading {args.years} years of historical data...")
    
    with create_progress() as progress:
        task = progress.add_task("Downloading data...", total=100)
        
        data = collector.download_yahoo_data(
            symbol=args.symbol,
            years=args.years
        )
        
        progress.update(task, completed=50)
        
        if data.empty:
            print_error(f"No data collected for {args.symbol}")
            return None
        
        # Preprocess data
        print_info("Preprocessing data...")
        data = preprocessor.clean_data(data)
        data = preprocessor.add_all_indicators(data)
        
        progress.update(task, completed=80)
        
        # Save data
        filepath = collector.save_data(data, args.symbol, 'historical')
        
        progress.update(task, completed=100)
    
    # Display summary in a nice table
    data_info = {
        'Total Records': len(data),
        'Date Range': f"{data.index[0].date()} to {data.index[-1].date()}",
        'Latest Close': data['close'].iloc[-1],
    }
    
    console.print(create_data_summary_table(data_info))
    print_success(f"Data saved to {filepath}")
    
    return data


def train_model(args: argparse.Namespace, config: dict) -> Any:
    """Train prediction model."""
    from src.models.predictor import StockPredictor
    
    print_header(f"Training {args.model.title()} Model for {args.symbol}")
    
    # Create predictor
    predictor = StockPredictor(args.symbol, args.model, config)
    
    # Load data
    print_info("Loading data...")
    predictor.load_data(years=args.years)
    
    # Train model with progress
    print_info(f"Training {args.model} model...")
    
    with create_progress() as progress:
        task = progress.add_task("Training model...", total=100)
        
        result = predictor.train()
        
        progress.update(task, completed=100)
    
    # Display results
    print_success("Training complete!")
    
    training_info = {
        'Model Type': result['model_type'],
        'Data Points': result['data_points'],
    }
    
    table = create_data_summary_table(training_info)
    console.print(table)
    
    # Save model if requested
    if args.save:
        model_path = predictor.save_model()
        print_success(f"Model saved to: {model_path}")
    
    return predictor


def make_predictions(args: argparse.Namespace, config: dict) -> Tuple[Any, Any]:
    """Make stock predictions."""
    from src.models.predictor import StockPredictor
    
    print_header(f"Making Predictions for {args.symbol}")
    
    # Create predictor
    predictor = StockPredictor(args.symbol, args.model, config)
    
    # Load data with progress
    print_info("Loading data...")
    
    with create_progress() as progress:
        task = progress.add_task("Loading and training...", total=100)
        
        predictor.load_data(years=args.years)
        progress.update(task, completed=40)
        
        print_info("Training model...")
        predictor.train()
        progress.update(task, completed=80)
        
        # Make predictions
        print_info(f"Making predictions for next {args.days} days...")
        predictions = predictor.predict(args.days)
        
        progress.update(task, completed=100)
    
    # Display predictions in a table
    lower_bound = predictions.get('lower_bound', predictions['predictions'])
    upper_bound = predictions.get('upper_bound', predictions['predictions'])
    
    table = create_predictions_table(
        predictions=predictions['predictions'].tolist() if hasattr(predictions['predictions'], 'tolist') else list(predictions['predictions']),
        lower_bound=lower_bound.tolist() if hasattr(lower_bound, 'tolist') else list(lower_bound),
        upper_bound=upper_bound.tolist() if hasattr(upper_bound, 'tolist') else list(upper_bound),
        title=f"Stock Predictions for {args.symbol}"
    )
    console.print(table)
    
    # Current price and change
    current = predictor.get_current_price()
    if current:
        pred = predictions['predictions'][0]
        change = pred - current
        change_pct = (change / current) * 100
        
        # Display as panel
        price_info = f"""
[cyan]Current Price:[/cyan]     [green]${current:.2f}[/green]
[cyan]Next Day Prediction:[/cyan] [green]${pred:.2f}[/green]
[cyan]Change:[/cyan]            [green]{change:+.2f} ({change_pct:+.2f}%)[/green]
        """
        print_panel(price_info.strip(), title="Price Summary", style="green")
    
    return predictor, predictions


def run_dashboard(args: argparse.Namespace, config: dict) -> None:
    """Run Streamlit dashboard."""
    import subprocess
    import os
    
    print_header("Starting Dashboard")
    
    print_info("Starting Streamlit dashboard...")
    print_info("If the dashboard doesn't open automatically, run:")
    console.print("  [cyan]streamlit run src/visualization/dashboard.py[/cyan]")
    print("\nOr open [link=http://localhost:8501]http://localhost:8501[/link] in your browser")
    
    # Change to project directory and run streamlit
    cmd = [sys.executable, "-m", "streamlit", "run", "src/visualization/dashboard.py"]
    subprocess.run(cmd, cwd=os.getcwd())


def analyze_sentiment(args: argparse.Namespace, config: dict) -> None:
    """Analyze sentiment from news/text."""
    from src.sentiment.analyzer import SentimentAnalyzer
    
    print_header("Analyzing Sentiment")
    
    # Sample texts (in real usage, would fetch from news sources)
    sample_texts = [
        "Stock market reaches all-time high amid positive economic data",
        "Company reports strong quarterly earnings, beating expectations",
        "Market faces uncertainty due to geopolitical concerns",
        "Analysts downgrade stock rating citing slowdown risks",
        "CEO announces strategic partnership, investors react positively"
    ]
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer(config.get('sentiment', {}))
    
    # Analyze with progress
    print_info(f"Analyzing {len(sample_texts)} sample texts...")
    
    with create_progress() as progress:
        task = progress.add_task("Analyzing sentiment...", total=len(sample_texts))
        results = analyzer.analyze_batch(sample_texts, method=args.method)
        progress.update(task, completed=100)
    
    # Display results in a table
    table = create_sentiment_table(results)
    console.print(table)
    
    # Summary
    summary = analyzer.get_sentiment_summary(results)
    
    summary_text = f"""
[green]Positive:[/green] {summary.get('positive_pct', 0):.1f}%
[yellow]Neutral:[/yellow]  {summary.get('neutral_pct', 0):.1f}%
[red]Negative:[/red]  {summary.get('negative_pct', 0):.1f}%
    """
    print_panel(summary_text.strip(), title="Sentiment Summary", style="cyan")


def evaluate_model(args: argparse.Namespace, config: dict) -> None:
    """Evaluate trained model."""
    from src.models.predictor import StockPredictor
    
    print_header(f"Evaluating {args.model.title()} Model for {args.symbol}")
    
    # Create predictor
    predictor = StockPredictor(args.symbol, args.model, config)
    
    # Load data with progress
    print_info("Loading data...")
    
    with create_progress() as progress:
        task = progress.add_task("Loading and evaluating...", total=100)
        
        predictor.load_data(years=args.years)
        progress.update(task, completed=30)
        
        print_info("Training model...")
        predictor.train()
        progress.update(task, completed=60)
        
        # Evaluate
        print_info("Evaluating model...")
        metrics = predictor.evaluate()
        progress.update(task, completed=100)
    
    # Display metrics in a table
    table = create_metrics_table(metrics, title=f"Model Evaluation: {args.symbol}")
    console.print(table)


# ============== Argument Parser Setup ==============

def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with comprehensive CLI help."""
    
    # Main parser with raw description formatter
    parser = argparse.ArgumentParser(
        prog='stock-prediction',
        description=textwrap.dedent('''
            Stock Market Prediction CLI
            ============================
            
            A comprehensive command-line interface for stock market prediction.
            Collect data, train models, make predictions, and analyze market sentiment.
            
            Use this tool to:
              • Download and preprocess historical stock data
              • Train machine learning models (ARIMA, LSTM, GRU, Ensemble)
              • Generate predictions for future stock prices
              • Evaluate model performance with various metrics
              • Analyze market sentiment from news and text
              • Visualize results in an interactive dashboard
        ''').strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''
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
            
            ========================================
            Related Commands
            ========================================
            
            data     - Collect and manage stock data (alias: collect)
            train    - Train prediction models
            predict  - Make stock price predictions
            evaluate - Evaluate model performance
            sentiment - Analyze market sentiment
            dashboard - Launch interactive dashboard
            
            For more information, visit the documentation.
        ''').strip()
    )
    
    # Note: argparse automatically adds -h/--help
    # We handle help display in main() function
    
    # Global options group
    global_group = parser.add_argument_group(
        'Global Options',
        'Options that work with any command'
    )
    global_group.add_argument(
        '-V', '--verbose',
        action='store_true',
        dest='verbose',
        help='Enable verbose/debug output for detailed logging'
    )
    global_group.add_argument(
        '-c', '--config',
        type=str,
        default='config.yaml',
        metavar='FILE',
        help='Path to configuration file (default: config.yaml)'
    )
    global_group.add_argument(
        '-i', '--interactive',
        action='store_true',
        dest='interactive',
        help='Run in interactive mode with guided prompts'
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(
        dest='command',
        title='Commands',
        description='Available commands for stock prediction',
        help='Run with COMMAND --help for detailed usage'
    )
    
    # ============== COLLECT Command ==============
    collect_parser = subparsers.add_parser(
        'collect',
        aliases=['data'],
        help='Collect historical stock data from Yahoo Finance',
        description=textwrap.dedent('''
            Collect Historical Stock Data
            =============================
            
            Downloads historical stock data from Yahoo Finance and saves it
            for later use in training and prediction.
        ''').strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''
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
        ''').strip()
    )
    
    # Collect arguments
    collect_group = collect_parser.add_argument_group(
        'Required Arguments',
        'Must be provided'
    )
    collect_group.add_argument(
        '--symbol', '-s',
        type=str,
        required=True,
        metavar='SYMBOL',
        help='Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)'
    )
    
    collect_opt = collect_parser.add_argument_group(
        'Optional Arguments',
        'May be specified'
    )
    collect_opt.add_argument(
        '--years', '-y',
        type=int,
        default=5,
        metavar='N',
        help='Number of years of historical data to collect (default: 5)'
    )
    collect_parser.set_defaults(func=collect_data)
    
    # ============== TRAIN Command ==============
    train_parser = subparsers.add_parser(
        'train',
        aliases=['model'],
        help='Train a prediction model on historical data',
        description=textwrap.dedent('''
            Train Prediction Model
            ======================
            
            Train a machine learning model using historical stock data.
            Supports multiple model types for different prediction approaches.
        ''').strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''
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
        ''').strip()
    )
    
    # Train arguments
    train_group = train_parser.add_argument_group(
        'Required Arguments',
        'Must be provided'
    )
    train_group.add_argument(
        '--symbol', '-s',
        type=str,
        required=True,
        metavar='SYMBOL',
        help='Stock ticker symbol to train on'
    )
    
    train_opt = train_parser.add_argument_group(
        'Model Options',
        'Configure the model type and behavior'
    )
    train_opt.add_argument(
        '--model', '-m',
        type=str,
        default='ensemble',
        choices=['arima', 'lstm', 'gru', 'ensemble'],
        metavar='TYPE',
        help='Model type to use (default: ensemble)'
    )
    train_opt.add_argument(
        '--years', '-y',
        type=int,
        default=5,
        metavar='N',
        help='Years of historical data to use (default: 5)'
    )
    train_opt.add_argument(
        '--save',
        action='store_true',
        help='Save trained model to disk for later use'
    )
    train_parser.set_defaults(func=train_model)
    
    # ============== PREDICT Command ==============
    predict_parser = subparsers.add_parser(
        'predict',
        aliases=['forecast', 'pred'],
        help='Generate stock price predictions',
        description=textwrap.dedent('''
            Make Stock Predictions
            =======================
            
            Generate predictions for future stock prices using trained models.
            Returns predictions with confidence intervals.
        ''').strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''
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
        ''').strip()
    )
    
    # Predict arguments
    predict_group = predict_parser.add_argument_group(
        'Required Arguments',
        'Must be provided'
    )
    predict_group.add_argument(
        '--symbol', '-s',
        type=str,
        required=True,
        metavar='SYMBOL',
        help='Stock ticker symbol to predict'
    )
    
    predict_opt = predict_parser.add_argument_group(
        'Prediction Options',
        'Configure prediction behavior'
    )
    predict_opt.add_argument(
        '--model', '-m',
        type=str,
        default='ensemble',
        choices=['arima', 'lstm', 'gru', 'ensemble'],
        metavar='TYPE',
        help='Model to use for prediction (default: ensemble)'
    )
    predict_opt.add_argument(
        '--years', '-y',
        type=int,
        default=5,
        metavar='N',
        help='Years of data to use for training (default: 5)'
    )
    predict_opt.add_argument(
        '--days', '-d',
        type=int,
        default=7,
        metavar='N',
        help='Number of days to predict ahead (default: 7)'
    )
    predict_parser.set_defaults(func=make_predictions)
    
    # ============== DASHBOARD Command ==============
    dashboard_parser = subparsers.add_parser(
        'dashboard',
        aliases=['ui', 'web'],
        help='Launch interactive Streamlit dashboard',
        description=textwrap.dedent('''
            Launch Interactive Dashboard
            ============================
            
            Opens a web-based dashboard for interactive stock analysis
            and visualization using Streamlit.
        ''').strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''
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
        ''').strip()
    )
    
    # Dashboard arguments
    dash_group = dashboard_parser.add_argument_group(
        'Dashboard Options',
        'Configure dashboard behavior'
    )
    dash_group.add_argument(
        '--symbol', '-s',
        type=str,
        default='AAPL',
        metavar='SYMBOL',
        help='Default stock symbol to display (default: AAPL)'
    )
    dash_group.add_argument(
        '--model', '-m',
        type=str,
        default='ensemble',
        choices=['arima', 'lstm', 'gru', 'ensemble'],
        metavar='TYPE',
        help='Default model type (default: ensemble)'
    )
    dash_group.add_argument(
        '--years', '-y',
        type=int,
        default=5,
        metavar='N',
        help='Years of data to load (default: 5)'
    )
    dashboard_parser.set_defaults(func=run_dashboard)
    
    # ============== SENTIMENT Command ==============
    sentiment_parser = subparsers.add_parser(
        'sentiment',
        aliases=['sent', 'analyze'],
        help='Analyze market sentiment from text',
        description=textwrap.dedent('''
            Analyze Market Sentiment
            =========================
            
            Analyze the sentiment of news articles, social media posts,
            or custom text to gauge market mood.
        ''').strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''
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
        ''').strip()
    )
    
    # Sentiment arguments
    sent_group = sentiment_parser.add_argument_group(
        'Sentiment Options',
        'Configure sentiment analysis'
    )
    sent_group.add_argument(
        '--method', '-m',
        type=str,
        default='vader',
        choices=['vader', 'textblob', 'combined'],
        metavar='METHOD',
        help='Sentiment analysis method (default: vader)'
    )
    sentiment_parser.set_defaults(func=analyze_sentiment)
    
    # ============== EVALUATE Command ==============
    eval_parser = subparsers.add_parser(
        'evaluate',
        aliases=['eval', 'metrics', 'test'],
        help='Evaluate model performance with metrics',
        description=textwrap.dedent('''
            Evaluate Model Performance
            ==========================
            
            Evaluate a trained model using various metrics including
            RMSE, MAE, MAPE, and R-squared.
        ''').strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''
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
        ''').strip()
    )
    
    # Evaluate arguments
    eval_group = eval_parser.add_argument_group(
        'Required Arguments',
        'Must be provided'
    )
    eval_group.add_argument(
        '--symbol', '-s',
        type=str,
        required=True,
        metavar='SYMBOL',
        help='Stock ticker symbol to evaluate'
    )
    
    eval_opt = eval_parser.add_argument_group(
        'Evaluation Options',
        'Configure evaluation behavior'
    )
    eval_opt.add_argument(
        '--model', '-m',
        type=str,
        default='ensemble',
        choices=['arima', 'lstm', 'gru', 'ensemble'],
        metavar='TYPE',
        help='Model type to evaluate (default: ensemble)'
    )
    eval_opt.add_argument(
        '--years', '-y',
        type=int,
        default=5,
        metavar='N',
        help='Years of data to use (default: 5)'
    )
    eval_parser.set_defaults(func=evaluate_model)
    
    return parser


# ============== Main Entry Point ==============

def run_interactive(config: dict) -> int:
    """Run CLI in interactive mode with guided prompts."""
    from src.cli.output import print_header, print_success, print_error, print_info
    
    print_header("Welcome to Pythia Stock Predictor - Interactive Mode")
    print_info("This guided mode will help you through the prediction workflow.\n")
    
    # Step 1: Choose action
    print_info("Available actions:")
    print_info("  1. Collect data - Download historical stock data")
    print_info("  2. Train model - Train a prediction model")
    print_info("  3. Make predictions - Generate price predictions")
    print_info("  4. Run dashboard - Launch the visualization dashboard")
    print_info("  5. Analyze sentiment - Analyze market sentiment")
    print_info("  6. Full workflow - Complete prediction pipeline")
    print_info("  0. Exit - Exit interactive mode\n")
    
    while True:
        try:
            choice = input("Enter your choice (0-6): ").strip()
            
            if choice == '0':
                print_info("Exiting interactive mode. Goodbye!")
                return 0
            
            elif choice == '1':
                # Collect data
                symbol = input("Enter stock symbol (e.g., AAPL): ").strip().upper()
                years = input("Enter number of years of data [1-10]: ").strip() or "5"
                
                # Create mock args namespace
                class Args:
                    pass
                args = Args()
                args.symbol = symbol
                args.years = int(years)
                args.source = 'yfinance'
                args.interactive = True
                
                collect_data(args, config)
                
            elif choice == '2':
                # Train model
                symbol = input("Enter stock symbol (e.g., AAPL): ").strip().upper()
                model = input("Enter model type [arima/lstm/gru/ensemble]: ").strip().lower() or "ensemble"
                
                class Args:
                    pass
                args = Args()
                args.symbol = symbol
                args.model = model
                args.epochs = 50
                args.save = True
                args.interactive = True
                
                train_model(args, config)
                
            elif choice == '3':
                # Make predictions
                symbol = input("Enter stock symbol (e.g., AAPL): ").strip().upper()
                model = input("Enter model type [arima/lstm/gru/ensemble]: ").strip().lower() or "ensemble"
                days = input("Enter number of days to predict [1-90]: ").strip() or "7"
                
                class Args:
                    pass
                args = Args()
                args.symbol = symbol
                args.model = model
                args.days = int(days)
                args.interactive = True
                
                make_predictions(args, config)
                
            elif choice == '4':
                # Run dashboard
                symbol = input("Enter stock symbol (e.g., AAPL): ").strip().upper()
                
                class Args:
                    pass
                args = Args()
                args.symbol = symbol
                args.model = "ensemble"
                args.port = 8501
                args.interactive = True
                
                run_dashboard(args, config)
                
            elif choice == '5':
                # Sentiment analysis
                text = input("Enter text to analyze (or press Enter for sample): ").strip()
                if not text:
                    text = "Stock market shows positive trends today with strong tech sector performance."
                
                class Args:
                    pass
                args = Args()
                args.text = text
                args.method = "vader"
                args.interactive = True
                
                analyze_sentiment(args, config)
                
            elif choice == '6':
                # Full workflow
                symbol = input("Enter stock symbol (e.g., AAPL): ").strip().upper()
                years = input("Enter number of years of data [1-10]: ").strip() or "5"
                model = input("Enter model type [arima/lstm/gru/ensemble]: ").strip().lower() or "ensemble"
                days = input("Enter number of days to predict [1-90]: ").strip() or "7"
                
                print_info("\n" + "="*50)
                print_header(f"Starting Full Workflow for {symbol}")
                print_info("="*50 + "\n")
                
                # Step 1: Collect
                print_info("Step 1/3: Collecting data...")
                class Args1:
                    pass
                args1 = Args1()
                args1.symbol = symbol
                args1.years = int(years)
                args1.source = 'yfinance'
                args1.interactive = True
                collect_data(args1, config)
                
                # Step 2: Train
                print_info("Step 2/3: Training model...")
                class Args2:
                    pass
                args2 = Args2()
                args2.symbol = symbol
                args2.model = model
                args2.epochs = 50
                args2.save = True
                args2.interactive = True
                train_model(args2, config)
                
                # Step 3: Predict
                print_info("Step 3/3: Making predictions...")
                class Args3:
                    pass
                args3 = Args3()
                args3.symbol = symbol
                args3.model = model
                args3.days = int(days)
                args3.interactive = True
                make_predictions(args3, config)
                
                print_success("\nFull workflow completed!")
                
            else:
                print_error("Invalid choice. Please enter a number between 0 and 6.")
                continue
            
            # Ask if user wants to continue
            print_info("\n" + "-"*50)
            continue_choice = input("Would you like to perform another action? (y/n): ").strip().lower()
            if continue_choice not in ('y', 'yes'):
                print_info("Exiting interactive mode. Goodbye!")
                return 0
                
        except KeyboardInterrupt:
            print_info("\n\nOperation cancelled. Exiting interactive mode.")
            return 130
        except Exception as e:
            print_error(f"Error: {e}")
            print_info("Please try again.")


def main() -> int:
    """Main entry point."""
    global VERBOSE_MODE
    
    # Setup basic logging first
    setup_logging()
    
    # Create and parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle interactive mode
    if args.interactive:
        print_welcome()
        config = load_config(args.config)
        return run_interactive(config)
    
    # Show welcome message if no command
    if args.command is None:
        print_welcome()
        parser.print_help()
        print_info("\nTip: Use --interactive or -i for guided mode!")
        return 0
    
    # Handle verbose mode
    VERBOSE_MODE = args.verbose
    if VERBOSE_MODE:
        setup_logging(logging.DEBUG)
        print_info("Verbose mode enabled")
    
    # Load configuration
    config = load_config(args.config)
    
    # Validate inputs
    is_valid, errors = validate_all(args)
    if not is_valid:
        print_error("Validation failed:")
        for error in errors:
            console.print(f"  * {error}")
        return 2  # Exit code 2 for validation errors
    
    # Execute command with exception handling
    try:
        # Call the appropriate function
        if hasattr(args, 'func'):
            args.func(args, config)
        
        print_success("Command completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print_warning("\nOperation cancelled by user")
        return 130  # Standard exit code for SIGINT
        
    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        if VERBOSE_MODE:
            import traceback
            console.print(traceback.format_exc())
        return 3  # Exit code 3 for file errors
        
    except ValueError as e:
        print_error(f"Invalid value: {e}")
        if VERBOSE_MODE:
            import traceback
            console.print(traceback.format_exc())
        return 4  # Exit code 4 for validation/value errors
        
    except ImportError as e:
        print_error(f"Import error: {e}")
        print_info("Check that all required dependencies are installed.")
        if VERBOSE_MODE:
            import traceback
            console.print(traceback.format_exc())
        return 5  # Exit code 5 for import errors
        
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        if VERBOSE_MODE:
            console.print_exception()
        return 1  # Exit code 1 for general errors


if __name__ == '__main__':
    sys.exit(main())
