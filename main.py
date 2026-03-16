"""
Stock Prediction CLI - Main Entry Point

Command-line interface for stock market prediction system.
"""

import sys
import argparse
import logging
from pathlib import Path

import yaml


def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    # Create logs directory if it doesn't exist
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(logs_dir / 'app.log')
        ]
    )


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load config file: {e}")
        return {}


def collect_data(args, config):
    """Collect historical stock data."""
    from src.data.collector import StockDataCollector
    from src.data.preprocessor import DataPreprocessor
    
    print(f"\n{'='*50}")
    print(f"Collecting data for {args.symbol}")
    print(f"{'='*50}\n")
    
    # Initialize collector
    collector = StockDataCollector(config.get('data', {}))
    preprocessor = DataPreprocessor(config.get('indicators', {}))
    
    # Download data
    print(f"Downloading {args.years} years of historical data...")
    data = collector.download_yahoo_data(
        symbol=args.symbol,
        years=args.years
    )
    
    if data.empty:
        print(f"Error: No data collected for {args.symbol}")
        return
        
    # Preprocess
    print("Preprocessing data...")
    data = preprocessor.clean_data(data)
    data = preprocessor.add_all_indicators(data)
    
    # Save data
    filepath = collector.save_data(data, args.symbol, 'historical')
    print(f"Data saved to {filepath}")
    
    # Display info
    print(f"\nData Summary:")
    print(f"  Total records: {len(data)}")
    print(f"  Date range: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"  Latest close: ${data['close'].iloc[-1]:.2f}")
    
    return data


def train_model(args, config):
    """Train prediction model."""
    from src.models.predictor import StockPredictor
    
    print(f"\n{'='*50}")
    print(f"Training {args.model} model for {args.symbol}")
    print(f"{'='*50}\n")
    
    # Create predictor
    predictor = StockPredictor(args.symbol, args.model, config)
    
    # Load data
    print("Loading data...")
    predictor.load_data(years=args.years)
    
    # Train model
    print(f"Training {args.model} model...")
    result = predictor.train()
    
    print(f"\nTraining complete!")
    print(f"  Model type: {result['model_type']}")
    print(f"  Data points: {result['data_points']}")
    
    # Save model
    if args.save:
        model_path = predictor.save_model()
        print(f"  Model saved to: {model_path}")
    
    return predictor


def make_predictions(args, config):
    """Make stock predictions."""
    from src.models.predictor import StockPredictor
    
    print(f"\n{'='*50}")
    print(f"Making predictions for {args.symbol}")
    print(f"{'='*50}\n")
    
    # Create predictor
    predictor = StockPredictor(args.symbol, args.model, config)
    
    # Load data
    print("Loading data...")
    predictor.load_data(years=args.years)
    
    # Train model
    print("Training model...")
    predictor.train()
    
    # Make predictions
    print(f"\nMaking predictions for next {args.days} days...")
    predictions = predictor.predict(args.days)
    
    # Display predictions
    print(f"\n{'Predictions':<15} {'Lower':<15} {'Upper':<15}")
    print("-" * 45)
    
    for i, (pred, lower, upper) in enumerate(zip(
        predictions['predictions'],
        predictions.get('lower_bound', predictions['predictions']),
        predictions.get('upper_bound', predictions['predictions'])
    )):
        print(f"Day {i+1}: ${pred:<14.2f} ${lower:<14.2f} ${upper:<14.2f}")
    
    # Current price
    current = predictor.get_current_price()
    if current:
        print(f"\nCurrent Price: ${current:.2f}")
        print(f"Next Day Prediction: ${predictions['predictions'][0]:.2f}")
        change = predictions['predictions'][0] - current
        change_pct = (change / current) * 100
        print(f"Change: {change:+.2f} ({change_pct:+.2f}%)")
    
    return predictor, predictions


def run_dashboard(args, config):
    """Run Streamlit dashboard."""
    import subprocess
    import sys
    import os
    
    print(f"\n{'='*50}")
    print(f"Starting Dashboard")
    print(f"{'='*50}\n")
    
    # Run dashboard using streamlit CLI
    print("Starting Streamlit dashboard...")
    print("If the dashboard doesn't open automatically, run:")
    print("  streamlit run src/visualization/dashboard.py")
    print("\nOr open http://localhost:8501 in your browser")
    
    # Change to project directory and run streamlit
    cmd = [sys.executable, "-m", "streamlit", "run", "src/visualization/dashboard.py"]
    subprocess.run(cmd, cwd=os.getcwd())


def analyze_sentiment(args, config):
    """Analyze sentiment from news/text."""
    from src.sentiment.analyzer import SentimentAnalyzer
    
    print(f"\n{'='*50}")
    print(f"Analyzing Sentiment")
    print(f"{'='*50}\n")
    
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
    
    # Analyze
    print(f"Analyzing {len(sample_texts)} sample texts...")
    results = analyzer.analyze_batch(sample_texts, method=args.method)
    
    # Display results
    print(f"\n{'Text':<50} {'Sentiment':<10} {'Score':<10}")
    print("-" * 70)
    
    for _, row in results.iterrows():
        text = row['text'][:47] + "..." if len(row['text']) > 50 else row['text']
        print(f"{text:<50} {row['sentiment']:<10} {row.get('compound', row.get('polarity', 0)):<10.3f}")
    
    # Summary
    summary = analyzer.get_sentiment_summary(results)
    print(f"\nSentiment Summary:")
    print(f"  Positive: {summary.get('positive_pct', 0):.1f}%")
    print(f"  Neutral: {summary.get('neutral_pct', 0):.1f}%")
    print(f"  Negative: {summary.get('negative_pct', 0):.1f}%")


def evaluate_model(args, config):
    """Evaluate trained model."""
    from src.models.predictor import StockPredictor
    
    print(f"\n{'='*50}")
    print(f"Evaluating {args.model} model for {args.symbol}")
    print(f"{'='*50}\n")
    
    # Create predictor
    predictor = StockPredictor(args.symbol, args.model, config)
    
    # Load data
    print("Loading data...")
    predictor.load_data(years=args.years)
    
    # Train model
    print("Training model...")
    predictor.train()
    
    # Evaluate
    print("Evaluating model...")
    metrics = predictor.evaluate()
    
    # Display metrics
    print(f"\nModel Evaluation Metrics:")
    print(f"  MAE:  ${metrics['mae']:.4f}")
    print(f"  RMSE: ${metrics['rmse']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    
    return metrics


def main():
    """Main entry point."""
    # Setup
    setup_logging()
    config = load_config()
    
    # Create parser
    parser = argparse.ArgumentParser(
        description='Stock Market Prediction CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect historical data
  python main.py collect --symbol AAPL --years 5

  # Train ARIMA model
  python main.py train --symbol AAPL --model arima

  # Make predictions
  python main.py predict --symbol AAPL --days 7

  # Run dashboard
  python main.py dashboard --symbol AAPL

  # Analyze sentiment
  python main.py sentiment --method vader
        """
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Collect command
    collect_parser = subparsers.add_parser('collect', help='Collect historical stock data')
    collect_parser.add_argument('--symbol', type=str, required=True, help='Stock symbol')
    collect_parser.add_argument('--years', type=int, default=5, help='Years of data to collect')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train prediction model')
    train_parser.add_argument('--symbol', type=str, required=True, help='Stock symbol')
    train_parser.add_argument('--model', type=str, default='ensemble', 
                            choices=['arima', 'lstm', 'gru', 'ensemble'],
                            help='Model type')
    train_parser.add_argument('--years', type=int, default=5, help='Years of data to use')
    train_parser.add_argument('--save', action='store_true', help='Save trained model')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make stock predictions')
    predict_parser.add_argument('--symbol', type=str, required=True, help='Stock symbol')
    predict_parser.add_argument('--model', type=str, default='ensemble',
                              choices=['arima', 'lstm', 'gru', 'ensemble'],
                              help='Model type')
    predict_parser.add_argument('--years', type=int, default=5, help='Years of data to use')
    predict_parser.add_argument('--days', type=int, default=7, help='Days to predict')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Run Streamlit dashboard')
    dashboard_parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol')
    dashboard_parser.add_argument('--model', type=str, default='ensemble',
                                 choices=['arima', 'lstm', 'gru', 'ensemble'],
                                 help='Model type')
    dashboard_parser.add_argument('--years', type=int, default=5, help='Years of data to use')
    
    # Sentiment command
    sentiment_parser = subparsers.add_parser('sentiment', help='Analyze sentiment')
    sentiment_parser.add_argument('--method', type=str, default='vader',
                                  choices=['vader', 'textblob', 'combined'],
                                  help='Sentiment analysis method')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--symbol', type=str, required=True, help='Stock symbol')
    eval_parser.add_argument('--model', type=str, default='ensemble',
                           choices=['arima', 'lstm', 'gru', 'ensemble'],
                           help='Model type')
    eval_parser.add_argument('--years', type=int, default=5, help='Years of data to use')
    
    # Parse args
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
        
    # Execute command
    if args.command == 'collect':
        collect_data(args, config)
    elif args.command == 'train':
        train_model(args, config)
    elif args.command == 'predict':
        make_predictions(args, config)
    elif args.command == 'dashboard':
        run_dashboard(args, config)
    elif args.command == 'sentiment':
        analyze_sentiment(args, config)
    elif args.command == 'evaluate':
        evaluate_model(args, config)


if __name__ == '__main__':
    main()
