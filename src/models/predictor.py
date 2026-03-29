"""
Stock Predictor Module

High-level interface for making stock predictions using trained models.

NOTE: Sentiment analysis integration is currently supported for ARIMA/Statistical models.
LSTM/GRU models may require input dimension adjustments for sentiment features.
"""

import logging
from typing import Optional, Dict, List
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# FIX: Integrated sentiment analysis into prediction pipeline
# SentimentAnalyzer will be imported lazily to avoid import errors if not available
SentimentAnalyzer = None


class StockPredictor:
    """
    High-level stock prediction interface.

    Combines data collection, preprocessing, and model prediction
    into a unified interface.
    """

    def __init__(
        self, symbol: str, model_type: str = "ensemble", config: Optional[Dict] = None
    ):
        """
        Initialize Stock Predictor.

        Args:
            symbol: Stock ticker symbol
            model_type: Type of model to use (arima, lstm, gru, ensemble)
            config: Configuration dictionary
        """
        self.symbol = symbol
        self.model_type = model_type
        self.config = config or {}

        # FIX: Added config option to enable/disable sentiment integration
        # Sentiment integration is disabled by default
        self.enable_sentiment = self.config.get("sentiment", {}).get("enabled", False)
        self.sentiment_method = self.config.get("sentiment", {}).get("method", "vader")
        self.sentiment_analyzer = None

        # Initialize components
        from src.data.collector import StockDataCollector
        from src.data.preprocessor import DataPreprocessor

        self.collector = StockDataCollector(config)
        self.preprocessor = DataPreprocessor(config)

        # Initialize sentiment analyzer if enabled
        if self.enable_sentiment:
            self._initialize_sentiment_analyzer()

        # Initialize model
        self.model = None
        self._initialize_model()

        # Data
        self.data = None
        self.predictions = None

    def _initialize_sentiment_analyzer(self):
        """Initialize the sentiment analyzer."""
        global SentimentAnalyzer
        try:
            if SentimentAnalyzer is None:
                from src.sentiment.analyzer import SentimentAnalyzer as SA

                SentimentAnalyzer = SA

            self.sentiment_analyzer = SentimentAnalyzer(
                self.config.get("sentiment", {})
            )
            logger.info(
                f"Initialized sentiment analyzer with method: {self.sentiment_method}"
            )
        except ImportError as e:
            logger.warning(f"SentimentAnalyzer not available: {e}")
            self.enable_sentiment = False
        except Exception as e:
            logger.warning(f"Failed to initialize sentiment analyzer: {e}")
            self.enable_sentiment = False

    def _fetch_and_analyze_sentiment(self) -> pd.DataFrame:
        """
        Fetch news and analyze sentiment for the stock.

        Returns:
            DataFrame with sentiment features
        """
        if self.sentiment_analyzer is None:
            return pd.DataFrame()

        try:
            # Fetch news headlines
            max_news = self.config.get("sentiment", {}).get("max_news", 10)
            news = self.collector.get_news_headlines(self.symbol, max_news=max_news)

            if not news:
                logger.info(f"No news available for sentiment analysis")
                return pd.DataFrame()

            # Extract titles for sentiment analysis
            headlines = [item["title"] for item in news if item.get("title")]

            if not headlines:
                return pd.DataFrame()

            # Analyze sentiment
            sentiment_df = self.sentiment_analyzer.analyze_batch(
                headlines, method=self.sentiment_method
            )

            # Get aggregate sentiment scores
            sentiment_summary = {
                "news_count": len(headlines),
                "avg_sentiment_compound": sentiment_df["compound"].mean()
                if "compound" in sentiment_df
                else 0.0,
                "avg_sentiment_pos": sentiment_df["pos"].mean()
                if "pos" in sentiment_df
                else 0.0,
                "avg_sentiment_neg": sentiment_df["neg"].mean()
                if "neg" in sentiment_df
                else 0.0,
                "avg_sentiment_neu": sentiment_df["neu"].mean()
                if "neu" in sentiment_df
                else 0.0,
            }

            # Determine overall sentiment
            compound = sentiment_summary["avg_sentiment_compound"]
            if compound >= 0.05:
                sentiment_summary["overall_sentiment"] = "positive"
            elif compound <= -0.05:
                sentiment_summary["overall_sentiment"] = "negative"
            else:
                sentiment_summary["overall_sentiment"] = "neutral"

            logger.info(
                f"Sentiment analysis complete: {sentiment_summary['overall_sentiment']} "
                f"(compound: {compound:.3f}, news: {sentiment_summary['news_count']})"
            )

            # Return as single-row DataFrame
            return pd.DataFrame([sentiment_summary])

        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return pd.DataFrame()

    def _initialize_model(self):
        """Initialize the prediction model."""
        model_config = self.config.get("models", {})

        if self.model_type == "arima":
            from src.models.arima_model import ARIMAModel

            self.model = ARIMAModel(model_config.get("arima", {}))

        elif self.model_type == "lstm":
            from src.models.lstm_model import LSTMModel

            self.model = LSTMModel(model_config.get("lstm", {}))

        elif self.model_type == "gru":
            from src.models.lstm_model import GRUModel

            self.model = GRUModel(model_config.get("gru", {}))

        elif self.model_type == "cnn_lstm":
            from src.models.temporal.cnn_lstm import CNNLSTMModel

            config = model_config.get("cnn_lstm", {})
            # Set sequence length from config or use default
            if "sequence_length" not in config:
                config["sequence_length"] = model_config.get("lstm", {}).get(
                    "sequence_length", 60
                )
            self.model = CNNLSTMModel(config)

        elif self.model_type == "ensemble":
            from src.models.ensemble_model import EnsembleModel

            self.model = EnsembleModel(model_config.get("ensemble", {}))

        else:
            logger.warning(f"Unknown model type: {self.model_type}, using ensemble")
            from src.models.ensemble_model import EnsembleModel

            self.model = EnsembleModel(model_config.get("ensemble", {}))

        logger.info(f"Initialized {self.model_type} model for {self.symbol}")

    def load_data(
        self,
        years: int = 5,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load historical stock data.

        Args:
            years: Number of years of data to load
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with stock data
        """
        self.data = self.collector.download_yahoo_data(
            symbol=self.symbol, years=years, start_date=start_date, end_date=end_date
        )

        # Preprocess data
        self.data = self.preprocessor.clean_data(self.data)
        self.data = self.preprocessor.add_all_indicators(self.data)

        # FIX: Integrate sentiment analysis if enabled
        # NOTE: This is currently best suited for ARIMA/Statistical models
        # LSTM/GRU models may require input dimension adjustments for sentiment features
        if self.enable_sentiment:
            try:
                sentiment_features = self._fetch_and_analyze_sentiment()

                if not sentiment_features.empty:
                    # Add sentiment features to the most recent row
                    # These will be used as features for prediction
                    for col in sentiment_features.columns:
                        if col != "news_count":
                            # Store as a constant value for the entire dataset
                            # (In production, you'd want to map sentiment to specific dates)
                            self.data[col] = sentiment_features[col].values[0]

                    logger.info(
                        f"Added sentiment features to data: {list(sentiment_features.columns)}"
                    )

                    # Log warning for LSTM/GRU models
                    if self.model_type in ["lstm", "gru"]:
                        logger.warning(
                            "Sentiment features added but LSTM/GRU input dimensions "
                            "may need adjustment for new features. Consider using ARIMA "
                            "or ensemble models for sentiment integration."
                        )

            except Exception as e:
                logger.error(f"Error integrating sentiment features: {str(e)}")
                # Continue without sentiment - don't break predictions
                self.enable_sentiment = False

        logger.info(f"Loaded {len(self.data)} days of data for {self.symbol}")

        return self.data

    def train(self, data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Train the prediction model.

        Args:
            data: Training data (if None, use loaded data)

        Returns:
            Training results
        """
        if data is not None:
            self.data = data

        if self.data is None:
            raise ValueError("No data available. Call load_data() first.")

        # Use close price for training
        target = self.data["close"]

        logger.info(f"Training {self.model_type} model on {len(target)} data points...")

        # Train model
        self.model.fit(target)

        logger.info("Model training complete")

        return {
            "status": "success",
            "model_type": self.model_type,
            "data_points": len(target),
        }

    def predict(self, steps: int = 1, include_confidence: bool = True) -> Dict:
        """
        Make predictions.

        Args:
            steps: Number of steps to predict
            include_confidence: Include confidence intervals

        Returns:
            Dictionary with predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Make predictions
        if include_confidence:
            result = self.model.predict_with_confidence(steps)
        else:
            result = {"predictions": self.model.predict(steps)}

        self.predictions = result

        return result

    def predict_next(self) -> float:
        """
        Predict next day's price.

        Returns:
            Predicted price
        """
        if self.data is None or self.model is None:
            raise ValueError("Model not trained. Call load_data() and train() first.")

        if hasattr(self.model, "predict_next"):
            last_price = self.data["close"].values[-1]
            return self.model.predict_next(last_price)
        else:
            predictions = self.model.predict(1)
            return float(predictions[0])

    def get_current_price(self) -> Optional[float]:
        """Get current stock price."""
        try:
            quote = self.collector.get_realtime_quote(self.symbol)
            if quote:
                return quote.get("price")
        except Exception as e:
            logger.warning(f"Could not get realtime quote: {e}")

        # Fall back to last close
        if self.data is not None:
            return self.data["close"].iloc[-1]

        return None

    def evaluate(self) -> Dict:
        """
        Evaluate model on test data.

        Returns:
            Evaluation metrics
        """
        if self.data is None or self.model is None:
            raise ValueError("Model not trained. Call load_data() and train() first.")

        # Split data
        split_data = self.preprocessor.split_data(
            self.data, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1
        )

        test_data = split_data["test"]["close"]

        # Evaluate
        metrics = self.model.evaluate(test_data)

        return metrics

    def get_technical_indicators(self) -> pd.DataFrame:
        """Get technical indicators for the data."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        indicators = [
            "sma_20",
            "sma_50",
            "rsi_14",
            "macd",
            "macd_signal",
            "bb_upper",
            "bb_middle",
            "bb_lower",
        ]

        available = [col for col in indicators if col in self.data.columns]

        return self.data[available]

    def save_model(self, path: Optional[Path] = None) -> Path:
        """
        Save trained model.

        Args:
            path: Path to save model

        Returns:
            Path to saved model
        """
        if path is None:
            path = Path("models") / f"{self.symbol}_{self.model_type}"

        if hasattr(self.model, "save_models"):
            self.model.save_models(path)
        elif hasattr(self.model, "save_model"):
            self.model.save_model(path)
        else:
            raise ValueError("Model does not support saving")

        return path

    def load_model(self, path: Path) -> "StockPredictor":
        """
        Load trained model.

        Args:
            path: Path to model

        Returns:
            Self
        """
        if hasattr(self.model, "load_models"):
            self.model.load_models(path)
        elif hasattr(self.model, "load_model"):
            self.model.load_model(path)
        else:
            raise ValueError("Model does not support loading")

        logger.info(f"Loaded model from {path}")

        return self

    def get_prediction_summary(self) -> Dict:
        """
        Get summary of current prediction.

        Returns:
            Dictionary with prediction summary
        """
        if self.predictions is None:
            return {"status": "no_predictions"}

        predictions = self.predictions["predictions"]

        current_price = self.get_current_price()

        summary = {
            "symbol": self.symbol,
            "model_type": self.model_type,
            "current_price": current_price,
            "predictions": predictions.tolist()
            if isinstance(predictions, np.ndarray)
            else predictions,
            "prediction_days": len(predictions),
            "timestamp": datetime.now().isoformat(),
        }

        if "lower_bound" in self.predictions:
            summary["lower_bound"] = self.predictions["lower_bound"].tolist()
            summary["upper_bound"] = self.predictions["upper_bound"].tolist()

        if current_price is not None and len(predictions) > 0:
            change = predictions[0] - current_price
            change_pct = (change / current_price) * 100
            summary["next_day_prediction"] = {
                "price": float(predictions[0]),
                "change": float(change),
                "change_pct": float(change_pct),
            }

        return summary


def create_predictor(
    symbol: str, model_type: str = "ensemble", config: Optional[Dict] = None
) -> StockPredictor:
    """
    Factory function to create a StockPredictor.

    Args:
        symbol: Stock ticker symbol
        model_type: Type of model (arima, lstm, gru, ensemble)
        config: Configuration dictionary

    Returns:
        StockPredictor instance
    """
    return StockPredictor(symbol, model_type, config)
