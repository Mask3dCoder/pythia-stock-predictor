"""
Stock Plotter Module

Creates interactive charts for stock data visualization.
"""

import logging
from typing import Optional, Dict, List

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class StockPlotter:
    """Creates stock visualization charts."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Stock Plotter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
    def plot_candlestick(
        self,
        df: pd.DataFrame,
        title: str = "Stock Price",
        save_path: Optional[str] = None
    ):
        """
        Plot candlestick chart.
        
        Args:
            df: DataFrame with OHLC data
            title: Chart title
            save_path: Path to save chart
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create figure
        fig = go.Figure(data=[
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            )
        ])
        
        # Update layout
        fig.update_layout(
            title=title,
            yaxis_title='Price ($)',
            xaxis_title='Date',
            template='plotly_dark',
            xaxis_rangeslider_visible=False
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved chart to {save_path}")
        else:
            fig.show()
            
    def plot_price_with_indicators(
        self,
        df: pd.DataFrame,
        indicators: Optional[List[str]] = None,
        title: str = "Stock Price with Indicators",
        save_path: Optional[str] = None
    ):
        """
        Plot price with technical indicators.
        
        Args:
            df: DataFrame with stock data
            indicators: List of indicator columns to plot
            title: Chart title
            save_path: Path to save chart
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Default indicators
        if indicators is None:
            indicators = ['sma_20', 'sma_50', 'sma_200']
            
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Add candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add moving averages
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        
        for i, indicator in enumerate(indicators):
            if indicator in df.columns:
                color = colors[i % len(colors)]
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[indicator],
                        mode='lines',
                        name=indicator,
                        line=dict(color=color)
                    ),
                    row=1, col=1
                )
                
        # Add volume
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker_color='gray',
                opacity=0.5
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            template='plotly_dark',
            height=800,
            xaxis_rangeslider_visible=False
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
            
    def plot_prediction(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        title: str = "Predictions vs Actual",
        save_path: Optional[str] = None
    ):
        """
        Plot predictions vs actual values.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            dates: Date index
            title: Chart title
            save_path: Path to save chart
        """
        import plotly.graph_objects as go
        
        # Create x-axis
        if dates is not None:
            x = dates
        else:
            x = np.arange(len(actual))
            
        # Create figure
        fig = go.Figure()
        
        # Add actual line
        fig.add_trace(go.Scatter(
            x=x,
            y=actual,
            mode='lines',
            name='Actual',
            line=dict(color='blue')
        ))
        
        # Add predicted line
        fig.add_trace(go.Scatter(
            x=x,
            y=predicted,
            mode='lines',
            name='Predicted',
            line=dict(color='red', dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price ($)',
            template='plotly_dark',
            hovermode='x unified'
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
            
    def plot_sentiment(
        self,
        df: pd.DataFrame,
        date_column: str = 'date',
        sentiment_column: str = 'sentiment',
        title: str = "Sentiment Analysis",
        save_path: Optional[str] = None
    ):
        """
        Plot sentiment over time.
        
        Args:
            df: DataFrame with sentiment data
            date_column: Date column name
            sentiment_column: Sentiment score column
            title: Chart title
            save_path: Path to save chart
        """
        import plotly.graph_objects as go
        
        # Create figure
        fig = go.Figure()
        
        # Sentiment line
        fig.add_trace(go.Scatter(
            x=df[date_column],
            y=df[sentiment_column],
            mode='lines+markers',
            name='Sentiment',
            line=dict(color='green'),
            marker=dict(
                size=8,
                color=df[sentiment_column],
                colorscale='RdYlGn',
                showscale=True
            )
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Sentiment Score',
            template='plotly_dark'
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
            
    def plot_correlation(
        self,
        df: pd.DataFrame,
        columns: List[str],
        title: str = "Feature Correlation",
        save_path: Optional[str] = None
    ):
        """
        Plot correlation heatmap.
        
        Args:
            df: DataFrame with data
            columns: Columns to correlate
            title: Chart title
            save_path: Path to save chart
        """
        import plotly.graph_objects as go
        import seaborn as sns
        
        # Calculate correlation
        corr = df[columns].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale='RdBu',
            zmid=0,
            text=corr.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title=title,
            template='plotly_dark',
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
            
    def plot_model_comparison(
        self,
        results: Dict[str, Dict],
        metric: str = 'mae',
        title: str = "Model Comparison",
        save_path: Optional[str] = None
    ):
        """
        Plot comparison of different models.
        
        Args:
            results: Dictionary of model results
            metric: Metric to compare
            title: Chart title
            save_path: Path to save chart
        """
        import plotly.graph_objects as go
        
        models = list(results.keys())
        values = [results[m].get(metric, 0) for m in models]
        
        fig = go.Figure(data=[
            go.Bar(
                x=models,
                y=values,
                marker_color=['blue', 'orange', 'green', 'red']
            )
        ])
        
        fig.update_layout(
            title=f"{title} - {metric.upper()}",
            xaxis_title='Model',
            yaxis_title=metric.upper(),
            template='plotly_dark'
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
            
    def create_dashboard_html(
        self,
        df: pd.DataFrame,
        predictions: Optional[Dict] = None,
        title: str = "Stock Analysis Dashboard"
    ) -> str:
        """
        Create complete HTML dashboard.
        
        Args:
            df: Stock data DataFrame
            predictions: Predictions dictionary
            title: Dashboard title
            
        Returns:
            HTML string
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            specs=[
                [{"colspan": 2}, None],
                [{"colspan": 1}, {"colspan": 1}],
                [{"colspan": 2}, None]
            ],
            subplot_titles=(
                'Price History', 'Volume', 
                'Technical Indicators', 'Model Predictions'
            ),
            vertical_spacing=0.1
        )
        
        # Add price chart
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Add volume
        colors = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green' 
                  for i in range(len(df))]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.5
            ),
            row=2, col=1
        )
        
        # Add SMA if available
        if 'sma_20' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['sma_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='orange')
                ),
                row=3, col=1
            )
            
        if 'sma_50' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['sma_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='green')
                ),
                row=3, col=1
            )
            
        # Add predictions if available
        if predictions and 'predictions' in predictions:
            pred = predictions['predictions']
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(pred)),
                    y=pred,
                    mode='lines',
                    name='Predictions',
                    line=dict(color='red', dash='dash')
                ),
                row=3, col=2
            )
            
        # Update layout
        fig.update_layout(
            title=title,
            template='plotly_dark',
            height=900,
            showlegend=True
        )
        
        return fig.to_html()
