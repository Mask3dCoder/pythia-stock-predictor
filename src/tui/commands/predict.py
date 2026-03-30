"""
PREDICT Command

AI-powered price prediction for a symbol.
Usage: PRED <symbol> or PREDICT <symbol>
"""

from typing import List
import math
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import numpy as np
from rich.table import Table
from rich.panel import Panel

from src.tui.commands.base import MLCommandHandler
from src.tui.app import CommandResult, PythiaTerminal


class PredictCommand(MLCommandHandler):
    """AI-powered price prediction."""
    
    @property
    def name(self) -> str:
        return "PREDICT"
    
    @property
    def aliases(self) -> List[str]:
        return ["PRED", "PREDICT", "FORECAST"]
    
    @property
    def description(self) -> str:
        return "AI-powered price prediction"
    
    @property
    def usage(self) -> str:
        return "PRED <symbol>  or  PREDICT <symbol>"
    
    @property
    def min_args(self) -> int:
        return 1
    
    async def execute(self, terminal: PythiaTerminal, args: List[str]) -> CommandResult:
        symbol = args[0].upper()
        return await self._predict(terminal, symbol)
    
    async def _predict(self, terminal: PythiaTerminal, symbol: str) -> CommandResult:
        """Generate price prediction."""
        try:
            # Fetch historical data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")
            
            if hist.empty:
                return CommandResult(False, f"No data available for {symbol}")
            
            # Get current price
            current_price = hist['Close'].iloc[-1]
            info = ticker.info
            company_name = info.get('shortName', info.get('longName', symbol))
            
            # Calculate predictions using multiple methods
            predictions = self._calculate_predictions(hist['Close'])
            
            # Create results table
            table = Table(title=f"🔮 {symbol} Price Predictions")
            table.add_column("Timeframe", style="cyan")
            table.add_column("Prediction", style="green", justify="right")
            table.add_column("Change", style="magenta", justify="right")
            table.add_column("Change %", style="magenta", justify="right")
            table.add_column("Method", style="yellow")
            
            for tf, (price, method) in predictions.items():
                change = price - current_price
                change_pct = (change / current_price) * 100
                change_style = "green" if change >= 0 else "red"
                
                table.add_row(
                    tf,
                    f"${price:.2f}",
                    f"[{change_style}]{change:+.2f}[/{change_style}]",
                    f"[{change_style}]{change_pct:+.2f}%[/{change_style}]",
                    method
                )
            
            # Add technical signals
            signals = self._generate_signals(hist)
            
            # Summary panel
            summary = Table(show_header=False)
            summary.add_column("Field", style="cyan")
            summary.add_column("Value", style="green")
            summary.add_row("Current Price", f"${current_price:.2f}")
            summary.add_row("Company", company_name)
            summary.add_row("Overall Signal", signals['overall'])
            summary.add_row("Confidence", signals['confidence'])
            
            panel = Panel(
                summary,
                title=f"[bold cyan]{symbol} Prediction Summary[/bold cyan]"
            )
            
            terminal.current_symbol = symbol
            
            return CommandResult(success=True, message="", panel=panel, table=table)
            
        except Exception as e:
            return CommandResult(False, f"Error generating prediction: {str(e)}")
    
    def _calculate_predictions(self, prices: pd.Series) -> dict:
        """Calculate predictions using multiple methods."""
        predictions = {}
        
        # Method 1: Linear Regression
        y = prices.values
        x = np.arange(len(y))
        coeffs = np.polyfit(x, y, 1)
        
        # Next 5, 10, 30 days
        for days, label in [(5, "5 Days"), (10, "10 Days"), (30, "30 Days")]:
            future_x = len(y) - 1 + days
            pred_price = np.polyval(coeffs, future_x)
            predictions[label] = (pred_price, "Linear Regression")
        
        # Method 2: Moving Average projection
        ma20 = prices.rolling(20).mean().iloc[-1]
        ma50 = prices.rolling(50).mean().iloc[-1]
        
        # Trend based on MA crossover
        trend = "bullish" if ma20 > ma50 else "bearish"
        adjustment = 1.01 if trend == "bullish" else 0.99
        
        # Apply to current price
        for days, label in [(5, "5 Days"), (10, "10 Days"), (30, "30 Days")]:
            factor = adjustment ** (days / 10)
            pred_price = prices.iloc[-1] * factor
            if label not in predictions:
                predictions[label] = (pred_price, "MA Projection")
        
        # Method 3: Exponential Smoothing
        from scipy.ndimage import uniform_filter1d
        smoothed = uniform_filter1d(prices.values, size=20)
        slope = (smoothed[-1] - smoothed[-20]) / 20
        
        for days, label in [(5, "5 Days"), (10, "10 Days"), (30, "30 Days")]:
            pred_price = prices.iloc[-1] + slope * days
            if label not in predictions:
                predictions[label] = (pred_price, "Exp. Smoothing")
        
        return predictions
    
    def _generate_signals(self, hist: pd.DataFrame) -> dict:
        """Generate trading signals based on technical analysis."""
        close = hist['Close']
        
        # Moving averages
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean() if len(close) >= 200 else None
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_current = rsi.iloc[-1]
        
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        
        # Signals
        signals = []
        
        # MA signals
        if ma20.iloc[-1] > ma50.iloc[-1]:
            signals.append(("MA Cross", "Bullish"))
        else:
            signals.append(("MA Cross", "Bearish"))
        
        if ma200 is not None and close.iloc[-1] > ma200.iloc[-1]:
            signals.append(("Price vs MA200", "Above"))
        else:
            signals.append(("Price vs MA200", "Below"))
        
        # RSI signals
        if rsi_current > 70:
            signals.append(("RSI", "Overbought"))
        elif rsi_current < 30:
            signals.append(("RSI", "Oversold"))
        else:
            signals.append(("RSI", "Neutral"))
        
        # MACD signals
        if macd.iloc[-1] > signal.iloc[-1]:
            signals.append(("MACD", "Bullish"))
        else:
            signals.append(("MACD", "Bearish"))
        
        # Determine overall
        bullish_count = sum(1 for _, s in signals if s in ["Bullish", "Above", "Oversold"])
        bearish_count = sum(1 for _, s in signals if s in ["Bearish", "Below", "Overbought"])
        
        if bullish_count > bearish_count:
            overall = "[green]BULLISH[/green]"
            confidence = f"{int(bullish_count/len(signals)*100)}%"
        elif bearish_count > bullish_count:
            overall = "[red]BEARISH[/red]"
            confidence = f"{int(bearish_count/len(signals)*100)}%"
        else:
            overall = "[yellow]NEUTRAL[/yellow]"
            confidence = "50%"
        
        return {
            'signals': signals,
            'overall': overall,
            'confidence': confidence
        }
