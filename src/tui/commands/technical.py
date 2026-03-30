"""
TECHNICAL INDICATORS Command

Display technical indicators for a symbol.
Usage: IN <indicator> <symbol> or INDICATORS <symbol>
"""

import asyncio
from typing import List
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import numpy as np
from rich.table import Table
from rich.panel import Panel

from src.tui.commands.base import AnalyticsCommandHandler
from src.tui.app import CommandResult, PythiaTerminal, CommandCategory


class TechnicalCommand(AnalyticsCommandHandler):
    """Display technical indicators."""
    
    @property
    def name(self) -> str:
        return "INDICATORS"
    
    @property
    def aliases(self) -> List[str]:
        return ["IN", "IND", "INDICATOR", "INDICATORS"]
    
    @property
    def description(self) -> str:
        return "Display technical indicators"
    
    @property
    def usage(self) -> str:
        return "IN [indicator] <symbol>  or  INDICATORS <symbol>"
    
    @property
    def min_args(self) -> int:
        return 1
    
    async def execute(self, terminal: PythiaTerminal, args: List[str]) -> CommandResult:
        # Check if first arg is indicator or symbol
        indicators = ['RSI', 'MACD', 'SMA', 'EMA', 'BB', 'ATR', 'STOCH', 'ALL']
        
        if args[0].upper() in indicators:
            indicator = args[0].upper()
            symbol = args[1].upper() if len(args) > 1 else terminal.current_symbol
        else:
            indicator = 'ALL'
            symbol = args[0].upper()
        
        if not symbol:
            return CommandResult(False, "Please specify a symbol")
        
        try:
            # Fetch data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="6mo")
            
            if hist.empty:
                return CommandResult(False, f"No data found for {symbol}")
            
            # Calculate indicators
            df = self._calculate_indicators(hist)
            
            # Create table
            table = Table(title=f"📊 {symbol} Technical Indicators")
            table.add_column("Indicator", style="cyan")
            table.add_column("Value", style="green")
            table.add_column("Signal", style="yellow")
            
            # RSI
            rsi = df['RSI'].iloc[-1]
            if rsi > 70:
                rsi_signal = "🔴 Overbought"
            elif rsi < 30:
                rsi_signal = "🟢 Oversold"
            else:
                rsi_signal = "🟡 Neutral"
            table.add_row("RSI (14)", f"{rsi:.1f}", rsi_signal)
            
            # MACD
            macd = df['MACD'].iloc[-1]
            signal = df['MACD_Signal'].iloc[-1]
            if macd > signal:
                macd_signal = "🟢 Bullish"
            else:
                macd_signal = "🔴 Bearish"
            table.add_row("MACD", f"{macd:.2f}", macd_signal)
            
            # SMA
            for period in [20, 50, 200]:
                sma = df[f'SMA_{period}'].iloc[-1]
                price = df['Close'].iloc[-1]
                if price > sma:
                    sma_signal = "🟢 Above"
                else:
                    sma_signal = "🔴 Below"
                table.add_row(f"SMA {period}", f"${sma:.2f}", sma_signal)
            
            # Bollinger Bands
            bb_upper = df['BB_Upper'].iloc[-1]
            bb_lower = df['BB_Lower'].iloc[-1]
            price = df['Close'].iloc[-1]
            
            if price > bb_upper:
                bb_signal = "🔴 Above Upper"
            elif price < bb_lower:
                bb_signal = "🟢 Below Lower"
            else:
                bb_signal = "🟡 Within Bands"
            table.add_row("Bollinger", f"${bb_lower:.2f}-${bb_upper:.2f}", bb_signal)
            
            # ATR
            atr = df['ATR'].iloc[-1]
            table.add_row("ATR (14)", f"${atr:.2f}", "Volatility")
            
            # Stochastic
            stoch_k = df['Stoch_K'].iloc[-1]
            stoch_d = df['Stoch_D'].iloc[-1]
            if stoch_k > 80:
                stoch_signal = "🔴 Overbought"
            elif stoch_k < 20:
                stoch_signal = "🟢 Oversold"
            else:
                stoch_signal = "🟡 Neutral"
            table.add_row(f"Stochastic", f"{stoch_k:.1f}/{stoch_d:.1f}", stoch_signal)
            
            panel = Panel(
                table,
                title=f"📈 {symbol} Technical Analysis",
                subtitle=f"Data from {df.index[-1].strftime('%Y-%m-%d')}"
            )
            
            return CommandResult(
                success=True,
                message="",
                panel=panel
            )
            
        except Exception as e:
            return CommandResult(False, f"Error calculating indicators: {str(e)}")
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # SMA
        df['SMA_20'] = close.rolling(20).mean()
        df['SMA_50'] = close.rolling(50).mean()
        df['SMA_200'] = close.rolling(200).mean()
        
        # EMA
        df['EMA_20'] = close.ewm(span=20).mean()
        
        # Bollinger Bands
        sma = close.rolling(20).std()
        df['BB_Upper'] = df['SMA_20'] + (sma * 2)
        df['BB_Lower'] = df['SMA_20'] - (sma * 2)
        
        # ATR
        high_low = high - low
        high_close = (high - close.shift()).abs()
        low_close = (low - close.shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        
        # Stochastic
        low_min = low.rolling(14).min()
        high_max = high.rolling(14).max()
        df['Stoch_K'] = 100 * (close - low_min) / (high_max - low_min)
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
        
        return df
