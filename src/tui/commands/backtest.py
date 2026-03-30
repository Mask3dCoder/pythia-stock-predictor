"""
BACKTEST Command

Run backtest for a trading strategy on a symbol.
Usage: BT <symbol> or BACKTEST <symbol>
"""

from typing import List
import math
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import numpy as np
from rich.table import Table
from rich.panel import Panel

from src.tui.commands.base import AnalyticsCommandHandler
from src.tui.app import CommandResult, PythiaTerminal


class BacktestCommand(AnalyticsCommandHandler):
    """Run backtest for trading strategies."""
    
    @property
    def name(self) -> str:
        return "BACKTEST"
    
    @property
    def aliases(self) -> List[str]:
        return ["BT", "BACKTEST", "BACK"]
    
    @property
    def description(self) -> str:
        return "Run backtest for trading strategies"
    
    @property
    def usage(self) -> str:
        return "BT <symbol> [days]  or  BACKTEST <symbol> [days]"
    
    @property
    def min_args(self) -> int:
        return 1
    
    async def execute(self, terminal: PythiaTerminal, args: List[str]) -> CommandResult:
        symbol = args[0].upper()
        days = int(args[1]) if len(args) > 1 else 365
        return await self._backtest(terminal, symbol, days)
    
    async def _backtest(self, terminal: PythiaTerminal, symbol: str, days: int) -> CommandResult:
        """Run backtest for a symbol."""
        try:
            # Fetch historical data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{days + 60}d")  # Extra for lookback
            
            if hist.empty or len(hist) < 60:
                return CommandResult(False, f"Insufficient data for {symbol}")
            
            # Get current price
            current_price = hist['Close'].iloc[-1]
            info = ticker.info
            company_name = info.get('shortName', info.get('longName', symbol))
            
            # Run multiple strategies
            results = {}
            
            # Strategy 1: Buy and Hold
            results['Buy & Hold'] = self._backtest_buy_and_hold(hist)
            
            # Strategy 2: MA Crossover
            results['MA Cross'] = self._backtest_ma_crossover(hist)
            
            # Strategy 3: RSI Mean Reversion
            results['RSI Reversion'] = self._backtest_rsi_reversion(hist)
            
            # Strategy 4: MACD
            results['MACD'] = self._backtest_macd(hist)
            
            # Create results table
            table = Table(title=f"📈 {symbol} Backtest Results ({days} days)")
            table.add_column("Strategy", style="cyan")
            table.add_column("Return %", style="green", justify="right")
            table.add_column("Max DD %", style="red", justify="right")
            table.add_column("Sharpe", style="yellow", justify="right")
            table.add_column("Trades", style="magenta", justify="right")
            
            best_return = -float('inf')
            best_strategy = None
            
            for strategy, metrics in results.items():
                return_pct = metrics['return_pct']
                if return_pct > best_return:
                    best_return = return_pct
                    best_strategy = strategy
                
                style = "green" if return_pct >= 0 else "red"
                table.add_row(
                    strategy,
                    f"[{style}]{return_pct:+.2f}%[/{style}]",
                    f"[red]{metrics['max_drawdown']:.2f}%[/red]",
                    f"{metrics['sharpe']:.2f}",
                    str(metrics['trades'])
                )
            
            # Summary panel
            summary = Table(show_header=False)
            summary.add_column("Field", style="cyan")
            summary.add_column("Value", style="green")
            summary.add_row("Symbol", symbol)
            summary.add_row("Company", company_name)
            summary.add_row("Period", f"{days} days")
            summary.add_row("Start Price", f"${hist['Close'].iloc[0]:.2f}")
            summary.add_row("End Price", f"${current_price:.2f}")
            summary.add_row("Best Strategy", f"[green]{best_strategy}[/green]")
            
            panel = Panel(
                summary,
                title=f"[bold cyan]{symbol} Backtest Summary[/bold cyan]"
            )
            
            terminal.current_symbol = symbol
            
            return CommandResult(success=True, message="", panel=panel, table=table)
            
        except Exception as e:
            return CommandResult(False, f"Error running backtest: {str(e)}")
    
    def _backtest_buy_and_hold(self, hist: pd.DataFrame) -> dict:
        """Simple buy and hold strategy."""
        prices = hist['Close']
        
        start_price = prices.iloc[0]
        end_price = prices.iloc[-1]
        
        total_return = (end_price - start_price) / start_price
        
        # Calculate max drawdown
        cummax = prices.cummax()
        drawdown = (prices - cummax) / cummax
        max_dd = drawdown.min() * 100
        
        # Calculate Sharpe (annualized)
        returns = prices.pct_change().dropna()
        sharpe = self._calculate_sharpe(returns)
        
        return {
            'return_pct': total_return * 100,
            'max_drawdown': abs(max_dd),
            'sharpe': sharpe,
            'trades': 1
        }
    
    def _backtest_ma_crossover(self, hist: pd.DataFrame) -> dict:
        """MA Crossover strategy."""
        prices = hist['Close'].copy()
        
        ma20 = prices.rolling(20).mean()
        ma50 = prices.rolling(50).mean()
        
        position = 0  # 0 = no position, 1 = long
        entry_price = 0
        trades = 0
        pnl_pct = 0
        portfolio_value = 10000
        peak = portfolio_value
        
        for i in range(50, len(prices)):
            if ma20.iloc[i] > ma50.iloc[i] and position == 0:
                # Buy signal
                position = 1
                entry_price = prices.iloc[i]
                trades += 1
            elif ma20.iloc[i] < ma50.iloc[i] and position == 1:
                # Sell signal
                position = 0
                exit_price = prices.iloc[i]
                pnl_pct += (exit_price - entry_price) / entry_price
                entry_price = 0
        
        # Close any open position
        if position == 1:
            pnl_pct += (prices.iloc[-1] - entry_price) / entry_price
        
        # Calculate drawdown
        returns = prices.pct_change().dropna()
        max_dd = self._calculate_max_drawdown(prices)
        
        sharpe = self._calculate_sharpe(returns) if len(returns) > 0 else 0
        
        return {
            'return_pct': pnl_pct * 100,
            'max_drawdown': max_dd * 100,
            'sharpe': sharpe,
            'trades': trades
        }
    
    def _backtest_rsi_reversion(self, hist: pd.DataFrame) -> dict:
        """RSI Mean Reversion strategy."""
        prices = hist['Close'].copy()
        
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        position = 0
        entry_price = 0
        trades = 0
        pnl_pct = 0
        
        for i in range(20, len(prices)):
            if rsi.iloc[i] < 30 and position == 0:
                # Oversold - Buy
                position = 1
                entry_price = prices.iloc[i]
                trades += 1
            elif rsi.iloc[i] > 70 and position == 1:
                # Overbought - Sell
                position = 0
                exit_price = prices.iloc[i]
                pnl_pct += (exit_price - entry_price) / entry_price
        
        if position == 1:
            pnl_pct += (prices.iloc[-1] - entry_price) / entry_price
        
        max_dd = self._calculate_max_drawdown(prices)
        returns = prices.pct_change().dropna()
        sharpe = self._calculate_sharpe(returns)
        
        return {
            'return_pct': pnl_pct * 100,
            'max_drawdown': max_dd * 100,
            'sharpe': sharpe,
            'trades': trades
        }
    
    def _backtest_macd(self, hist: pd.DataFrame) -> dict:
        """MACD strategy."""
        prices = hist['Close'].copy()
        
        ema12 = prices.ewm(span=12, adjust=False).mean()
        ema26 = prices.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        
        position = 0
        entry_price = 0
        trades = 0
        pnl_pct = 0
        
        for i in range(30, len(prices)):
            if macd.iloc[i] > signal.iloc[i] and position == 0:
                position = 1
                entry_price = prices.iloc[i]
                trades += 1
            elif macd.iloc[i] < signal.iloc[i] and position == 1:
                position = 0
                exit_price = prices.iloc[i]
                pnl_pct += (exit_price - entry_price) / entry_price
        
        if position == 1:
            pnl_pct += (prices.iloc[-1] - entry_price) / entry_price
        
        max_dd = self._calculate_max_drawdown(prices)
        returns = prices.pct_change().dropna()
        sharpe = self._calculate_sharpe(returns)
        
        return {
            'return_pct': pnl_pct * 100,
            'max_drawdown': max_dd * 100,
            'sharpe': sharpe,
            'trades': trades
        }
    
    def _calculate_sharpe(self, returns: pd.Series, risk_free: float = 0.05) -> float:
        """Calculate Sharpe ratio."""
        if returns.empty or returns.std() == 0:
            return 0
        mean_return = returns.mean() * 252
        std_return = returns.std() * math.sqrt(252)
        return (mean_return - risk_free) / std_return
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cummax = prices.cummax()
        drawdown = (prices - cummax) / cummax
        return abs(drawdown.min())
