"""
RISK Command

Display risk metrics for a symbol or portfolio.
Usage: RSK <symbol> or RSK
"""

from typing import List
from datetime import datetime, timedelta
import math

import yfinance as yf
import pandas as pd
import numpy as np
from rich.table import Table
from rich.panel import Panel

from src.tui.commands.base import DataCommandHandler
from src.tui.app import CommandResult, PythiaTerminal


class RiskCommand(DataCommandHandler):
    """Display risk metrics and analysis."""
    
    @property
    def name(self) -> str:
        return "RISK"
    
    @property
    def aliases(self) -> List[str]:
        return ["RSK", "RISK"]
    
    @property
    def description(self) -> str:
        return "Display risk metrics and analysis"
    
    @property
    def usage(self) -> str:
        return "RSK <symbol>  or  RSK (portfolio)"
    
    @property
    def requires_symbol(self) -> bool:
        return False
    
    async def execute(self, terminal: PythiaTerminal, args: List[str]) -> CommandResult:
        if args:
            return await self._analyze_symbol(args[0].upper())
        else:
            return await self._analyze_portfolio(terminal)
    
    async def _analyze_symbol(self, symbol: str) -> CommandResult:
        """Analyze risk for a single symbol."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2y")
            
            if hist.empty:
                return CommandResult(False, f"No data available for {symbol}")
            
            # Calculate returns
            returns = hist['Close'].pct_change().dropna()
            
            # Calculate metrics
            volatility = returns.std() * math.sqrt(252) * 100
            sharpe = self._calculate_sharpe(returns)
            var_95 = self._calculate_var(returns, 0.95)
            cvar_95 = self._calculate_cvar(returns, 0.95)
            max_dd = self._calculate_max_drawdown(hist['Close'])
            beta = await self._calculate_beta(symbol)
            
            # Beta to market
            beta_value = beta if beta is not None else 1.0
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() * math.sqrt(252)
            sortino = (returns.mean() * 252) / downside_std if downside_std > 0 else 0
            
            # Create metrics table
            table = Table(title=f"📊 {symbol} Risk Metrics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green", justify="right")
            table.add_column("Interpretation", style="yellow")
            
            # Volatility
            if volatility < 15:
                vol_interp = "Low"
                vol_style = "green"
            elif volatility < 30:
                vol_interp = "Moderate"
                vol_style = "yellow"
            else:
                vol_interp = "High"
                vol_style = "red"
            
            table.add_row(
                "Volatility (Annual)",
                f"[{vol_style}]{volatility:.1f}%[/{vol_style}]",
                vol_interp
            )
            
            # Sharpe
            if sharpe > 1:
                sharpe_interp = "Good"
                sharpe_style = "green"
            elif sharpe > 0:
                sharpe_interp = "Low"
                sharpe_style = "yellow"
            else:
                sharpe_interp = "Poor"
                sharpe_style = "red"
            
            table.add_row(
                "Sharpe Ratio",
                f"[{sharpe_style}]{sharpe:.2f}[/{sharpe_style}]",
                sharpe_interp
            )
            
            # Sortino
            table.add_row(
                "Sortino Ratio",
                f"{sortino:.2f}",
                "Good" if sortino > 1 else "Low"
            )
            
            # VaR
            table.add_row(
                "VaR (95%)",
                f"{var_95*100:.2f}%",
                "Daily loss at 95% confidence"
            )
            
            # CVaR
            table.add_row(
                "CVaR (95%)",
                f"{cvar_95*100:.2f}%",
                "Expected loss beyond VaR"
            )
            
            # Max Drawdown
            table.add_row(
                "Max Drawdown",
                f"[red]{max_dd*100:.2f}%[/red]",
                "Largest peak-to-trough"
            )
            
            # Beta
            beta_style = "green" if beta_value < 1.2 else "yellow" if beta_value < 1.5 else "red"
            table.add_row(
                "Beta",
                f"[{beta_style}]{beta_value:.2f}[/{beta_style}]",
                "Market sensitivity"
            )
            
            panel = Panel(
                f"[bold]Analysis Period:[/bold] 2 years\n"
                f"[bold]Trading Days:[/bold] {len(returns)}",
                title=f"[bold cyan]{symbol} Risk Analysis[/bold cyan]"
            )
            
            return CommandResult(success=True, message="", panel=panel, table=table)
            
        except Exception as e:
            return CommandResult(False, f"Error calculating risk: {str(e)}")
    
    async def _analyze_portfolio(self, terminal: PythiaTerminal) -> CommandResult:
        """Analyze portfolio risk."""
        portfolio = terminal.portfolio
        
        if not portfolio:
            return CommandResult(
                False,
                "No positions in portfolio. Add positions first: PORT ADD <symbol> <shares> <cost>"
            )
        
        # Calculate portfolio-level risk
        symbols = list(portfolio.keys())
        total_value = sum(
            p['shares'] * p.get('current_price', p['cost'] / p['shares'])
            for p in portfolio.values()
        )
        
        # Weights
        weights = []
        for symbol, pos in portfolio.items():
            shares = pos['shares']
            cost_per_share = pos['cost'] / shares if shares > 0 else 0
            value = shares * cost_per_share
            weights.append(value / total_value if total_value > 0 else 0)
        
        # Calculate portfolio volatility (simplified - assumes no correlation)
        port_vol = 0
        for i, (symbol, pos) in enumerate(portfolio.items()):
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1y")
                if not hist.empty:
                    returns = hist['Close'].pct_change().dropna()
                    vol = returns.std() * math.sqrt(252)
                    port_vol += weights[i] * vol
            except Exception:
                pass
        
        port_vol *= 100
        
        # Create summary table
        table = Table(title="📊 Portfolio Risk Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")
        
        table.add_row("Positions", str(len(portfolio)))
        table.add_row("Total Value", f"${total_value:.2f}")
        table.add_row("Portfolio Volatility", f"{port_vol:.1f}%")
        table.add_row("Diversification", "Single Stock" if len(portfolio) < 5 else "Diversified")
        
        panel = Panel(
            "[yellow]Note: Portfolio risk is simplified. Use BT command for full analysis.[/yellow]",
            title="[bold cyan]Portfolio Risk[/bold cyan]"
        )
        
        return CommandResult(success=True, message="", panel=panel, table=table)
    
    def _calculate_sharpe(self, returns: pd.Series, risk_free: float = 0.05) -> float:
        """Calculate Sharpe ratio."""
        if returns.empty:
            return 0
        mean_return = returns.mean() * 252
        std_return = returns.std() * math.sqrt(252)
        if std_return == 0:
            return 0
        return (mean_return - risk_free) / std_return
    
    def _calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        if returns.empty:
            return 0
        return abs(returns.quantile(1 - confidence))
    
    def _calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        if returns.empty:
            return 0
        var = self._calculate_var(returns, confidence)
        return abs(returns[returns <= -var].mean())
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if prices.empty:
            return 0
        cummax = prices.cummax()
        drawdown = (prices - cummax) / cummax
        return abs(drawdown.min())
    
    async def _calculate_beta(self, symbol: str) -> float:
        """Calculate beta relative to market."""
        try:
            # Get SPY as market proxy
            ticker = yf.Ticker(symbol)
            market = yf.Ticker("SPY")
            
            hist = ticker.history(period="2y")['Close']
            market_hist = market.history(period="2y")['Close']
            
            if hist.empty or market_hist.empty:
                return None
            
            # Align data
            combined = pd.DataFrame({'stock': hist, 'market': market_hist}).dropna()
            
            if len(combined) < 30:
                return None
            
            # Calculate returns
            stock_returns = combined['stock'].pct_change().dropna()
            market_returns = combined['market'].pct_change().dropna()
            
            # Calculate covariance and variance
            covariance = stock_returns.cov(market_returns)
            market_variance = market_returns.var()
            
            if market_variance == 0:
                return None
            
            beta = covariance / market_variance
            return beta
            
        except Exception:
            return None
