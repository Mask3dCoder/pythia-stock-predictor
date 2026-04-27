"""
Backtesting Engine

Comprehensive backtesting system with:
- Realistic transaction costs (commission, slippage, spread)
- Position management
- Performance metrics
- Walk-forward validation
"""

import logging
from typing import Optional, Dict, List, Tuple, Callable, Any
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PositionSide(Enum):
    """Position direction."""
    LONG = 1
    SHORT = -1
    FLAT = 0


class OrderType(Enum):
    """Order types."""
    MARKET = 'market'
    LIMIT = 'limit'
    STOP = 'stop'


class TransactionCosts:
    """
    Transaction cost calculator.
    
    Includes:
    - Commission (percentage or fixed)
    - Slippage (percentage or fixed)
    - Bid-ask spread
    - Market impact (for large orders)
    """
    
    def __init__(
        self,
        commission_pct: float = 0.001,
        commission_fixed: float = 0.0,
        slippage_pct: float = 0.0005,
        slippage_fixed: float = 0.0,
        spread_pct: float = 0.0002,
        mincommission: float = 1.0
    ):
        """
        Initialize transaction costs.
        
        Args:
            commission_pct: Commission as percentage of trade value
            commission_fixed: Fixed commission per trade
            slippage_pct: Slippage as percentage
            slippage_fixed: Fixed slippage per trade
            spread_pct: Bid-ask spread as percentage
            mincommission: Minimum commission per trade
        """
        self.commission_pct = commission_pct
        self.commission_fixed = commission_fixed
        self.slippage_pct = slippage_pct
        self.slippage_fixed = slippage_fixed
        self.spread_pct = spread_pct
        self.mincommission = mincommission
        
    def calculate(
        self,
        price: float,
        quantity: float,
        direction: int
    ) -> float:
        """
        Calculate total transaction cost.
        
        Args:
            price: Execution price
            quantity: Number of shares
            direction: 1 for buy, -1 for sell
            
        Returns:
            Total cost
        """
        trade_value = abs(price * quantity)
        
        # Commission
        commission = max(
            trade_value * self.commission_pct + self.commission_fixed,
            self.mincommission
        )
        
        # Slippage
        slippage = trade_value * self.slippage_pct + self.slippage_fixed
        
        # Spread (only applies to round trip)
        spread = trade_value * self.spread_pct
        
        total_cost = commission + slippage + spread
        
        return total_cost
    
    def apply_to_trades(
        self,
        trades: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply transaction costs to trade log.
        
        Args:
            trades: DataFrame with trade information
            
        Returns:
            Updated trades DataFrame with costs
        """
        trades = trades.copy()
        
        costs = trades.apply(
            lambda row: self.calculate(
                row['price'],
                row['quantity'],
                row['direction']
            ),
            axis=1
        )
        
        trades['cost'] = costs
        trades['net_pnl'] = trades['pnl'] - costs
        
        return trades


class BacktestEngine:
    """
    Comprehensive backtesting engine.
    
    Features:
    - Long/short trading
    - Position sizing
    - Stop loss / take profit
    - Transaction costs
    - Performance analytics
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_costs: Optional[TransactionCosts] = None,
        max_position_pct: float = 1.0,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        allow_shorting: bool = True
    ):
        """
        Initialize backtest engine.
        
        Args:
            initial_capital: Starting capital
            transaction_costs: Transaction cost calculator
            max_position_pct: Maximum position size (% of capital)
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            allow_shorting: Allow short positions
        """
        self.initial_capital = initial_capital
        self.transaction_costs = transaction_costs or TransactionCosts()
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.allow_shorting = allow_shorting
        
        # State
        self.capital = initial_capital
        self.position = 0  # Shares held
        self.position_side = PositionSide.FLAT
        self.entry_price = 0.0
        
        # Trade log
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
        self.dates: List[datetime] = []
        
    def reset(self) -> None:
        """Reset the engine to initial state."""
        self.capital = self.initial_capital
        self.position = 0
        self.position_side = PositionSide.FLAT
        self.entry_price = 0.0
        self.trades = []
        self.equity_curve = []
        self.dates = []
    
    def generate_signals(
        self,
        predictions: np.ndarray,
        prices: np.ndarray,
        threshold: float = 0.0
    ) -> np.ndarray:
        """
        Generate trading signals from predictions.
        
        Args:
            predictions: Model predictions
            prices: Actual prices
            threshold: Threshold for signal generation
            
        Returns:
            Array of signals (-1, 0, 1)
        """
        signals = np.zeros(len(predictions))
        
        # Simple signal: predict direction
        returns = predictions - prices
        
        # Buy signal
        signals[returns > threshold] = 1
        
        # Sell/short signal
        if self.allow_shorting:
            signals[returns < -threshold] = -1
            
        return signals
    
    def run(
        self,
        prices: pd.Series,
        signals: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> Dict[str, Any]:
        """
        Run backtest.
        
        Args:
            prices: Price series
            signals: Trading signals
            dates: Date index
            
        Returns:
            Dictionary with results
        """
        self.reset()
        
        if dates is None:
            dates = pd.date_range(start='2020-01-01', periods=len(prices), freq='D')
            
        for i in range(len(prices)):
            price = prices.iloc[i]
            signal = signals[i]
            date = dates[i] if i < len(dates) else None
            
            # Update equity
            current_equity = self.capital + self.position * price
            self.equity_curve.append(current_equity)
            if date:
                self.dates.append(date)
            
            # Process signal
            if signal > 0 and self.position_side == PositionSide.FLAT:
                # Buy signal - go long
                self._open_position(price, 1, date)
                
            elif signal < 0 and self.position_side == PositionSide.FLAT and self.allow_shorting:
                # Short signal
                self._open_position(price, -1, date)
                
            elif signal == 0 and self.position_side != PositionSide.FLAT:
                # Flat signal - close position
                self._close_position(price, date)
                
            # Check stop loss / take profit
            if self.position_side != PositionSide.FLAT:
                self._check_stops(price, date)
                
        # Close any remaining position
        if self.position_side != PositionSide.FLAT:
            self._close_position(prices.iloc[-1], dates[-1] if dates is not None else None)
            
        return self._calculate_metrics()
    
    def _open_position(
        self,
        price: float,
        direction: int,
        date: Optional[datetime]
    ) -> None:
        """Open a position."""
        # Calculate position size
        max_shares = int(self.capital * self.max_position_pct / price)
        
        if max_shares <= 0:
            return
            
        # Calculate cost
        cost = self.transaction_costs.calculate(price, max_shares, direction)
        
        # Update capital
        if direction > 0:
            # Long: pay for shares plus transaction cost
            self.capital -= (price * max_shares + cost)
        else:
            # Short: receive money
            self.capital += price * max_shares - cost
            
        self.position = max_shares * direction
        self.position_side = PositionSide.LONG if direction > 0 else PositionSide.SHORT
        self.entry_price = price
        
        self.trades.append({
            'date': date,
            'action': 'open',
            'direction': direction,
            'price': price,
            'quantity': max_shares,
            'capital': self.capital
        })
    
    def _close_position(
        self,
        price: float,
        date: Optional[datetime]
    ) -> None:
        """Close current position."""
        if self.position == 0:
            return
            
        direction = -1 if self.position > 0 else 1
        
        # Calculate P&L
        pnl = self.position * (price - self.entry_price)
        
        # Calculate cost
        cost = self.transaction_costs.calculate(
            price, 
            abs(self.position),
            direction
        )
        
        # Update capital (same formula for long and short)
        # For longs: position > 0 → receive money from selling
        # For shorts: position < 0 → pay money to buy back
        self.capital += self.position * price - cost
            
        self.trades.append({
            'date': date,
            'action': 'close',
            'direction': direction,
            'price': price,
            'quantity': abs(self.position),
            'pnl': pnl,
            'cost': cost,
            'net_pnl': pnl - cost,
            'capital': self.capital
        })
        
        self.position = 0
        self.position_side = PositionSide.FLAT
        self.entry_price = 0.0
    
    def _check_stops(
        self,
        price: float,
        date: Optional[datetime]
    ) -> None:
        """Check stop loss and take profit."""
        if self.position == 0:
            return
            
        pnl_pct = (price - self.entry_price) / self.entry_price
        
        if self.position < 0:
            pnl_pct = -pnl_pct
            
        # Stop loss
        if self.stop_loss_pct and pnl_pct <= -self.stop_loss_pct:
            self._close_position(price, date)
            self.trades[-1]['exit_reason'] = 'stop_loss'
            
        # Take profit
        elif self.take_profit_pct and pnl_pct >= self.take_profit_pct:
            self._close_position(price, date)
            self.trades[-1]['exit_reason'] = 'take_profit'
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics."""
        if not self.equity_curve:
            return {}
            
        equity = np.array(self.equity_curve)
        
        # Returns
        returns = np.diff(equity) / equity[:-1]
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        
        # Basic metrics
        total_return = (equity[-1] - self.initial_capital) / self.initial_capital
        annual_return = total_return * 252 / len(equity)
        
        # Risk metrics
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
        
        # Sharpe ratio
        if volatility > 0:
            sharpe = annual_return / volatility
        else:
            sharpe = 0.0
            
        # Max drawdown
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        # Calmar ratio
        if max_drawdown != 0:
            calmar = annual_return / abs(max_drawdown)
        else:
            calmar = 0.0
            
        # Win rate
        closed_trades = [t for t in self.trades if t.get('action') == 'close']
        if closed_trades:
            wins = sum(1 for t in closed_trades if t.get('net_pnl', 0) > 0)
            win_rate = wins / len(closed_trades)
        else:
            win_rate = 0.0
            
        # Profit factor
        gross_profit = sum(t.get('net_pnl', 0) for t in closed_trades if t.get('net_pnl', 0) > 0)
        gross_loss = abs(sum(t.get('net_pnl', 0) for t in closed_trades if t.get('net_pnl', 0) < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': equity[-1],
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(closed_trades),
            'equity_curve': equity,
            'dates': self.dates,
            'trades': pd.DataFrame(closed_trades)
        }


class WalkForwardBacktest:
    """
    Walk-forward backtesting with expanding window.
    
    Trains model on historical data, tests on forward window,
    then rolls forward and repeats.
    """
    
    def __init__(
        self,
        train_period: int = 252,
        test_period: int = 21,
        step_period: Optional[int] = None
    ):
        """
        Initialize walk-forward backtest.
        
        Args:
            train_period: Training window size (days)
            test_period: Test window size (days)
            step_period: Step between windows
        """
        self.train_period = train_period
        self.test_period = test_period
        self.step_period = step_period or test_period
        
    def run(
        self,
        prices: pd.Series,
        model_factory: Callable,
        feature_extractor: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run walk-forward backtest.
        
        Args:
            prices: Price series
            model_factory: Function to create and train model
            feature_extractor: Function to extract features
            
        Returns:
            Aggregated results
        """
        all_results = []
        
        start = self.train_period
        
        while start + self.test_period <= len(prices):
            train_data = prices.iloc[max(0, start - self.train_period):start]
            test_data = prices.iloc[start:start + self.test_period]
            
            try:
                # Train model
                model = model_factory(train_data)
                
                # Generate predictions
                if feature_extractor:
                    features = feature_extractor(test_data)
                else:
                    features = test_data.values
                    
                predictions = model.predict(features)
                
                # Generate signals
                signals = np.sign(predictions - test_data.values)
                
                # Run backtest for this window
                engine = BacktestEngine(initial_capital=100000)
                result = engine.run(test_data, signals)
                
                result['train_period'] = (start - self.train_period, start)
                result['test_period'] = (start, start + self.test_period)
                
                all_results.append(result)
                
            except Exception as e:
                logger.warning(f"Error in walk-forward window {start}: {e}")
                
            start += self.step_period
            
        # Aggregate results
        return self._aggregate_results(all_results)
    
    def _aggregate_results(
        self,
        results: List[Dict]
    ) -> Dict[str, Any]:
        """Aggregate results across windows."""
        if not results:
            return {}
            
        return {
            'n_windows': len(results),
            'avg_return': np.mean([r['total_return'] for r in results]),
            'avg_sharpe': np.mean([r['sharpe_ratio'] for r in results]),
            'avg_drawdown': np.mean([r['max_drawdown'] for r in results]),
            'win_rate': np.mean([r['win_rate'] for r in results]),
            'total_trades': sum(r['total_trades'] for r in results),
            'windows': results
        }
