"""
Risk Management Layer

Implements position sizing based on:
- Model uncertainty
- Market volatility
- Portfolio risk limits
- Kelly criterion
"""

import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Risk metrics container."""
    position_size: float
    stop_loss: float
    take_profit: float
    max_risk: float
    kelly_fraction: float
    var_95: float
    confidence: float


class RiskManager:
    """
    Risk management layer for position sizing and risk control.
    
    Features:
    - Kelly criterion for optimal position sizing
    - Volatility-based sizing
    - Uncertainty-based sizing
    - Value at Risk (VaR) calculation
    - Maximum drawdown protection
    """
    
    def __init__(
        self,
        max_position_pct: float = 0.2,
        max_portfolio_risk: float = 0.06,
        min_stop_loss_pct: float = 0.02,
        max_stop_loss_pct: float = 0.1,
        use_kelly: bool = True,
        kelly_fraction: float = 0.25,
        use_var: bool = True,
        confidence_level: float = 0.95
    ):
        """
        Initialize risk manager.
        
        Args:
            max_position_pct: Maximum position size (% of portfolio)
            max_portfolio_risk: Maximum portfolio risk per trade
            min_stop_loss_pct: Minimum stop loss percentage
            max_stop_loss_pct: Maximum stop loss percentage
            use_kelly: Use Kelly criterion
            kelly_fraction: Fraction of Kelly to use (conservative)
            use_var: Use VaR for sizing
            confidence_level: Confidence level for VaR
        """
        self.max_position_pct = max_position_pct
        self.max_portfolio_risk = max_portfolio_risk
        self.min_stop_loss_pct = min_stop_loss_pct
        self.max_stop_loss_pct = max_stop_loss_pct
        self.use_kelly = use_kelly
        self.kelly_fraction = kelly_fraction
        self.use_var = use_var
        self.confidence_level = confidence_level
        
    def calculate_position_size(
        self,
        price: float,
        portfolio_value: float,
        volatility: Optional[float] = None,
        prediction_std: Optional[float] = None,
        historical_returns: Optional[pd.Series] = None
    ) -> RiskMetrics:
        """
        Calculate optimal position size.
        
        Args:
            price: Current price
            portfolio_value: Total portfolio value
            volatility: Current volatility (optional)
            prediction_std: Prediction uncertainty (optional)
            historical_returns: Historical returns series (optional)
            
        Returns:
            RiskMetrics with position sizing
        """
        # Base position size
        position_pct = self.max_position_pct
        
        # Adjust for volatility
        if volatility is not None:
            # Reduce position in high volatility
            avg_vol = 0.02  # ~2% daily
            vol_scalar = avg_vol / max(volatility, avg_vol)
            position_pct *= np.clip(vol_scalar, 0.2, 1.5)
            
        # Adjust for prediction uncertainty
        if prediction_std is not None and prediction_std > 0:
            # Reduce position when uncertainty is high
            # Assume 2% std is baseline
            uncertainty_scalar = 0.02 / max(prediction_std, 0.02)
            position_pct *= np.clip(uncertainty_scalar, 0.3, 1.2)
            
        # Calculate Kelly fraction
        kelly_frac = 0.0
        if self.use_kelly and historical_returns is not None:
            win_rate = (historical_returns > 0).mean()
            avg_win = historical_returns[historical_returns > 0].mean() if (historical_returns > 0).any() else 0
            avg_loss = abs(historical_returns[historical_returns < 0].mean()) if (historical_returns < 0).any() else 1
            
            if avg_loss > 0:
                win_loss_ratio = avg_win / avg_loss
                kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
                kelly = max(0, kelly)  # No negative Kelly
                
                # Use fractional Kelly
                kelly_frac = min(kelly * self.kelly_fraction, 0.25)
                
        # Calculate stop loss
        if volatility is not None:
            stop_loss = max(
                self.min_stop_loss_pct,
                min(self.max_stop_loss_pct, volatility * 2)
            )
        else:
            stop_loss = self.min_stop_loss_pct
            
        # Take profit (2:1 ratio)
        take_profit = stop_loss * 2
        
        # Value at Risk
        var_95 = 0.0
        if self.use_var and historical_returns is not None:
            var_95 = np.percentile(historical_returns, (1 - self.confidence_level) * 100)
            
        # Risk metrics
        max_risk = min(position_pct * stop_loss, self.max_portfolio_risk)
        position_value = portfolio_value * position_pct
        shares = int(position_value / price)
        
        confidence = 1.0
        if prediction_std is not None and volatility is not None:
            confidence = 1 - min(prediction_std / (volatility * 2), 0.9)
            
        return RiskMetrics(
            position_size=shares,
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_risk=max_risk,
            kelly_fraction=kelly_frac,
            var_95=var_95,
            confidence=confidence
        )
    
    def adjust_for_uncertainty(
        self,
        base_position: float,
        uncertainty: float,
        max_uncertainty: float = 0.5
    ) -> float:
        """
        Adjust position size based on prediction uncertainty.
        
        Args:
            base_position: Base position size
            uncertainty: Prediction uncertainty (0-1)
            max_uncertainty: Maximum uncertainty threshold
            
        Returns:
            Adjusted position size
        """
        # Linear scaling based on uncertainty
        uncertainty_factor = 1 - min(uncertainty / max_uncertainty, 0.8)
        
        return base_position * uncertainty_factor
    
    def calculate_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        method: str = 'historical'
    ) -> float:
        """
        Calculate Value at Risk.
        
        Args:
            returns: Historical returns
            confidence: Confidence level
            method: 'historical', 'parametric', or 'monte_carlo'
            
        Returns:
            VaR estimate
        """
        if method == 'historical':
            return np.percentile(returns, (1 - confidence) * 100)
            
        elif method == 'parametric':
            # Assume normal distribution
            mu = returns.mean()
            sigma = returns.std()
            z = abs(pd.Series({0.90: 1.28, 0.95: 1.65, 0.99: 2.33}).get(confidence, 1.65))
            return mu - z * sigma
            
        elif method == 'monte_carlo':
            # Simple Monte Carlo
            n_samples = 10000
            mu = returns.mean()
            sigma = returns.std()
            simulated = np.random.normal(mu, sigma, n_samples)
            return np.percentile(simulated, (1 - confidence) * 100)
            
        return 0.0
    
    def check_risk_limits(
        self,
        position_value: float,
        portfolio_value: float,
        daily_var: float,
        current_drawdown: float
    ) -> Tuple[bool, str]:
        """
        Check if position violates risk limits.
        
        Args:
            position_value: Current position value
            portfolio_value: Total portfolio value
            daily_var: Daily VaR
            current_drawdown: Current drawdown
            
        Returns:
            (is_valid, reason)
        """
        # Position size limit
        position_pct = position_value / portfolio_value
        if position_pct > self.max_position_pct:
            return False, f"Position size {position_pct:.1%} exceeds max {self.max_position_pct:.1%}"
            
        # Drawdown limit
        max_drawdown = 0.2  # 20% max drawdown
        if current_drawdown > max_drawdown:
            return False, f"Drawdown {current_drawdown:.1%} exceeds max {max_drawdown:.1%}"
            
        # VaR limit
        max_var = 0.05  # 5% daily VaR
        if abs(daily_var) > max_var:
            return False, f"VaR {daily_var:.1%} exceeds max {max_var:.1%}"
            
        return True, "OK"
    
    def get_portfolio_risk(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        correlations: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Calculate portfolio-level risk.
        
        Args:
            positions: Dict of symbol -> shares
            prices: Dict of symbol -> price
            correlations: Optional correlation matrix
            
        Returns:
            Portfolio volatility
        """
        # Calculate position values
        values = {sym: shares * prices.get(sym, 0) for sym, shares in positions.items()}
        total_value = sum(values.values())
        
        if total_value == 0:
            return 0.0
            
        # Weights
        weights = np.array([v / total_value for v in values.values()])
        
        # Simple portfolio volatility (assuming 20% annual vol per position)
        # In production, you'd calculate actual volatilities and correlations
        vol = 0.20 / np.sqrt(252)  # Daily vol
        
        portfolio_vol = vol * np.sqrt(np.sum(weights ** 2))
        
        return portfolio_vol


class DynamicRiskManager(RiskManager):
    """
    Dynamic risk manager that adapts to market conditions.
    
    Adjusts risk parameters based on:
    - Market regime (trending vs ranging)
    - Volatility regime
    - Recent performance
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.market_regime = 'normal'
        self.volatility_regime = 'normal'
        self.recent_performance = 1.0
        
    def detect_regime(
        self,
        prices: pd.Series,
        lookback: int = 20
    ) -> str:
        """
        Detect current market regime.
        
        Args:
            prices: Price series
            lookback: Lookback period
            
        Returns:
            Regime: 'trending', 'ranging', or 'volatile'
        """
        returns = prices.pct_change()
        
        # Trend strength
        ma = prices.rolling(10).mean()
        trend = (prices - ma) / ma
        
        # Volatility
        vol = returns.rolling(lookback).std()
        
        # High volatility
        if vol.iloc[-1] > vol.quantile(0.75):
            return 'volatile'
            
        # Trending
        if abs(trend.iloc[-1]) > 0.02:
            return 'trending'
            
        return 'ranging'
    
    def adjust_for_regime(
        self,
        regime: str
    ) -> Dict[str, float]:
        """
        Adjust risk parameters for market regime.
        
        Args:
            regime: Market regime
            
        Returns:
            Adjusted risk parameters
        """
        adjustments = {
            'trending': {
                'position_mult': 1.2,
                'stop_mult': 1.5,
                'risk_mult': 1.2
            },
            'ranging': {
                'position_mult': 0.8,
                'stop_mult': 1.0,
                'risk_mult': 0.8
            },
            'volatile': {
                'position_mult': 0.5,
                'stop_mult': 2.0,
                'risk_mult': 0.5
            }
        }
        
        adj = adjustments.get(regime, adjustments['ranging'])
        
        return {
            'max_position_pct': self.max_position_pct * adj['position_mult'],
            'stop_loss_pct': self.max_stop_loss_pct * adj['stop_mult'],
            'max_portfolio_risk': self.max_portfolio_risk * adj['risk_mult']
        }
    
    def update_performance(
        self,
        returns: float
    ) -> None:
        """
        Update recent performance tracking.
        
        Args:
            returns: Recent returns
        """
        self.recent_performance = 0.95 * self.recent_performance + 0.05 * returns
        
    def get_adjusted_limits(self) -> Dict[str, float]:
        """Get risk limits adjusted for performance."""
        # Reduce risk after losses
        perf_factor = max(0.5, min(1.5, self.recent_performance))
        
        return {
            'position_mult': perf_factor,
            'stop_mult': 1.0,
            'risk_mult': perf_factor
        }
