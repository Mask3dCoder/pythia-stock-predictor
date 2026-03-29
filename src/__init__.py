"""
Pythia Stock Predictor - Enhanced Version

A production-ready stock prediction system with:
- Advanced feature engineering
- State-of-the-art models (CNN-LSTM with attention)
- Robust validation (walk-forward, purged K-fold)
- Bayesian hyperparameter optimization
- Uncertainty quantification
- Risk management
- Docker deployment
- Hydra configuration
"""

__version__ = "3.0.0"
__author__ = "Pythia Team"

from src.models.base import BaseModel, TimeSeriesModel, EnsembleModelBase
from src.models.registry import ModelRegistry, create_model, register_model
from src.features.pipeline import FeaturePipeline
from src.features.advanced_indicators import AdvancedIndicators
from src.features.alternative_data import AlternativeDataIntegrator
from src.validation.validators import (
    WalkForwardValidator,
    PurgedKFoldValidator,
    FinancialLossFunctions,
    FinancialEarlyStopping
)
from src.optimization.hyperopt import HyperparameterOptimizer, AutoMLPipeline
from src.backtest.engine import BacktestEngine, WalkForwardBacktest, TransactionCosts
from src.backtest.risk_manager import RiskManager, DynamicRiskManager
from src.uncertainty.quantification import (
    MonteCarloDropout,
    DeepEnsemble,
    ConformalPrediction,
    EnsembleUncertaintyCombiner
)

__all__ = [
    # Models
    'BaseModel',
    'TimeSeriesModel',
    'EnsembleModelBase',
    'ModelRegistry',
    'create_model',
    'register_model',
    
    # Features
    'FeaturePipeline',
    'AdvancedIndicators',
    'AlternativeDataIntegrator',
    
    # Validation
    'WalkForwardValidator',
    'PurgedKFoldValidator',
    'FinancialLossFunctions',
    'FinancialEarlyStopping',
    
    # Optimization
    'HyperparameterOptimizer',
    'AutoMLPipeline',
    
    # Backtesting
    'BacktestEngine',
    'WalkForwardBacktest',
    'TransactionCosts',
    
    # Risk Management
    'RiskManager',
    'DynamicRiskManager',
    
    # Uncertainty
    'MonteCarloDropout',
    'DeepEnsemble',
    'ConformalPrediction',
    'EnsembleUncertaintyCombiner'
]
