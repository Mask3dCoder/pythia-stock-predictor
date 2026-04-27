"""
Model Registry

Factory pattern implementation for creating and managing prediction models.
Supports dynamic model registration and easy model swapping.
"""

import logging
from typing import Dict, Type, Optional, Any, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Central registry for all prediction models.
    
    Provides a factory pattern for creating models by name,
    with support for custom model registration.
    """
    
    _models: Dict[str, Type] = {}
    _model_configs: Dict[str, Dict[str, Any]] = {}
    _factories: Dict[str, Callable] = {}
    
    @classmethod
    def register(
        cls,
        name: str,
        model_class: Optional[Type] = None,
        config: Optional[Dict[str, Any]] = None,
        factory: Optional[Callable] = None
    ) -> Callable:
        """
        Register a model class or factory function.
        
        Can be used as a decorator or called directly.
        
        Args:
            name: Model identifier
            model_class: Model class to register
            config: Default configuration for the model
            factory: Factory function to create model instances
            
        Returns:
            Decorator function if used as decorator
        """
        def decorator(model_class: Type) -> Type:
            cls._models[name.lower()] = model_class
            if config:
                cls._model_configs[name.lower()] = config
            logger.info(f"Registered model: {name}")
            return model_class
            
        if model_class is not None:
            return decorator(model_class)
        elif factory is not None:
            cls._factories[name.lower()] = factory
            if config:
                cls._model_configs[name.lower()] = config
            logger.info(f"Registered model factory: {name}")
            return factory
        else:
            return decorator
            
    @classmethod
    def create(
        cls,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Create a model instance by name.
        
        Args:
            name: Model identifier
            config: Model configuration
            **kwargs: Additional arguments passed to model constructor
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model name is not registered
        """
        name = name.lower()
        
        # Merge default config with provided config
        model_config = cls._model_configs.get(name, {}).copy()
        if config:
            model_config.update(config)
            
        # Check for factory function first
        if name in cls._factories:
            return cls._factories[name](model_config, **kwargs)
            
        # Check for registered class
        if name not in cls._models:
            available = list(cls._models.keys()) + list(cls._factories.keys())
            raise ValueError(
                f"Model '{name}' not found. Available models: {available}"
            )
            
        model_class = cls._models[name]
        return model_class(model_config, **kwargs)
    
    @classmethod
    def list_models(cls) -> Dict[str, Any]:
        """
        List all registered models.
        
        Returns:
            Dictionary of model names and their classes/factories
        """
        result = {}
        for name in set(list(cls._models.keys()) + list(cls._factories.keys())):
            if name in cls._models:
                result[name] = {
                    'type': 'class',
                    'class': cls._models[name].__name__,
                    'config': cls._model_configs.get(name, {})
                }
            else:
                result[name] = {
                    'type': 'factory',
                    'config': cls._model_configs.get(name, {})
                }
        return result
    
    @classmethod
    def get_model_class(cls, name: str) -> Optional[Type]:
        """Get model class by name."""
        return cls._models.get(name.lower())
    
    @classmethod
    def get_config(cls, name: str) -> Dict[str, Any]:
        """Get default configuration for a model."""
        return cls._model_configs.get(name.lower(), {}).copy()
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a model is registered."""
        return name.lower() in cls._models or name.lower() in cls._factories
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered models (mainly for testing)."""
        cls._models.clear()
        cls._model_configs.clear()
        cls._factories.clear()


def register_model(name: str, config: Optional[Dict[str, Any]] = None):
    """
    Decorator to register a model class.
    
    Usage:
        @register_model('my_model', config={'param1': 'value1'})
        class MyModel(BaseModel):
            ...
    """
    def decorator(cls):
        return ModelRegistry.register(name, cls, config)
    return decorator


def create_model(name: str, config: Optional[Dict[str, Any]] = None, **kwargs):
    """
    Convenience function to create a model.
    
    Usage:
        model = create_model('lstm', config={'sequence_length': 60})
    """
    return ModelRegistry.create(name, config, **kwargs)


# Built-in model registrations
# These will be imported and registered when the models module is loaded

def _register_builtin_models():
    """Register all built-in models."""
    from src.models.arima_model import ARIMAModel
    from src.models.lstm_model import LSTMModel, GRUModel
    from src.models.ensemble_model import EnsembleModel
    
    ModelRegistry.register('arima', ARIMAModel, {
        'order': [5, 1, 0],
        'auto_fit': True
    })
    
    ModelRegistry.register('lstm', LSTMModel, {
        'sequence_length': 60,
        'lstm_units': [50, 50],
        'dropout': 0.2,
        'epochs': 50
    })
    
    ModelRegistry.register('gru', GRUModel, {
        'sequence_length': 60,
        'lstm_units': [50, 50],
        'dropout': 0.2,
        'epochs': 50
    })
    
    ModelRegistry.register('ensemble', EnsembleModel, {
        'weights': {'arima': 0.33, 'lstm': 0.34, 'gru': 0.33}
    })
    
    logger.info("Registered built-in models: arima, lstm, gru, ensemble")


# Auto-register models on import
try:
    _register_builtin_models()
except ImportError as e:
    logger.warning(f"Could not register builtin models: {e}")
