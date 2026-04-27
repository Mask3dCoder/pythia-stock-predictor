"""
CLI Input Validation Module

Provides robust input validation for the Stock Prediction CLI.
"""

import re
from typing import Tuple, Optional, List


# ============== Validation Constants ==============

VALID_SYMBOL_PATTERN = re.compile(r'^[A-Z]{1,5}$')
VALID_MODEL_TYPES = {'arima', 'lstm', 'gru', 'ensemble', 'cnn_lstm'}
VALID_SENTIMENT_METHODS = {'vader', 'textblob', 'combined'}

MIN_YEARS = 1
MAX_YEARS = 50
MIN_DAYS = 1
MAX_DAYS = 365


# ============== Validation Functions ==============

def validate_symbol(symbol: str) -> Tuple[bool, Optional[str]]:
    """
    Validate stock symbol format.
    
    Args:
        symbol: Stock ticker symbol to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not symbol:
        return False, "Symbol cannot be empty"
    
    # Remove any whitespace
    symbol = symbol.strip().upper()
    
    # Check against pattern (1-5 uppercase letters)
    if not VALID_SYMBOL_PATTERN.match(symbol):
        return False, "Symbol must be 1-5 uppercase letters (e.g., AAPL, MSFT, GOOGL)"
    
    return True, None


def validate_model_type(model: str) -> Tuple[bool, Optional[str]]:
    """
    Validate model type.
    
    Args:
        model: Model type to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not model:
        return False, "Model type cannot be empty"
    
    model_lower = model.lower()
    
    if model_lower not in VALID_MODEL_TYPES:
        valid_options = ', '.join(sorted(VALID_MODEL_TYPES))
        return False, f"Invalid model type '{model}'. Choose from: {valid_options}"
    
    return True, None


def validate_years(years: int) -> Tuple[bool, Optional[str]]:
    """
    Validate years parameter.
    
    Args:
        years: Number of years to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(years, int):
        return False, "Years must be an integer"
    
    if not MIN_YEARS <= years <= MAX_YEARS:
        return False, f"Years must be between {MIN_YEARS} and {MAX_YEARS}"
    
    return True, None


def validate_days(days: int) -> Tuple[bool, Optional[str]]:
    """
    Validate days parameter.
    
    Args:
        days: Number of days to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(days, int):
        return False, "Days must be an integer"
    
    if not MIN_DAYS <= days <= MAX_DAYS:
        return False, f"Days must be between {MIN_DAYS} and {MAX_DAYS}"
    
    return True, None


def validate_sentiment_method(method: str) -> Tuple[bool, Optional[str]]:
    """
    Validate sentiment analysis method.
    
    Args:
        method: Sentiment method to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not method:
        return False, "Sentiment method cannot be empty"
    
    method_lower = method.lower()
    
    if method_lower not in VALID_SENTIMENT_METHODS:
        valid_options = ', '.join(sorted(VALID_SENTIMENT_METHODS))
        return False, f"Invalid sentiment method '{method}'. Choose from: {valid_options}"
    
    return True, None


def validate_config_path(path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate configuration file path.
    
    Args:
        path: Path to config file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not path:
        return False, "Config path cannot be empty"
    
    import os
    from pathlib import Path
    
    path_obj = Path(path)
    
    # Check if file exists
    if not path_obj.exists():
        return False, f"Config file not found: {path}"
    
    # Check if it's a file
    if not path_obj.is_file():
        return False, f"Config path is not a file: {path}"
    
    # Check extension
    if path_obj.suffix not in ['.yaml', '.yml']:
        return False, "Config file must be a YAML file (.yaml or .yml)"
    
    return True, None


def validate_positive_number(value: int, name: str = "value") -> Tuple[bool, Optional[str]]:
    """
    Validate that a number is positive.
    
    Args:
        value: Number to validate
        name: Name of the parameter for error messages
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(value, (int, float)):
        return False, f"{name} must be a number"
    
    if value <= 0:
        return False, f"{name} must be positive"
    
    return True, None


def validate_percentage(value: float, name: str = "value") -> Tuple[bool, Optional[str]]:
    """
    Validate that a value is a valid percentage (0-100).
    
    Args:
        value: Value to validate
        name: Name of the parameter for error messages
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(value, (int, float)):
        return False, f"{name} must be a number"
    
    if not 0 <= value <= 100:
        return False, f"{name} must be between 0 and 100"
    
    return True, None


# ============== Validation Decorator ==============

def validate_args(**validators):
    """
    Decorator to validate command arguments.
    
    Args:
        **validators: Mapping of argument names to validator functions
        
    Example:
        @validate_args(
            symbol=validate_symbol,
            days=validate_days
        )
        def predict_command(args):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get the args object
            args_dict = kwargs.get('args') or (args[0] if args else None)
            
            if args_dict is None:
                return func(*args, **kwargs)
            
            # Run validators
            for arg_name, validator in validators.items():
                if hasattr(args_dict, arg_name):
                    value = getattr(args_dict, arg_name)
                    is_valid, error_msg = validator(value)
                    
                    if not is_valid:
                        from .output import print_error, console
                        console.print(f"[error]Validation Error:[/error] {error_msg}")
                        import sys
                        sys.exit(2)  # Exit code 2 for validation errors
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ============== Batch Validation ==============

def validate_all(args) -> Tuple[bool, List[str]]:
    """
    Validate all arguments from parsed args namespace.
    
    Args:
        args: Parsed argparse namespace
        
    Returns:
        Tuple of (all_valid, list_of_errors)
    """
    errors = []
    
    # Validate symbol if present
    if hasattr(args, 'symbol') and args.symbol:
        is_valid, error = validate_symbol(args.symbol)
        if not is_valid:
            errors.append(f"symbol: {error}")
    
    # Validate model if present
    if hasattr(args, 'model') and args.model:
        is_valid, error = validate_model_type(args.model)
        if not is_valid:
            errors.append(f"model: {error}")
    
    # Validate years if present
    if hasattr(args, 'years') and args.years is not None:
        is_valid, error = validate_years(args.years)
        if not is_valid:
            errors.append(f"years: {error}")
    
    # Validate days if present
    if hasattr(args, 'days') and args.days is not None:
        is_valid, error = validate_days(args.days)
        if not is_valid:
            errors.append(f"days: {error}")
    
    # Validate sentiment method if present
    if hasattr(args, 'method') and args.method:
        is_valid, error = validate_sentiment_method(args.method)
        if not is_valid:
            errors.append(f"method: {error}")
    
    return len(errors) == 0, errors
