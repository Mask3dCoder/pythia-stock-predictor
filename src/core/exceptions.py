"""
Custom Exceptions Module

Defines a comprehensive hierarchy of custom exceptions for the Stock Prediction CLI.
Provides detailed error handling with error codes for better debugging and user feedback.
"""

from typing import Optional, Any, Dict
from datetime import datetime


class PredictionError(Exception):
    """
    Base exception for all prediction-related errors.
    
    Attributes:
        message: Human-readable error message
        error_code: Machine-readable error code
        details: Additional context about the error
        timestamp: When the error occurred
    """
    
    ERROR_CODES = {
        'PREDICTION_ERROR': 'P001',
        'DATA_COLLECTION_ERROR': 'D001',
        'MODEL_TRAINING_ERROR': 'M001',
        'VALIDATION_ERROR': 'V001',
        'CONFIGURATION_ERROR': 'C001',
        'API_ERROR': 'A001',
    }
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.ERROR_CODES.get(self.__class__.__name__.upper(), 'P000')
        self.details = details or {}
        self.timestamp = timestamp or datetime.now()
    
    def __str__(self) -> str:
        base_msg = f"[{self.error_code}] {self.message}"
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            base_msg += f" ({details_str})"
        return base_msg
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            'error': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }


class DataCollectionError(PredictionError):
    """
    Raised when data collection fails.
    
    Common causes:
    - Network connectivity issues
    - Invalid API keys
    - Symbol not found
    - Rate limiting by data provider
    """
    
    ERROR_CODES = {
        'NETWORK_ERROR': 'D001',
        'API_ERROR': 'D002',
        'INVALID_SYMBOL': 'D003',
        'NO_DATA_AVAILABLE': 'D004',
        'RATE_LIMITED': 'D005',
        'DATA_VALIDATION_ERROR': 'D006',
    }
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        symbol: Optional[str] = None
    ):
        super().__init__(
            message,
            error_code=error_code or 'D001',
            details=details or {}
        )
        if symbol:
            self.details['symbol'] = symbol


class ModelTrainingError(PredictionError):
    """
    Raised when model training fails.
    
    Common causes:
    - Insufficient data
    - Invalid model configuration
    - Training algorithm convergence failure
    - Hardware resource limitations
    """
    
    ERROR_CODES = {
        'INSUFFICIENT_DATA': 'M001',
        'INVALID_CONFIG': 'M002',
        'CONVERGENCE_FAILURE': 'M003',
        'MEMORY_ERROR': 'M004',
        'TENSORFLOW_ERROR': 'M005',
        'VALIDATION_FAILURE': 'M006',
    }
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        model_type: Optional[str] = None
    ):
        super().__init__(
            message,
            error_code=error_code or 'M001',
            details=details or {}
        )
        if model_type:
            self.details['model_type'] = model_type


class ValidationError(PredictionError):
    """
    Raised when input validation fails.
    
    Common causes:
    - Invalid stock symbol format
    - Out-of-range parameters (years, days)
    - Invalid date formats
    - Missing required fields
    """
    
    ERROR_CODES = {
        'INVALID_SYMBOL': 'V001',
        'INVALID_PARAMETER': 'V002',
        'INVALID_DATE': 'V003',
        'MISSING_REQUIRED_FIELD': 'V004',
        'OUT_OF_RANGE': 'V005',
    }
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        field: Optional[str] = None
    ):
        super().__init__(
            message,
            error_code=error_code or 'V001',
            details=details or {}
        )
        if field:
            self.details['field'] = field


class ConfigurationError(PredictionError):
    """
    Raised when configuration is invalid or missing.
    
    Common causes:
    - Missing required config files
    - Invalid YAML syntax
    - Invalid parameter values in config
    - Missing environment variables
    """
    
    ERROR_CODES = {
        'FILE_NOT_FOUND': 'C001',
        'INVALID_YAML': 'C002',
        'INVALID_PARAMETER': 'C003',
        'MISSING_ENV_VAR': 'C004',
        'SCHEMA_VALIDATION_ERROR': 'C005',
    }
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None
    ):
        super().__init__(
            message,
            error_code=error_code or 'C001',
            details=details or {}
        )
        if config_path:
            self.details['config_path'] = config_path


class APIError(PredictionError):
    """
    Raised when API operations fail.
    
    Common causes:
    - Invalid API key
    - Rate limiting
    - Server errors
    - Invalid request format
    """
    
    ERROR_CODES = {
        'INVALID_API_KEY': 'A001',
        'RATE_LIMIT_EXCEEDED': 'A002',
        'SERVER_ERROR': 'A003',
        'INVALID_REQUEST': 'A004',
        'SERVICE_UNAVAILABLE': 'A005',
        'TIMEOUT': 'A006',
    }
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        status_code: Optional[int] = None
    ):
        super().__init__(
            message,
            error_code=error_code or 'A001',
            details=details or {}
        )
        if status_code:
            self.details['status_code'] = status_code


# Utility functions for error handling

def format_exception(exc: Exception) -> Dict[str, Any]:
    """
    Format any exception into a consistent dictionary response.
    
    Args:
        exc: The exception to format
        
    Returns:
        Dictionary with error details
    """
    if isinstance(exc, PredictionError):
        return exc.to_dict()
    
    return {
        'error': exc.__class__.__name__,
        'message': str(exc),
        'error_code': 'P999',
        'details': {},
        'timestamp': datetime.now().isoformat()
    }


def retryable_error(exc: Exception) -> bool:
    """
    Determine if an error is retryable.
    
    Args:
        exc: The exception to check
        
    Returns:
        True if the operation should be retried
    """
    if isinstance(exc, DataCollectionError):
        return exc.error_code in ['D001', 'D002', 'D005']  # Network, API, Rate limit
    
    if isinstance(exc, APIError):
        return exc.error_code in ['A002', 'A003', 'A006']  # Rate limit, Server, Timeout
    
    return False
