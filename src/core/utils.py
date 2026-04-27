"""
Utilities Module

Provides utility functions and decorators for retry logic, caching, 
timing, and common operations.
"""

import time
import hashlib
import logging
import functools
from typing import Callable, Any, Optional, TypeVar, Type, Tuple, List
from pathlib import Path
from datetime import datetime, timedelta
from collections import OrderedDict
import threading
import json

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar('T')


# Retry Decorator

def retry_with_backoff(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable] = None
):
    """
    Decorator to retry a function with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Factor to multiply wait time by after each retry
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback function called on each retry
        
    Returns:
        Decorated function
        
    Example:
        @retry_with_backoff(max_retries=3, backoff_factor=2.0)
        def fetch_data(url):
            return requests.get(url)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries: {e}"
                        )
                        raise
                    
                    wait_time = backoff_factor ** attempt
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    
                    if on_retry:
                        on_retry(attempt, e)
                    
                    time.sleep(wait_time)
            
            raise last_exception
        
        return wrapper
    return decorator


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Decorator implementing circuit breaker pattern.
    
    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before attempting recovery
        exceptions: Tuple of exception types to count as failures
        
    Returns:
        Decorated function
        
    Example:
        @circuit_breaker(failure_threshold=5, recovery_timeout=60)
        def unstable_api_call():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        state = {
            'failures': 0,
            'last_failure_time': None,
            'state': 'closed'  # closed, open, half-open
        }
        lock = threading.Lock()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            with lock:
                if state['state'] == 'open':
                    if time.time() - state['last_failure_time'] >= recovery_timeout:
                        logger.info(f"Circuit breaker for {func.__name__} entering half-open state")
                        state['state'] = 'half-open'
                    else:
                        raise RuntimeError(f"Circuit breaker is OPEN for {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                with lock:
                    state['failures'] = 0
                    state['state'] = 'closed'
                return result
            except exceptions as e:
                with lock:
                    state['failures'] += 1
                    state['last_failure_time'] = time.time()
                    
                    if state['failures'] >= failure_threshold:
                        logger.warning(f"Circuit breaker OPENED for {func.__name__}")
                        state['state'] = 'open'
                
                raise
        
        return wrapper
    return decorator


# Cache Classes

class LRUCache:
    """
    Thread-safe LRU (Least Recently Used) cache.
    
    Attributes:
        max_size: Maximum number of items to cache
    """
    
    def __init__(self, max_size: int = 100):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of items
        """
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Set item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
    
    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)


class DataCache:
    """
    Specialized cache for pandas DataFrames with disk persistence.
    
    Attributes:
        max_size_mb: Maximum cache size in megabytes
        cache_dir: Directory for disk cache
    """
    
    def __init__(self, max_size_mb: int = 500, cache_dir: str = ".cache"):
        """
        Initialize data cache.
        
        Args:
            max_size_mb: Maximum size in MB
            cache_dir: Directory for disk cache
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self._memory_cache = LRUCache(max_size=50)
        self._lock = threading.RLock()
        
        # Load cache index
        self._index_file = self.cache_dir / "cache_index.json"
        self._load_index()
    
    def _load_index(self):
        """Load cache index from disk."""
        if self._index_file.exists():
            try:
                with open(self._index_file, 'r') as f:
                    self._index = json.load(f)
            except Exception:
                self._index = {}
        else:
            self._index = {}
    
    def _save_index(self):
        """Save cache index to disk."""
        with open(self._index_file, 'w') as f:
            json.dump(self._index, f)
    
    def _get_cache_key(self, symbol: str, years: int, start_date: Optional[str] = None) -> str:
        """Generate cache key for data."""
        key_str = f"{symbol}_{years}_{start_date}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, symbol: str, years: int, start_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Get cached data.
        
        Args:
            symbol: Stock symbol
            years: Number of years
            start_date: Start date
            
        Returns:
            Cached DataFrame or None
        """
        key = self._get_cache_key(symbol, years, start_date)
        
        # Check memory cache
        result = self._memory_cache.get(key)
        if result is not None:
            logger.debug(f"Cache hit (memory): {symbol}")
            return result
        
        # Check disk cache
        cache_file = self.cache_dir / f"{key}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                self._memory_cache.set(key, df)
                logger.debug(f"Cache hit (disk): {symbol}")
                return df
            except Exception as e:
                logger.warning(f"Error loading cache: {e}")
        
        return None
    
    def set(self, df: pd.DataFrame, symbol: str, years: int, start_date: Optional[str] = None) -> None:
        """
        Cache DataFrame.
        
        Args:
            df: DataFrame to cache
            symbol: Stock symbol
            years: Number of years
            start_date: Start date
        """
        key = self._get_cache_key(symbol, years, start_date)
        
        # Store in memory
        self._memory_cache.set(key, df)
        
        # Store on disk
        try:
            cache_file = self.cache_dir / f"{key}.parquet"
            df.to_parquet(cache_file)
            self._index[key] = {
                'symbol': symbol,
                'years': years,
                'size_bytes': cache_file.stat().st_size,
                'created': datetime.now().isoformat()
            }
            self._save_index()
            logger.debug(f"Cached data: {symbol}")
        except Exception as e:
            logger.warning(f"Error saving cache: {e}")
    
    def clear(self) -> None:
        """Clear all cached data."""
        self._memory_cache.clear()
        
        for cache_file in self.cache_dir.glob("*.parquet"):
            try:
                cache_file.unlink()
            except Exception:
                pass
        
        self._index = {}
        self._save_index()


# Timing Utilities

def timing_decorator(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to measure
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        
        logger.info(f"{func.__name__} took {elapsed:.2f}s")
        
        return result
    
    return wrapper


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = "Operation"):
        """
        Initialize timer.
        
        Args:
            name: Name of the operation being timed
        """
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        """Start timer."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        """Stop timer and log elapsed time."""
        self.elapsed = time.time() - self.start_time
        logger.info(f"{self.name} took {self.elapsed:.2f}s")


# Data Validation

def validate_ohlc(df: pd.DataFrame) -> bool:
    """
    Validate OHLC (Open, High, Low, Close) data relationships.
    
    Args:
        df: DataFrame with OHLC columns
        
    Returns:
        True if valid, False otherwise
    """
    required_cols = ['open', 'high', 'low', 'close']
    
    if not all(col in df.columns for col in required_cols):
        return False
    
    # High >= Open, Close, Low
    valid = (
        (df['high'] >= df['open']).all() and
        (df['high'] >= df['close']).all() and
        (df['high'] >= df['low']).all() and
        (df['low'] <= df['open']).all() and
        (df['low'] <= df['close']).all()
    )
    
    return valid


def validate_positive(df: pd.DataFrame, columns: Optional[List[str]] = None) -> bool:
    """
    Validate that specified columns contain only positive values.
    
    Args:
        df: DataFrame to validate
        columns: List of columns (if None, checks numeric columns)
        
    Returns:
        True if valid, False otherwise
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    return (df[columns] > 0).all().all()


# Data Formatting

def format_currency(value: float, currency: str = "$") -> str:
    """
    Format value as currency.
    
    Args:
        value: Numeric value
        currency: Currency symbol
        
    Returns:
        Formatted string
    """
    return f"{currency}{value:,.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format value as percentage.
    
    Args:
        value: Numeric value (e.g., 0.05 for 5%)
        decimals: Number of decimal places
        
    Returns:
        Formatted string
    """
    return f"{value * 100:.{decimals}f}%"


def format_large_number(value: float) -> str:
    """
    Format large numbers with K, M, B suffixes.
    
    Args:
        value: Numeric value
        
    Returns:
        Formatted string
    """
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    elif abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    elif abs(value) >= 1_000:
        return f"{value / 1_000:.2f}K"
    else:
        return f"{value:.2f}"


# Safe Operations

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division fails
        
    Returns:
        Result of division or default
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default


def safe_mean(values: List[float], default: float = 0.0) -> float:
    """
    Safely calculate mean, returning default for empty list.
    
    Args:
        values: List of values
        default: Default value if list is empty
        
    Returns:
        Mean or default
    """
    if not values:
        return default
    
    return sum(values) / len(values)


def safe_percentile(data: np.ndarray, percentile: float, default: float = 0.0) -> float:
    """
    Safely calculate percentile.
    
    Args:
        data: NumPy array
        percentile: Percentile to calculate (0-100)
        default: Default value if calculation fails
        
    Returns:
        Percentile value or default
    """
    try:
        if len(data) == 0:
            return default
        return float(np.percentile(data, percentile))
    except Exception:
        return default


# Global instances

data_cache = DataCache()
