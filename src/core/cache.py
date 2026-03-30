"""
Caching Module

Provides caching functionality with:
- In-memory LRU cache (default)
- Redis support (optional)
- TTL management
- Cache invalidation
- Thread-safe operations
"""

import logging
import time
import hashlib
import json
from typing import Any, Optional, Callable, Dict
from dataclasses import dataclass, field
from functools import wraps
from collections import OrderedDict
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached item."""
    value: Any
    timestamp: float
    ttl: Optional[float] = None
    hits: int = 0
    
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl


class InMemoryCache:
    """Thread-safe in-memory LRU cache."""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 300):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = Lock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self._stats['misses'] += 1
                return None
            
            entry = self._cache[key]
            
            if entry.is_expired():
                del self._cache[key]
                self._stats['misses'] += 1
                return None
            
            entry.hits += 1
            self._cache.move_to_end(key)
            self._stats['hits'] += 1
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Set value in cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
            elif len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
                self._stats['evictions'] += 1
            
            ttl = ttl if ttl is not None else self._default_ttl
            
            self._cache[key] = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl=ttl
            )
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._stats = {'hits': 0, 'misses': 0, 'evictions': 0}
    
    def invalidate_prefix(self, prefix: str):
        """Invalidate all keys starting with prefix."""
        with self._lock:
            keys_to_delete = [k for k in self._cache if k.startswith(prefix)]
            for key in keys_to_delete:
                del self._cache[key]
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self._lock:
            total = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total if total > 0 else 0
            
            return {
                'size': len(self._cache),
                'max_size': self._max_size,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'hit_rate': hit_rate,
                'evictions': self._stats['evictions'],
            }


class CacheManager:
    """Manages caching with optional Redis backend."""
    
    _instance: Optional['CacheManager'] = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, use_redis: bool = False, redis_url: Optional[str] = None):
        if self._initialized:
            return
        
        self._initialized = True
        self._use_redis = use_redis and redis_url is not None
        self._redis_url = redis_url
        
        if self._use_redis:
            try:
                import redis
                self._redis = redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                logger.info("Redis cache backend initialized")
            except Exception as e:
                logger.warning(f"Redis unavailable, using in-memory cache: {e}")
                self._use_redis = False
                self._cache = InMemoryCache()
        else:
            self._cache = InMemoryCache()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if self._use_redis:
            try:
                value = self._redis.get(key)
                if value:
                    return json.loads(value)
            except Exception as e:
                logger.warning(f"Redis get error: {e}")
            return None
        
        return self._cache.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Set value in cache."""
        if self._use_redis:
            try:
                self._redis.setex(key, int(ttl or 300), json.dumps(value))
            except Exception as e:
                logger.warning(f"Redis set error: {e}")
            return
        
        self._cache.set(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if self._use_redis:
            try:
                return bool(self._redis.delete(key))
            except Exception as e:
                logger.warning(f"Redis delete error: {e}")
            return False
        
        return self._cache.delete(key)
    
    def clear(self):
        """Clear all cache."""
        if self._use_redis:
            try:
                self._redis.flushdb()
            except Exception as e:
                logger.warning(f"Redis clear error: {e}")
            return
        
        self._cache.clear()
    
    def invalidate_prefix(self, prefix: str):
        """Invalidate all keys starting with prefix."""
        if self._use_redis:
            try:
                keys = self._redis.keys(f"{prefix}*")
                if keys:
                    self._redis.delete(*keys)
            except Exception as e:
                logger.warning(f"Redis invalidate error: {e}")
            return
        
        self._cache.invalidate_prefix(prefix)
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        if self._use_redis:
            try:
                info = self._redis.info('stats')
                return {
                    'backend': 'redis',
                    'keys': self._redis.dbsize(),
                    'hits': info.get('keyspace_hits', 0),
                    'misses': info.get('keyspace_misses', 0),
                }
            except Exception:
                pass
        
        stats = self._cache.get_stats()
        stats['backend'] = 'memory'
        return stats


def cache_key(*args, **kwargs) -> str:
    """Generate cache key from arguments."""
    key_data = json.dumps({'args': args, 'kwargs': kwargs}, sort_keys=True)
    return hashlib.md5(key_data.encode()).hexdigest()


def cached(ttl: float = 300, key_prefix: str = ''):
    """Decorator to cache function results."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = CacheManager()
            
            cache_key_val = f"{key_prefix}:{func.__name__}:{cache_key(*args, **kwargs)}"
            
            cached_value = manager.get(cache_key_val)
            if cached_value is not None:
                return cached_value
            
            result = func(*args, **kwargs)
            
            manager.set(cache_key_val, result, ttl)
            
            return result
        
        return wrapper
    return decorator


def invalidate_cache(prefix: str):
    """Invalidate cache entries matching prefix."""
    manager = CacheManager()
    manager.invalidate_prefix(prefix)


def get_cache_stats() -> Dict:
    """Get cache statistics."""
    manager = CacheManager()
    return manager.get_stats()


cache = CacheManager()
