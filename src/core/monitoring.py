"""
Monitoring Module

Provides monitoring and observability:
- Metrics collection
- Health checks
- Performance tracking
- Structured logging
"""

import logging
import time
import psutil
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from threading import Lock
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """Represents a metric data point."""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Collects and stores metrics."""
    
    def __init__(self, max_points: int = 1000):
        self._metrics: Dict[str, list] = defaultdict(list)
        self._max_points = max_points
        self._lock = Lock()
    
    def record(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        with self._lock:
            metric = Metric(
                name=name,
                value=value,
                timestamp=time.time(),
                tags=tags or {}
            )
            
            self._metrics[name].append(metric)
            
            if len(self._metrics[name]) > self._max_points:
                self._metrics[name] = self._metrics[name][-self._max_points:]
    
    def get(self, name: str, since: Optional[float] = None) -> list:
        """Get metrics by name."""
        with self._lock:
            metrics = self._metrics.get(name, [])
            
            if since:
                metrics = [m for m in metrics if m.timestamp >= since]
            
            return metrics
    
    def get_latest(self, name: str) -> Optional[Metric]:
        """Get latest metric value."""
        metrics = self.get(name)
        return metrics[-1] if metrics else None
    
    def get_average(self, name: str, since: Optional[float] = None) -> Optional[float]:
        """Get average metric value."""
        metrics = self.get(name, since)
        
        if not metrics:
            return None
        
        return sum(m.value for m in metrics) / len(metrics)
    
    def clear(self, name: Optional[str] = None):
        """Clear metrics."""
        with self._lock:
            if name:
                self._metrics.pop(name, None)
            else:
                self._metrics.clear()
    
    def get_all(self) -> Dict:
        """Get all metrics summary."""
        with self._lock:
            summary = {}
            
            for name, metrics in self._metrics.items():
                if metrics:
                    latest = metrics[-1]
                    summary[name] = {
                        'latest': latest.value,
                        'avg': sum(m.value for m in metrics) / len(metrics),
                        'min': min(m.value for m in metrics),
                        'max': max(m.value for m in metrics),
                        'count': len(metrics),
                    }
            
            return summary


class HealthChecker:
    """Performs health checks on system components."""
    
    def __init__(self):
        self._checks: Dict[str, callable] = {}
    
    def register_check(self, name: str, check_fn: callable):
        """Register a health check function."""
        self._checks[name] = check_fn
    
    def check_all(self) -> Dict:
        """Run all health checks."""
        results = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'checks': {}
        }
        
        for name, check_fn in self._checks.items():
            try:
                result = check_fn()
                results['checks'][name] = {
                    'status': 'healthy' if result else 'unhealthy',
                    'healthy': result,
                }
                
                if not result:
                    results['status'] = 'unhealthy'
            except Exception as e:
                results['checks'][name] = {
                    'status': 'error',
                    'error': str(e),
                }
                results['status'] = 'error'
        
        return results


class PerformanceTracker:
    """Tracks performance of operations."""
    
    def __init__(self):
        self._collector = MetricsCollector()
    
    def track(self, operation: str):
        """Decorator to track operation performance."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    
                    duration = time.time() - start_time
                    self._collector.record(
                        f"duration_{operation}",
                        duration,
                        {'operation': operation, 'status': 'success'}
                    )
                    
                    self._collector.record(
                        f"count_{operation}",
                        1,
                        {'operation': operation, 'status': 'success'}
                    )
                    
                    return result
                    
                except Exception as e:
                    duration = time.time() - start_time
                    
                    self._collector.record(
                        f"duration_{operation}",
                        duration,
                        {'operation': operation, 'status': 'error'}
                    )
                    
                    self._collector.record(
                        f"count_{operation}",
                        1,
                        {'operation': operation, 'status': 'error'}
                    )
                    
                    raise
            
            return wrapper
        return decorator
    
    def get_stats(self, operation: str) -> Dict:
        """Get performance stats for an operation."""
        duration_metrics = self._collector.get(f"duration_{operation}")
        count_metrics = self._collector.get(f"count_{operation}")
        
        if not duration_metrics:
            return {}
        
        durations = [m.value for m in duration_metrics]
        successes = sum(1 for m in count_metrics if m.tags.get('status') == 'success')
        errors = sum(1 for m in count_metrics if m.tags.get('status') == 'error')
        
        return {
            'count': len(durations),
            'success': successes,
            'errors': errors,
            'success_rate': successes / (successes + errors) if (successes + errors) > 0 else 0,
            'avg_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
        }


class SystemMonitor:
    """Monitors system resources."""
    
    @staticmethod
    def get_system_info() -> Dict:
        """Get system information."""
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_mb': psutil.virtual_memory().available / (1024 * 1024),
            'disk_percent': psutil.disk_usage('/').percent,
            'process_count': len(psutil.pids()),
        }
    
    @staticmethod
    def get_process_info() -> Dict:
        """Get current process information."""
        process = psutil.Process()
        
        return {
            'cpu_percent': process.cpu_percent(),
            'memory_mb': process.memory_info().rss / (1024 * 1024),
            'threads': process.num_threads(),
            'open_files': len(process.open_files()),
            'connections': len(process.connections()),
        }


metrics = MetricsCollector()
health = HealthChecker()
performance = PerformanceTracker()
monitor = SystemMonitor()


def track_time(operation: str):
    """Decorator to track execution time."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start
                
                metrics.record(f"execution_time_{operation}", duration)
                metrics.record(f"success_{operation}", 1)
                
                return result
                
            except Exception as e:
                duration = time.time() - start
                metrics.record(f"execution_time_{operation}", duration)
                metrics.record(f"error_{operation}", 1)
                raise
        
        return wrapper
    return decorator


def health_check(name: str):
    """Decorator to register a health check."""
    def decorator(func):
        health.register_check(name, func)
        return func
    return decorator


def get_health_status() -> Dict:
    """Get overall health status."""
    system_info = SystemMonitor.get_system_info()
    
    checks = health.check_all()
    
    memory_ok = system_info['memory_percent'] < 90
    disk_ok = system_info['disk_percent'] < 90
    
    if 'memory' not in checks['checks']:
        checks['checks']['memory'] = {'status': 'healthy' if memory_ok else 'unhealthy'}
    if 'disk' not in checks['checks']:
        checks['checks']['disk'] = {'status': 'healthy' if disk_ok else 'unhealthy'}
    
    if not memory_ok or not disk_ok:
        checks['status'] = 'degraded'
    
    return {
        **checks,
        'system': system_info,
    }
