# performance_utils.py
import time
import functools
import hashlib
import json
import os
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class MemoryCache:
    """Simple in-memory cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.cache: Dict[str, Dict] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
    
    def _is_expired(self, item: Dict) -> bool:
        """Check if cache item is expired."""
        return time.time() > item['expires_at']
    
    def get(self, key: str) -> Any:
        """Get item from cache."""
        if key in self.cache:
            item = self.cache[key]
            if not self._is_expired(item):
                item['hits'] += 1
                return item['value']
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set item in cache."""
        # Evict old items if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_expired()
            if len(self.cache) >= self.max_size:
                # Remove least recently used
                oldest_key = min(self.cache.keys(), 
                               key=lambda k: self.cache[k]['accessed_at'])
                del self.cache[oldest_key]
        
        expires_at = time.time() + (ttl or self.default_ttl)
        self.cache[key] = {
            'value': value,
            'expires_at': expires_at,
            'accessed_at': time.time(),
            'hits': 0
        }
    
    def _evict_expired(self) -> None:
        """Remove expired items."""
        current_time = time.time()
        expired_keys = [
            key for key, item in self.cache.items() 
            if current_time > item['expires_at']
        ]
        for key in expired_keys:
            del self.cache[key]
    
    def clear(self) -> None:
        """Clear all cache."""
        self.cache.clear()
    
    def stats(self) -> Dict:
        """Get cache statistics."""
        total_hits = sum(item['hits'] for item in self.cache.values())
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'total_hits': total_hits,
            'hit_rate': total_hits / max(1, len(self.cache))
        }

# Global cache instance
cache = MemoryCache(max_size=500, default_ttl=1800)  # 30 minutes TTL

def cached_analysis(cache_key_func=None, ttl=1800):
    """Decorator for caching analysis results."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                # Default cache key based on function name and hashed arguments
                arg_str = json.dumps([str(arg) for arg in args] + 
                                   [f"{k}={v}" for k, v in sorted(kwargs.items())], 
                                   sort_keys=True)
                cache_key = f"{func.__name__}:{hashlib.md5(arg_str.encode()).hexdigest()}"
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.info(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            logger.info(f"{func.__name__} executed in {execution_time:.2f}s")
            
            # Cache the result
            cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator

class PerformanceMonitor:
    """Performance monitoring utilities."""
    
    def __init__(self):
        self.metrics = {}
    
    def record_metric(self, name: str, value: float, labels: Dict = None):
        """Record a performance metric."""
        if name not in self.metrics:
            self.metrics[name] = []
        
        metric_entry = {
            'timestamp': time.time(),
            'value': value,
            'labels': labels or {}
        }
        self.metrics[name].append(metric_entry)
        
        # Keep only recent metrics (last 1000 entries)
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-1000:]
    
    def get_metrics_summary(self, name: str, window_minutes: int = 60) -> Dict:
        """Get summary statistics for a metric within time window."""
        if name not in self.metrics:
            return {}
        
        current_time = time.time()
        window_start = current_time - (window_minutes * 60)
        
        recent_metrics = [
            m for m in self.metrics[name] 
            if m['timestamp'] >= window_start
        ]
        
        if not recent_metrics:
            return {}
        
        values = [m['value'] for m in recent_metrics]
        
        return {
            'count': len(values),
            'avg': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'p95': sorted(values)[int(len(values) * 0.95)] if len(values) > 0 else 0,
            'window_minutes': window_minutes
        }
    
    def get_all_metrics_summary(self) -> Dict:
        """Get summary for all metrics."""
        return {
            name: self.get_metrics_summary(name) 
            for name in self.metrics.keys()
        }

# Global performance monitor
perf_monitor = PerformanceMonitor()

def monitor_performance(metric_name: str):
    """Decorator to monitor function performance."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                success = True
            except Exception as e:
                success = False
                raise
            finally:
                execution_time = time.time() - start_time
                perf_monitor.record_metric(
                    metric_name,
                    execution_time,
                    {'function': func.__name__, 'success': success}
                )
            return result
        return wrapper
    return decorator

def generate_cache_key(resume_text: str, jd_text: str, analyzer_name: str) -> str:
    """Generate cache key for analysis results."""
    content = f"{analyzer_name}:{resume_text}:{jd_text}"
    return hashlib.md5(content.encode()).hexdigest()

class ResourceManager:
    """Manage system resources and limits."""
    
    @staticmethod
    def get_memory_usage() -> Dict:
        """Get current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                'rss': memory_info.rss,  # Resident Set Size
                'vms': memory_info.vms,  # Virtual Memory Size
                'percent': process.memory_percent()
            }
        except ImportError:
            return {'error': 'psutil not available'}
    
    @staticmethod
    def check_resource_limits() -> Dict:
        """Check if system is within resource limits."""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent,
                'warnings': {
                    'high_cpu': cpu_percent > 80,
                    'high_memory': memory.percent > 85,
                    'high_disk': disk.percent > 90
                }
            }
        except ImportError:
            return {'error': 'psutil not available'}