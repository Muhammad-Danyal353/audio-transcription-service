"""
Performance timing utilities for the audio transcription service.

Provides comprehensive timing and performance monitoring capabilities
for tracking execution times, memory usage, and system metrics.
"""

import time
import functools
import threading
import uuid
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from contextlib import contextmanager
from collections import defaultdict

from ..core.logging import get_logger


logger = get_logger(__name__)


@dataclass
class TimingMetrics:
    """Container for timing and performance metrics."""
    
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    memory_before_mb: Optional[float] = None
    memory_after_mb: Optional[float] = None
    memory_delta_mb: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate duration if end_time is set."""
        if self.end_time is not None:
            self.duration_ms = (self.end_time - self.start_time) * 1000


class PerformanceMonitor:
    """
    Thread-safe performance monitoring system.
    
    Tracks execution times, memory usage, and provides detailed
    performance analytics for the audio processing pipeline.
    """
    
    def __init__(self):
        """Initialize performance monitor."""
        self._metrics: List[TimingMetrics] = []
        self._lock = threading.Lock()
        self._active_operations: Dict[str, TimingMetrics] = {}
    
    def start_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start timing an operation.
        
        Args:
            operation_name: Name of the operation to time
            metadata: Optional metadata to attach to the operation
            
        Returns:
            Operation ID for tracking
        """
        operation_id = f"{operation_name}_{uuid.uuid4().hex[:8]}"
        
        with self._lock:
            metrics = TimingMetrics(
                operation_name=operation_name,
                start_time=time.perf_counter(),
                memory_before_mb=self._get_memory_usage_mb(),
                metadata=metadata or {}
            )
            self._active_operations[operation_id] = metrics
        
        logger.debug(f"Started operation: {operation_name} (ID: {operation_id})")
        return operation_id
    
    def end_operation(self, operation_id: str, success: bool = True, error_message: Optional[str] = None) -> TimingMetrics:
        """
        End timing an operation.
        
        Args:
            operation_id: Operation ID returned by start_operation
            success: Whether the operation succeeded
            error_message: Error message if operation failed
            
        Returns:
            Completed timing metrics
        """
        with self._lock:
            if operation_id not in self._active_operations:
                logger.warning(f"Operation {operation_id} not found in active operations")
                return None
            
            metrics = self._active_operations.pop(operation_id)
            metrics.end_time = time.perf_counter()
            metrics.success = success
            metrics.error_message = error_message
            metrics.memory_after_mb = self._get_memory_usage_mb()
            
            if metrics.memory_before_mb is not None and metrics.memory_after_mb is not None:
                metrics.memory_delta_mb = metrics.memory_after_mb - metrics.memory_before_mb
            
            self._metrics.append(metrics)
        
        if metrics and metrics.operation_name and metrics.duration_ms is not None:
            logger.debug(
                f"Completed operation: {metrics.operation_name} "
                f"({metrics.duration_ms:.2f}ms, success: {success})"
            )
        return metrics
    
    def _get_memory_usage_mb(self) -> Optional[float]:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return None
    
    def get_operation_stats(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for operations.
        
        Args:
            operation_name: Filter by operation name (None for all)
            
        Returns:
            Dictionary with operation statistics
        """
        with self._lock:
            metrics = self._metrics.copy()
        
        if operation_name:
            metrics = [m for m in metrics if m.operation_name == operation_name]
        
        if not metrics:
            return {"count": 0}
        
        durations = [m.duration_ms for m in metrics if m.duration_ms is not None]
        success_count = sum(1 for m in metrics if m.success)
        
        stats = {
            "count": len(metrics),
            "success_count": success_count,
            "failure_count": len(metrics) - success_count,
            "success_rate": success_count / len(metrics) if metrics else 0,
        }
        
        if durations:
            stats.update({
                "avg_duration_ms": sum(durations) / len(durations),
                "min_duration_ms": min(durations),
                "max_duration_ms": max(durations),
                "total_duration_ms": sum(durations),
            })
        
        return stats
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss_mb": memory_info.rss / (1024 * 1024),
                "vms_mb": memory_info.vms / (1024 * 1024),
                "percent": process.memory_percent(),
            }
        except ImportError:
            return {"error": "psutil not available"}
    
    def clear_metrics(self) -> None:
        """Clear all stored metrics."""
        with self._lock:
            self._metrics.clear()
            self._active_operations.clear()
        logger.info("Cleared all performance metrics")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        with self._lock:
            metrics = self._metrics.copy()
        
        if not metrics:
            return {"message": "No metrics available"}
        
        # Group by operation name
        operation_groups = defaultdict(list)
        for metric in metrics:
            operation_groups[metric.operation_name].append(metric)
        
        summary = {
            "total_operations": len(metrics),
            "unique_operations": len(operation_groups),
            "operations": {}
        }
        
        for op_name, op_metrics in operation_groups.items():
            durations = [m.duration_ms for m in op_metrics if m.duration_ms is not None]
            success_count = sum(1 for m in op_metrics if m.success)
            
            op_stats = {
                "count": len(op_metrics),
                "success_count": success_count,
                "success_rate": success_count / len(op_metrics),
            }
            
            if durations:
                op_stats.update({
                    "avg_duration_ms": sum(durations) / len(durations),
                    "min_duration_ms": min(durations),
                    "max_duration_ms": max(durations),
                })
            
            summary["operations"][op_name] = op_stats
        
        return summary


# Global performance monitor instance
_performance_monitor = PerformanceMonitor()


def time_operation(operation_name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
    """
    Decorator to automatically time function execution.
    
    Args:
        operation_name: Custom name for the operation (defaults to function name)
        metadata: Additional metadata to attach
        
    Example:
        @time_operation("audio_processing")
        def process_audio(file_path: str):
            # Function implementation
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            operation_id = _performance_monitor.start_operation(op_name, metadata)
            
            try:
                result = func(*args, **kwargs)
                _performance_monitor.end_operation(operation_id, success=True)
                return result
            except Exception as e:
                _performance_monitor.end_operation(
                    operation_id, 
                    success=False, 
                    error_message=str(e)
                )
                raise
        
        return wrapper
    return decorator


@contextmanager
def time_context(operation_name: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Context manager for timing code blocks.
    
    Args:
        operation_name: Name of the operation
        metadata: Additional metadata to attach
        
    Example:
        with time_context("audio_processing"):
            # Code block to time
            pass
    """
    op_name = operation_name or "unnamed_operation"
    operation_id = _performance_monitor.start_operation(op_name, metadata)
    
    try:
        yield
    except Exception as e:
        _performance_monitor.end_operation(
            operation_id, 
            success=False, 
            error_message=str(e)
        )
        raise
    finally:
        _performance_monitor.end_operation(operation_id, success=True)


def get_performance_monitor() -> PerformanceMonitor:
    """
    Get the global performance monitor instance.
    
    Returns:
        The global PerformanceMonitor instance
    """
    return _performance_monitor


def get_operation_stats(operation_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to get operation statistics.
    
    Args:
        operation_name: Filter by operation name (None for all)
        
    Returns:
        Dictionary with operation statistics
    """
    return _performance_monitor.get_operation_stats(operation_name)


def get_memory_stats() -> Dict[str, Any]:
    """
    Convenience function to get memory usage statistics.
    
    Returns:
        Dictionary with memory statistics
    """
    return _performance_monitor.get_memory_stats()


def get_performance_summary() -> Dict[str, Any]:
    """
    Convenience function to get comprehensive performance summary.
    
    Returns:
        Dictionary with performance summary
    """
    return _performance_monitor.get_summary()


def clear_performance_metrics() -> None:
    """
    Convenience function to clear all performance metrics.
    """
    _performance_monitor.clear_metrics()


def export_metrics_to_dict() -> Dict[str, Any]:
    """
    Export all metrics to a dictionary format for serialization.
    
    Returns:
        Dictionary containing all metrics data
    """
    with _performance_monitor._lock:
        metrics_data = []
        for metric in _performance_monitor._metrics:
            metrics_data.append({
                "operation_name": metric.operation_name,
                "start_time": metric.start_time,
                "end_time": metric.end_time,
                "duration_ms": metric.duration_ms,
                "memory_before_mb": metric.memory_before_mb,
                "memory_after_mb": metric.memory_after_mb,
                "memory_delta_mb": metric.memory_delta_mb,
                "success": metric.success,
                "error_message": metric.error_message,
                "metadata": metric.metadata
            })
        
        return {
            "metrics": metrics_data,
            "summary": _performance_monitor.get_summary(),
            "memory_stats": _performance_monitor.get_memory_stats(),
            "export_timestamp": time.time()
        }


def get_percentile_stats(operation_name: Optional[str] = None, percentiles: List[float] = None) -> Dict[str, Any]:
    """
    Get percentile statistics for operation durations.
    
    Args:
        operation_name: Filter by operation name (None for all)
        percentiles: List of percentiles to calculate (default: [50, 90, 95, 99])
        
    Returns:
        Dictionary with percentile statistics
    """
    if percentiles is None:
        percentiles = [50, 90, 95, 99]
    
    with _performance_monitor._lock:
        metrics = _performance_monitor._metrics.copy()
    
    if operation_name:
        metrics = [m for m in metrics if m.operation_name == operation_name]
    
    durations = [m.duration_ms for m in metrics if m.duration_ms is not None]
    
    if not durations:
        return {"error": "No duration data available"}
    
    durations.sort()
    n = len(durations)
    
    percentile_stats = {}
    for p in percentiles:
        if p < 0 or p > 100:
            continue
        
        index = (p / 100) * (n - 1)
        if index.is_integer():
            percentile_stats[f"p{p}"] = durations[int(index)]
        else:
            lower = durations[int(index)]
            upper = durations[min(int(index) + 1, n - 1)]
            percentile_stats[f"p{p}"] = lower + (upper - lower) * (index - int(index))
    
    return {
        "operation_name": operation_name or "all_operations",
        "total_samples": n,
        "percentiles": percentile_stats
    }


def get_system_resources() -> Dict[str, Any]:
    """
    Get comprehensive system resource information.
    
    Returns:
        Dictionary with system resource data
    """
    try:
        import psutil
        
        # CPU information
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        # Memory information
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk information
        disk = psutil.disk_usage('/')
        
        # Network information (if available)
        try:
            network = psutil.net_io_counters()
            network_info = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
        except (AttributeError, OSError):
            network_info = {"error": "Network stats not available"}
        
        return {
            "cpu": {
                "percent": cpu_percent,
                "count": cpu_count,
                "frequency_mhz": cpu_freq.current if cpu_freq else None,
                "frequency_max_mhz": cpu_freq.max if cpu_freq else None
            },
            "memory": {
                "total_mb": memory.total / (1024 * 1024),
                "available_mb": memory.available / (1024 * 1024),
                "used_mb": memory.used / (1024 * 1024),
                "percent": memory.percent
            },
            "swap": {
                "total_mb": swap.total / (1024 * 1024),
                "used_mb": swap.used / (1024 * 1024),
                "free_mb": swap.free / (1024 * 1024),
                "percent": swap.percent
            },
            "disk": {
                "total_mb": disk.total / (1024 * 1024),
                "used_mb": disk.used / (1024 * 1024),
                "free_mb": disk.free / (1024 * 1024),
                "percent": (disk.used / disk.total) * 100
            },
            "network": network_info,
            "timestamp": time.time()
        }
    except ImportError:
        return {"error": "psutil not available"}


def reset_performance_monitor() -> None:
    """
    Reset the global performance monitor (clear all data and reinitialize).
    """
    global _performance_monitor
    _performance_monitor.clear_metrics()
    _performance_monitor = PerformanceMonitor()
    logger.info("Performance monitor reset")


# Additional utility functions for common timing patterns

def time_async_operation(operation_name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
    """
    Decorator for timing async functions.
    
    Args:
        operation_name: Custom name for the operation (defaults to function name)
        metadata: Additional metadata to attach
        
    Example:
        @time_async_operation("async_audio_processing")
        async def process_audio_async(file_path: str):
            # Async function implementation
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            operation_id = _performance_monitor.start_operation(op_name, metadata)
            
            try:
                result = await func(*args, **kwargs)
                _performance_monitor.end_operation(operation_id, success=True)
                return result
            except Exception as e:
                _performance_monitor.end_operation(
                    operation_id, 
                    success=False, 
                    error_message=str(e)
                )
                raise
        
        return wrapper
    return decorator


class TimingContext:
    """
    Enhanced context manager with additional features for timing operations.
    
    Provides more control over timing operations with support for
    nested operations and custom metadata updates.
    """
    
    def __init__(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize timing context.
        
        Args:
            operation_name: Name of the operation
            metadata: Initial metadata to attach
        """
        self.operation_name = operation_name
        self.operation_id = None
        self.metadata = metadata or {}
    
    def __enter__(self):
        """Enter the timing context."""
        self.operation_id = _performance_monitor.start_operation(
            self.operation_name, 
            self.metadata
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the timing context."""
        if self.operation_id:
            success = exc_type is None
            error_message = str(exc_val) if exc_val else None
            _performance_monitor.end_operation(
                self.operation_id, 
                success=success, 
                error_message=error_message
            )
    
    def update_metadata(self, **kwargs):
        """
        Update metadata for the current operation.
        
        Args:
            **kwargs: Metadata key-value pairs to update
        """
        if self.operation_id and self.operation_id in _performance_monitor._active_operations:
            with _performance_monitor._lock:
                if self.operation_id in _performance_monitor._active_operations:
                    _performance_monitor._active_operations[self.operation_id].metadata.update(kwargs)