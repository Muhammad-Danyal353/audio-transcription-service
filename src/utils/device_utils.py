"""
Device detection and management utilities.

Handles GPU availability checking, device selection, and memory management
for optimal performance across different hardware configurations.
"""

import torch
import logging
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from ..core.config import settings
from ..core.logging import get_logger, log_execution_time


logger = get_logger(__name__)


@dataclass
class DeviceInfo:
    """Information about the selected compute device."""

    device_type: str  # 'cuda', 'mps', or 'cpu'
    device_name: str
    device_index: Optional[int]
    memory_available: Optional[int]  # in bytes
    memory_total: Optional[int]  # in bytes
    supports_half_precision: bool
    supports_quantization: bool

    @property
    def torch_device(self) -> torch.device:
        """Get torch device object."""
        if self.device_index is not None:
            return torch.device(f"{self.device_type}:{self.device_index}")
        return torch.device(self.device_type)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "type": self.device_type,
            "name": self.device_name,
            "index": self.device_index,
            "memory_available_gb": (
                round(self.memory_available / (1024**3), 2)
                if self.memory_available
                else None
            ),
            "memory_total_gb": (
                round(self.memory_total / (1024**3), 2) if self.memory_total else None
            ),
            "supports_half_precision": self.supports_half_precision,
            "supports_quantization": self.supports_quantization,
        }


class DeviceManager:
    """
    Manages device selection and configuration for model inference.

    Singleton class that handles GPU/CPU detection and provides optimal
    device configuration based on available hardware.
    """

    _instance: Optional["DeviceManager"] = None
    _device_info: Optional[DeviceInfo] = None

    def __new__(cls) -> "DeviceManager":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize device manager."""
        if self._device_info is None:
            self._device_info = self._detect_device()

    @log_execution_time(name="device_detection")
    def _detect_device(self) -> DeviceInfo:
        """
        Detect and configure the best available device.

        Returns:
            DeviceInfo object with device configuration
        """
        # Check if GPU usage is enabled in settings
        if not settings.enable_gpu:
            logger.info("GPU disabled by configuration, using CPU")
            return self._get_cpu_info()

        # Try CUDA (NVIDIA GPUs)
        if torch.cuda.is_available():
            return self._get_cuda_info()

        # Try MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            return self._get_mps_info()

        # Fallback to CPU
        logger.info("No GPU available, using CPU")
        return self._get_cpu_info()

    def _get_cuda_info(self) -> DeviceInfo:
        """Get CUDA device information."""
        device_index = 0  # Use first GPU by default

        # Get device properties
        props = torch.cuda.get_device_properties(device_index)
        memory_total = props.total_memory
        memory_available = memory_total - torch.cuda.memory_allocated(device_index)

        # Check capabilities
        major, minor = props.major, props.minor
        supports_half = major >= 7  # Volta and newer
        supports_int8 = major >= 7  # Tensor cores support

        logger.info(
            f"Using CUDA device: {props.name} "
            f"(compute {major}.{minor}, "
            f"{memory_total / (1024**3):.1f}GB memory)"
        )

        return DeviceInfo(
            device_type="cuda",
            device_name=props.name,
            device_index=device_index,
            memory_available=memory_available,
            memory_total=memory_total,
            supports_half_precision=supports_half,
            supports_quantization=supports_int8,
        )

    def _get_mps_info(self) -> DeviceInfo:
        """Get Apple Silicon MPS device information."""
        logger.info("Using Apple Silicon MPS device")

        return DeviceInfo(
            device_type="mps",
            device_name="Apple Silicon GPU",
            device_index=None,
            memory_available=None,  # MPS doesn't provide memory info
            memory_total=None,
            supports_half_precision=True,
            supports_quantization=False,  # Limited quantization support
        )

    def _get_cpu_info(self) -> DeviceInfo:
        """Get CPU device information."""
        import psutil

        # Get CPU and memory info
        cpu_count = torch.get_num_threads()
        memory_info = psutil.virtual_memory()

        logger.info(
            f"Using CPU with {cpu_count} threads, "
            f"{memory_info.total / (1024**3):.1f}GB RAM"
        )

        return DeviceInfo(
            device_type="cpu",
            device_name=f"CPU ({cpu_count} threads)",
            device_index=None,
            memory_available=memory_info.available,
            memory_total=memory_info.total,
            supports_half_precision=False,  # CPU doesn't benefit from FP16
            supports_quantization=True,  # CPU supports INT8
        )

    @property
    def device_info(self) -> DeviceInfo:
        """Get current device information."""
        return self._device_info

    @property
    def device(self) -> torch.device:
        """Get torch device for model loading."""
        return self._device_info.torch_device

    @property
    def dtype(self) -> torch.dtype:
        """
        Get optimal data type for the current device.

        Returns:
            torch.float16 for GPU with FP16 support, torch.float32 otherwise
        """
        if self._device_info.supports_half_precision and settings.use_quantization:
            return torch.float16
        return torch.float32

    def estimate_memory_requirement(
        self, model_size_mb: float, batch_size: int = 1, overhead_factor: float = 1.5
    ) -> Tuple[bool, float]:
        """
        Estimate if model can fit in memory.

        Args:
            model_size_mb: Model size in megabytes
            batch_size: Batch size for inference
            overhead_factor: Safety factor for memory overhead

        Returns:
            Tuple of (can_fit, required_memory_mb)
        """
        required_bytes = model_size_mb * 1024 * 1024 * batch_size * overhead_factor

        if self._device_info.memory_available:
            can_fit = required_bytes < self._device_info.memory_available
        else:
            # For CPU, assume we can use 50% of available RAM
            import psutil

            available = psutil.virtual_memory().available * 0.5
            can_fit = required_bytes < available

        return can_fit, required_bytes / (1024 * 1024)

    def clear_cache(self) -> None:
        """Clear GPU memory cache if applicable."""
        if self._device_info.device_type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("Cleared CUDA cache")
        elif self._device_info.device_type == "mps":
            # MPS doesn't have explicit cache clearing yet
            pass

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get current memory statistics.

        Returns:
            Dictionary with memory usage information
        """
        stats = {"device_type": self._device_info.device_type}

        if self._device_info.device_type == "cuda":
            stats.update(
                {
                    "allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
                    "reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
                    "free_mb": (
                        self._device_info.memory_total - torch.cuda.memory_allocated()
                    )
                    / (1024 * 1024),
                }
            )
        elif self._device_info.device_type == "cpu":
            import psutil

            mem = psutil.virtual_memory()
            stats.update(
                {
                    "used_mb": mem.used / (1024 * 1024),
                    "available_mb": mem.available / (1024 * 1024),
                    "percent_used": mem.percent,
                }
            )

        return stats


# Global device manager instance
device_manager = DeviceManager()


def get_device() -> torch.device:
    """
    Get the optimal device for computation.

    Returns:
        torch.device object
    """
    return device_manager.device


def get_device_info() -> DeviceInfo:
    """
    Get information about the current device.

    Returns:
        DeviceInfo object
    """
    return device_manager.device_info


def get_optimal_dtype() -> torch.dtype:
    """
    Get optimal data type for the current device.

    Returns:
        torch.dtype (float16 or float32)
    """
    return device_manager.dtype
