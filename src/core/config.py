"""
Configuration management for the audio transcription service.

This module handles all configuration parameters using pydantic-settings
for type safety and environment variable support.
"""

from typing import Optional, Literal
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """
    Application settings with environment variable support.

    All settings can be overridden via environment variables with the same name.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # API Configuration
    app_name: str = "Audio Transcription Service"
    app_version: str = "1.0.0"
    api_v1_prefix: str = "/v1"
    debug_mode: bool = Field(default=False, description="Enable debug mode")

    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of worker processes")

    # Model Configuration
    model_dir: Path = Field(
        default=Path("models"), description="Directory for storing downloaded models"
    )
    whisper_model_size: Literal["tiny", "base", "small", "medium", "large"] = Field(
        default="small",
        description="Whisper model size (trade-off between speed and accuracy)",
    )
    demucs_model: str = Field(
        default="htdemucs", description="Demucs model name for source separation"
    )

    # Audio Processing Configuration
    max_file_size_mb: int = Field(
        default=100, description="Maximum audio file size in MB"
    )
    target_sample_rate: int = Field(
        default=16000, description="Target sample rate for audio processing"
    )
    chunk_size_seconds: int = Field(
        default=30, description="Audio chunk size for processing long files"
    )
    chunk_overlap_seconds: float = Field(
        default=2.0, description="Overlap between chunks for better stitching"
    )

    # Feature Flags
    enable_gpu: bool = Field(
        default=True, description="Enable GPU acceleration if available"
    )
    enable_separation_by_default: bool = Field(
        default=True, description="Enable vocal separation by default"
    )
    enable_vad: bool = Field(
        default=True, description="Enable Voice Activity Detection"
    )
    enable_diarization: bool = Field(
        default=False,
        description="Enable speaker diarization (requires HuggingFace token)",
    )

    # Performance Configuration
    model_loading_strategy: Literal["lazy", "eager"] = Field(
        default="lazy", description="Model loading strategy (lazy for faster startup)"
    )
    cache_models: bool = Field(
        default=True, description="Keep models in memory after first load"
    )
    use_quantization: bool = Field(
        default=False, description="Use int8 quantization for faster inference"
    )

    # External Services
    huggingface_token: Optional[str] = Field(
        default=None, description="HuggingFace token for diarization models"
    )

    # Logging Configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    log_format: Literal["json", "text"] = Field(
        default="json", description="Log output format"
    )

    @field_validator("model_dir")
    @classmethod
    def create_model_dir(cls, v: Path) -> Path:
        """Ensure model directory exists."""
        v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("max_file_size_mb")
    @classmethod
    def validate_file_size(cls, v: int) -> int:
        """Validate maximum file size."""
        if v < 1:
            raise ValueError("Maximum file size must be at least 1 MB")
        if v > 500:
            raise ValueError("Maximum file size cannot exceed 500 MB")
        return v

    @property
    def max_file_size_bytes(self) -> int:
        """Get maximum file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024

    @property
    def whisper_model_path(self) -> Path:
        """Get Whisper model path."""
        return self.model_dir / "whisper" / self.whisper_model_size

    @property
    def demucs_model_path(self) -> Path:
        """Get Demucs model path."""
        return self.model_dir / "demucs" / self.demucs_model


# Global settings instance
settings = Settings()
