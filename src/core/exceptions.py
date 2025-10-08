"""
Custom exception classes for the audio transcription service.

These exceptions provide specific error handling for different failure modes
in the audio processing pipeline.
"""

from typing import Optional, Dict, Any


class AudioTranscriptionError(Exception):
    """Base exception for all audio transcription errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the exception.

        Args:
            message: Error message
            error_code: Optional error code for API responses
            details: Optional additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "AUDIO_ERROR"
        self.details = details or {}


class AudioFormatError(AudioTranscriptionError):
    """Raised when audio format is invalid or unsupported."""

    def __init__(self, message: str, format_info: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="INVALID_FORMAT",
            details={"format": format_info} if format_info else {},
        )


class FileSizeError(AudioTranscriptionError):
    """Raised when audio file exceeds size limits."""

    def __init__(self, actual_size: int, max_size: int):
        super().__init__(
            message=f"File size {actual_size} bytes exceeds maximum {max_size} bytes",
            error_code="FILE_TOO_LARGE",
            details={"actual_size": actual_size, "max_size": max_size},
        )


class AudioProcessingError(AudioTranscriptionError):
    """Raised when audio processing fails."""

    def __init__(self, message: str, stage: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="PROCESSING_ERROR",
            details={"stage": stage} if stage else {},
        )


class ModelLoadError(AudioTranscriptionError):
    """Raised when model loading fails."""

    def __init__(self, model_name: str, reason: str):
        super().__init__(
            message=f"Failed to load model {model_name}: {reason}",
            error_code="MODEL_LOAD_ERROR",
            details={"model": model_name, "reason": reason},
        )


class TranscriptionError(AudioTranscriptionError):
    """Raised when transcription fails."""

    def __init__(self, message: str, model: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="TRANSCRIPTION_ERROR",
            details={"model": model} if model else {},
        )


class SeparationError(AudioTranscriptionError):
    """Raised when audio separation fails."""

    def __init__(self, message: str, fallback_used: bool = False):
        super().__init__(
            message=message,
            error_code="SEPARATION_ERROR",
            details={"fallback_used": fallback_used},
        )


class VADError(AudioTranscriptionError):
    """Raised when Voice Activity Detection fails."""

    def __init__(self, message: str):
        super().__init__(message=message, error_code="VAD_ERROR")


class DiarizationError(AudioTranscriptionError):
    """Raised when speaker diarization fails."""

    def __init__(self, message: str):
        super().__init__(message=message, error_code="DIARIZATION_ERROR")
