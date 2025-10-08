"""
Audio utility functions for format conversion, validation, and I/O operations.

Provides comprehensive audio handling including format detection, conversion,
resampling, and temporary file management.
"""

import tempfile
import subprocess
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
from contextlib import contextmanager
import torch
import torchaudio

from ..core.config import settings
from ..core.exceptions import AudioFormatError, AudioProcessingError, FileSizeError
from ..core.logging import get_logger, log_execution_time


logger = get_logger(__name__)

# Supported audio formats
SUPPORTED_FORMATS = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", 
    ".opus", ".webm", ".aac", ".wma"
}

# Format to MIME type mapping
FORMAT_TO_MIME = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".webm": "audio/webm",
    ".aac": "audio/aac",
    ".wma": "audio/x-ms-wma"
}


@contextmanager
def temporary_audio_file(suffix: str = ".wav"):
    """
    Context manager for temporary audio files.
    
    Args:
        suffix: File extension for the temporary file
        
    Yields:
        Path to temporary file
    """
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            temp_file = Path(f.name)
        yield temp_file
    finally:
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_file}: {e}")


def detect_audio_format(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Detect audio format and properties using ffprobe.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Dictionary with format information
        
    Raises:
        AudioFormatError: If format detection fails
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise AudioFormatError(f"File not found: {file_path}")
    
    try:
        # Use ffprobe to get format info
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(file_path)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            raise AudioFormatError(
                f"ffprobe failed: {result.stderr}",
                format_info=file_path.suffix
            )
        
        info = json.loads(result.stdout)

        streams = info.get("streams", [])
        audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
        if not audio_streams:
            raise AudioFormatError(
                "No audio stream found", format_info=str(file_path.suffix)
            )

        audio_stream = audio_streams[0]
        fmt = info.get("format", {})

        return {
            "path": str(file_path),
            "extension": file_path.suffix.lower(),
            "format_name": fmt.get("format_name"),
            "format_long_name": fmt.get("format_long_name"),
            "duration": float(fmt.get("duration") or audio_stream.get("duration") or 0.0),
            "bit_rate": int(fmt.get("bit_rate") or audio_stream.get("bit_rate") or 0),
            "codec_name": audio_stream.get("codec_name"),
            "sample_rate": int(audio_stream.get("sample_rate") or 0),
            "channels": int(audio_stream.get("channels") or 0),
            "channel_layout": audio_stream.get("channel_layout"),
            "tags": fmt.get("tags", {}),
        }
    except AudioFormatError:
        raise
    except Exception as e:
        raise AudioFormatError(f"Failed to detect audio format: {e}", format_info=str(file_path))


def _ext(path: Union[str, Path]) -> str:
    return Path(path).suffix.lower()


def is_supported_format(path: Union[str, Path]) -> bool:
    return _ext(path) in SUPPORTED_FORMATS


def get_mime_type(path: Union[str, Path]) -> str:
    return FORMAT_TO_MIME.get(_ext(path), "application/octet-stream")


@log_execution_time(name="validate_audio_file")
def validate_audio_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    file_path = Path(file_path)

    if not file_path.exists() or not file_path.is_file():
        raise AudioFormatError(f"File not found: {file_path}")

    if not is_supported_format(file_path):
        raise AudioFormatError(
            f"Unsupported file extension: {_ext(file_path)}",
            format_info=_ext(file_path),
        )

    file_size = file_path.stat().st_size
    if file_size > settings.max_file_size_bytes:
        raise FileSizeError(actual_size=file_size, max_size=settings.max_file_size_bytes)

    info = detect_audio_format(file_path)
    if info.get("channels", 0) <= 0:
        raise AudioFormatError("Invalid channel count", format_info=str(info.get("channels")))

    return info


def _run_ffmpeg(cmd: list) -> None:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise AudioProcessingError(
                f"ffmpeg failed: {result.stderr}", stage="ffmpeg_conversion"
            )
    except AudioProcessingError:
        raise
    except Exception as e:
        raise AudioProcessingError(f"ffmpeg execution error: {e}", stage="ffmpeg")


@log_execution_time(name="convert_to_wav")
def convert_to_wav(
    src_path: Union[str, Path],
    dst_path: Union[str, Path],
    sample_rate: Optional[int] = None,
    mono: bool = True,
) -> Path:
    src_path = Path(src_path)
    dst_path = Path(dst_path)

    sr = sample_rate or settings.target_sample_rate

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src_path),
        "-ac",
        "1" if mono else "2",
        "-ar",
        str(sr),
        "-c:a",
        "pcm_s16le",
        str(dst_path),
    ]
    _run_ffmpeg(cmd)
    return dst_path


@log_execution_time(name="ensure_wav")
def ensure_wav(
    src_path: Union[str, Path],
    target_sample_rate: Optional[int] = None,
    mono: bool = True,
) -> Path:
    src_path = Path(src_path)
    target_sample_rate = target_sample_rate or settings.target_sample_rate

    try:
        info = detect_audio_format(src_path)
    except AudioFormatError:
        # Try anyway to convert using ffmpeg
        info = {"extension": _ext(src_path), "sample_rate": 0, "channels": 0}

    already_wav = _ext(src_path) == ".wav"
    needs_resample = (info.get("sample_rate") or 0) != int(target_sample_rate)
    needs_channels = mono and (info.get("channels") or 2) != 1

    if already_wav and not needs_resample and not needs_channels:
        return src_path

    with temporary_audio_file(suffix=".wav") as tmp_wav:
        return convert_to_wav(src_path, tmp_wav, sample_rate=target_sample_rate, mono=mono)


def _torchaudio_resampler(orig_sr: int, target_sr: int):
    return torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)


@log_execution_time(name="load_audio_as_tensor")
def load_audio_as_tensor(
    file_path: Union[str, Path], target_sample_rate: Optional[int] = None, mono: bool = True
) -> Tuple[torch.Tensor, int]:
    file_path = ensure_wav(file_path, target_sample_rate=target_sample_rate or settings.target_sample_rate, mono=mono)

    waveform, sample_rate = torchaudio.load(str(file_path))

    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    desired_sr = target_sample_rate or settings.target_sample_rate
    if sample_rate != desired_sr:
        resampler = _torchaudio_resampler(sample_rate, desired_sr)
        waveform = resampler(waveform)
        sample_rate = desired_sr

    return waveform, sample_rate


@log_execution_time(name="resample_audio_tensor")
def resample_audio_tensor(
    waveform: torch.Tensor, original_sr: int, target_sr: int
) -> torch.Tensor:
    if original_sr == target_sr:
        return waveform
    resampler = _torchaudio_resampler(original_sr, target_sr)
    return resampler(waveform)


@log_execution_time(name="save_wav_from_tensor")
def save_wav_from_tensor(
    waveform: torch.Tensor, sample_rate: int, dst_path: Union[str, Path]
) -> Path:
    dst_path = Path(dst_path)
    # Ensure shape [channels, time]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    torchaudio.save(str(dst_path), waveform, sample_rate=sample_rate, encoding="PCM_S", bits_per_sample=16)
    return dst_path
