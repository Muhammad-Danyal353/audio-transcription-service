"""
Pytest configuration and fixtures for the audio transcription service tests.

This module provides shared fixtures, test configuration, and utilities
for all test modules in the test suite.
"""

import pytest
import asyncio
import tempfile
import os
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
from unittest.mock import Mock, AsyncMock, patch
import json

from src.core.config import settings
from src.core.logging import get_logger
from src.services.audio_processor import AudioProcessor, ProcessingConfig, ProcessingResult
from src.services.transcriber import TranscriptionResult, TranscriptSegment, WordSegment
from src.services.separator import SeparationResult
from src.services.diarization import DiarizationResult, SpeakerSegment
from src.services.vad import VoiceSegment
from src.api.models import TranscribeConfig, TranscribeResponse
from src.utils.timing import clear_performance_metrics

logger = get_logger(__name__)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def clear_metrics():
    """Clear performance metrics before each test."""
    clear_performance_metrics()
    yield
    clear_performance_metrics()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_audio_data():
    """Generate sample audio data for testing."""
    # Generate 5 seconds of sine wave at 440Hz (A note)
    sample_rate = 16000
    duration = 5.0
    frequency = 440.0
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    return audio, sample_rate


@pytest.fixture
def sample_audio_file(temp_dir, sample_audio_data):
    """Create a sample audio file for testing."""
    audio, sample_rate = sample_audio_data
    
    # Save as WAV file
    import soundfile as sf
    audio_file = temp_dir / "sample.wav"
    sf.write(audio_file, audio, sample_rate)
    
    return audio_file, audio, sample_rate


@pytest.fixture
def sample_audio_file_stereo(temp_dir, sample_audio_data):
    """Create a stereo sample audio file for testing."""
    audio, sample_rate = sample_audio_data
    
    # Create stereo audio
    stereo_audio = np.stack([audio, audio * 0.5], axis=0)
    
    # Save as WAV file
    import soundfile as sf
    audio_file = temp_dir / "sample_stereo.wav"
    sf.write(audio_file, stereo_audio.T, sample_rate)
    
    return audio_file, stereo_audio, sample_rate


@pytest.fixture
def sample_audio_file_long(temp_dir):
    """Create a long sample audio file for testing chunking."""
    sample_rate = 16000
    duration = 60.0  # 1 minute
    
    # Generate multiple sine waves
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # A note
    
    # Add some variation
    audio += 0.5 * np.sin(2 * np.pi * 880 * t).astype(np.float32)  # A octave
    
    # Save as WAV file
    import soundfile as sf
    audio_file = temp_dir / "sample_long.wav"
    sf.write(audio_file, audio, sample_rate)
    
    return audio_file, audio, sample_rate


@pytest.fixture
def mock_transcription_result():
    """Create a mock transcription result for testing."""
    segments = [
        TranscriptSegment(
            id=0,
            start=0.0,
            end=2.5,
            text="Hello, this is a test transcription.",
            confidence=0.95,
            words=[
                WordSegment(word="Hello", start=0.0, end=0.5, probability=0.98),
                WordSegment(word="this", start=0.6, end=0.8, probability=0.95),
                WordSegment(word="is", start=0.9, end=1.0, probability=0.99),
                WordSegment(word="a", start=1.1, end=1.2, probability=0.97),
                WordSegment(word="test", start=1.3, end=1.8, probability=0.94),
                WordSegment(word="transcription", start=1.9, end=2.5, probability=0.91)
            ]
        ),
        TranscriptSegment(
            id=1,
            start=2.5,
            end=5.0,
            text="This is the second segment.",
            confidence=0.92,
            words=[
                WordSegment(word="This", start=2.5, end=2.8, probability=0.96),
                WordSegment(word="is", start=2.9, end=3.0, probability=0.99),
                WordSegment(word="the", start=3.1, end=3.3, probability=0.98),
                WordSegment(word="second", start=3.4, end=3.9, probability=0.93),
                WordSegment(word="segment", start=4.0, end=4.5, probability=0.90)
            ]
        )
    ]
    
    return TranscriptionResult(
        text="Hello, this is a test transcription. This is the second segment.",
        language="en",
        language_probability=0.95,
        segments=segments,
        duration=5.0,
        metadata={"model_size": "small", "device": "cpu"}
    )


@pytest.fixture
def mock_separation_result(sample_audio_data):
    """Create a mock separation result for testing."""
    audio, sample_rate = sample_audio_data
    
    return SeparationResult(
        vocals=audio * 0.8,  # Simulate separated vocals
        accompaniment=audio * 0.2,  # Simulate accompaniment
        sample_rate=sample_rate,
        duration=5.0,
        metadata={"method": "demucs", "model": "htdemucs"}
    )


@pytest.fixture
def mock_diarization_result():
    """Create a mock diarization result for testing."""
    segments = [
        SpeakerSegment(speaker="SPEAKER_00", start=0.0, end=2.5, confidence=0.95),
        SpeakerSegment(speaker="SPEAKER_01", start=2.5, end=5.0, confidence=0.92)
    ]
    
    return DiarizationResult(
        speakers=["SPEAKER_00", "SPEAKER_01"],
        segments=segments,
        duration=5.0,
        num_speakers=2,
        metadata={"method": "pyannote", "model": "speaker-diarization-3.1"}
    )


@pytest.fixture
def mock_voice_segments():
    """Create mock voice activity segments for testing."""
    return [
        VoiceSegment(start_time=0.0, end_time=2.5, confidence=0.95, is_speech=True),
        VoiceSegment(start_time=2.5, end_time=3.0, confidence=0.1, is_speech=False),
        VoiceSegment(start_time=3.0, end_time=5.0, confidence=0.92, is_speech=True)
    ]


@pytest.fixture
def mock_processing_result(
    mock_transcription_result,
    mock_separation_result,
    mock_diarization_result,
    mock_voice_segments
):
    """Create a mock processing result for testing."""
    return ProcessingResult(
        transcription=mock_transcription_result,
        separation=mock_separation_result,
        diarization=mock_diarization_result,
        voice_segments=mock_voice_segments,
        config=ProcessingConfig(),
        processing_time=3.2,
        device_info={"device": "cpu", "cuda_available": False},
        performance_metrics={"total_operations": 5},
        quality_metrics={"transcription": {"avg_confidence": 0.93}}
    )


@pytest.fixture
def mock_transcribe_config():
    """Create a mock transcription configuration for testing."""
    return TranscribeConfig(
        enable_separation=True,
        enable_vad=True,
        enable_diarization=False,
        language="en",
        word_timestamps=True
    )


@pytest.fixture
def mock_transcribe_response(mock_processing_result, mock_transcribe_config):
    """Create a mock transcription response for testing."""
    from src.api.models import convert_processing_result_to_response
    
    return convert_processing_result_to_response(
        mock_processing_result,
        "test-request-id",
        mock_transcribe_config
    )


@pytest.fixture
def mock_audio_processor():
    """Create a mock audio processor for testing."""
    processor = Mock(spec=AudioProcessor)
    processor.device = "cpu"
    processor.device_info = {"device": "cpu", "cuda_available": False}
    
    return processor


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx client for testing."""
    client = AsyncMock()
    return client


@pytest.fixture
def test_config():
    """Create test configuration."""
    return {
        "app_name": "Audio Transcription Service Test",
        "app_version": "1.0.0-test",
        "debug_mode": True,
        "max_file_size_mb": 10,
        "target_sample_rate": 16000,
        "whisper_model_size": "tiny",  # Use tiny model for faster tests
        "enable_gpu": False,  # Disable GPU for consistent testing
        "enable_separation_by_default": False,  # Disable for faster tests
        "enable_vad": False,  # Disable for faster tests
        "enable_diarization": False,  # Disable for faster tests
        "chunk_size_seconds": 10,  # Smaller chunks for testing
        "chunk_overlap_seconds": 1.0
    }


@pytest.fixture
def mock_settings(test_config):
    """Mock settings with test configuration."""
    with patch('src.core.config.settings') as mock_settings:
        for key, value in test_config.items():
            setattr(mock_settings, key, value)
        yield mock_settings


@pytest.fixture
def sample_audio_files(temp_dir):
    """Create multiple sample audio files for batch testing."""
    files = []
    
    # Create different types of audio files
    sample_rate = 16000
    
    # Short audio (5 seconds)
    audio_short = np.sin(2 * np.pi * 440 * np.linspace(0, 5, sample_rate * 5)).astype(np.float32)
    file_short = temp_dir / "short.wav"
    import soundfile as sf
    sf.write(file_short, audio_short, sample_rate)
    files.append(file_short)
    
    # Medium audio (15 seconds)
    audio_medium = np.sin(2 * np.pi * 440 * np.linspace(0, 15, sample_rate * 15)).astype(np.float32)
    file_medium = temp_dir / "medium.wav"
    sf.write(file_medium, audio_medium, sample_rate)
    files.append(file_medium)
    
    # Long audio (35 seconds) - will trigger chunking
    audio_long = np.sin(2 * np.pi * 440 * np.linspace(0, 35, sample_rate * 35)).astype(np.float32)
    file_long = temp_dir / "long.wav"
    sf.write(file_long, audio_long, sample_rate)
    files.append(file_long)
    
    return files


@pytest.fixture
def invalid_audio_file(temp_dir):
    """Create an invalid audio file for testing error handling."""
    invalid_file = temp_dir / "invalid.txt"
    invalid_file.write_text("This is not an audio file")
    return invalid_file


@pytest.fixture
def large_audio_file(temp_dir):
    """Create a large audio file for testing file size limits."""
    sample_rate = 16000
    duration = 300.0  # 5 minutes
    
    # Generate audio data
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    # Save as WAV file
    import soundfile as sf
    audio_file = temp_dir / "large.wav"
    sf.write(audio_file, audio, sample_rate)
    
    return audio_file


@pytest.fixture
def mock_fastapi_app():
    """Create a mock FastAPI app for testing."""
    from fastapi import FastAPI
    app = FastAPI()
    return app


@pytest.fixture
def mock_request():
    """Create a mock FastAPI request for testing."""
    request = Mock()
    request.state.request_id = "test-request-id"
    request.method = "POST"
    request.url = "http://localhost:8000/v1/transcribe"
    request.client.host = "127.0.0.1"
    request.headers = {"user-agent": "test-client"}
    return request


@pytest.fixture
def mock_upload_file(temp_dir, sample_audio_file):
    """Create a mock upload file for testing."""
    from fastapi import UploadFile
    
    file_path, _, _ = sample_audio_file
    
    with open(file_path, "rb") as f:
        content = f.read()
    
    upload_file = Mock(spec=UploadFile)
    upload_file.filename = "test.wav"
    upload_file.size = len(content)
    upload_file.read = AsyncMock(return_value=content)
    
    return upload_file


@pytest.fixture
def mock_upload_file_large(temp_dir, large_audio_file):
    """Create a mock large upload file for testing."""
    from fastapi import UploadFile
    
    file_path = large_audio_file
    
    with open(file_path, "rb") as f:
        content = f.read()
    
    upload_file = Mock(spec=UploadFile)
    upload_file.filename = "large.wav"
    upload_file.size = len(content)
    upload_file.read = AsyncMock(return_value=content)
    
    return upload_file


@pytest.fixture
def mock_upload_file_invalid(temp_dir, invalid_audio_file):
    """Create a mock invalid upload file for testing."""
    from fastapi import UploadFile
    
    file_path = invalid_audio_file
    
    with open(file_path, "rb") as f:
        content = f.read()
    
    upload_file = Mock(spec=UploadFile)
    upload_file.filename = "invalid.txt"
    upload_file.size = len(content)
    upload_file.read = AsyncMock(return_value=content)
    
    return upload_file


# Test utilities

def create_test_audio(duration: float = 5.0, sample_rate: int = 16000, frequency: float = 440.0) -> np.ndarray:
    """Create test audio data."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    return np.sin(2 * np.pi * frequency * t).astype(np.float32)


def create_test_audio_file(temp_dir: Path, duration: float = 5.0, sample_rate: int = 16000, 
                          frequency: float = 440.0, filename: str = "test.wav") -> Path:
    """Create a test audio file."""
    audio = create_test_audio(duration, sample_rate, frequency)
    
    import soundfile as sf
    audio_file = temp_dir / filename
    sf.write(audio_file, audio, sample_rate)
    
    return audio_file


def assert_audio_files_equal(file1: Path, file2: Path, tolerance: float = 1e-6):
    """Assert that two audio files are equal within tolerance."""
    import soundfile as sf
    
    audio1, sr1 = sf.read(file1)
    audio2, sr2 = sf.read(file2)
    
    assert sr1 == sr2, f"Sample rates differ: {sr1} vs {sr2}"
    assert audio1.shape == audio2.shape, f"Audio shapes differ: {audio1.shape} vs {audio2.shape}"
    assert np.allclose(audio1, audio2, atol=tolerance), "Audio data differs beyond tolerance"


def assert_transcription_quality(result: TranscriptionResult, min_confidence: float = 0.8):
    """Assert that transcription result meets quality requirements."""
    assert result.text, "Transcription text should not be empty"
    assert result.language, "Language should be detected"
    assert result.language_probability > 0.5, "Language probability should be reasonable"
    assert len(result.segments) > 0, "Should have at least one segment"
    
    for segment in result.segments:
        assert segment.confidence >= min_confidence, f"Segment confidence too low: {segment.confidence}"
        assert segment.text.strip(), "Segment text should not be empty"
        assert segment.start < segment.end, "Segment start should be before end"


def assert_processing_result_quality(result: ProcessingResult):
    """Assert that processing result meets quality requirements."""
    assert result.transcription, "Should have transcription result"
    assert result.processing_time > 0, "Processing time should be positive"
    assert result.device_info, "Should have device info"
    
    # Check transcription quality
    assert_transcription_quality(result.transcription)
    
    # Check optional results
    if result.separation:
        assert result.separation.has_vocals, "Separation should produce vocals"
        assert result.separation.duration > 0, "Separation duration should be positive"
    
    if result.diarization:
        assert result.diarization.has_speakers, "Diarization should detect speakers"
        assert result.diarization.num_speakers > 0, "Should detect at least one speaker"
    
    if result.voice_segments:
        assert len(result.voice_segments) > 0, "Should have voice segments"
        for segment in result.voice_segments:
            assert segment.start_time < segment.end_time, "Voice segment timing should be valid"


# Pytest configuration

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "requires_models: marks tests that require downloaded models"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add slow marker to tests that might take a long time
        if "integration" in item.nodeid or "batch" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Add integration marker to integration tests
        if "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Add unit marker to unit tests
        if any(module in item.nodeid for module in ["test_audio_utils", "test_separator", "test_transcriber", "test_vad"]):
            item.add_marker(pytest.mark.unit)
        
        # Add requires_models marker to tests that need models
        if any(keyword in item.nodeid for keyword in ["whisper", "demucs", "pyannote"]):
            item.add_marker(pytest.mark.requires_models)
