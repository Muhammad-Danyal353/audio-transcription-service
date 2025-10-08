"""
Unit tests for audio utilities.

This module tests the audio utility functions including format conversion,
resampling, validation, and file I/O operations.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

from src.utils.audio_utils import (
    load_audio, save_audio, resample_audio, validate_audio,
    get_audio_info, validate_audio_file, convert_audio_format
)


class TestAudioUtils:
    """Test cases for audio utility functions."""
    
    def test_load_audio_wav(self, sample_audio_file):
        """Test loading WAV audio file."""
        file_path, expected_audio, expected_sr = sample_audio_file
        
        audio, sample_rate = load_audio(file_path)
        
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert sample_rate == expected_sr
        assert np.allclose(audio, expected_audio, atol=1e-6)
    
    def test_load_audio_stereo(self, sample_audio_file_stereo):
        """Test loading stereo audio file."""
        file_path, expected_audio, expected_sr = sample_audio_file_stereo
        
        audio, sample_rate = load_audio(file_path)
        
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert sample_rate == expected_sr
        assert audio.ndim == 2  # Should be stereo
        assert audio.shape[0] == 2  # 2 channels
        assert np.allclose(audio, expected_audio, atol=1e-6)
    
    def test_load_audio_nonexistent(self, temp_dir):
        """Test loading non-existent audio file."""
        nonexistent_file = temp_dir / "nonexistent.wav"
        
        with pytest.raises(FileNotFoundError):
            load_audio(nonexistent_file)
    
    def test_save_audio(self, temp_dir, sample_audio_data):
        """Test saving audio file."""
        audio, sample_rate = sample_audio_data
        output_file = temp_dir / "output.wav"
        
        save_audio(audio, sample_rate, output_file)
        
        assert output_file.exists()
        
        # Verify the saved file can be loaded
        loaded_audio, loaded_sr = load_audio(output_file)
        assert loaded_sr == sample_rate
        assert np.allclose(loaded_audio, audio, atol=1e-6)
    
    def test_save_audio_stereo(self, temp_dir, sample_audio_data):
        """Test saving stereo audio file."""
        audio, sample_rate = sample_audio_data
        stereo_audio = np.stack([audio, audio * 0.5], axis=0)
        output_file = temp_dir / "output_stereo.wav"
        
        save_audio(stereo_audio, sample_rate, output_file)
        
        assert output_file.exists()
        
        # Verify the saved file can be loaded
        loaded_audio, loaded_sr = load_audio(output_file)
        assert loaded_sr == sample_rate
        assert loaded_audio.ndim == 2
        assert loaded_audio.shape[0] == 2
        assert np.allclose(loaded_audio, stereo_audio, atol=1e-6)
    
    def test_resample_audio(self, sample_audio_data):
        """Test audio resampling."""
        audio, original_sr = sample_audio_data
        target_sr = 8000
        
        resampled_audio = resample_audio(audio, original_sr, target_sr)
        
        assert isinstance(resampled_audio, np.ndarray)
        assert resampled_audio.dtype == np.float32
        
        # Check that the duration is approximately the same
        original_duration = len(audio) / original_sr
        resampled_duration = len(resampled_audio) / target_sr
        assert abs(original_duration - resampled_duration) < 0.1
    
    def test_resample_audio_same_rate(self, sample_audio_data):
        """Test resampling to the same sample rate."""
        audio, sample_rate = sample_audio_data
        
        resampled_audio = resample_audio(audio, sample_rate, sample_rate)
        
        assert np.array_equal(resampled_audio, audio)
    
    def test_resample_audio_stereo(self, sample_audio_data):
        """Test resampling stereo audio."""
        audio, original_sr = sample_audio_data
        stereo_audio = np.stack([audio, audio * 0.5], axis=0)
        target_sr = 8000
        
        resampled_audio = resample_audio(stereo_audio, original_sr, target_sr)
        
        assert isinstance(resampled_audio, np.ndarray)
        assert resampled_audio.ndim == 2
        assert resampled_audio.shape[0] == 2
        
        # Check that the duration is approximately the same
        original_duration = len(audio) / original_sr
        resampled_duration = len(resampled_audio[0]) / target_sr
        assert abs(original_duration - resampled_duration) < 0.1
    
    def test_validate_audio_valid(self, sample_audio_data):
        """Test validating valid audio data."""
        audio, sample_rate = sample_audio_data
        
        # Should not raise any exception
        validate_audio(audio, sample_rate)
    
    def test_validate_audio_invalid_dtype(self, sample_audio_data):
        """Test validating audio with invalid dtype."""
        audio, sample_rate = sample_audio_data
        audio_int = (audio * 32767).astype(np.int16)
        
        with pytest.raises(ValueError, match="Audio must be float32"):
            validate_audio(audio_int, sample_rate)
    
    def test_validate_audio_invalid_shape(self, sample_audio_data):
        """Test validating audio with invalid shape."""
        audio, sample_rate = sample_audio_data
        audio_3d = audio.reshape(1, 1, -1)
        
        with pytest.raises(ValueError, match="Audio must be 1D or 2D"):
            validate_audio(audio_3d, sample_rate)
    
    def test_validate_audio_invalid_sample_rate(self, sample_audio_data):
        """Test validating audio with invalid sample rate."""
        audio, _ = sample_audio_data
        
        with pytest.raises(ValueError, match="Sample rate must be positive"):
            validate_audio(audio, 0)
        
        with pytest.raises(ValueError, match="Sample rate must be positive"):
            validate_audio(audio, -1)
    
    def test_validate_audio_empty(self):
        """Test validating empty audio."""
        empty_audio = np.array([], dtype=np.float32)
        
        with pytest.raises(ValueError, match="Audio cannot be empty"):
            validate_audio(empty_audio, 16000)
    
    def test_get_audio_info(self, sample_audio_file):
        """Test getting audio file information."""
        file_path, _, _ = sample_audio_file
        
        info = get_audio_info(file_path)
        
        assert isinstance(info, dict)
        assert "duration" in info
        assert "sample_rate" in info
        assert "channels" in info
        assert "format" in info
        
        assert info["sample_rate"] == 16000
        assert info["channels"] == 1
        assert info["duration"] == 5.0
        assert info["format"] == "WAV"
    
    def test_get_audio_info_stereo(self, sample_audio_file_stereo):
        """Test getting stereo audio file information."""
        file_path, _, _ = sample_audio_file_stereo
        
        info = get_audio_info(file_path)
        
        assert info["channels"] == 2
        assert info["sample_rate"] == 16000
        assert info["duration"] == 5.0
    
    def test_get_audio_info_nonexistent(self, temp_dir):
        """Test getting info for non-existent file."""
        nonexistent_file = temp_dir / "nonexistent.wav"
        
        with pytest.raises(FileNotFoundError):
            get_audio_info(nonexistent_file)
    
    def test_validate_audio_file_valid(self, sample_audio_file):
        """Test validating valid audio file."""
        file_path, _, _ = sample_audio_file
        
        # Should not raise any exception
        validate_audio_file(file_path)
    
    def test_validate_audio_file_invalid_format(self, invalid_audio_file):
        """Test validating invalid audio file."""
        with pytest.raises(ValueError, match="Unsupported audio format"):
            validate_audio_file(invalid_audio_file)
    
    def test_validate_audio_file_nonexistent(self, temp_dir):
        """Test validating non-existent file."""
        nonexistent_file = temp_dir / "nonexistent.wav"
        
        with pytest.raises(FileNotFoundError):
            validate_audio_file(nonexistent_file)
    
    def test_convert_audio_format(self, temp_dir, sample_audio_file):
        """Test converting audio format."""
        input_file, _, _ = sample_audio_file
        output_file = temp_dir / "output.mp3"
        
        # Mock ffmpeg conversion
        with patch('src.utils.audio_utils.subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            
            convert_audio_format(input_file, output_file, "mp3")
            
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert "ffmpeg" in args[0]
            assert str(input_file) in args
            assert str(output_file) in args
    
    def test_convert_audio_format_failure(self, temp_dir, sample_audio_file):
        """Test audio format conversion failure."""
        input_file, _, _ = sample_audio_file
        output_file = temp_dir / "output.mp3"
        
        # Mock ffmpeg failure
        with patch('src.utils.audio_utils.subprocess.run') as mock_run:
            mock_run.return_value.returncode = 1
            
            with pytest.raises(RuntimeError, match="Audio conversion failed"):
                convert_audio_format(input_file, output_file, "mp3")
    
    def test_convert_audio_format_unsupported(self, temp_dir, sample_audio_file):
        """Test converting to unsupported format."""
        input_file, _, _ = sample_audio_file
        output_file = temp_dir / "output.xyz"
        
        with pytest.raises(ValueError, match="Unsupported output format"):
            convert_audio_format(input_file, output_file, "xyz")
    
    def test_audio_normalization(self, temp_dir):
        """Test audio normalization."""
        # Create audio with high amplitude
        audio = np.array([2.0, -2.0, 1.5, -1.5], dtype=np.float32)
        sample_rate = 16000
        
        # Save and load to test normalization
        input_file = temp_dir / "input.wav"
        save_audio(audio, sample_rate, input_file)
        
        loaded_audio, loaded_sr = load_audio(input_file)
        
        # Audio should be normalized to [-1, 1] range
        assert np.max(np.abs(loaded_audio)) <= 1.0
        assert loaded_sr == sample_rate
    
    def test_audio_edge_cases(self, temp_dir):
        """Test audio utilities with edge cases."""
        # Test with very short audio
        short_audio = np.array([0.1, -0.1], dtype=np.float32)
        sample_rate = 16000
        
        short_file = temp_dir / "short.wav"
        save_audio(short_audio, sample_rate, short_file)
        
        loaded_audio, loaded_sr = load_audio(short_file)
        assert np.allclose(loaded_audio, short_audio, atol=1e-6)
        assert loaded_sr == sample_rate
        
        # Test with silence
        silent_audio = np.zeros(1000, dtype=np.float32)
        
        silent_file = temp_dir / "silent.wav"
        save_audio(silent_audio, sample_rate, silent_file)
        
        loaded_audio, loaded_sr = load_audio(silent_file)
        assert np.allclose(loaded_audio, silent_audio, atol=1e-6)
        assert loaded_sr == sample_rate
    
    def test_audio_file_size_limits(self, temp_dir, mock_settings):
        """Test audio file size validation."""
        # Create a large audio file
        large_audio = np.random.randn(16000 * 60).astype(np.float32)  # 1 minute
        sample_rate = 16000
        
        large_file = temp_dir / "large.wav"
        save_audio(large_audio, sample_rate, large_file)
        
        # Test with different size limits
        with patch('src.utils.audio_utils.settings') as mock_settings:
            mock_settings.max_file_size_bytes = 1024 * 1024  # 1MB
            
            with pytest.raises(ValueError, match="File too large"):
                validate_audio_file(large_file)
    
    @pytest.mark.parametrize("sample_rate", [8000, 16000, 22050, 44100, 48000])
    def test_resample_different_rates(self, sample_audio_data, sample_rate):
        """Test resampling to different sample rates."""
        audio, original_sr = sample_audio_data
        
        resampled_audio = resample_audio(audio, original_sr, sample_rate)
        
        assert isinstance(resampled_audio, np.ndarray)
        assert resampled_audio.dtype == np.float32
        
        # Check that the duration is approximately the same
        original_duration = len(audio) / original_sr
        resampled_duration = len(resampled_audio) / sample_rate
        assert abs(original_duration - resampled_duration) < 0.1
    
    def test_audio_utils_error_handling(self, temp_dir):
        """Test error handling in audio utilities."""
        # Test with corrupted file
        corrupted_file = temp_dir / "corrupted.wav"
        corrupted_file.write_bytes(b"corrupted audio data")
        
        with pytest.raises((ValueError, RuntimeError)):
            load_audio(corrupted_file)
        
        # Test with invalid sample rate
        audio = np.array([0.1, -0.1], dtype=np.float32)
        
        with pytest.raises(ValueError):
            save_audio(audio, -1, temp_dir / "invalid.wav")
        
        with pytest.raises(ValueError):
            save_audio(audio, 0, temp_dir / "invalid.wav")
