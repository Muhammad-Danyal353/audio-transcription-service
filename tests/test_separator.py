"""
Unit tests for the vocal separation service.

This module tests the vocal separation functionality including Demucs integration,
spectral subtraction fallback, and quality metrics calculation.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.services.separator import (
    DemucsSeparator, SpectralSubtractionSeparator, VocalSeparator,
    SeparationResult, get_separator_instance
)


class TestDemucsSeparator:
    """Test cases for Demucs separator."""
    
    @pytest.fixture
    def mock_demucs_model(self):
        """Mock Demucs model for testing."""
        model = Mock()
        model.eval.return_value = None
        model.to.return_value = model
        
        # Mock separated output
        separated = Mock()
        separated.__getitem__ = Mock(return_value=Mock())
        separated.__getitem__.return_value.cpu.return_value.numpy.return_value = np.array([0.1, 0.2, 0.3])
        
        model.return_value = [separated]
        return model
    
    @pytest.fixture
    def mock_demucs_api(self, mock_demucs_model):
        """Mock Demucs API for testing."""
        with patch('src.services.separator.demucs.api') as mock_api:
            mock_api.load_model.return_value = mock_demucs_model
            yield mock_api
    
    def test_demucs_separator_init(self, mock_demucs_api):
        """Test Demucs separator initialization."""
        separator = DemucsSeparator("htdemucs", "cpu")
        
        assert separator.model_name == "htdemucs"
        assert separator.device == "cpu"
        assert separator.sample_rate == 44100
        assert separator.model is not None
    
    def test_demucs_separator_init_failure(self):
        """Test Demucs separator initialization failure."""
        with patch('src.services.separator.demucs.api', side_effect=ImportError):
            with pytest.raises(ImportError):
                DemucsSeparator("htdemucs", "cpu")
    
    def test_demucs_separate_sources(self, mock_demucs_api, sample_audio_data):
        """Test Demucs source separation."""
        audio, sample_rate = sample_audio_data
        separator = DemucsSeparator("htdemucs", "cpu")
        
        # Mock the model's __call__ method
        mock_output = Mock()
        mock_output.__getitem__ = Mock(return_value=Mock())
        mock_output.__getitem__.return_value.cpu.return_value.numpy.return_value = np.array([0.1, 0.2, 0.3])
        
        separator.model.return_value = [mock_output]
        
        result = separator.separate_sources(audio, sample_rate, ['vocals', 'drums'])
        
        assert isinstance(result, dict)
        assert 'vocals' in result
        assert 'drums' in result
        assert isinstance(result['vocals'], np.ndarray)
        assert isinstance(result['drums'], np.ndarray)
    
    def test_demucs_separate_vocals(self, mock_demucs_api, sample_audio_data):
        """Test Demucs vocal separation."""
        audio, sample_rate = sample_audio_data
        separator = DemucsSeparator("htdemucs", "cpu")
        
        # Mock the separate_sources method
        with patch.object(separator, 'separate_sources') as mock_separate:
            mock_separate.return_value = {'vocals': np.array([0.1, 0.2, 0.3])}
            
            result = separator.separate_vocals(audio, sample_rate)
            
            assert isinstance(result, np.ndarray)
            mock_separate.assert_called_once_with(audio, sample_rate, ['vocals'])
    
    def test_demucs_resampling(self, mock_demucs_api, sample_audio_data):
        """Test Demucs with different sample rates."""
        audio, sample_rate = sample_audio_data
        separator = DemucsSeparator("htdemucs", "cpu")
        
        # Mock resample_audio
        with patch('src.services.separator.resample_audio') as mock_resample:
            mock_resample.return_value = audio
            mock_resample.side_effect = lambda x, sr1, sr2: x  # Return input unchanged
            
            # Mock the model
            mock_output = Mock()
            mock_output.__getitem__ = Mock(return_value=Mock())
            mock_output.__getitem__.return_value.cpu.return_value.numpy.return_value = audio
            separator.model.return_value = [mock_output]
            
            result = separator.separate_sources(audio, sample_rate, ['vocals'])
            
            # Should call resample_audio for input
            mock_resample.assert_called()


class TestSpectralSubtractionSeparator:
    """Test cases for spectral subtraction separator."""
    
    def test_spectral_separator_init(self):
        """Test spectral subtraction separator initialization."""
        separator = SpectralSubtractionSeparator(alpha=2.0, beta=0.01)
        
        assert separator.alpha == 2.0
        assert separator.beta == 0.01
    
    def test_spectral_separate_vocals(self, sample_audio_data):
        """Test spectral subtraction vocal separation."""
        audio, sample_rate = sample_audio_data
        separator = SpectralSubtractionSeparator()
        
        # Mock librosa functions
        with patch('src.services.separator.librosa.stft') as mock_stft, \
             patch('src.services.separator.librosa.istft') as mock_istft:
            
            # Mock STFT
            mock_stft.return_value = np.array([[1+1j, 2+2j], [3+3j, 4+4j]])
            
            # Mock ISTFT
            mock_istft.return_value = audio * 0.8  # Simulate noise reduction
            
            result = separator.separate_vocals(audio, sample_rate)
            
            assert isinstance(result, np.ndarray)
            assert result.dtype == np.float32
            assert len(result) == len(audio)
    
    def test_spectral_separate_vocals_stereo(self, sample_audio_data):
        """Test spectral subtraction with stereo audio."""
        audio, sample_rate = sample_audio_data
        stereo_audio = np.stack([audio, audio * 0.5], axis=0)
        separator = SpectralSubtractionSeparator()
        
        # Mock librosa functions
        with patch('src.services.separator.librosa.stft') as mock_stft, \
             patch('src.services.separator.librosa.istft') as mock_istft:
            
            mock_stft.return_value = np.array([[1+1j, 2+2j], [3+3j, 4+4j]])
            mock_istft.return_value = audio * 0.8
            
            result = separator.separate_vocals(stereo_audio, sample_rate)
            
            assert isinstance(result, np.ndarray)
            assert result.dtype == np.float32
            assert len(result) == len(audio)  # Should be mono output
    
    def test_spectral_separate_vocals_no_librosa(self, sample_audio_data):
        """Test spectral subtraction without librosa."""
        audio, sample_rate = sample_audio_data
        separator = SpectralSubtractionSeparator()
        
        # Mock ImportError for librosa
        with patch('src.services.separator.librosa', side_effect=ImportError):
            result = separator.separate_vocals(audio, sample_rate)
            
            # Should return original audio unchanged
            assert np.array_equal(result, audio)
    
    def test_spectral_separate_vocals_error(self, sample_audio_data):
        """Test spectral subtraction error handling."""
        audio, sample_rate = sample_audio_data
        separator = SpectralSubtractionSeparator()
        
        # Mock librosa to raise an exception
        with patch('src.services.separator.librosa.stft', side_effect=Exception("Test error")):
            result = separator.separate_vocals(audio, sample_rate)
            
            # Should return original audio unchanged
            assert np.array_equal(result, audio)


class TestVocalSeparator:
    """Test cases for the main vocal separator service."""
    
    def test_vocal_separator_init_demucs(self, mock_demucs_api):
        """Test vocal separator initialization with Demucs."""
        with patch('src.services.separator.DemucsSeparator') as mock_demucs_class:
            mock_demucs_instance = Mock()
            mock_demucs_class.return_value = mock_demucs_instance
            
            separator = VocalSeparator(method="demucs")
            
            assert separator.method == "demucs"
            assert separator.demucs_separator is not None
            assert separator.spectral_separator is not None
    
    def test_vocal_separator_init_spectral(self):
        """Test vocal separator initialization with spectral only."""
        separator = VocalSeparator(method="spectral")
        
        assert separator.method == "spectral"
        assert separator.demucs_separator is None
        assert separator.spectral_separator is not None
    
    def test_vocal_separator_init_auto(self, mock_demucs_api):
        """Test vocal separator initialization with auto method."""
        with patch('src.services.separator.DemucsSeparator') as mock_demucs_class:
            mock_demucs_instance = Mock()
            mock_demucs_class.return_value = mock_demucs_instance
            
            separator = VocalSeparator(method="auto")
            
            assert separator.method == "auto"
            assert separator.demucs_separator is not None
            assert separator.spectral_separator is not None
    
    def test_vocal_separator_init_failure(self):
        """Test vocal separator initialization failure."""
        with patch('src.services.separator.DemucsSeparator', side_effect=Exception("Test error")):
            separator = VocalSeparator(method="auto")
            
            # Should still initialize with spectral fallback
            assert separator.demucs_separator is None
            assert separator.spectral_separator is not None
    
    def test_separate_vocals_demucs(self, mock_demucs_api, sample_audio_data):
        """Test vocal separation using Demucs."""
        audio, sample_rate = sample_audio_data
        
        with patch('src.services.separator.DemucsSeparator') as mock_demucs_class:
            mock_demucs_instance = Mock()
            mock_demucs_class.return_value = mock_demucs_instance
            
            # Mock separation result
            mock_result = SeparationResult(
                vocals=audio * 0.8,
                accompaniment=audio * 0.2,
                sample_rate=sample_rate,
                duration=5.0,
                metadata={"method": "demucs"}
            )
            mock_demucs_instance.separate_vocals.return_value = mock_result
            
            separator = VocalSeparator(method="demucs")
            result = separator.separate_vocals(audio, sample_rate)
            
            assert isinstance(result, SeparationResult)
            assert result.has_vocals
            assert result.metadata["method"] == "demucs"
    
    def test_separate_vocals_spectral(self, sample_audio_data):
        """Test vocal separation using spectral subtraction."""
        audio, sample_rate = sample_audio_data
        
        separator = VocalSeparator(method="spectral")
        
        # Mock spectral separation
        with patch.object(separator.spectral_separator, 'separate_vocals') as mock_spectral:
            mock_spectral.return_value = audio * 0.8
            
            result = separator.separate_vocals(audio, sample_rate)
            
            assert isinstance(result, SeparationResult)
            assert result.has_vocals
            assert result.metadata["method"] == "spectral"
    
    def test_separate_vocals_auto_demucs_first(self, mock_demucs_api, sample_audio_data):
        """Test auto method with Demucs working."""
        audio, sample_rate = sample_audio_data
        
        with patch('src.services.separator.DemucsSeparator') as mock_demucs_class:
            mock_demucs_instance = Mock()
            mock_demucs_class.return_value = mock_demucs_instance
            
            # Mock successful Demucs separation
            mock_result = SeparationResult(
                vocals=audio * 0.8,
                accompaniment=audio * 0.2,
                sample_rate=sample_rate,
                duration=5.0,
                metadata={"method": "demucs"}
            )
            mock_demucs_instance.separate_vocals.return_value = mock_result
            
            separator = VocalSeparator(method="auto")
            result = separator.separate_vocals(audio, sample_rate)
            
            assert isinstance(result, SeparationResult)
            assert result.metadata["method"] == "demucs"
            # Spectral should not be called
            separator.spectral_separator.separate_vocals.assert_not_called()
    
    def test_separate_vocals_auto_demucs_fallback(self, sample_audio_data):
        """Test auto method with Demucs failure and spectral fallback."""
        audio, sample_rate = sample_audio_data
        
        with patch('src.services.separator.DemucsSeparator', side_effect=Exception("Demucs failed")):
            separator = VocalSeparator(method="auto")
            
            # Mock spectral separation
            with patch.object(separator.spectral_separator, 'separate_vocals') as mock_spectral:
                mock_spectral.return_value = audio * 0.8
                
                result = separator.separate_vocals(audio, sample_rate)
                
                assert isinstance(result, SeparationResult)
                assert result.metadata["method"] == "spectral"
                mock_spectral.assert_called_once()
    
    def test_separate_vocals_disabled(self, sample_audio_data, mock_settings):
        """Test vocal separation when disabled."""
        audio, sample_rate = sample_audio_data
        
        with patch('src.services.separator.settings') as mock_settings:
            mock_settings.enable_separation_by_default = False
            
            separator = VocalSeparator()
            result = separator.separate_vocals(audio, sample_rate)
            
            assert isinstance(result, SeparationResult)
            assert not result.metadata["enabled"]
            assert result.vocals is audio  # Should return original audio
    
    def test_separate_vocals_error_fallback(self, sample_audio_data):
        """Test vocal separation error fallback."""
        audio, sample_rate = sample_audio_data
        
        separator = VocalSeparator(method="demucs")
        
        # Mock both separators to fail
        with patch.object(separator, 'demucs_separator', None), \
             patch.object(separator.spectral_separator, 'separate_vocals', side_effect=Exception("Spectral failed")):
            
            result = separator.separate_vocals(audio, sample_rate)
            
            assert isinstance(result, SeparationResult)
            assert result.metadata["fallback"]
            assert result.vocals is audio  # Should return original audio
    
    def test_separate_sources(self, mock_demucs_api, sample_audio_data):
        """Test separating multiple sources."""
        audio, sample_rate = sample_audio_data
        
        with patch('src.services.separator.DemucsSeparator') as mock_demucs_class:
            mock_demucs_instance = Mock()
            mock_demucs_class.return_value = mock_demucs_instance
            
            # Mock separation result
            mock_result = {
                'vocals': audio * 0.8,
                'drums': audio * 0.1,
                'bass': audio * 0.05,
                'other': audio * 0.05
            }
            mock_demucs_instance.separate_sources.return_value = mock_result
            
            separator = VocalSeparator(method="demucs")
            result = separator.separate_sources(audio, sample_rate, ['vocals', 'drums'])
            
            assert isinstance(result, dict)
            assert 'vocals' in result
            assert 'drums' in result
            mock_demucs_instance.separate_sources.assert_called_once()
    
    def test_get_separation_quality_metrics(self, sample_audio_data):
        """Test separation quality metrics calculation."""
        audio, sample_rate = sample_audio_data
        separated = audio * 0.8
        
        separator = VocalSeparator()
        metrics = separator.get_separation_quality_metrics(audio, separated)
        
        assert isinstance(metrics, dict)
        assert "snr_db" in metrics
        assert "spectral_distance" in metrics
        assert "signal_power" in metrics
        assert "noise_power" in metrics
        
        assert isinstance(metrics["snr_db"], float)
        assert isinstance(metrics["signal_power"], float)
        assert isinstance(metrics["noise_power"], float)
    
    def test_get_separation_quality_metrics_error(self, sample_audio_data):
        """Test separation quality metrics error handling."""
        audio, sample_rate = sample_audio_data
        separated = audio * 0.8
        
        separator = VocalSeparator()
        
        # Mock librosa to raise an exception
        with patch('src.services.separator.librosa.stft', side_effect=Exception("Test error")):
            metrics = separator.get_separation_quality_metrics(audio, separated)
            
            assert "error" in metrics
            assert "Test error" in metrics["error"]


class TestSeparationResult:
    """Test cases for SeparationResult dataclass."""
    
    def test_separation_result_creation(self, sample_audio_data):
        """Test creating a separation result."""
        audio, sample_rate = sample_audio_data
        
        result = SeparationResult(
            vocals=audio * 0.8,
            accompaniment=audio * 0.2,
            sample_rate=sample_rate,
            duration=5.0,
            metadata={"method": "test"}
        )
        
        assert result.has_vocals
        assert result.sample_rate == sample_rate
        assert result.duration == 5.0
        assert result.metadata["method"] == "test"
    
    def test_separation_result_no_vocals(self, sample_audio_data):
        """Test separation result with no vocals."""
        audio, sample_rate = sample_audio_data
        
        result = SeparationResult(
            vocals=np.zeros_like(audio),
            accompaniment=audio,
            sample_rate=sample_rate,
            duration=5.0,
            metadata={"method": "test"}
        )
        
        assert not result.has_vocals


class TestGlobalSeparatorInstance:
    """Test cases for global separator instance."""
    
    def test_get_separator_instance(self):
        """Test getting global separator instance."""
        # Clear any existing instance
        import src.services.separator
        src.services.separator._separator_instance = None
        
        instance1 = get_separator_instance()
        instance2 = get_separator_instance()
        
        assert instance1 is instance2  # Should be the same instance
        assert isinstance(instance1, VocalSeparator)
