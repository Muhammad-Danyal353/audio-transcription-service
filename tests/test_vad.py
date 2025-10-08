"""
Unit tests for the Voice Activity Detection (VAD) service.

This module tests the VAD functionality including WebRTC VAD, Silero VAD,
and the unified VAD interface.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.services.vad import (
    WebRTCVAD, SileroVAD, VoiceActivityDetector, VoiceSegment,
    get_vad_instance
)


class TestWebRTCVAD:
    """Test cases for WebRTC VAD."""
    
    @pytest.fixture
    def mock_webrtc_vad(self):
        """Mock WebRTC VAD for testing."""
        vad = Mock()
        vad.is_speech.return_value = True
        return vad
    
    @pytest.fixture
    def mock_webrtc_module(self, mock_webrtc_vad):
        """Mock WebRTC VAD module for testing."""
        with patch('src.services.vad.webrtcvad') as mock_webrtc:
            mock_webrtc.Vad.return_value = mock_webrtc_vad
            yield mock_webrtc
    
    def test_webrtc_vad_init(self, mock_webrtc_module):
        """Test WebRTC VAD initialization."""
        vad = WebRTCVAD(aggressiveness=3, sample_rate=16000)
        
        assert vad.sample_rate == 16000
        assert vad.frame_duration_ms == 30
        assert vad.frame_size == 480  # 16000 * 30 / 1000
        assert vad.vad is not None
    
    def test_webrtc_vad_init_failure(self):
        """Test WebRTC VAD initialization failure."""
        with patch('src.services.vad.webrtcvad', side_effect=ImportError):
            with pytest.raises(ImportError):
                WebRTCVAD(aggressiveness=3, sample_rate=16000)
    
    def test_webrtc_vad_detect_voice_activity(self, mock_webrtc_module, sample_audio_data):
        """Test WebRTC VAD voice activity detection."""
        audio, sample_rate = sample_audio_data
        vad = WebRTCVAD(aggressiveness=3, sample_rate=16000)
        
        # Mock the VAD to return speech for all frames
        vad.vad.is_speech.return_value = True
        
        segments = vad.detect_voice_activity(audio)
        
        assert isinstance(segments, list)
        assert len(segments) > 0
        assert all(isinstance(seg, VoiceSegment) for seg in segments)
        assert all(seg.is_speech for seg in segments)
    
    def test_webrtc_vad_detect_voice_activity_no_speech(self, mock_webrtc_module, sample_audio_data):
        """Test WebRTC VAD with no speech detected."""
        audio, sample_rate = sample_audio_data
        vad = WebRTCVAD(aggressiveness=3, sample_rate=16000)
        
        # Mock the VAD to return no speech for all frames
        vad.vad.is_speech.return_value = False
        
        segments = vad.detect_voice_activity(audio)
        
        assert isinstance(segments, list)
        assert len(segments) > 0
        assert all(isinstance(seg, VoiceSegment) for seg in segments)
        assert all(not seg.is_speech for seg in segments)
    
    def test_webrtc_vad_detect_voice_activity_mixed(self, mock_webrtc_module, sample_audio_data):
        """Test WebRTC VAD with mixed speech and silence."""
        audio, sample_rate = sample_audio_data
        vad = WebRTCVAD(aggressiveness=3, sample_rate=16000)
        
        # Mock the VAD to return alternating speech/silence
        call_count = 0
        def mock_is_speech(frame_bytes, sample_rate):
            nonlocal call_count
            call_count += 1
            return call_count % 2 == 1  # Alternate between True and False
        
        vad.vad.is_speech.side_effect = mock_is_speech
        
        segments = vad.detect_voice_activity(audio)
        
        assert isinstance(segments, list)
        assert len(segments) > 0
        assert any(seg.is_speech for seg in segments)
        assert any(not seg.is_speech for seg in segments)
    
    def test_webrtc_vad_merge_segments(self, mock_webrtc_module):
        """Test WebRTC VAD segment merging."""
        vad = WebRTCVAD(aggressiveness=3, sample_rate=16000)
        
        segments = [
            VoiceSegment(start_time=0.0, end_time=1.0, confidence=1.0, is_speech=True),
            VoiceSegment(start_time=1.0, end_time=1.2, confidence=1.0, is_speech=True),  # Small gap
            VoiceSegment(start_time=1.2, end_time=2.0, confidence=1.0, is_speech=True),
            VoiceSegment(start_time=3.0, end_time=4.0, confidence=1.0, is_speech=True),  # Large gap
        ]
        
        merged = vad._merge_segments(segments, gap_threshold=0.5)
        
        assert len(merged) == 2  # Should merge first 3 segments, keep last one separate
        assert merged[0].start_time == 0.0
        assert merged[0].end_time == 2.0
        assert merged[1].start_time == 3.0
        assert merged[1].end_time == 4.0
    
    def test_webrtc_vad_error_handling(self, mock_webrtc_module, sample_audio_data):
        """Test WebRTC VAD error handling."""
        audio, sample_rate = sample_audio_data
        vad = WebRTCVAD(aggressiveness=3, sample_rate=16000)
        
        # Mock the VAD to raise an exception
        vad.vad.is_speech.side_effect = Exception("VAD error")
        
        segments = vad.detect_voice_activity(audio)
        
        # Should handle errors gracefully and return empty list or partial results
        assert isinstance(segments, list)


class TestSileroVAD:
    """Test cases for Silero VAD."""
    
    @pytest.fixture
    def mock_silero_model(self):
        """Mock Silero VAD model for testing."""
        model = Mock()
        model.eval.return_value = None
        model.to.return_value = model
        return model
    
    @pytest.fixture
    def mock_silero_module(self, mock_silero_model):
        """Mock Silero VAD module for testing."""
        with patch('src.services.vad.silero_vad') as mock_silero:
            mock_silero.load_silero_vad.return_value = mock_silero_model
            mock_silero.get_speech_timestamps.return_value = [
                {"start": 0, "end": 2500, "confidence": 0.95},
                {"start": 3000, "end": 5000, "confidence": 0.92}
            ]
            yield mock_silero
    
    def test_silero_vad_init(self, mock_silero_module):
        """Test Silero VAD initialization."""
        vad = SileroVAD(device="cpu")
        
        assert vad.device == "cpu"
        assert vad.sample_rate == 16000
        assert vad.model is not None
        assert vad.get_speech_timestamps is not None
    
    def test_silero_vad_init_failure(self):
        """Test Silero VAD initialization failure."""
        with patch('src.services.vad.silero_vad', side_effect=ImportError):
            with pytest.raises(ImportError):
                SileroVAD(device="cpu")
    
    def test_silero_vad_detect_voice_activity(self, mock_silero_module, sample_audio_data):
        """Test Silero VAD voice activity detection."""
        audio, sample_rate = sample_audio_data
        vad = SileroVAD(device="cpu")
        
        segments = vad.detect_voice_activity(audio, threshold=0.5)
        
        assert isinstance(segments, list)
        assert len(segments) == 2
        assert all(isinstance(seg, VoiceSegment) for seg in segments)
        assert all(seg.is_speech for seg in segments)
        assert segments[0].start_time == 0.0
        assert segments[0].end_time == 2.5
        assert segments[1].start_time == 3.0
        assert segments[1].end_time == 5.0
    
    def test_silero_vad_detect_voice_activity_no_speech(self, mock_silero_module, sample_audio_data):
        """Test Silero VAD with no speech detected."""
        audio, sample_rate = sample_audio_data
        vad = SileroVAD(device="cpu")
        
        # Mock to return no speech timestamps
        vad.get_speech_timestamps.return_value = []
        
        segments = vad.detect_voice_activity(audio)
        
        assert isinstance(segments, list)
        assert len(segments) == 0
    
    def test_silero_vad_detect_voice_activity_parameters(self, mock_silero_module, sample_audio_data):
        """Test Silero VAD with different parameters."""
        audio, sample_rate = sample_audio_data
        vad = SileroVAD(device="cpu")
        
        segments = vad.detect_voice_activity(
            audio,
            threshold=0.7,
            min_speech_duration_ms=500,
            min_silence_duration_ms=200
        )
        
        # Verify that get_speech_timestamps was called with correct parameters
        vad.get_speech_timestamps.assert_called_once()
        call_args = vad.get_speech_timestamps.call_args
        assert call_args[1]["threshold"] == 0.7
        assert call_args[1]["min_speech_duration_ms"] == 500
        assert call_args[1]["min_silence_duration_ms"] == 200
    
    def test_silero_vad_model_not_loaded(self, sample_audio_data):
        """Test Silero VAD with model not loaded."""
        audio, sample_rate = sample_audio_data
        
        with patch('src.services.vad.silero_vad', side_effect=ImportError):
            vad = SileroVAD(device="cpu")
            vad.model = None  # Simulate model not loaded
        
        with pytest.raises(RuntimeError, match="Silero VAD model not loaded"):
            vad.detect_voice_activity(audio)


class TestVoiceActivityDetector:
    """Test cases for the main VAD service."""
    
    def test_vad_detector_init_webrtc(self, mock_webrtc_module):
        """Test VAD detector initialization with WebRTC."""
        with patch('src.services.vad.WebRTCVAD') as mock_webrtc_class:
            mock_webrtc_instance = Mock()
            mock_webrtc_class.return_value = mock_webrtc_instance
            
            detector = VoiceActivityDetector(method="webrtc")
            
            assert detector.method == "webrtc"
            assert detector.webrtc_vad is not None
            assert detector.silero_vad is None
    
    def test_vad_detector_init_silero(self, mock_silero_module):
        """Test VAD detector initialization with Silero."""
        with patch('src.services.vad.SileroVAD') as mock_silero_class:
            mock_silero_instance = Mock()
            mock_silero_class.return_value = mock_silero_instance
            
            detector = VoiceActivityDetector(method="silero")
            
            assert detector.method == "silero"
            assert detector.webrtc_vad is None
            assert detector.silero_vad is not None
    
    def test_vad_detector_init_auto(self, mock_webrtc_module, mock_silero_module):
        """Test VAD detector initialization with auto method."""
        with patch('src.services.vad.WebRTCVAD') as mock_webrtc_class, \
             patch('src.services.vad.SileroVAD') as mock_silero_class:
            
            mock_webrtc_instance = Mock()
            mock_silero_instance = Mock()
            mock_webrtc_class.return_value = mock_webrtc_instance
            mock_silero_class.return_value = mock_silero_instance
            
            detector = VoiceActivityDetector(method="auto")
            
            assert detector.method == "auto"
            assert detector.webrtc_vad is not None
            assert detector.silero_vad is not None
    
    def test_vad_detector_init_failure(self):
        """Test VAD detector initialization failure."""
        with patch('src.services.vad.WebRTCVAD', side_effect=Exception("WebRTC failed")), \
             patch('src.services.vad.SileroVAD', side_effect=Exception("Silero failed")):
            
            with pytest.raises(RuntimeError, match="No VAD methods available"):
                VoiceActivityDetector(method="auto")
    
    def test_detect_voice_activity_webrtc(self, mock_webrtc_module, sample_audio_data):
        """Test voice activity detection using WebRTC."""
        audio, sample_rate = sample_audio_data
        
        with patch('src.services.vad.WebRTCVAD') as mock_webrtc_class:
            mock_webrtc_instance = Mock()
            mock_webrtc_class.return_value = mock_webrtc_instance
            
            # Mock detection result
            mock_segments = [
                VoiceSegment(start_time=0.0, end_time=2.0, confidence=0.95, is_speech=True),
                VoiceSegment(start_time=3.0, end_time=5.0, confidence=0.92, is_speech=True)
            ]
            mock_webrtc_instance.detect_voice_activity.return_value = mock_segments
            
            detector = VoiceActivityDetector(method="webrtc")
            result = detector.detect_voice_activity(audio, method="webrtc")
            
            assert result == mock_segments
            mock_webrtc_instance.detect_voice_activity.assert_called_once()
    
    def test_detect_voice_activity_silero(self, mock_silero_module, sample_audio_data):
        """Test voice activity detection using Silero."""
        audio, sample_rate = sample_audio_data
        
        with patch('src.services.vad.SileroVAD') as mock_silero_class:
            mock_silero_instance = Mock()
            mock_silero_class.return_value = mock_silero_instance
            
            # Mock detection result
            mock_segments = [
                VoiceSegment(start_time=0.0, end_time=2.5, confidence=0.95, is_speech=True),
                VoiceSegment(start_time=3.0, end_time=5.0, confidence=0.92, is_speech=True)
            ]
            mock_silero_instance.detect_voice_activity.return_value = mock_segments
            
            detector = VoiceActivityDetector(method="silero")
            result = detector.detect_voice_activity(audio, method="silero", threshold=0.5)
            
            assert result == mock_segments
            mock_silero_instance.detect_voice_activity.assert_called_once_with(audio, threshold=0.5)
    
    def test_detect_voice_activity_auto_silero_first(self, mock_webrtc_module, mock_silero_module, sample_audio_data):
        """Test auto method with Silero working."""
        audio, sample_rate = sample_audio_data
        
        with patch('src.services.vad.SileroVAD') as mock_silero_class:
            mock_silero_instance = Mock()
            mock_silero_class.return_value = mock_silero_instance
            
            # Mock successful Silero detection
            mock_segments = [
                VoiceSegment(start_time=0.0, end_time=2.5, confidence=0.95, is_speech=True)
            ]
            mock_silero_instance.detect_voice_activity.return_value = mock_segments
            
            detector = VoiceActivityDetector(method="auto")
            result = detector.detect_voice_activity(audio, threshold=0.5)
            
            assert result == mock_segments
            mock_silero_instance.detect_voice_activity.assert_called_once()
            # WebRTC should not be called
            detector.webrtc_vad.detect_voice_activity.assert_not_called()
    
    def test_detect_voice_activity_auto_silero_fallback(self, mock_webrtc_module, sample_audio_data):
        """Test auto method with Silero failure and WebRTC fallback."""
        audio, sample_rate = sample_audio_data
        
        with patch('src.services.vad.SileroVAD', side_effect=Exception("Silero failed")):
            with patch('src.services.vad.WebRTCVAD') as mock_webrtc_class:
                mock_webrtc_instance = Mock()
                mock_webrtc_class.return_value = mock_webrtc_instance
                
                # Mock WebRTC detection
                mock_segments = [
                    VoiceSegment(start_time=0.0, end_time=2.0, confidence=0.95, is_speech=True)
                ]
                mock_webrtc_instance.detect_voice_activity.return_value = mock_segments
                
                detector = VoiceActivityDetector(method="auto")
                result = detector.detect_voice_activity(audio)
                
                assert result == mock_segments
                mock_webrtc_instance.detect_voice_activity.assert_called_once()
    
    def test_detect_voice_activity_disabled(self, sample_audio_data, mock_settings):
        """Test voice activity detection when disabled."""
        audio, sample_rate = sample_audio_data
        
        with patch('src.services.vad.settings') as mock_settings:
            mock_settings.enable_vad = False
            
            detector = VoiceActivityDetector()
            result = detector.detect_voice_activity(audio)
            
            assert len(result) == 1
            assert result[0].start_time == 0.0
            assert result[0].end_time == len(audio) / sample_rate
            assert result[0].is_speech == True
            assert result[0].confidence == 1.0
    
    def test_detect_voice_activity_error_fallback(self, sample_audio_data):
        """Test voice activity detection error fallback."""
        audio, sample_rate = sample_audio_data
        
        detector = VoiceActivityDetector(method="webrtc")
        
        # Mock both VAD methods to fail
        with patch.object(detector, 'webrtc_vad', None), \
             patch.object(detector, 'silero_vad', None):
            
            with pytest.raises(RuntimeError, match="VAD method 'webrtc' not available"):
                detector.detect_voice_activity(audio, method="webrtc")
    
    def test_filter_silence(self, sample_audio_data, mock_voice_segments):
        """Test silence filtering."""
        audio, sample_rate = sample_audio_data
        detector = VoiceActivityDetector()
        
        filtered_audio = detector.filter_silence(audio, mock_voice_segments, padding=0.1)
        
        assert isinstance(filtered_audio, np.ndarray)
        assert filtered_audio.dtype == np.float32
        # Should be shorter than original due to silence removal
        assert len(filtered_audio) <= len(audio)
    
    def test_filter_silence_no_segments(self, sample_audio_data):
        """Test silence filtering with no segments."""
        audio, sample_rate = sample_audio_data
        detector = VoiceActivityDetector()
        
        filtered_audio = detector.filter_silence(audio, [])
        
        assert len(filtered_audio) == 0
    
    def test_get_voice_activity_summary(self, mock_voice_segments):
        """Test voice activity summary generation."""
        detector = VoiceActivityDetector()
        summary = detector.get_voice_activity_summary(mock_voice_segments)
        
        assert isinstance(summary, dict)
        assert "total_segments" in summary
        assert "speech_segments" in summary
        assert "total_speech_time" in summary
        assert "total_time" in summary
        assert "speech_ratio" in summary
        assert "avg_segment_duration" in summary
        
        assert summary["total_segments"] == 3
        assert summary["speech_segments"] == 2
        assert summary["total_speech_time"] == 4.5  # 2.5 + 2.0
        assert summary["total_time"] == 5.0
        assert summary["speech_ratio"] == 0.9  # 4.5 / 5.0
        assert summary["avg_segment_duration"] == 2.25  # 4.5 / 2
    
    def test_get_voice_activity_summary_empty(self):
        """Test voice activity summary with empty segments."""
        detector = VoiceActivityDetector()
        summary = detector.get_voice_activity_summary([])
        
        assert summary == {"total_segments": 0, "total_speech_time": 0.0}


class TestVoiceSegment:
    """Test cases for VoiceSegment dataclass."""
    
    def test_voice_segment_creation(self):
        """Test creating a voice segment."""
        segment = VoiceSegment(
            start_time=0.0,
            end_time=2.5,
            confidence=0.95,
            is_speech=True
        )
        
        assert segment.start_time == 0.0
        assert segment.end_time == 2.5
        assert segment.confidence == 0.95
        assert segment.is_speech == True
        assert segment.duration == 2.5
    
    def test_voice_segment_duration_calculation(self):
        """Test voice segment duration calculation."""
        segment = VoiceSegment(
            start_time=1.0,
            end_time=3.5,
            confidence=0.92,
            is_speech=True
        )
        
        assert segment.duration == 2.5


class TestGlobalVADInstance:
    """Test cases for global VAD instance."""
    
    def test_get_vad_instance(self):
        """Test getting global VAD instance."""
        # Clear any existing instance
        import src.services.vad
        src.services.vad._vad_instance = None
        
        instance1 = get_vad_instance()
        instance2 = get_vad_instance()
        
        assert instance1 is instance2  # Should be the same instance
        assert isinstance(instance1, VoiceActivityDetector)
