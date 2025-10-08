"""
Unit tests for the ASR transcription service.

This module tests the speech-to-text transcription functionality including
Whisper integration, chunking, language detection, and quality metrics.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.services.transcriber import (
    WhisperTranscriber, AudioTranscriber, TranscriptionResult,
    TranscriptSegment, WordSegment, get_transcriber_instance
)


class TestWhisperTranscriber:
    """Test cases for Whisper transcriber."""
    
    @pytest.fixture
    def mock_whisper_model(self):
        """Mock Whisper model for testing."""
        model = Mock()
        model.transcribe.return_value = {
            "text": "Hello, this is a test transcription.",
            "language": "en",
            "language_probability": 0.95,
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.5,
                    "text": "Hello, this is a test transcription.",
                    "tokens": [1, 2, 3, 4, 5],
                    "temperature": 0.0,
                    "avg_logprob": -0.1,
                    "compression_ratio": 1.2,
                    "no_speech_prob": 0.01,
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 0.98},
                        {"word": "this", "start": 0.6, "end": 0.8, "probability": 0.95},
                        {"word": "is", "start": 0.9, "end": 1.0, "probability": 0.99},
                        {"word": "a", "start": 1.1, "end": 1.2, "probability": 0.97},
                        {"word": "test", "start": 1.3, "end": 1.8, "probability": 0.94},
                        {"word": "transcription", "start": 1.9, "end": 2.5, "probability": 0.91}
                    ]
                }
            ]
        }
        return model
    
    @pytest.fixture
    def mock_whisper_module(self, mock_whisper_model):
        """Mock Whisper module for testing."""
        with patch('src.services.transcriber.whisper') as mock_whisper:
            mock_whisper.load_model.return_value = mock_whisper_model
            yield mock_whisper
    
    def test_whisper_transcriber_init(self, mock_whisper_module):
        """Test Whisper transcriber initialization."""
        transcriber = WhisperTranscriber("small", "cpu")
        
        assert transcriber.model_size == "small"
        assert transcriber.device == "cpu"
        assert transcriber.sample_rate == 16000
        assert transcriber.model is not None
    
    def test_whisper_transcriber_init_failure(self):
        """Test Whisper transcriber initialization failure."""
        with patch('src.services.transcriber.whisper', side_effect=ImportError):
            with pytest.raises(ImportError):
                WhisperTranscriber("small", "cpu")
    
    def test_whisper_transcribe(self, mock_whisper_module, sample_audio_data):
        """Test Whisper transcription."""
        audio, sample_rate = sample_audio_data
        transcriber = WhisperTranscriber("small", "cpu")
        
        result = transcriber.transcribe(audio, sample_rate, language="en")
        
        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello, this is a test transcription."
        assert result.language == "en"
        assert result.language_probability == 0.95
        assert len(result.segments) == 1
        assert result.segments[0].confidence > 0.8
    
    def test_whisper_transcribe_with_word_timestamps(self, mock_whisper_module, sample_audio_data):
        """Test Whisper transcription with word timestamps."""
        audio, sample_rate = sample_audio_data
        transcriber = WhisperTranscriber("small", "cpu")
        
        result = transcriber.transcribe(audio, sample_rate, word_timestamps=True)
        
        assert result.has_words
        assert len(result.segments[0].words) == 6
        assert result.segments[0].words[0].word == "Hello"
        assert result.segments[0].words[0].start == 0.0
        assert result.segments[0].words[0].end == 0.5
    
    def test_whisper_transcribe_without_word_timestamps(self, mock_whisper_module, sample_audio_data):
        """Test Whisper transcription without word timestamps."""
        audio, sample_rate = sample_audio_data
        transcriber = WhisperTranscriber("small", "cpu")
        
        # Mock model to return no words
        transcriber.model.transcribe.return_value = {
            "text": "Hello, this is a test transcription.",
            "language": "en",
            "language_probability": 0.95,
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.5,
                    "text": "Hello, this is a test transcription.",
                    "tokens": [1, 2, 3, 4, 5],
                    "temperature": 0.0,
                    "avg_logprob": -0.1,
                    "compression_ratio": 1.2,
                    "no_speech_prob": 0.01,
                    "words": []  # No words
                }
            ]
        }
        
        result = transcriber.transcribe(audio, sample_rate, word_timestamps=False)
        
        assert not result.has_words
        assert len(result.segments[0].words) == 0
    
    def test_whisper_transcribe_resampling(self, mock_whisper_module, sample_audio_data):
        """Test Whisper transcription with resampling."""
        audio, sample_rate = sample_audio_data
        transcriber = WhisperTranscriber("small", "cpu")
        
        # Mock resample_audio
        with patch('src.services.transcriber.resample_audio') as mock_resample:
            mock_resample.return_value = audio
            
            result = transcriber.transcribe(audio, sample_rate)
            
            # Should call resample_audio for input
            mock_resample.assert_called_once()
    
    def test_whisper_transcribe_chunked(self, mock_whisper_module, sample_audio_file_long):
        """Test Whisper transcription with chunking."""
        file_path, audio, sample_rate = sample_audio_file_long
        transcriber = WhisperTranscriber("small", "cpu")
        
        # Mock resample_audio to return input unchanged
        with patch('src.services.transcriber.resample_audio', side_effect=lambda x, sr1, sr2: x):
            result = transcriber.transcribe_chunked(audio, sample_rate, chunk_size=10, overlap=1.0)
            
            assert isinstance(result, TranscriptionResult)
            assert result.metadata["chunked"]
            assert result.metadata["chunk_size"] == 10
            assert result.metadata["overlap"] == 1.0
    
    def test_whisper_transcribe_chunked_short_audio(self, mock_whisper_module, sample_audio_data):
        """Test Whisper transcription with short audio (no chunking)."""
        audio, sample_rate = sample_audio_data
        transcriber = WhisperTranscriber("small", "cpu")
        
        # Mock the regular transcribe method
        with patch.object(transcriber, 'transcribe') as mock_transcribe:
            mock_result = TranscriptionResult(
                text="Short audio",
                language="en",
                language_probability=0.95,
                segments=[],
                duration=5.0,
                metadata={}
            )
            mock_transcribe.return_value = mock_result
            
            result = transcriber.transcribe_chunked(audio, sample_rate, chunk_size=30)
            
            # Should call regular transcribe for short audio
            mock_transcribe.assert_called_once()
            assert result == mock_result
    
    def test_whisper_transcribe_error(self, mock_whisper_module, sample_audio_data):
        """Test Whisper transcription error handling."""
        audio, sample_rate = sample_audio_data
        transcriber = WhisperTranscriber("small", "cpu")
        
        # Mock model to raise an exception
        transcriber.model.transcribe.side_effect = Exception("Transcription failed")
        
        with pytest.raises(Exception, match="Transcription failed"):
            transcriber.transcribe(audio, sample_rate)
    
    def test_parse_whisper_result(self, mock_whisper_module):
        """Test parsing Whisper result."""
        transcriber = WhisperTranscriber("small", "cpu")
        
        whisper_result = {
            "text": "Hello world",
            "language": "en",
            "language_probability": 0.95,
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "Hello world",
                    "tokens": [1, 2],
                    "temperature": 0.0,
                    "avg_logprob": -0.1,
                    "compression_ratio": 1.2,
                    "no_speech_prob": 0.01,
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 0.98},
                        {"word": "world", "start": 0.5, "end": 1.0, "probability": 0.95}
                    ]
                }
            ]
        }
        
        result = transcriber._parse_whisper_result(whisper_result, 1.0)
        
        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.language_probability == 0.95
        assert len(result.segments) == 1
        assert len(result.segments[0].words) == 2
        assert result.segments[0].words[0].word == "Hello"
        assert result.segments[0].words[1].word == "world"


class TestAudioTranscriber:
    """Test cases for the main audio transcriber service."""
    
    def test_audio_transcriber_init(self, mock_whisper_module):
        """Test audio transcriber initialization."""
        with patch('src.services.transcriber.WhisperTranscriber') as mock_whisper_class:
            mock_whisper_instance = Mock()
            mock_whisper_class.return_value = mock_whisper_instance
            
            transcriber = AudioTranscriber("small", "cpu")
            
            assert transcriber.model_size == "small"
            assert transcriber.device == "cpu"
            assert transcriber.whisper_transcriber is not None
    
    def test_audio_transcriber_init_failure(self):
        """Test audio transcriber initialization failure."""
        with patch('src.services.transcriber.WhisperTranscriber', side_effect=Exception("Init failed")):
            with pytest.raises(Exception, match="Init failed"):
                AudioTranscriber("small", "cpu")
    
    def test_transcribe(self, mock_whisper_module, sample_audio_data):
        """Test audio transcription."""
        audio, sample_rate = sample_audio_data
        
        with patch('src.services.transcriber.WhisperTranscriber') as mock_whisper_class:
            mock_whisper_instance = Mock()
            mock_whisper_class.return_value = mock_whisper_instance
            
            # Mock transcription result
            mock_result = TranscriptionResult(
                text="Test transcription",
                language="en",
                language_probability=0.95,
                segments=[],
                duration=5.0,
                metadata={}
            )
            mock_whisper_instance.transcribe.return_value = mock_result
            
            transcriber = AudioTranscriber("small", "cpu")
            result = transcriber.transcribe(audio, sample_rate, language="en")
            
            assert isinstance(result, TranscriptionResult)
            assert result.text == "Test transcription"
            mock_whisper_instance.transcribe.assert_called_once()
    
    def test_transcribe_chunked(self, mock_whisper_module, sample_audio_file_long):
        """Test audio transcription with chunking."""
        file_path, audio, sample_rate = sample_audio_file_long
        
        with patch('src.services.transcriber.WhisperTranscriber') as mock_whisper_class:
            mock_whisper_instance = Mock()
            mock_whisper_class.return_value = mock_whisper_instance
            
            # Mock chunked transcription result
            mock_result = TranscriptionResult(
                text="Long transcription",
                language="en",
                language_probability=0.95,
                segments=[],
                duration=60.0,
                metadata={"chunked": True}
            )
            mock_whisper_instance.transcribe_chunked.return_value = mock_result
            
            transcriber = AudioTranscriber("small", "cpu")
            result = transcriber.transcribe(audio, sample_rate, chunk_long_audio=True)
            
            assert isinstance(result, TranscriptionResult)
            assert result.metadata["chunked"]
            mock_whisper_instance.transcribe_chunked.assert_called_once()
    
    def test_transcribe_file(self, mock_whisper_module, sample_audio_file):
        """Test transcribing audio file."""
        file_path, _, _ = sample_audio_file
        
        with patch('src.services.transcriber.WhisperTranscriber') as mock_whisper_class:
            mock_whisper_instance = Mock()
            mock_whisper_class.return_value = mock_whisper_instance
            
            # Mock transcription result
            mock_result = TranscriptionResult(
                text="File transcription",
                language="en",
                language_probability=0.95,
                segments=[],
                duration=5.0,
                metadata={}
            )
            mock_whisper_instance.transcribe.return_value = mock_result
            
            transcriber = AudioTranscriber("small", "cpu")
            result = transcriber.transcribe_file(file_path, language="en")
            
            assert isinstance(result, TranscriptionResult)
            assert result.text == "File transcription"
    
    def test_transcribe_file_error(self, mock_whisper_module, temp_dir):
        """Test transcribing non-existent file."""
        nonexistent_file = temp_dir / "nonexistent.wav"
        
        with patch('src.services.transcriber.WhisperTranscriber') as mock_whisper_class:
            mock_whisper_instance = Mock()
            mock_whisper_class.return_value = mock_whisper_instance
            
            transcriber = AudioTranscriber("small", "cpu")
            
            with pytest.raises(FileNotFoundError):
                transcriber.transcribe_file(nonexistent_file)
    
    def test_get_supported_languages(self, mock_whisper_module):
        """Test getting supported languages."""
        with patch('src.services.transcriber.whisper.tokenizer.LANGUAGES', {'en': 'english', 'es': 'spanish'}):
            transcriber = AudioTranscriber("small", "cpu")
            languages = transcriber.get_supported_languages()
            
            assert isinstance(languages, list)
            assert 'en' in languages
            assert 'es' in languages
    
    def test_get_supported_languages_no_whisper(self):
        """Test getting supported languages without Whisper."""
        with patch('src.services.transcriber.whisper', side_effect=ImportError):
            transcriber = AudioTranscriber("small", "cpu")
            languages = transcriber.get_supported_languages()
            
            assert languages == []
    
    def test_detect_language(self, mock_whisper_module, sample_audio_data):
        """Test language detection."""
        audio, sample_rate = sample_audio_data
        
        with patch('src.services.transcriber.WhisperTranscriber') as mock_whisper_class:
            mock_whisper_instance = Mock()
            mock_whisper_class.return_value = mock_whisper_instance
            
            # Mock language detection result
            mock_result = TranscriptionResult(
                text="Test",
                language="en",
                language_probability=0.95,
                segments=[],
                duration=5.0,
                metadata={}
            )
            mock_whisper_instance.transcribe.return_value = mock_result
            
            transcriber = AudioTranscriber("small", "cpu")
            language, confidence = transcriber.detect_language(audio, sample_rate)
            
            assert language == "en"
            assert confidence == 0.95
    
    def test_detect_language_error(self, mock_whisper_module, sample_audio_data):
        """Test language detection error handling."""
        audio, sample_rate = sample_audio_data
        
        with patch('src.services.transcriber.WhisperTranscriber') as mock_whisper_class:
            mock_whisper_instance = Mock()
            mock_whisper_class.return_value = mock_whisper_instance
            mock_whisper_instance.transcribe.side_effect = Exception("Detection failed")
            
            transcriber = AudioTranscriber("small", "cpu")
            language, confidence = transcriber.detect_language(audio, sample_rate)
            
            assert language == "unknown"
            assert confidence == 0.0
    
    def test_get_transcription_quality_metrics(self, mock_transcription_result):
        """Test transcription quality metrics calculation."""
        transcriber = AudioTranscriber("small", "cpu")
        metrics = transcriber.get_transcription_quality_metrics(mock_transcription_result)
        
        assert isinstance(metrics, dict)
        assert "avg_confidence" in metrics
        assert "avg_logprob" in metrics
        assert "avg_compression_ratio" in metrics
        assert "avg_no_speech_prob" in metrics
        assert "num_segments" in metrics
        assert "total_duration" in metrics
        assert "language" in metrics
        assert "language_probability" in metrics
        assert "has_word_timestamps" in metrics
        
        assert metrics["num_segments"] == 2
        assert metrics["total_duration"] == 5.0
        assert metrics["language"] == "en"
        assert metrics["language_probability"] == 0.95
        assert metrics["has_word_timestamps"] == True
    
    def test_get_transcription_quality_metrics_no_segments(self):
        """Test transcription quality metrics with no segments."""
        result = TranscriptionResult(
            text="",
            language="en",
            language_probability=0.95,
            segments=[],
            duration=0.0,
            metadata={}
        )
        
        transcriber = AudioTranscriber("small", "cpu")
        metrics = transcriber.get_transcription_quality_metrics(result)
        
        assert metrics == {"error": "No segments available"}


class TestTranscriptionResult:
    """Test cases for TranscriptionResult dataclass."""
    
    def test_transcription_result_creation(self):
        """Test creating a transcription result."""
        segments = [
            TranscriptSegment(
                id=0,
                start=0.0,
                end=2.5,
                text="Hello world",
                confidence=0.95,
                words=[
                    WordSegment(word="Hello", start=0.0, end=1.0, probability=0.98),
                    WordSegment(word="world", start=1.0, end=2.0, probability=0.95)
                ]
            )
        ]
        
        result = TranscriptionResult(
            text="Hello world",
            language="en",
            language_probability=0.95,
            segments=segments,
            duration=2.5,
            metadata={"model_size": "small"}
        )
        
        assert result.has_segments
        assert result.has_words
        assert result.segments[0].duration == 2.5
        assert result.segments[0].confidence == 0.95
    
    def test_transcription_result_no_segments(self):
        """Test transcription result with no segments."""
        result = TranscriptionResult(
            text="Hello world",
            language="en",
            language_probability=0.95,
            segments=[],
            duration=2.5,
            metadata={}
        )
        
        assert not result.has_segments
        assert not result.has_words


class TestTranscriptSegment:
    """Test cases for TranscriptSegment dataclass."""
    
    def test_transcript_segment_creation(self):
        """Test creating a transcript segment."""
        words = [
            WordSegment(word="Hello", start=0.0, end=1.0, probability=0.98),
            WordSegment(word="world", start=1.0, end=2.0, probability=0.95)
        ]
        
        segment = TranscriptSegment(
            id=0,
            start=0.0,
            end=2.0,
            text="Hello world",
            confidence=0.95,
            words=words
        )
        
        assert segment.duration == 2.0
        assert segment.confidence == 0.95
        assert len(segment.words) == 2
    
    def test_transcript_segment_confidence_calculation(self):
        """Test transcript segment confidence calculation."""
        # Test with positive avg_logprob
        segment = TranscriptSegment(
            id=0,
            start=0.0,
            end=2.0,
            text="Hello world",
            avg_logprob=0.0,
            words=[]
        )
        assert segment.confidence == 1.0
        
        # Test with negative avg_logprob
        segment = TranscriptSegment(
            id=0,
            start=0.0,
            end=2.0,
            text="Hello world",
            avg_logprob=-0.5,
            words=[]
        )
        assert segment.confidence == 0.5
        
        # Test with very negative avg_logprob
        segment = TranscriptSegment(
            id=0,
            start=0.0,
            end=2.0,
            text="Hello world",
            avg_logprob=-2.0,
            words=[]
        )
        assert segment.confidence == 0.0


class TestWordSegment:
    """Test cases for WordSegment dataclass."""
    
    def test_word_segment_creation(self):
        """Test creating a word segment."""
        word = WordSegment(
            word="Hello",
            start=0.0,
            end=1.0,
            probability=0.98
        )
        
        assert word.word == "Hello"
        assert word.start == 0.0
        assert word.end == 1.0
        assert word.probability == 0.98


class TestGlobalTranscriberInstance:
    """Test cases for global transcriber instance."""
    
    def test_get_transcriber_instance(self):
        """Test getting global transcriber instance."""
        # Clear any existing instance
        import src.services.transcriber
        src.services.transcriber._transcriber_instance = None
        
        instance1 = get_transcriber_instance()
        instance2 = get_transcriber_instance()
        
        assert instance1 is instance2  # Should be the same instance
        assert isinstance(instance1, AudioTranscriber)
