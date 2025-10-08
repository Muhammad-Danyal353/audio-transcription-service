"""
ASR (Automatic Speech Recognition) transcription service.

This module provides speech-to-text capabilities using OpenAI's Whisper models
with support for multiple languages, chunking, and word-level timestamps.
"""

import numpy as np
import torch
import torchaudio
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import os
import json

from ..core.config import settings
from ..core.logging import get_logger
from ..utils.timing import time_operation, time_context
from ..utils.device_utils import get_device, get_device_info
from ..utils.audio_utils import load_audio_as_tensor, save_wav_from_tensor, resample_audio_tensor

logger = get_logger(__name__)


@dataclass
class WordSegment:
    """Represents a word-level segment with timing information."""
    
    word: str
    start: float
    end: float
    probability: float = 1.0


@dataclass
class TranscriptSegment:
    """Represents a transcript segment with timing and confidence information."""
    
    id: int
    start: float
    end: float
    text: str
    tokens: List[int] = field(default_factory=list)
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0
    words: List[WordSegment] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        """Get segment duration in seconds."""
        return self.end - self.start
    
    @property
    def confidence(self) -> float:
        """Get segment confidence score."""
        return max(0.0, min(1.0, 1.0 + self.avg_logprob))


@dataclass
class TranscriptionResult:
    """Complete transcription result with metadata."""
    
    text: str
    language: str
    language_probability: float
    segments: List[TranscriptSegment]
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_segments(self) -> bool:
        """Check if result has segment information."""
        return len(self.segments) > 0
    
    @property
    def has_words(self) -> bool:
        """Check if result has word-level information."""
        return any(segment.words for segment in self.segments)


class WhisperTranscriber:
    """
    Whisper-based ASR transcription implementation.
    
    Uses OpenAI's Whisper models for high-quality speech recognition.
    """
    
    def __init__(self, model_size: str = None, device: Optional[str] = None):
        """
        Initialize Whisper transcriber.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to run model on (auto-detect if None)
        """
        self.model_size = model_size or settings.whisper_model_size
        self.device = device or get_device()
        self.model = None
        self.sample_rate = 16000  # Whisper's native sample rate
        
        logger.info(f"Initializing Whisper transcriber with model: {self.model_size}")
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model."""
        try:
            import whisper
            
            # Load the model
            self.model = whisper.load_model(self.model_size, device=self.device)
            
            logger.info(f"Whisper model '{self.model_size}' loaded successfully on {self.device}")
            
        except ImportError:
            logger.error("Whisper not available. Install with: pip install openai-whisper")
            raise
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    @time_operation("whisper_transcription")
    def transcribe(self, audio: np.ndarray, 
                  sample_rate: int,
                  language: Optional[str] = None,
                  task: str = "transcribe",
                  word_timestamps: bool = True,
                  **kwargs) -> TranscriptionResult:
        """
        Transcribe audio using Whisper.
        
        Args:
            audio: Audio array (1D, float32)
            sample_rate: Audio sample rate
            language: Language code (auto-detect if None)
            task: Task type ("transcribe" or "translate")
            word_timestamps: Enable word-level timestamps
            **kwargs: Additional Whisper parameters
            
        Returns:
            TranscriptionResult with full transcription data
        """
        if self.model is None:
            raise RuntimeError("Whisper model not loaded")
        
        # Resample to Whisper's sample rate if needed
        if sample_rate != self.sample_rate:
            logger.info(f"Resampling from {sample_rate}Hz to {self.sample_rate}Hz")
            # Convert to tensor for resampling, then back to numpy
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            resampled_tensor = resample_audio_tensor(audio_tensor, sample_rate, self.sample_rate)
            audio = resampled_tensor.squeeze().numpy()
        
        # Prepare Whisper options
        options = {
            "language": language,
            "task": task,
            "word_timestamps": word_timestamps,
            "fp16": self.device != "cpu",  # Use fp16 on GPU
            **kwargs
        }
        
        with time_context("whisper_inference"):
            try:
                # Run transcription
                result = self.model.transcribe(audio, **options)
                
                # Parse result
                return self._parse_whisper_result(result, len(audio) / self.sample_rate)
                
            except Exception as e:
                logger.error(f"Whisper transcription failed: {e}")
                raise
    
    def _parse_whisper_result(self, result: Dict[str, Any], duration: float) -> TranscriptionResult:
        """Parse Whisper result into TranscriptionResult."""
        # Extract segments
        segments = []
        for i, segment in enumerate(result.get("segments", [])):
            # Extract word-level information
            words = []
            for word_info in segment.get("words", []):
                words.append(WordSegment(
                    word=word_info["word"],
                    start=word_info["start"],
                    end=word_info["end"],
                    probability=word_info.get("probability", 1.0)
                ))
            
            segments.append(TranscriptSegment(
                id=i,
                start=segment["start"],
                end=segment["end"],
                text=segment["text"].strip(),
                tokens=segment.get("tokens", []),
                temperature=segment.get("temperature", 0.0),
                avg_logprob=segment.get("avg_logprob", 0.0),
                compression_ratio=segment.get("compression_ratio", 0.0),
                no_speech_prob=segment.get("no_speech_prob", 0.0),
                words=words
            ))
        
        return TranscriptionResult(
            text=result["text"].strip(),
            language=result.get("language", "unknown"),
            language_probability=result.get("language_probability", 1.0),
            segments=segments,
            duration=duration,
            metadata={
                "model_size": self.model_size,
                "device": str(self.device),
                "sample_rate": self.sample_rate
            }
        )
    
    def transcribe_chunked(self, audio: np.ndarray, 
                          sample_rate: int,
                          chunk_size: int = None,
                          overlap: float = None,
                          **kwargs) -> TranscriptionResult:
        """
        Transcribe long audio by chunking.
        
        Args:
            audio: Audio array (1D, float32)
            sample_rate: Audio sample rate
            chunk_size: Chunk size in seconds (default from config)
            overlap: Overlap between chunks in seconds (default from config)
            **kwargs: Additional transcription parameters
            
        Returns:
            Combined transcription result
        """
        if chunk_size is None:
            chunk_size = settings.chunk_size_seconds
        if overlap is None:
            overlap = settings.chunk_overlap_seconds
        
        duration = len(audio) / sample_rate
        
        # If audio is short enough, transcribe directly
        if duration <= chunk_size:
            return self.transcribe(audio, sample_rate, **kwargs)
        
        logger.info(f"Chunking audio ({duration:.1f}s) into {chunk_size}s chunks with {overlap}s overlap")
        
        chunk_samples = int(chunk_size * sample_rate)
        overlap_samples = int(overlap * sample_rate)
        
        all_segments = []
        segment_id_offset = 0
        
        for i in range(0, len(audio), chunk_samples - overlap_samples):
            chunk = audio[i:i + chunk_samples]
            chunk_start_time = i / sample_rate
            
            if len(chunk) < sample_rate:  # Skip very short chunks
                continue
            
            logger.debug(f"Transcribing chunk {i//chunk_samples + 1} ({chunk_start_time:.1f}s - {chunk_start_time + len(chunk)/sample_rate:.1f}s)")
            
            try:
                chunk_result = self.transcribe(chunk, sample_rate, **kwargs)
                
                # Adjust segment timestamps
                for segment in chunk_result.segments:
                    segment.id += segment_id_offset
                    segment.start += chunk_start_time
                    segment.end += chunk_start_time
                    
                    # Adjust word timestamps
                    for word in segment.words:
                        word.start += chunk_start_time
                        word.end += chunk_start_time
                
                all_segments.extend(chunk_result.segments)
                segment_id_offset += len(chunk_result.segments)
                
            except Exception as e:
                logger.warning(f"Failed to transcribe chunk at {chunk_start_time:.1f}s: {e}")
                continue
        
        # Combine all text
        combined_text = " ".join(segment.text for segment in all_segments)
        
        return TranscriptionResult(
            text=combined_text,
            language=all_segments[0].language if all_segments else "unknown",
            language_probability=1.0,  # Approximate
            segments=all_segments,
            duration=duration,
            metadata={
                "model_size": self.model_size,
                "device": str(self.device),
                "chunked": True,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "num_chunks": len(all_segments)
            }
        )


class AudioTranscriber:
    """
    Main audio transcription service.
    
    Provides a unified interface for transcription with automatic
    chunking, language detection, and error handling.
    """
    
    def __init__(self, model_size: str = None, device: Optional[str] = None):
        """
        Initialize audio transcriber.
        
        Args:
            model_size: Whisper model size
            device: Device to run model on
        """
        self.model_size = model_size or settings.whisper_model_size
        self.device = device or get_device()
        self.whisper_transcriber = None
        
        logger.info(f"Initializing audio transcriber with model: {self.model_size}")
        self._initialize_transcriber()
    
    def _initialize_transcriber(self):
        """Initialize transcription methods."""
        try:
            self.whisper_transcriber = WhisperTranscriber(
                model_size=self.model_size,
                device=self.device
            )
            logger.info("Whisper transcriber initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Whisper transcriber: {e}")
            raise
    
    @time_operation("audio_transcription")
    def transcribe(self, audio: np.ndarray, 
                  sample_rate: int,
                  language: Optional[str] = None,
                  task: str = "transcribe",
                  word_timestamps: bool = True,
                  chunk_long_audio: bool = True,
                  **kwargs) -> TranscriptionResult:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio array (1D, float32)
            sample_rate: Audio sample rate
            language: Language code (auto-detect if None)
            task: Task type ("transcribe" or "translate")
            word_timestamps: Enable word-level timestamps
            chunk_long_audio: Enable automatic chunking for long audio
            **kwargs: Additional transcription parameters
            
        Returns:
            TranscriptionResult with full transcription data
        """
        if self.whisper_transcriber is None:
            raise RuntimeError("Transcriber not initialized")
        
        duration = len(audio) / sample_rate
        logger.info(f"Transcribing audio ({duration:.1f}s) with language: {language or 'auto'}")
        
        try:
            if chunk_long_audio and duration > settings.chunk_size_seconds:
                return self.whisper_transcriber.transcribe_chunked(
                    audio, sample_rate, **kwargs
                )
            else:
                return self.whisper_transcriber.transcribe(
                    audio, sample_rate, language, task, word_timestamps, **kwargs
                )
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            # Return empty result as fallback
            return TranscriptionResult(
                text="",
                language="unknown",
                language_probability=0.0,
                segments=[],
                duration=duration,
                metadata={"error": str(e), "fallback": True}
            )
    
    def transcribe_file(self, file_path: Union[str, Path],
                       language: Optional[str] = None,
                       **kwargs) -> TranscriptionResult:
        """
        Transcribe audio file.
        
        Args:
            file_path: Path to audio file
            language: Language code (auto-detect if None)
            **kwargs: Additional transcription parameters
            
        Returns:
            TranscriptionResult with full transcription data
        """
        try:
            # Load audio as tensor and convert to numpy
            waveform, sample_rate = load_audio_as_tensor(file_path)
            audio = waveform.squeeze().numpy()  # Convert to 1D numpy array
            return self.transcribe(audio, sample_rate, language, **kwargs)
        except Exception as e:
            logger.error(f"Failed to transcribe file {file_path}: {e}")
            raise
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        try:
            import whisper
            return list(whisper.tokenizer.LANGUAGES.keys())
        except ImportError:
            return []
    
    def detect_language(self, audio: np.ndarray, sample_rate: int) -> Tuple[str, float]:
        """
        Detect language of audio.
        
        Args:
            audio: Audio array (1D, float32)
            sample_rate: Audio sample rate
            
        Returns:
            Tuple of (language_code, confidence)
        """
        try:
            # Use first 30 seconds for language detection
            max_samples = min(len(audio), 30 * sample_rate)
            audio_sample = audio[:max_samples]
            
            result = self.transcribe(audio_sample, sample_rate, word_timestamps=False)
            return result.language, result.language_probability
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "unknown", 0.0
    
    def get_transcription_quality_metrics(self, result: TranscriptionResult) -> Dict[str, Any]:
        """
        Calculate quality metrics for transcription results.
        
        Args:
            result: Transcription result
            
        Returns:
            Dictionary with quality metrics
        """
        if not result.segments:
            return {"error": "No segments available"}
        
        # Calculate average confidence
        confidences = [segment.confidence for segment in result.segments]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Calculate average log probability
        logprobs = [segment.avg_logprob for segment in result.segments]
        avg_logprob = np.mean(logprobs) if logprobs else 0.0
        
        # Calculate compression ratio
        compression_ratios = [segment.compression_ratio for segment in result.segments]
        avg_compression_ratio = np.mean(compression_ratios) if compression_ratios else 0.0
        
        # Calculate no-speech probability
        no_speech_probs = [segment.no_speech_prob for segment in result.segments]
        avg_no_speech_prob = np.mean(no_speech_probs) if no_speech_probs else 0.0
        
        return {
            "avg_confidence": avg_confidence,
            "avg_logprob": avg_logprob,
            "avg_compression_ratio": avg_compression_ratio,
            "avg_no_speech_prob": avg_no_speech_prob,
            "num_segments": len(result.segments),
            "total_duration": result.duration,
            "language": result.language,
            "language_probability": result.language_probability,
            "has_word_timestamps": result.has_words
        }


# Global transcriber instance
_transcriber_instance = None


def get_transcriber_instance() -> AudioTranscriber:
    """Get or create global transcriber instance."""
    global _transcriber_instance
    if _transcriber_instance is None:
        _transcriber_instance = AudioTranscriber()
    return _transcriber_instance
