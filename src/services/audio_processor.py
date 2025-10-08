"""
Main audio processing pipeline orchestrator.

This module coordinates the entire audio processing pipeline including
vocal separation, voice activity detection, transcription, and diarization.
"""

import time
import numpy as np
import torch
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import os

from ..core.config import settings
from ..core.logging import get_logger
from ..utils.timing import time_operation, time_context, get_performance_summary
from ..utils.device_utils import get_device, get_device_info
from ..utils.audio_utils import load_audio_as_tensor, save_wav_from_tensor, resample_audio_tensor, validate_audio_file
from .vad import get_vad_instance, VoiceSegment
from .separator import get_separator_instance, SeparationResult
from .transcriber import get_transcriber_instance, TranscriptionResult, TranscriptSegment
from .diarization import get_diarizer_instance, DiarizationResult, SpeakerSegment

logger = get_logger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for audio processing pipeline."""
    
    # Feature flags
    enable_separation: bool = True
    enable_vad: bool = True
    enable_diarization: bool = False
    
    # Processing parameters
    language: Optional[str] = None
    task: str = "transcribe"  # "transcribe" or "translate"
    word_timestamps: bool = True
    
    # Separation parameters
    separation_method: Optional[str] = None
    
    # VAD parameters
    vad_method: Optional[str] = None
    vad_threshold: float = 0.5
    
    # Diarization parameters
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    
    # Chunking parameters
    chunk_long_audio: bool = True
    chunk_size_seconds: Optional[int] = None
    chunk_overlap_seconds: Optional[float] = None
    
    # Quality parameters
    min_segment_duration: float = 0.1
    max_segment_duration: float = 30.0
    
    def __post_init__(self):
        """Apply default values from settings."""
        if self.chunk_size_seconds is None:
            self.chunk_size_seconds = settings.chunk_size_seconds
        if self.chunk_overlap_seconds is None:
            self.chunk_overlap_seconds = settings.chunk_overlap_seconds


@dataclass
class ProcessingResult:
    """Complete result of audio processing pipeline."""
    
    # Core results
    transcription: TranscriptionResult
    separation: Optional[SeparationResult] = None
    diarization: Optional[DiarizationResult] = None
    voice_segments: Optional[List[VoiceSegment]] = None
    
    # Processing metadata
    config: ProcessingConfig = field(default_factory=ProcessingConfig)
    processing_time: float = 0.0
    device_info: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_speaker_info(self) -> bool:
        """Check if result has speaker diarization information."""
        return self.diarization is not None and self.diarization.has_speakers
    
    @property
    def has_separation_info(self) -> bool:
        """Check if result has vocal separation information."""
        return self.separation is not None and self.separation.has_vocals
    
    def get_speaker_transcript(self) -> List[Dict[str, Any]]:
        """Get transcript with speaker information mapped."""
        if not self.has_speaker_info:
            return [{"segment": seg, "speaker": "UNKNOWN"} for seg in self.transcription.segments]
        
        return self.diarization.map_speakers_to_transcript(
            self.diarization, self.transcription.segments
        )


class AudioProcessor:
    """
    Main audio processing pipeline orchestrator.
    
    Coordinates the entire audio processing pipeline including:
    1. Audio validation and preprocessing
    2. Vocal separation (optional)
    3. Voice activity detection (optional)
    4. Speech transcription
    5. Speaker diarization (optional)
    6. Result formatting and quality assessment
    """
    
    def __init__(self):
        """Initialize audio processor with all required services."""
        self.device = get_device()
        self.device_info = get_device_info()
        
        # Initialize services
        self.vad = get_vad_instance()
        self.separator = get_separator_instance()
        self.transcriber = get_transcriber_instance()
        self.diarizer = get_diarizer_instance()
        
        logger.info(f"Audio processor initialized on device: {self.device}")
        logger.info(f"Device info: {self.device_info}")
    
    def _validate_audio_array(self, audio: np.ndarray, sample_rate: int) -> None:
        """Validate audio array parameters."""
        if audio is None or len(audio) == 0:
            raise ValueError("Audio array is empty")
        if sample_rate <= 0:
            raise ValueError(f"Invalid sample rate: {sample_rate}")
        if not isinstance(audio, np.ndarray):
            raise ValueError("Audio must be a numpy array")
        if audio.dtype not in [np.float32, np.float64]:
            logger.warning(f"Audio dtype {audio.dtype} may cause issues, consider using float32")
    
    @time_operation("audio_processing_pipeline")
    def process_audio(self, audio: np.ndarray, 
                     sample_rate: int,
                     config: Optional[ProcessingConfig] = None) -> ProcessingResult:
        """
        Process audio through the complete pipeline.
        
        Args:
            audio: Audio array (1D, float32, normalized to [-1, 1])
            sample_rate: Audio sample rate
            config: Processing configuration (uses defaults if None)
            
        Returns:
            ProcessingResult with complete processing results
        """
        if config is None:
            config = ProcessingConfig()
        
        start_time = time.perf_counter()
        logger.info(f"Starting audio processing pipeline ({len(audio)/sample_rate:.1f}s audio)")
        
        try:
            # Step 1: Audio validation and preprocessing
            processed_audio, sample_rate = self._preprocess_audio(audio, sample_rate, config)
            
            # Step 2: Vocal separation (optional)
            separation_result = None
            if config.enable_separation:
                separation_result = self._separate_vocals(processed_audio, sample_rate, config)
                if separation_result and separation_result.has_vocals:
                    processed_audio = separation_result.vocals
            
            # Step 3: Voice activity detection (optional)
            voice_segments = None
            if config.enable_vad:
                voice_segments = self._detect_voice_activity(processed_audio, sample_rate, config)
                if voice_segments:
                    # Filter audio based on voice activity
                    processed_audio = self.vad.filter_silence(processed_audio, voice_segments)
            
            # Step 4: Speech transcription
            transcription_result = self._transcribe_audio(processed_audio, sample_rate, config)
            
            # Step 5: Speaker diarization (optional)
            diarization_result = None
            if config.enable_diarization:
                diarization_result = self._diarize_speakers(processed_audio, sample_rate, config)
            
            # Step 6: Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(
                audio, sample_rate, transcription_result, separation_result, diarization_result
            )
            
            # Step 7: Get performance metrics
            performance_metrics = get_performance_summary()
            
            processing_time = time.perf_counter() - start_time
            
            result = ProcessingResult(
                transcription=transcription_result,
                separation=separation_result,
                diarization=diarization_result,
                voice_segments=voice_segments,
                config=config,
                processing_time=processing_time,
                device_info=self.device_info,
                performance_metrics=performance_metrics,
                quality_metrics=quality_metrics
            )
            
            logger.info(f"Audio processing completed in {processing_time:.2f}s")
            logger.info(f"Transcription: {len(transcription_result.text)} chars, {len(transcription_result.segments)} segments")
            if result.has_speaker_info:
                logger.info(f"Speakers detected: {result.diarization.num_speakers}")
            
            return result
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            raise
    
    def process_file(self, file_path: Union[str, Path],
                    config: Optional[ProcessingConfig] = None) -> ProcessingResult:
        """
        Process audio file through the complete pipeline.
        
        Args:
            file_path: Path to audio file
            config: Processing configuration (uses defaults if None)
            
        Returns:
            ProcessingResult with complete processing results
        """
        try:
            # Load audio as tensor and convert to numpy
            waveform, sample_rate = load_audio_as_tensor(file_path)
            audio = waveform.squeeze().numpy()  # Convert to 1D numpy array
            return self.process_audio(audio, sample_rate, config)
        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            raise
    
    @time_context("audio_preprocessing")
    def _preprocess_audio(self, audio: np.ndarray, sample_rate: int, 
                         config: ProcessingConfig) -> Tuple[np.ndarray, int]:
        """Preprocess audio for processing pipeline."""
        logger.debug("Preprocessing audio")
        
        # Validate audio
        self._validate_audio_array(audio, sample_rate)
        
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        
        # Resample to target sample rate if needed
        if sample_rate != settings.target_sample_rate:
            logger.info(f"Resampling from {sample_rate}Hz to {settings.target_sample_rate}Hz")
            # Convert to tensor for resampling, then back to numpy
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            resampled_tensor = resample_audio_tensor(audio_tensor, sample_rate, settings.target_sample_rate)
            audio = resampled_tensor.squeeze().numpy()
            sample_rate = settings.target_sample_rate
        
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        return audio, sample_rate
    
    @time_context("vocal_separation")
    def _separate_vocals(self, audio: np.ndarray, sample_rate: int,
                        config: ProcessingConfig) -> Optional[SeparationResult]:
        """Separate vocals from audio."""
        if not config.enable_separation:
            return None
        
        logger.debug("Separating vocals")
        try:
            return self.separator.separate_vocals(
                audio, sample_rate, method=config.separation_method
            )
        except Exception as e:
            logger.warning(f"Vocal separation failed: {e}")
            return None
    
    @time_context("voice_activity_detection")
    def _detect_voice_activity(self, audio: np.ndarray, sample_rate: int,
                              config: ProcessingConfig) -> Optional[List[VoiceSegment]]:
        """Detect voice activity in audio."""
        if not config.enable_vad:
            return None
        
        logger.debug("Detecting voice activity")
        try:
            return self.vad.detect_voice_activity(
                audio, method=config.vad_method, threshold=config.vad_threshold
            )
        except Exception as e:
            logger.warning(f"Voice activity detection failed: {e}")
            return None
    
    @time_context("speech_transcription")
    def _transcribe_audio(self, audio: np.ndarray, sample_rate: int,
                         config: ProcessingConfig) -> TranscriptionResult:
        """Transcribe audio to text."""
        logger.debug("Transcribing audio")
        
        return self.transcriber.transcribe(
            audio, sample_rate,
            language=config.language,
            task=config.task,
            word_timestamps=config.word_timestamps,
            chunk_long_audio=config.chunk_long_audio
        )
    
    @time_context("speaker_diarization")
    def _diarize_speakers(self, audio: np.ndarray, sample_rate: int,
                         config: ProcessingConfig) -> Optional[DiarizationResult]:
        """Perform speaker diarization on audio."""
        if not config.enable_diarization:
            return None
        
        logger.debug("Performing speaker diarization")
        try:
            return self.diarizer.diarize(
                audio, sample_rate,
                num_speakers=config.num_speakers,
                min_speakers=config.min_speakers,
                max_speakers=config.max_speakers
            )
        except Exception as e:
            logger.warning(f"Speaker diarization failed: {e}")
            return None
    
    def _calculate_quality_metrics(self, original_audio: np.ndarray, sample_rate: int,
                                 transcription_result: TranscriptionResult,
                                 separation_result: Optional[SeparationResult],
                                 diarization_result: Optional[DiarizationResult]) -> Dict[str, Any]:
        """Calculate quality metrics for the processing results."""
        metrics = {}
        
        # Transcription quality
        if transcription_result.segments:
            metrics["transcription"] = self.transcriber.get_transcription_quality_metrics(transcription_result)
        
        # Separation quality
        if separation_result and separation_result.has_vocals:
            metrics["separation"] = self.separator.get_separation_quality_metrics(
                original_audio, separation_result.vocals
            )
        
        # Diarization quality
        if diarization_result and diarization_result.has_speakers:
            metrics["diarization"] = self.diarizer.get_diarization_summary(diarization_result)
        
        # VAD quality
        if hasattr(self, '_last_voice_segments') and self._last_voice_segments:
            metrics["vad"] = self.vad.get_voice_activity_summary(self._last_voice_segments)
        
        return metrics
    
    def get_processing_capabilities(self) -> Dict[str, Any]:
        """Get information about processing capabilities."""
        return {
            "device": str(self.device),  # Convert torch.device to string
            "device_info": self.device_info.__dict__ if hasattr(self.device_info, '__dict__') else self.device_info,
            "services": {
                "vad": {
                    "available": self.vad is not None,
                    "methods": ["webrtc", "silero"] if self.vad else []
                },
                "separation": {
                    "available": self.separator is not None,
                    "methods": ["demucs", "spectral"] if self.separator else []
                },
                "transcription": {
                    "available": self.transcriber is not None,
                    "model_size": settings.whisper_model_size,
                    "supported_languages": self.transcriber.get_supported_languages() if self.transcriber else []
                },
                "diarization": {
                    "available": self.diarizer is not None and settings.enable_diarization,
                    "methods": ["pyannote", "simple"] if self.diarizer else []
                }
            },
            "configuration": {
                "max_file_size_mb": settings.max_file_size_mb,
                "target_sample_rate": settings.target_sample_rate,
                "chunk_size_seconds": settings.chunk_size_seconds,
                "enable_gpu": settings.enable_gpu
            }
        }
    
    def validate_config(self, config: ProcessingConfig) -> List[str]:
        """
        Validate processing configuration.
        
        Args:
            config: Processing configuration to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check language support
        if config.language and self.transcriber:
            supported_languages = self.transcriber.get_supported_languages()
            if config.language not in supported_languages:
                errors.append(f"Language '{config.language}' not supported")
        
        # Check speaker limits
        if config.enable_diarization:
            if config.min_speakers and config.max_speakers:
                if config.min_speakers > config.max_speakers:
                    errors.append("min_speakers cannot be greater than max_speakers")
            if config.num_speakers and config.num_speakers < 1:
                errors.append("num_speakers must be at least 1")
        
        # Check chunking parameters
        if config.chunk_size_seconds and config.chunk_size_seconds < 1:
            errors.append("chunk_size_seconds must be at least 1")
        if config.chunk_overlap_seconds and config.chunk_overlap_seconds < 0:
            errors.append("chunk_overlap_seconds cannot be negative")
        
        return errors


# Global audio processor instance
_processor_instance = None


def get_processor_instance() -> AudioProcessor:
    """Get or create global audio processor instance."""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = AudioProcessor()
    return _processor_instance
