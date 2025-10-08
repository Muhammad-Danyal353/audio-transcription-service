"""
Speaker diarization service for the audio transcription pipeline.

This module provides speaker diarization capabilities using pyannote.audio
to identify and separate different speakers in audio recordings.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import os

from ..core.config import settings
from ..core.logging import get_logger
from ..utils.timing import time_operation, time_context
from ..utils.device_utils import get_device, get_device_info
from ..utils.audio_utils import load_audio_as_tensor, save_wav_from_tensor, resample_audio_tensor

logger = get_logger(__name__)


@dataclass
class SpeakerSegment:
    """Represents a speaker segment with timing and speaker information."""
    
    speaker: str
    start: float
    end: float
    confidence: float = 1.0
    
    @property
    def duration(self) -> float:
        """Get segment duration in seconds."""
        return self.end - self.start


@dataclass
class DiarizationResult:
    """Result of speaker diarization."""
    
    speakers: List[str]
    segments: List[SpeakerSegment]
    duration: float
    num_speakers: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_speakers(self) -> bool:
        """Check if diarization found any speakers."""
        return len(self.speakers) > 0
    
    def get_speaker_segments(self, speaker: str) -> List[SpeakerSegment]:
        """Get all segments for a specific speaker."""
        return [segment for segment in self.segments if segment.speaker == speaker]
    
    def get_speaker_duration(self, speaker: str) -> float:
        """Get total duration for a specific speaker."""
        segments = self.get_speaker_segments(speaker)
        return sum(segment.duration for segment in segments)


class PyannoteDiarizer:
    """
    Pyannote.audio-based speaker diarization implementation.
    
    Uses HuggingFace's pyannote.audio models for speaker diarization.
    """
    
    def __init__(self, model_name: str = "pyannote/speaker-diarization-3.1", 
                 device: Optional[str] = None):
        """
        Initialize Pyannote diarizer.
        
        Args:
            model_name: Pyannote model name
            device: Device to run model on (auto-detect if None)
        """
        self.model_name = model_name
        self.device = device or get_device()
        self.pipeline = None
        self.sample_rate = 16000  # Pyannote's preferred sample rate
        
        logger.info(f"Initializing Pyannote diarizer with model: {model_name}")
        self._load_model()
    
    def _load_model(self):
        """Load Pyannote diarization pipeline."""
        try:
            from pyannote.audio import Pipeline
            
            # Check for HuggingFace token
            if not settings.huggingface_token:
                logger.warning("No HuggingFace token provided. Diarization may not work.")
                logger.info("Set HUGGINGFACE_TOKEN environment variable to enable diarization.")
                raise ValueError("HuggingFace token required for diarization")
            
            # Load the pipeline
            self.pipeline = Pipeline.from_pretrained(
                self.model_name,
                use_auth_token=settings.huggingface_token
            )
            
            # Move to device if possible
            if hasattr(self.pipeline, 'to'):
                self.pipeline = self.pipeline.to(self.device)
            
            logger.info(f"Pyannote diarization pipeline loaded successfully on {self.device}")
            
        except ImportError:
            logger.error("pyannote.audio not available. Install with: pip install pyannote.audio")
            raise
        except Exception as e:
            logger.error(f"Failed to load Pyannote diarization pipeline: {e}")
            raise
    
    @time_operation("pyannote_diarization")
    def diarize(self, audio: np.ndarray, sample_rate: int,
                num_speakers: Optional[int] = None,
                min_speakers: Optional[int] = None,
                max_speakers: Optional[int] = None) -> DiarizationResult:
        """
        Perform speaker diarization on audio.
        
        Args:
            audio: Audio array (1D, float32)
            sample_rate: Audio sample rate
            num_speakers: Expected number of speakers (optional)
            min_speakers: Minimum number of speakers (optional)
            max_speakers: Maximum number of speakers (optional)
            
        Returns:
            DiarizationResult with speaker segments
        """
        if self.pipeline is None:
            raise RuntimeError("Pyannote diarization pipeline not loaded")
        
        # Resample to Pyannote's preferred sample rate if needed
        if sample_rate != self.sample_rate:
            logger.info(f"Resampling from {sample_rate}Hz to {self.sample_rate}Hz")
            # Convert to tensor for resampling, then back to numpy
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            resampled_tensor = resample_audio_tensor(audio_tensor, sample_rate, self.sample_rate)
            audio = resampled_tensor.squeeze().numpy()
        
        # Save audio to temporary file (Pyannote expects file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            # Convert to tensor for saving
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            save_wav_from_tensor(audio_tensor, self.sample_rate, tmp_file.name)
            temp_path = tmp_file.name
        
        try:
            with time_context("pyannote_inference"):
                # Run diarization
                diarization = self.pipeline(
                    temp_path,
                    num_speakers=num_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers
                )
            
            # Parse results
            return self._parse_diarization_result(diarization, len(audio) / self.sample_rate)
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except OSError:
                pass
    
    def _parse_diarization_result(self, diarization, duration: float) -> DiarizationResult:
        """Parse Pyannote diarization result."""
        segments = []
        speakers = set()
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(SpeakerSegment(
                speaker=speaker,
                start=turn.start,
                end=turn.end,
                confidence=1.0  # Pyannote doesn't provide confidence scores
            ))
            speakers.add(speaker)
        
        # Sort segments by start time
        segments.sort(key=lambda x: x.start)
        
        return DiarizationResult(
            speakers=sorted(list(speakers)),
            segments=segments,
            duration=duration,
            num_speakers=len(speakers),
            metadata={
                "model": self.model_name,
                "device": str(self.device),
                "sample_rate": self.sample_rate
            }
        )


class SimpleDiarizer:
    """
    Simple fallback diarization implementation.
    
    Uses basic audio analysis to detect speaker changes.
    This is a placeholder implementation for when pyannote is not available.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize simple diarizer.
        
        Args:
            device: Device to run on (not used in simple implementation)
        """
        self.device = device or get_device()
        logger.info("Simple diarizer initialized (fallback mode)")
    
    @time_operation("simple_diarization")
    def diarize(self, audio: np.ndarray, sample_rate: int,
                num_speakers: Optional[int] = None,
                min_speakers: Optional[int] = None,
                max_speakers: Optional[int] = None) -> DiarizationResult:
        """
        Perform simple speaker diarization.
        
        This is a placeholder implementation that treats the entire audio
        as a single speaker segment.
        
        Args:
            audio: Audio array (1D, float32)
            sample_rate: Audio sample rate
            num_speakers: Expected number of speakers (ignored)
            min_speakers: Minimum number of speakers (ignored)
            max_speakers: Maximum number of speakers (ignored)
            
        Returns:
            DiarizationResult with single speaker segment
        """
        duration = len(audio) / sample_rate
        
        # Create a single speaker segment for the entire audio
        segments = [SpeakerSegment(
            speaker="SPEAKER_00",
            start=0.0,
            end=duration,
            confidence=0.5  # Low confidence for fallback
        )]
        
        return DiarizationResult(
            speakers=["SPEAKER_00"],
            segments=segments,
            duration=duration,
            num_speakers=1,
            metadata={
                "method": "simple_fallback",
                "device": str(self.device),
                "note": "This is a fallback implementation. Install pyannote.audio for proper diarization."
            }
        )


class SpeakerDiarizer:
    """
    Main speaker diarization service.
    
    Provides a unified interface for speaker diarization with automatic
    fallback and configuration management.
    """
    
    def __init__(self, method: str = "auto", **kwargs):
        """
        Initialize speaker diarizer.
        
        Args:
            method: Diarization method ("pyannote", "simple", or "auto")
            **kwargs: Additional arguments for specific methods
        """
        self.method = method
        self.pyannote_diarizer = None
        self.simple_diarizer = None
        self.device = get_device()
        
        logger.info(f"Initializing speaker diarizer with method: {method}")
        self._initialize_diarizers(**kwargs)
    
    def _initialize_diarizers(self, **kwargs):
        """Initialize diarization methods based on configuration."""
        if not settings.enable_diarization:
            logger.info("Speaker diarization disabled")
            return
        
        try:
            if self.method in ["pyannote", "auto"]:
                self.pyannote_diarizer = PyannoteDiarizer(
                    model_name=kwargs.get('model_name', "pyannote/speaker-diarization-3.1"),
                    device=self.device
                )
                logger.info("Pyannote diarizer initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Pyannote diarizer: {e}")
            self.pyannote_diarizer = None
        
        # Always initialize simple diarizer as fallback
        self.simple_diarizer = SimpleDiarizer(device=self.device)
        logger.info("Simple diarizer initialized")
    
    @time_operation("speaker_diarization")
    def diarize(self, audio: np.ndarray, sample_rate: int,
                method: Optional[str] = None,
                num_speakers: Optional[int] = None,
                min_speakers: Optional[int] = None,
                max_speakers: Optional[int] = None,
                **kwargs) -> DiarizationResult:
        """
        Perform speaker diarization on audio.
        
        Args:
            audio: Audio array (1D, float32)
            sample_rate: Audio sample rate
            method: Override diarization method
            num_speakers: Expected number of speakers
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            **kwargs: Additional arguments for diarization methods
            
        Returns:
            DiarizationResult with speaker segments
        """
        if not settings.enable_diarization:
            logger.info("Speaker diarization disabled, returning single speaker")
            duration = len(audio) / sample_rate
            return DiarizationResult(
                speakers=["SPEAKER_00"],
                segments=[SpeakerSegment(
                    speaker="SPEAKER_00",
                    start=0.0,
                    end=duration,
                    confidence=1.0
                )],
                duration=duration,
                num_speakers=1,
                metadata={"method": "disabled", "enabled": False}
            )
        
        # Determine which method to use
        use_method = method or self.method
        
        try:
            if use_method == "pyannote" and self.pyannote_diarizer:
                return self.pyannote_diarizer.diarize(
                    audio, sample_rate, num_speakers, min_speakers, max_speakers
                )
            elif use_method == "simple" and self.simple_diarizer:
                return self.simple_diarizer.diarize(
                    audio, sample_rate, num_speakers, min_speakers, max_speakers
                )
            elif use_method == "auto":
                # Try Pyannote first, fallback to simple
                if self.pyannote_diarizer:
                    try:
                        return self.pyannote_diarizer.diarize(
                            audio, sample_rate, num_speakers, min_speakers, max_speakers
                        )
                    except Exception as e:
                        logger.warning(f"Pyannote diarization failed, falling back to simple: {e}")
                
                if self.simple_diarizer:
                    return self.simple_diarizer.diarize(
                        audio, sample_rate, num_speakers, min_speakers, max_speakers
                    )
            
            raise RuntimeError(f"Diarization method '{use_method}' not available")
            
        except Exception as e:
            logger.error(f"Speaker diarization failed: {e}")
            # Return single speaker as fallback
            duration = len(audio) / sample_rate
            return DiarizationResult(
                speakers=["SPEAKER_00"],
                segments=[SpeakerSegment(
                    speaker="SPEAKER_00",
                    start=0.0,
                    end=duration,
                    confidence=0.0
                )],
                duration=duration,
                num_speakers=1,
                metadata={"method": "fallback", "error": str(e)}
            )
    
    def map_speakers_to_transcript(self, diarization_result: DiarizationResult,
                                 transcript_segments: List[Any]) -> List[Dict[str, Any]]:
        """
        Map speaker information to transcript segments.
        
        Args:
            diarization_result: Result from speaker diarization
            transcript_segments: List of transcript segments
            
        Returns:
            List of transcript segments with speaker information
        """
        if not diarization_result.has_speakers:
            # No speaker information available
            return [{"segment": seg, "speaker": "UNKNOWN"} for seg in transcript_segments]
        
        speaker_segments = diarization_result.segments
        mapped_segments = []
        
        for transcript_seg in transcript_segments:
            # Find overlapping speaker segments
            overlapping_speakers = []
            for speaker_seg in speaker_segments:
                # Check for overlap
                if (speaker_seg.start < transcript_seg.end and 
                    speaker_seg.end > transcript_seg.start):
                    overlap_start = max(speaker_seg.start, transcript_seg.start)
                    overlap_end = min(speaker_seg.end, transcript_seg.end)
                    overlap_duration = overlap_end - overlap_start
                    overlapping_speakers.append((speaker_seg.speaker, overlap_duration))
            
            # Choose speaker with maximum overlap
            if overlapping_speakers:
                best_speaker = max(overlapping_speakers, key=lambda x: x[1])[0]
            else:
                best_speaker = "UNKNOWN"
            
            mapped_segments.append({
                "segment": transcript_seg,
                "speaker": best_speaker
            })
        
        return mapped_segments
    
    def get_diarization_summary(self, result: DiarizationResult) -> Dict[str, Any]:
        """
        Get summary statistics for diarization results.
        
        Args:
            result: Diarization result
            
        Returns:
            Dictionary with summary statistics
        """
        if not result.has_speakers:
            return {"num_speakers": 0, "total_segments": 0}
        
        speaker_stats = {}
        for speaker in result.speakers:
            segments = result.get_speaker_segments(speaker)
            total_duration = result.get_speaker_duration(speaker)
            
            speaker_stats[speaker] = {
                "num_segments": len(segments),
                "total_duration": total_duration,
                "speaking_ratio": total_duration / result.duration if result.duration > 0 else 0.0
            }
        
        return {
            "num_speakers": result.num_speakers,
            "total_segments": len(result.segments),
            "total_duration": result.duration,
            "speakers": speaker_stats,
            "method": result.metadata.get("method", "unknown")
        }


# Global diarizer instance
_diarizer_instance = None


def get_diarizer_instance() -> SpeakerDiarizer:
    """Get or create global diarizer instance."""
    global _diarizer_instance
    if _diarizer_instance is None:
        _diarizer_instance = SpeakerDiarizer()
    return _diarizer_instance
