"""
Voice Activity Detection (VAD) service for the audio transcription pipeline.

This module provides voice activity detection capabilities using WebRTC VAD
and Silero VAD models to identify speech segments in audio files.
"""

import numpy as np
import torch
import torchaudio
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

from ..core.config import settings
from ..core.logging import get_logger
from ..utils.timing import time_operation, time_context
from ..utils.device_utils import get_device, get_device_info

logger = get_logger(__name__)


@dataclass
class VoiceSegment:
    """Represents a voice activity segment."""
    
    start_time: float
    end_time: float
    confidence: float
    is_speech: bool
    
    @property
    def duration(self) -> float:
        """Get segment duration in seconds."""
        return self.end_time - self.start_time


class WebRTCVAD:
    """
    WebRTC Voice Activity Detection implementation.
    
    Uses the webrtcvad library for lightweight, real-time VAD.
    """
    
    def __init__(self, aggressiveness: int = 3, sample_rate: int = 16000):
        """
        Initialize WebRTC VAD.
        
        Args:
            aggressiveness: VAD aggressiveness (0-3, higher = more aggressive)
            sample_rate: Audio sample rate (must be 8000, 16000, 32000, or 48000)
        """
        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(aggressiveness)
            self.sample_rate = sample_rate
            self.frame_duration_ms = 30  # 30ms frames
            self.frame_size = int(sample_rate * self.frame_duration_ms / 1000)
            logger.info(f"WebRTC VAD initialized with aggressiveness {aggressiveness}")
        except ImportError:
            logger.error("webrtcvad not available. Install with: pip install webrtcvad")
            raise
    
    @time_operation("webrtc_vad_detection")
    def detect_voice_activity(self, audio: np.ndarray) -> List[VoiceSegment]:
        """
        Detect voice activity in audio using WebRTC VAD.
        
        Args:
            audio: Audio array (1D, float32, normalized to [-1, 1])
            
        Returns:
            List of voice segments
        """
        # Convert to 16-bit PCM for WebRTC VAD
        audio_int16 = (audio * 32767).astype(np.int16)
        
        segments = []
        frame_size = self.frame_size
        hop_size = frame_size // 2  # 50% overlap
        
        for i in range(0, len(audio_int16) - frame_size + 1, hop_size):
            frame = audio_int16[i:i + frame_size]
            
            # WebRTC VAD expects exactly the right frame size
            if len(frame) != frame_size:
                continue
            
            try:
                is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)
                start_time = i / self.sample_rate
                end_time = (i + frame_size) / self.sample_rate
                
                segments.append(VoiceSegment(
                    start_time=start_time,
                    end_time=end_time,
                    confidence=1.0 if is_speech else 0.0,
                    is_speech=is_speech
                ))
            except Exception as e:
                logger.warning(f"VAD frame processing error: {e}")
                continue
        
        # Merge adjacent speech segments
        merged_segments = self._merge_segments(segments)
        logger.info(f"WebRTC VAD detected {len(merged_segments)} voice segments")
        return merged_segments
    
    def _merge_segments(self, segments: List[VoiceSegment], 
                       gap_threshold: float = 0.5) -> List[VoiceSegment]:
        """Merge adjacent speech segments with small gaps."""
        if not segments:
            return []
        
        merged = []
        current_segment = segments[0]
        
        for segment in segments[1:]:
            if (segment.is_speech and current_segment.is_speech and 
                segment.start_time - current_segment.end_time <= gap_threshold):
                # Merge segments
                current_segment.end_time = segment.end_time
                current_segment.confidence = max(current_segment.confidence, segment.confidence)
            else:
                merged.append(current_segment)
                current_segment = segment
        
        merged.append(current_segment)
        return merged


class SileroVAD:
    """
    Silero Voice Activity Detection implementation.
    
    Uses the Silero VAD model for more accurate voice activity detection.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize Silero VAD.
        
        Args:
            model_path: Path to Silero VAD model (optional)
            device: Device to run model on (auto-detect if None)
        """
        self.device = device or get_device()
        self.model = None
        self.utils = None
        self.sample_rate = 16000
        
        logger.info(f"Initializing Silero VAD on device: {self.device}")
        self._load_model(model_path)
    
    def _load_model(self, model_path: Optional[str] = None):
        """Load Silero VAD model."""
        try:
            import torch
            from silero_vad import load_silero_vad, get_speech_timestamps
            
            # Load model
            self.model = load_silero_vad(use_onnx=False)
            self.model.to(self.device)
            self.model.eval()
            
            # Get utility functions
            self.get_speech_timestamps = get_speech_timestamps
            
            logger.info("Silero VAD model loaded successfully")
            
        except ImportError:
            logger.error("Silero VAD not available. Install with: pip install silero-vad")
            raise
        except Exception as e:
            logger.error(f"Failed to load Silero VAD model: {e}")
            raise
    
    @time_operation("silero_vad_detection")
    def detect_voice_activity(self, audio: np.ndarray, 
                            threshold: float = 0.5,
                            min_speech_duration_ms: int = 250,
                            min_silence_duration_ms: int = 100) -> List[VoiceSegment]:
        """
        Detect voice activity using Silero VAD.
        
        Args:
            audio: Audio array (1D, float32, normalized to [-1, 1])
            threshold: Speech detection threshold (0.0-1.0)
            min_speech_duration_ms: Minimum speech duration in milliseconds
            min_silence_duration_ms: Minimum silence duration in milliseconds
            
        Returns:
            List of voice segments
        """
        if self.model is None:
            raise RuntimeError("Silero VAD model not loaded")
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float().to(self.device)
        
        # Get speech timestamps
        speech_timestamps = self.get_speech_timestamps(
            audio_tensor,
            self.model,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            sampling_rate=self.sample_rate
        )
        
        # Convert to VoiceSegment objects
        segments = []
        for timestamp in speech_timestamps:
            segments.append(VoiceSegment(
                start_time=timestamp['start'] / self.sample_rate,
                end_time=timestamp['end'] / self.sample_rate,
                confidence=timestamp.get('confidence', 1.0),
                is_speech=True
            ))
        
        logger.info(f"Silero VAD detected {len(segments)} voice segments")
        return segments


class VoiceActivityDetector:
    """
    Main Voice Activity Detection service.
    
    Provides a unified interface for different VAD implementations
    with automatic fallback and configuration management.
    """
    
    def __init__(self, method: str = "auto", **kwargs):
        """
        Initialize VAD service.
        
        Args:
            method: VAD method ("webrtc", "silero", or "auto")
            **kwargs: Additional arguments for specific VAD implementations
        """
        self.method = method
        self.webrtc_vad = None
        self.silero_vad = None
        self.device = get_device()
        
        logger.info(f"Initializing VAD service with method: {method}")
        self._initialize_vad_methods(**kwargs)
    
    def _initialize_vad_methods(self, **kwargs):
        """Initialize VAD methods based on configuration."""
        try:
            if self.method in ["webrtc", "auto"]:
                self.webrtc_vad = WebRTCVAD(
                    aggressiveness=kwargs.get('aggressiveness', 3),
                    sample_rate=settings.target_sample_rate
                )
                logger.info("WebRTC VAD initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize WebRTC VAD: {e}")
            self.webrtc_vad = None
        
        try:
            if self.method in ["silero", "auto"]:
                self.silero_vad = SileroVAD(
                    model_path=kwargs.get('model_path'),
                    device=self.device
                )
                logger.info("Silero VAD initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Silero VAD: {e}")
            self.silero_vad = None
        
        if not self.webrtc_vad and not self.silero_vad:
            raise RuntimeError("No VAD methods available")
    
    @time_operation("vad_detect_voice_activity")
    def detect_voice_activity(self, audio: np.ndarray, 
                            method: Optional[str] = None,
                            **kwargs) -> List[VoiceSegment]:
        """
        Detect voice activity in audio.
        
        Args:
            audio: Audio array (1D, float32, normalized to [-1, 1])
            method: Override VAD method ("webrtc" or "silero")
            **kwargs: Additional arguments for VAD methods
            
        Returns:
            List of voice segments
        """
        if not settings.enable_vad:
            logger.info("VAD disabled, returning full audio as single segment")
            return [VoiceSegment(
                start_time=0.0,
                end_time=len(audio) / settings.target_sample_rate,
                confidence=1.0,
                is_speech=True
            )]
        
        # Determine which method to use
        use_method = method or self.method
        
        if use_method == "silero" and self.silero_vad:
            return self.silero_vad.detect_voice_activity(audio, **kwargs)
        elif use_method == "webrtc" and self.webrtc_vad:
            return self.webrtc_vad.detect_voice_activity(audio)
        elif use_method == "auto":
            # Try Silero first, fallback to WebRTC
            if self.silero_vad:
                try:
                    return self.silero_vad.detect_voice_activity(audio, **kwargs)
                except Exception as e:
                    logger.warning(f"Silero VAD failed, falling back to WebRTC: {e}")
            
            if self.webrtc_vad:
                return self.webrtc_vad.detect_voice_activity(audio)
        
        raise RuntimeError(f"VAD method '{use_method}' not available")
    
    def filter_silence(self, audio: np.ndarray, 
                      segments: List[VoiceSegment],
                      padding: float = 0.1) -> np.ndarray:
        """
        Remove silence from audio based on VAD segments.
        
        Args:
            audio: Original audio array
            segments: Voice activity segments
            padding: Padding around speech segments (seconds)
            
        Returns:
            Audio with silence removed
        """
        if not segments:
            return np.array([])
        
        sample_rate = settings.target_sample_rate
        filtered_audio = []
        
        for segment in segments:
            if not segment.is_speech:
                continue
            
            start_sample = max(0, int((segment.start_time - padding) * sample_rate))
            end_sample = min(len(audio), int((segment.end_time + padding) * sample_rate))
            
            filtered_audio.append(audio[start_sample:end_sample])
        
        if filtered_audio:
            return np.concatenate(filtered_audio)
        else:
            return np.array([])
    
    def get_voice_activity_summary(self, segments: List[VoiceSegment]) -> Dict[str, Any]:
        """
        Get summary statistics for voice activity.
        
        Args:
            segments: Voice activity segments
            
        Returns:
            Dictionary with summary statistics
        """
        if not segments:
            return {"total_segments": 0, "total_speech_time": 0.0}
        
        speech_segments = [s for s in segments if s.is_speech]
        total_speech_time = sum(s.duration for s in speech_segments)
        total_time = max(s.end_time for s in segments) if segments else 0.0
        
        return {
            "total_segments": len(segments),
            "speech_segments": len(speech_segments),
            "total_speech_time": total_speech_time,
            "total_time": total_time,
            "speech_ratio": total_speech_time / total_time if total_time > 0 else 0.0,
            "avg_segment_duration": total_speech_time / len(speech_segments) if speech_segments else 0.0
        }


# Global VAD instance
_vad_instance = None


def get_vad_instance() -> VoiceActivityDetector:
    """Get or create global VAD instance."""
    global _vad_instance
    if _vad_instance is None:
        _vad_instance = VoiceActivityDetector()
    return _vad_instance
