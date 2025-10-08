"""
Pydantic data models for the audio transcription API.

This module defines all request and response models used by the FastAPI endpoints,
providing type safety, validation, and automatic OpenAPI documentation.
"""

from typing import List, Dict, Any, Optional, Union, Literal
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
import numpy as np

from ..services.audio_processor import ProcessingConfig, ProcessingResult
from ..services.transcriber import TranscriptSegment, WordSegment
from ..services.separator import SeparationResult
from ..services.diarization import DiarizationResult, SpeakerSegment
from ..services.vad import VoiceSegment
from ..utils.timing import TimingMetrics


def convert_numpy_types(obj):
    """Recursively convert NumPy types to Python native types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


class TaskType(str, Enum):
    """Supported transcription task types."""
    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"


class TranscribeConfig(BaseModel):
    """Configuration for audio transcription request."""
    
    # Feature flags
    enable_separation: bool = Field(
        default=True, 
        description="Enable vocal separation to improve transcription quality"
    )
    enable_vad: bool = Field(
        default=True, 
        description="Enable voice activity detection to remove silence"
    )
    enable_diarization: bool = Field(
        default=False, 
        description="Enable speaker diarization (requires HuggingFace token)"
    )
    
    # Processing parameters
    language: Optional[str] = Field(
        default=None, 
        description="Language code for transcription (auto-detect if not specified)",
        pattern=r"^[a-z]{2}(-[A-Z]{2})?$"
    )
    task: TaskType = Field(
        default=TaskType.TRANSCRIBE, 
        description="Task type: transcribe or translate to English"
    )
    word_timestamps: bool = Field(
        default=True, 
        description="Include word-level timestamps in response"
    )
    
    # Separation parameters
    separation_method: Optional[Literal["demucs", "spectral", "auto"]] = Field(
        default=None, 
        description="Vocal separation method to use"
    )
    
    # VAD parameters
    vad_method: Optional[Literal["webrtc", "silero", "auto"]] = Field(
        default=None, 
        description="Voice activity detection method to use"
    )
    vad_threshold: float = Field(
        default=0.5, 
        ge=0.0, 
        le=1.0, 
        description="VAD confidence threshold"
    )
    
    # Diarization parameters
    num_speakers: Optional[int] = Field(
        default=None, 
        ge=1, 
        le=10, 
        description="Expected number of speakers"
    )
    min_speakers: Optional[int] = Field(
        default=None, 
        ge=1, 
        le=10, 
        description="Minimum number of speakers"
    )
    max_speakers: Optional[int] = Field(
        default=None, 
        ge=1, 
        le=10, 
        description="Maximum number of speakers"
    )
    
    # Chunking parameters
    chunk_long_audio: bool = Field(
        default=True, 
        description="Enable automatic chunking for long audio files"
    )
    chunk_size_seconds: Optional[int] = Field(
        default=None, 
        ge=1, 
        le=300, 
        description="Chunk size in seconds for long audio processing"
    )
    chunk_overlap_seconds: Optional[float] = Field(
        default=None, 
        ge=0.0, 
        le=10.0, 
        description="Overlap between chunks in seconds"
    )
    
    @model_validator(mode='after')
    def validate_speaker_limits(self):
        """Validate speaker diarization parameters."""
        min_speakers = self.min_speakers
        max_speakers = self.max_speakers
        num_speakers = self.num_speakers
        
        if min_speakers and max_speakers and min_speakers > max_speakers:
            raise ValueError("min_speakers cannot be greater than max_speakers")
        
        if num_speakers and min_speakers and num_speakers < min_speakers:
            raise ValueError("num_speakers cannot be less than min_speakers")
        
        if num_speakers and max_speakers and num_speakers > max_speakers:
            raise ValueError("num_speakers cannot be greater than max_speakers")
        
        return self


class WordSegmentModel(BaseModel):
    """Word-level segment with timing information."""
    
    word: str = Field(description="The transcribed word")
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    probability: float = Field(
        default=1.0, 
        ge=0.0, 
        le=1.0, 
        description="Confidence probability for this word"
    )


class TranscriptSegmentModel(BaseModel):
    """Transcript segment with timing and confidence information."""
    
    id: int = Field(description="Segment identifier")
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    text: str = Field(description="Transcribed text for this segment")
    confidence: float = Field(
        ge=0.0, 
        le=1.0, 
        description="Confidence score for this segment"
    )
    words: List[WordSegmentModel] = Field(
        default_factory=list, 
        description="Word-level segments (if word_timestamps enabled)"
    )
    
    @property
    def duration(self) -> float:
        """Get segment duration in seconds."""
        return self.end - self.start


class VoiceSegmentModel(BaseModel):
    """Voice activity segment."""
    
    start_time: float = Field(description="Start time in seconds")
    end_time: float = Field(description="End time in seconds")
    confidence: float = Field(
        ge=0.0, 
        le=1.0, 
        description="Confidence score for voice activity"
    )
    is_speech: bool = Field(description="Whether this segment contains speech")
    
    @property
    def duration(self) -> float:
        """Get segment duration in seconds."""
        return self.end_time - self.start_time


class SpeakerSegmentModel(BaseModel):
    """Speaker diarization segment."""
    
    speaker: str = Field(description="Speaker identifier")
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    confidence: float = Field(
        default=1.0, 
        ge=0.0, 
        le=1.0, 
        description="Confidence score for speaker identification"
    )
    
    @property
    def duration(self) -> float:
        """Get segment duration in seconds."""
        return self.end - self.start


class PipelineInfo(BaseModel):
    """Information about the processing pipeline stages."""
    
    separation: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Vocal separation information and metadata"
    )
    vad: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Voice activity detection information and metadata"
    )
    transcription: Dict[str, Any] = Field(
        description="Transcription information and metadata"
    )
    diarization: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Speaker diarization information and metadata"
    )


class TimingMetricsModel(BaseModel):
    """Performance timing metrics."""
    
    operation_name: str = Field(description="Name of the operation")
    duration_ms: float = Field(description="Duration in milliseconds")
    memory_before_mb: Optional[float] = Field(
        default=None, 
        description="Memory usage before operation (MB)"
    )
    memory_after_mb: Optional[float] = Field(
        default=None, 
        description="Memory usage after operation (MB)"
    )
    memory_delta_mb: Optional[float] = Field(
        default=None, 
        description="Memory delta (MB)"
    )
    success: bool = Field(description="Whether the operation succeeded")
    error_message: Optional[str] = Field(
        default=None, 
        description="Error message if operation failed"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional operation metadata"
    )


class QualityMetricsModel(BaseModel):
    """Quality metrics for processing results."""
    
    transcription: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Transcription quality metrics"
    )
    separation: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Separation quality metrics"
    )
    diarization: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Diarization quality metrics"
    )
    vad: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="VAD quality metrics"
    )


class TranscribeResponse(BaseModel):
    """Complete response model for transcription requests."""
    
    # Core results
    text: str = Field(description="Complete transcribed text")
    language: str = Field(description="Detected or specified language")
    language_probability: float = Field(
        ge=0.0, 
        le=1.0, 
        description="Confidence in language detection"
    )
    segments: List[TranscriptSegmentModel] = Field(
        description="Transcribed segments with timing information"
    )
    
    # Optional results
    voice_segments: Optional[List[VoiceSegmentModel]] = Field(
        default=None, 
        description="Voice activity segments (if VAD enabled)"
    )
    speaker_segments: Optional[List[SpeakerSegmentModel]] = Field(
        default=None, 
        description="Speaker diarization segments (if diarization enabled)"
    )
    
    # Processing metadata
    pipeline_info: PipelineInfo = Field(description="Information about processing pipeline")
    processing_time: float = Field(description="Total processing time in seconds")
    device_info: Dict[str, Any] = Field(description="Device and system information")
    
    # Quality and performance metrics
    quality_metrics: QualityMetricsModel = Field(description="Quality assessment metrics")
    timing_metrics: List[TimingMetricsModel] = Field(
        default_factory=list, 
        description="Detailed timing metrics for each operation"
    )
    
    # Request metadata
    request_id: str = Field(description="Unique request identifier")
    timestamp: datetime = Field(description="Request processing timestamp")
    config: TranscribeConfig = Field(description="Processing configuration used")
    
    @property
    def has_speaker_info(self) -> bool:
        """Check if response has speaker diarization information."""
        return self.speaker_segments is not None and len(self.speaker_segments) > 0
    
    @property
    def has_word_timestamps(self) -> bool:
        """Check if response has word-level timestamps."""
        return any(segment.words for segment in self.segments)
    
    @property
    def total_duration(self) -> float:
        """Get total audio duration in seconds."""
        if self.segments:
            return max(segment.end for segment in self.segments)
        return 0.0


class ErrorResponse(BaseModel):
    """Error response model."""
    
    model_config = {
        "json_encoders": {datetime: lambda v: v.isoformat()},
        "json_schema_serialization_defaults_required": True
    }
    
    error: str = Field(description="Error type or code")
    message: str = Field(description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Additional error details"
    )
    request_id: str = Field(description="Request identifier for tracking")
    timestamp: datetime = Field(description="Error timestamp")


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: Literal["healthy", "unhealthy"] = Field(description="Service health status")
    timestamp: datetime = Field(description="Health check timestamp")
    version: str = Field(description="Service version")
    uptime: float = Field(description="Service uptime in seconds")
    services: Dict[str, Dict[str, Any]] = Field(description="Individual service health status")


class MetricsResponse(BaseModel):
    """Performance metrics response model."""
    
    timestamp: datetime = Field(description="Metrics timestamp")
    system_resources: Dict[str, Any] = Field(description="System resource usage")
    performance_summary: Dict[str, Any] = Field(description="Performance summary")
    operation_stats: Dict[str, Any] = Field(description="Operation statistics")
    memory_stats: Dict[str, Any] = Field(description="Memory usage statistics")


class CapabilitiesResponse(BaseModel):
    """Service capabilities response model."""
    
    device: str = Field(description="Processing device (CPU/GPU)")
    device_info: Dict[str, Any] = Field(description="Device information")
    services: Dict[str, Dict[str, Any]] = Field(description="Available services and methods")
    configuration: Dict[str, Any] = Field(description="Service configuration")
    supported_languages: List[str] = Field(description="Supported language codes")
    max_file_size_mb: int = Field(description="Maximum supported file size in MB")


# Utility functions for model conversion

def convert_processing_result_to_response(
    result: ProcessingResult, 
    request_id: str, 
    config: TranscribeConfig
) -> TranscribeResponse:
    """Convert ProcessingResult to TranscribeResponse."""
    
    # Convert transcript segments
    segments = []
    for segment in result.transcription.segments:
        words = []
        for word in segment.words:
            words.append(WordSegmentModel(
                word=word.word,
                start=float(word.start),
                end=float(word.end),
                probability=float(word.probability)
            ))
        
        segments.append(TranscriptSegmentModel(
            id=segment.id,
            start=float(segment.start),
            end=float(segment.end),
            text=segment.text,
            confidence=float(segment.confidence),
            words=words
        ))
    
    # Convert voice segments
    voice_segments = None
    if result.voice_segments:
        voice_segments = [
            VoiceSegmentModel(
                start_time=float(seg.start_time),
                end_time=float(seg.end_time),
                confidence=float(seg.confidence),
                is_speech=seg.is_speech
            )
            for seg in result.voice_segments
        ]
    
    # Convert speaker segments
    speaker_segments = None
    if result.diarization and result.diarization.segments:
        speaker_segments = [
            SpeakerSegmentModel(
                speaker=seg.speaker,
                start=float(seg.start),
                end=float(seg.end),
                confidence=float(seg.confidence)
            )
            for seg in result.diarization.segments
        ]
    
    # Build pipeline info
    pipeline_info = PipelineInfo(
        transcription=convert_numpy_types(result.transcription.metadata),
        separation=convert_numpy_types(result.separation.metadata) if result.separation else None,
        vad={"enabled": result.voice_segments is not None} if result.voice_segments else None,
        diarization=convert_numpy_types(result.diarization.metadata) if result.diarization else None
    )
    
    # Convert timing metrics
    timing_metrics = []
    if "operations" in result.performance_metrics:
        for op_name, op_stats in result.performance_metrics["operations"].items():
            timing_metrics.append(TimingMetricsModel(
                operation_name=op_name,
                duration_ms=float(op_stats.get("avg_duration_ms", 0.0)),
                success=float(op_stats.get("success_rate", 0.0)) > 0.5,
                metadata=convert_numpy_types(op_stats)
            ))
    
    return TranscribeResponse(
        text=result.transcription.text,
        language=result.transcription.language,
        language_probability=float(result.transcription.language_probability),
        segments=segments,
        voice_segments=voice_segments,
        speaker_segments=speaker_segments,
        pipeline_info=pipeline_info,
        processing_time=float(result.processing_time),
        device_info=result.device_info.__dict__ if hasattr(result.device_info, '__dict__') else result.device_info,
        quality_metrics=QualityMetricsModel(**convert_numpy_types(result.quality_metrics)),
        timing_metrics=timing_metrics,
        request_id=request_id,
        timestamp=datetime.utcnow(),
        config=config
    )
