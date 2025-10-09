from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class TranscribeConfig(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    language_hint: Optional[str] = Field(default=None, description="Language code hint, e.g., 'en'")
    enable_separation: bool = Field(default=True, description="Run separation/noise reduction before ASR")
    diarize: bool = Field(default=False, description="Optional diarization flag (not implemented in minimal)")
    model_size: str = Field(default="small", description="Whisper/faster-whisper model size")
    target_sr: int = Field(default=16000, description="Target sample rate for processing")


class Segment(BaseModel):
    start: float
    end: float
    text: str


class PipelineInfo(BaseModel):
    separation: Dict[str, Any]
    transcription: Dict[str, Any]


class TranscribeResponse(BaseModel):
    request_id: str
    duration_sec: float
    sample_rate: int
    pipeline: PipelineInfo
    segments: List[Segment]
    text: str
    language: str
    timings_ms: Dict[str, float]