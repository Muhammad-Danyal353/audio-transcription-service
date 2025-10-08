"""
API routes for the audio transcription service.

This module defines all FastAPI endpoints including the main transcription
endpoint, health checks, and metrics endpoints.
"""

import asyncio
import tempfile
import os
from typing import Optional
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request, Depends
from fastapi.responses import JSONResponse

from ..core.config import settings
from ..core.logging import get_logger
from ..core.exceptions import AudioTranscriptionError
from ..services.audio_processor import get_processor_instance, ProcessingConfig
from ..utils.audio_utils import validate_audio_file, detect_audio_format
from ..utils.timing import get_performance_summary, get_system_resources
from .models import (
    TranscribeConfig, TranscribeResponse, ErrorResponse, HealthResponse,
    MetricsResponse, CapabilitiesResponse, convert_processing_result_to_response
)

logger = get_logger(__name__)

# Create router
router = APIRouter()

# Service startup time for uptime calculation
_startup_time = None


def get_startup_time():
    """Get service startup time."""
    global _startup_time
    if _startup_time is None:
        import time
        _startup_time = time.time()
    return _startup_time


@router.post(
    "/transcribe",
    response_model=TranscribeResponse,
    summary="Transcribe audio file",
    description="Transcribe an audio file to text with optional vocal separation, VAD, and speaker diarization"
)
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(..., description="Audio file to transcribe"),
    config: Optional[str] = Form(None, description="JSON configuration string"),
    enable_separation: Optional[bool] = Form(None, description="Enable vocal separation"),
    enable_vad: Optional[bool] = Form(None, description="Enable voice activity detection"),
    enable_diarization: Optional[bool] = Form(None, description="Enable speaker diarization"),
    language: Optional[str] = Form(None, description="Language code for transcription"),
    word_timestamps: Optional[bool] = Form(None, description="Include word-level timestamps")
) -> TranscribeResponse:
    """
    Transcribe audio file to text.
    
    This endpoint accepts an audio file and optional configuration parameters
    to perform speech-to-text transcription with various processing options.
    
    Args:
        file: Audio file to transcribe (supports WAV, MP3, FLAC, etc.)
        config: JSON string with processing configuration
        enable_separation: Enable vocal separation
        enable_vad: Enable voice activity detection
        enable_diarization: Enable speaker diarization
        language: Language code (e.g., 'en', 'es', 'fr')
        word_timestamps: Include word-level timestamps
        
    Returns:
        TranscribeResponse with transcription results and metadata
        
    Raises:
        HTTPException: For various error conditions (400, 413, 422, 500)
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="No file provided"
            )
        
        # Check file size
        file_size = 0
        if hasattr(file, 'size') and file.size:
            file_size = file.size
        elif hasattr(file, 'content_length') and file.content_length:
            file_size = file.content_length
        
        if file_size > settings.max_file_size_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB"
            )
        
        # Parse configuration
        processing_config = TranscribeConfig()
        
        # Override with form parameters if provided
        if enable_separation is not None:
            processing_config.enable_separation = enable_separation
        if enable_vad is not None:
            processing_config.enable_vad = enable_vad
        if enable_diarization is not None:
            processing_config.enable_diarization = enable_diarization
        if language is not None:
            processing_config.language = language
        if word_timestamps is not None:
            processing_config.word_timestamps = word_timestamps
        
        # Parse JSON config if provided
        if config:
            try:
                import json
                config_dict = json.loads(config)
                processing_config = TranscribeConfig(**config_dict)
            except (json.JSONDecodeError, ValueError) as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid configuration JSON: {str(e)}"
                )
        
        # Validate configuration
        processor = get_processor_instance()
        config_errors = processor.validate_config(processing_config)
        if config_errors:
            raise HTTPException(
                status_code=422,
                detail=f"Configuration validation failed: {'; '.join(config_errors)}"
            )
        
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Validate audio file
            audio_info = detect_audio_format(tmp_file_path)
            logger.info(
                f"Processing audio file: {file.filename}",
                extra={
                    "request_id": request_id,
                    "file_size": len(content),
                    "duration": audio_info.get("duration", 0),
                    "sample_rate": audio_info.get("sample_rate", 0),
                    "channels": audio_info.get("channels", 0)
                }
            )
            
            # Process audio
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: processor.process_file(tmp_file_path, processing_config)
            )
            
            # Convert result to response model
            response = convert_processing_result_to_response(
                result, request_id, processing_config
            )
            
            logger.info(
                f"Transcription completed successfully",
                extra={
                    "request_id": request_id,
                    "processing_time": result.processing_time,
                    "text_length": len(result.transcription.text),
                    "segments_count": len(result.transcription.segments),
                    "has_speakers": response.has_speaker_info
                }
            )
            
            return response
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except OSError:
                pass
                
    except HTTPException:
        raise
    except AudioTranscriptionError as e:
        logger.error(
            f"Audio processing error: {e}",
            extra={"request_id": request_id, "error_code": e.error_code}
        )
        raise HTTPException(status_code=e.status_code, detail=str(e))
    except Exception as e:
        logger.error(
            f"Unexpected error during transcription: {e}",
            extra={"request_id": request_id},
            exc_info=True
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of the service and its components"
)
async def health_check() -> HealthResponse:
    """
    Perform health check on the service.
    
    Returns:
        HealthResponse with service health status and component information
    """
    import time
    from datetime import datetime
    
    current_time = time.time()
    startup_time = get_startup_time()
    uptime = current_time - startup_time
    
    # Check service components
    services_status = {}
    overall_healthy = True
    
    try:
        # Check audio processor
        processor = get_processor_instance()
        capabilities = processor.get_processing_capabilities()
        
        services_status["audio_processor"] = {
            "status": "healthy",
            "device": capabilities["device"],
            "services_available": {
                "vad": capabilities["services"]["vad"]["available"],
                "separation": capabilities["services"]["separation"]["available"],
                "transcription": capabilities["services"]["transcription"]["available"],
                "diarization": capabilities["services"]["diarization"]["available"]
            }
        }
        
    except Exception as e:
        services_status["audio_processor"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        overall_healthy = False
    
    # Check system resources
    try:
        system_resources = get_system_resources()
        if "error" in system_resources:
            services_status["system_resources"] = {
                "status": "warning",
                "message": "Limited system resource monitoring"
            }
        else:
            services_status["system_resources"] = {
                "status": "healthy",
                "memory_usage_percent": system_resources.get("memory", {}).get("percent", 0)
            }
    except Exception as e:
        services_status["system_resources"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        overall_healthy = False
    
    return HealthResponse(
        status="healthy" if overall_healthy else "unhealthy",
        timestamp=datetime.utcnow(),
        version=settings.app_version,
        uptime=uptime,
        services=services_status
    )


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Performance metrics",
    description="Get detailed performance metrics and system resource usage"
)
async def get_metrics() -> MetricsResponse:
    """
    Get performance metrics and system resource usage.
    
    Returns:
        MetricsResponse with comprehensive performance data
    """
    from datetime import datetime
    
    try:
        # Get performance summary
        performance_summary = get_performance_summary()
        
        # Get system resources
        system_resources = get_system_resources()
        
        # Get operation statistics
        from ..utils.timing import get_operation_stats
        operation_stats = get_operation_stats()
        
        # Get memory statistics
        from ..utils.timing import get_memory_stats
        memory_stats = get_memory_stats()
        
        return MetricsResponse(
            timestamp=datetime.utcnow(),
            system_resources=system_resources,
            performance_summary=performance_summary,
            operation_stats=operation_stats,
            memory_stats=memory_stats
        )
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


@router.get(
    "/capabilities",
    response_model=CapabilitiesResponse,
    summary="Service capabilities",
    description="Get information about service capabilities and configuration"
)
async def get_capabilities() -> CapabilitiesResponse:
    """
    Get service capabilities and configuration information.
    
    Returns:
        CapabilitiesResponse with service capabilities and supported features
    """
    try:
        processor = get_processor_instance()
        capabilities = processor.get_processing_capabilities()
        
        # Get supported languages
        supported_languages = []
        if capabilities["services"]["transcription"]["available"]:
            supported_languages = capabilities["services"]["transcription"]["supported_languages"]
        
        return CapabilitiesResponse(
            device=capabilities["device"],
            device_info=capabilities["device_info"],
            services=capabilities["services"],
            configuration=capabilities["configuration"],
            supported_languages=supported_languages,
            max_file_size_mb=settings.max_file_size_mb
        )
        
    except Exception as e:
        logger.error(f"Failed to get capabilities: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve capabilities")


@router.get(
    "/",
    summary="Service information",
    description="Get basic service information and available endpoints"
)
async def root() -> dict:
    """
    Get basic service information.
    
    Returns:
        Dictionary with service information and available endpoints
    """
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "description": "Audio transcription microservice with vocal separation, VAD, and speaker diarization",
        "endpoints": {
            "transcribe": "/v1/transcribe",
            "health": "/v1/health",
            "metrics": "/v1/metrics",
            "capabilities": "/v1/capabilities",
            "docs": "/docs",
            "openapi": "/openapi.json"
        },
        "features": [
            "Speech-to-text transcription",
            "Vocal separation",
            "Voice activity detection",
            "Speaker diarization",
            "Multi-language support",
            "Word-level timestamps",
            "Performance monitoring"
        ]
    }


@router.options("/{path:path}")
async def options_handler(path: str) -> JSONResponse:
    """
    Handle OPTIONS requests for CORS preflight.
    
    Args:
        path: Request path
        
    Returns:
        JSONResponse with CORS headers
    """
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "86400"
        }
    )
