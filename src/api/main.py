"""
FastAPI application entry point for the audio transcription service.

This module creates and configures the FastAPI application with all routes,
middleware, exception handlers, and OpenAPI documentation.
"""

import time
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..core.config import settings
from ..core.logging import get_logger, setup_logging
from ..core.exceptions import AudioTranscriptionError
from ..utils.timing import clear_performance_metrics
from .routes import router
from .middleware import setup_middleware
from .models import ErrorResponse

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events for the FastAPI application.
    """
    # Startup
    logger.info("Starting Audio Transcription Service")
    logger.info(f"Version: {settings.app_version}")
    logger.info(f"Debug mode: {settings.debug_mode}")
    logger.info(f"GPU enabled: {settings.enable_gpu}")
    
    # Clear any existing performance metrics
    clear_performance_metrics()
    
    # Initialize services (lazy loading will handle actual initialization)
    try:
        from ..services.audio_processor import get_processor_instance
        processor = get_processor_instance()
        capabilities = processor.get_processing_capabilities()
        logger.info(f"Audio processor initialized on device: {capabilities['device']}")
        logger.info(f"Available services: {list(capabilities['services'].keys())}")
    except Exception as e:
        logger.error(f"Failed to initialize audio processor: {e}")
    
    logger.info("Service startup completed")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Audio Transcription Service")
    logger.info("Service shutdown completed")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    # Setup logging
    setup_logging()
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="""
        Audio Transcription Microservice
        
        A high-performance audio transcription service that provides:
        
        * **Speech-to-Text Transcription** using OpenAI's Whisper models
        * **Vocal Separation** using Demucs for improved transcription quality
        * **Voice Activity Detection** to remove silence and noise
        * **Speaker Diarization** to identify different speakers
        * **Multi-language Support** with automatic language detection
        * **Word-level Timestamps** for precise timing information
        * **Performance Monitoring** with detailed metrics
        
        ## Features
        
        - Support for multiple audio formats (WAV, MP3, FLAC, etc.)
        - Configurable processing pipeline with optional components
        - GPU acceleration when available
        - Comprehensive error handling and logging
        - RESTful API with OpenAPI documentation
        - Health checks and performance metrics
        """,
        docs_url="/docs" if settings.debug_mode else None,
        redoc_url="/redoc" if settings.debug_mode else None,
        openapi_url="/openapi.json" if settings.debug_mode else None,
        lifespan=lifespan
    )
    
    # Add CORS middleware - allow all origins for now
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # Setup custom middleware
    app = setup_middleware(app, enable_rate_limiting=not settings.debug_mode)
    
    # Include API routes
    app.include_router(router, prefix=settings.api_v1_prefix)
    
    # Add exception handlers
    setup_exception_handlers(app)
    
    # Customize OpenAPI schema
    customize_openapi_schema(app)
    
    return app


def setup_exception_handlers(app: FastAPI):
    """Setup custom exception handlers for the application."""
    
    @app.exception_handler(AudioTranscriptionError)
    async def audio_transcription_error_handler(request: Request, exc: AudioTranscriptionError):
        """Handle custom audio transcription errors."""
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        logger.warning(
            f"Audio transcription error: {exc}",
            extra={
                "request_id": request_id,
                "error_code": exc.error_code,
                "status_code": exc.status_code
            }
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=exc.error_code,
                message=str(exc),
                details=exc.details,
                request_id=request_id,
                timestamp=time.time()
            ).model_dump()
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors."""
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        logger.warning(
            f"Validation error: {exc}",
            extra={"request_id": request_id}
        )
        
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "message": "Request validation failed",
                "details": {"errors": exc.errors()},
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions."""
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        logger.warning(
            f"HTTP error: {exc.status_code} - {exc.detail}",
            extra={"request_id": request_id, "status_code": exc.status_code}
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "http_error",
                "message": str(exc.detail),
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        logger.error(
            f"Unexpected error: {exc}",
            extra={"request_id": request_id},
            exc_info=True
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred",
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }
        )


def customize_openapi_schema(app: FastAPI):
    """Customize the OpenAPI schema with additional information."""
    
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title=settings.app_name,
            version=settings.app_version,
            description=app.description,
            routes=app.routes,
        )
        
        # Add custom information
        openapi_schema["info"]["contact"] = {
            "name": "Audio Transcription Service",
            "url": "https://github.com/your-org/audio-transcription-service",
        }
        
        openapi_schema["info"]["license"] = {
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        }
        
        # Add server information
        openapi_schema["servers"] = [
            {
                "url": f"http://{settings.host}:{settings.port}",
                "description": "Development server"
            }
        ]
        
        # Add security schemes
        openapi_schema["components"]["securitySchemes"] = {
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "API key for authentication (optional)"
            }
        }
        
        # Add examples for request/response models
        add_examples_to_schema(openapi_schema)
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi


def add_examples_to_schema(openapi_schema: dict):
    """Add examples to OpenAPI schema for better documentation."""
    
    # Example for TranscribeConfig
    if "components" in openapi_schema and "schemas" in openapi_schema["components"]:
        schemas = openapi_schema["components"]["schemas"]
        
        if "TranscribeConfig" in schemas:
            schemas["TranscribeConfig"]["example"] = {
                "enable_separation": True,
                "enable_vad": True,
                "enable_diarization": False,
                "language": "en",
                "task": "transcribe",
                "word_timestamps": True,
                "separation_method": "auto",
                "vad_method": "auto",
                "vad_threshold": 0.5,
                "chunk_long_audio": True,
                "chunk_size_seconds": 30,
                "chunk_overlap_seconds": 2.0
            }
        
        if "TranscribeResponse" in schemas:
            schemas["TranscribeResponse"]["example"] = {
                "text": "Hello, this is a sample transcription.",
                "language": "en",
                "language_probability": 0.95,
                "segments": [
                    {
                        "id": 0,
                        "start": 0.0,
                        "end": 2.5,
                        "text": "Hello, this is a sample transcription.",
                        "confidence": 0.92,
                        "words": [
                            {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 0.98},
                            {"word": "this", "start": 0.6, "end": 0.8, "probability": 0.95},
                            {"word": "is", "start": 0.9, "end": 1.0, "probability": 0.99},
                            {"word": "a", "start": 1.1, "end": 1.2, "probability": 0.97},
                            {"word": "sample", "start": 1.3, "end": 1.8, "probability": 0.94},
                            {"word": "transcription", "start": 1.9, "end": 2.5, "probability": 0.91}
                        ]
                    }
                ],
                "processing_time": 3.2,
                "request_id": "123e4567-e89b-12d3-a456-426614174000",
                "timestamp": "2024-01-01T12:00:00Z"
            }


# Create the application instance
app = create_app()


# Health check endpoint for load balancers
@app.get("/health")
async def health_check():
    """Simple health check endpoint for load balancers."""
    return {"status": "healthy", "timestamp": time.time()}


if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug_mode,
        workers=1 if settings.debug_mode else settings.workers,
        log_level=settings.log_level.lower(),
        access_log=True
    )
