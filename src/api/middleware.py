"""
API middleware for request logging, timing, and error handling.

This module provides middleware components for the FastAPI application
including request logging, performance timing, and error standardization.
"""

import time
import uuid
from typing import Callable, Dict, Any
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ..core.logging import get_logger
from ..core.exceptions import AudioTranscriptionError
from ..utils.timing import time_context, get_performance_summary

logger = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware to generate and propagate request IDs.
    
    Adds a unique request ID to each request for tracking and logging purposes.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = get_logger(__name__)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and add request ID."""
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Add to request state
        request.state.request_id = request_id
        
        # Add to headers for response
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for comprehensive request logging.
    
    Logs request details, processing time, and response status.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = get_logger(__name__)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request details and processing time."""
        start_time = time.perf_counter()
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        # Log request start
        self.logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
                "content_length": request.headers.get("content-length")
            }
        )
        
        # Process request
        try:
            response = await call_next(request)
            processing_time = time.perf_counter() - start_time
            
            # Log successful response
            self.logger.info(
                "Request completed",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "processing_time": processing_time,
                    "response_size": response.headers.get("content-length")
                }
            )
            
            # Add timing header
            response.headers["X-Processing-Time"] = str(processing_time)
            
            return response
            
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            
            # Log error
            self.logger.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "processing_time": processing_time
                },
                exc_info=True
            )
            
            raise


class PerformanceTimingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for performance timing and monitoring.
    
    Tracks request processing time and integrates with the timing utility.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = get_logger(__name__)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Track request performance timing."""
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        # Start timing context
        with time_context(f"api_request_{request.method.lower()}", {
            "request_id": request_id,
            "endpoint": request.url.path,
            "method": request.method
        }):
            response = await call_next(request)
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for standardized error handling.
    
    Catches exceptions and converts them to standardized error responses.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = get_logger(__name__)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle errors and return standardized responses."""
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        try:
            response = await call_next(request)
            return response
            
        except AudioTranscriptionError as e:
            # Handle custom application errors
            self.logger.warning(
                "Application error",
                extra={
                    "request_id": request_id,
                    "error_code": e.error_code,
                    "error_message": str(e)
                }
            )
            
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": e.error_code,
                    "message": str(e),
                    "details": e.details,
                    "request_id": request_id,
                    "timestamp": time.time()
                }
            )
            
        except ValueError as e:
            # Handle validation errors
            self.logger.warning(
                "Validation error",
                extra={
                    "request_id": request_id,
                    "error_message": str(e)
                }
            )
            
            return JSONResponse(
                status_code=400,
                content={
                    "error": "validation_error",
                    "message": str(e),
                    "request_id": request_id,
                    "timestamp": time.time()
                }
            )
            
        except FileNotFoundError as e:
            # Handle file not found errors
            self.logger.warning(
                "File not found",
                extra={
                    "request_id": request_id,
                    "error_message": str(e)
                }
            )
            
            return JSONResponse(
                status_code=404,
                content={
                    "error": "file_not_found",
                    "message": "Requested file not found",
                    "request_id": request_id,
                    "timestamp": time.time()
                }
            )
            
        except Exception as e:
            # Handle unexpected errors
            self.logger.error(
                "Unexpected error",
                extra={
                    "request_id": request_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                },
                exc_info=True
            )
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": "internal_server_error",
                    "message": "An unexpected error occurred",
                    "request_id": request_id,
                    "timestamp": time.time()
                }
            )


class CORSHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware for CORS headers.
    
    Adds appropriate CORS headers to responses.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = get_logger(__name__)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add CORS headers to response."""
        response = await call_next(request)
        
        # Add CORS headers
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Max-Age"] = "86400"
        
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware for security headers.
    
    Adds security-related headers to responses.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = get_logger(__name__)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response."""
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """
    Basic rate limiting middleware.
    
    Implements simple rate limiting based on client IP.
    """
    
    def __init__(self, app: ASGIApp, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.client_requests: Dict[str, list] = {}
        self.logger = get_logger(__name__)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply rate limiting to requests."""
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Clean old requests (older than 1 minute)
        if client_ip in self.client_requests:
            self.client_requests[client_ip] = [
                req_time for req_time in self.client_requests[client_ip]
                if current_time - req_time < 60
            ]
        else:
            self.client_requests[client_ip] = []
        
        # Check rate limit
        if len(self.client_requests[client_ip]) >= self.requests_per_minute:
            self.logger.warning(
                "Rate limit exceeded",
                extra={
                    "client_ip": client_ip,
                    "requests_count": len(self.client_requests[client_ip])
                }
            )
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Rate limit exceeded. Maximum {self.requests_per_minute} requests per minute.",
                    "timestamp": current_time
                },
                headers={"Retry-After": "60"}
            )
        
        # Add current request
        self.client_requests[client_ip].append(current_time)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = self.requests_per_minute - len(self.client_requests[client_ip])
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(current_time + 60))
        
        return response


def setup_middleware(app: ASGIApp, enable_rate_limiting: bool = True) -> ASGIApp:
    """
    Setup all middleware for the FastAPI application.
    
    Args:
        app: FastAPI application instance
        enable_rate_limiting: Whether to enable rate limiting
        
    Returns:
        Application with middleware configured
    """
    # Add middleware in reverse order (last added is first executed)
    
    # Security headers (outermost)
    app.add_middleware(SecurityHeadersMiddleware)
    
    # CORS headers
    app.add_middleware(CORSHeadersMiddleware)
    
    # Rate limiting (if enabled)
    if enable_rate_limiting:
        app.add_middleware(RateLimitingMiddleware, requests_per_minute=60)
    
    # Error handling
    app.add_middleware(ErrorHandlingMiddleware)
    
    # Performance timing
    app.add_middleware(PerformanceTimingMiddleware)
    
    # Request logging
    app.add_middleware(RequestLoggingMiddleware)
    
    # Request ID (innermost)
    app.add_middleware(RequestIDMiddleware)
    
    return app
