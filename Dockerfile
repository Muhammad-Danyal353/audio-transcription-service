# Multi-stage build for optimized image size
# Stage 1: Builder - Install dependencies and download models
FROM python:3.11.9-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies required for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download Whisper models to avoid download during runtime
# This significantly reduces cold start time
RUN python -c "import whisper; whisper.load_model('base')" && \
    python -c "import whisper; whisper.load_model('small')"

# Pre-download Demucs models
RUN python -c "from demucs.pretrained import get_model; get_model('htdemucs')"

# Stage 2: Runtime - Create minimal runtime image
FROM python:3.11.9-slim as runtime

# Set working directory
WORKDIR /app

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy pre-downloaded models from builder
COPY --from=builder /root/.cache /root/.cache

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_DIR=/root/.cache \
    MAX_FILE_SIZE_MB=100 \
    WHISPER_MODEL_SIZE=small \
    TARGET_SAMPLE_RATE=16000 \
    ENABLE_GPU=auto

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

# Run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

