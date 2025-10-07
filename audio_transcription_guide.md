# Audio Transcription Microservice - Implementation Guide

## 📁 Project Structure

```
audio-transcription-service/
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI application entry
│   │   ├── routes.py               # API endpoints
│   │   ├── models.py               # Pydantic request/response models
│   │   └── middleware.py           # Logging, error handling
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py               # Configuration management
│   │   ├── logging.py              # Structured logging setup
│   │   └── exceptions.py           # Custom exceptions
│   ├── services/
│   │   ├── __init__.py
│   │   ├── audio_processor.py      # Main pipeline orchestrator
│   │   ├── separator.py            # Vocal/noise separation
│   │   ├── transcriber.py          # ASR (Whisper)
│   │   ├── vad.py                  # Voice Activity Detection
│   │   └── diarization.py          # Speaker diarization (optional)
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── audio_utils.py          # Audio I/O, format conversion
│   │   ├── device_utils.py         # GPU/CPU detection
│   │   └── timing.py               # Performance metrics
│   └── cli/
│       ├── __init__.py
│       └── client.py               # CLI utility
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Pytest fixtures
│   ├── test_api.py                 # API endpoint tests
│   ├── test_separator.py           # Separation tests
│   ├── test_transcriber.py         # Transcription tests
│   └── test_integration.py         # End-to-end tests
├── sample_audio/
│   ├── clean_speech.wav            # Clear single speaker
│   ├── noisy_speech.wav            # Background noise
│   ├── music_with_vocals.mp3       # Music separation test
│   ├── multi_speaker.wav           # Multiple speakers
│   └── README.md                   # Expected behavior documentation
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── docs/
│   ├── ADR.md                      # Architecture Decision Record
│   ├── API.md                      # API documentation
│   └── openapi.yaml                # OpenAPI specification
├── .env.example
├── .gitignore
├── .dockerignore
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml                  # ruff, mypy, black config
├── README.md
└── PROGRESS.md                     # Step-by-step tracking
```

---

## 🎯 Implementation Roadmap

### Phase 1: Project Setup (Day 1)

#### ✅ Step 1.1: Initialize Repository
- [ ] Create GitHub repository
- [ ] Clone locally
- [ ] Create initial project structure
- [ ] Setup `.gitignore` (Python, IDE, models/, .env)
- [ ] Create `README.md` with basic overview

#### ✅ Step 1.2: Environment Setup
- [ ] Create `requirements.txt`:
  ```
  fastapi==0.104.1
  uvicorn[standard]==0.24.0
  python-multipart==0.0.6
  pydantic==2.5.0
  pydantic-settings==2.1.0
  torch==2.1.0
  torchaudio==2.1.0
  openai-whisper==20231117
  demucs==4.0.1
  librosa==0.10.1
  soundfile==0.12.1
  numpy==1.24.3
  python-dotenv==1.0.0
  httpx==0.25.0
  ```
- [ ] Create `requirements-dev.txt`:
  ```
  pytest==7.4.3
  pytest-asyncio==0.21.1
  pytest-cov==4.1.0
  httpx==0.25.0
  ruff==0.1.6
  black==23.11.0
  mypy==1.7.0
  ```
- [ ] Create `pyproject.toml` for tooling config
- [ ] Setup virtual environment: `python -m venv venv`

#### ✅ Step 1.3: Docker Setup
- [ ] Create `docker/Dockerfile` (multi-stage build)
- [ ] Create `docker/docker-compose.yml`
- [ ] Create `.dockerignore`
- [ ] Test basic Docker build

---

### Phase 2: Core Infrastructure (Day 1-2)

#### ✅ Step 2.1: Configuration Management
**File: `src/core/config.py`**
- [ ] Create Settings class with pydantic-settings
- [ ] Environment variables: `MODEL_DIR`, `MAX_FILE_SIZE`, `ENABLE_GPU`
- [ ] Model size configurations: `WHISPER_MODEL_SIZE`
- [ ] Audio processing params: `TARGET_SAMPLE_RATE`, `CHUNK_SIZE`

#### ✅ Step 2.2: Logging & Observability
**File: `src/core/logging.py`**
- [ ] Setup structured JSON logging
- [ ] Request ID generation and propagation
- [ ] Performance timing decorators
- [ ] Log levels based on environment

#### ✅ Step 2.3: Device Detection
**File: `src/utils/device_utils.py`**
- [ ] GPU availability check (CUDA/MPS)
- [ ] Auto-select device (GPU → CPU fallback)
- [ ] Memory estimation utilities
- [ ] Quantization support detection

#### ✅ Step 2.4: Audio Utilities
**File: `src/utils/audio_utils.py`**
- [ ] Audio format detection (ffprobe)
- [ ] Format conversion (ffmpeg wrapper)
- [ ] Resampling to target sample rate
- [ ] Audio validation (duration, channels, sample rate)
- [ ] Temporary file management

---

### Phase 3: Audio Processing Pipeline (Day 2-3)

#### ✅ Step 3.1: Voice Activity Detection
**File: `src/services/vad.py`**
- [ ] Implement WebRTC VAD or Silero VAD
- [ ] Chunk audio into voiced/unvoiced segments
- [ ] Remove silence with configurable threshold
- [ ] Return timestamps for voiced regions

#### ✅ Step 3.2: Vocal Separation
**File: `src/services/separator.py`**
- [ ] Initialize Demucs model (lazy loading)
- [ ] Implement `separate_vocals()` method
- [ ] Handle separation failures gracefully
- [ ] Fallback: basic noise suppression (spectral subtraction)
- [ ] Add timing metrics
- [ ] Support enable/disable via config

**Decision**: Use Demucs hybrid transformer (htdemucs) for quality vs speed balance

#### ✅ Step 3.3: ASR Transcription
**File: `src/services/transcriber.py`**
- [ ] Initialize Whisper model (configurable size: tiny/base/small/medium)
- [ ] Implement `transcribe()` method
- [ ] Handle language detection/hints
- [ ] Return segments with timestamps
- [ ] Add word-level timestamps if available
- [ ] Implement chunking for long audio (>30s)
- [ ] Overlap stitching for chunk boundaries

**Decision**: Use Whisper Small as default (good accuracy/speed trade-off)

#### ✅ Step 3.4: Speaker Diarization (Optional)
**File: `src/services/diarization.py`**
- [ ] Integrate pyannote.audio (if time permits)
- [ ] Map speaker labels to transcript segments
- [ ] Handle HuggingFace token requirement
- [ ] Make fully optional with feature flag

#### ✅ Step 3.5: Pipeline Orchestrator
**File: `src/services/audio_processor.py`**
- [ ] Create `AudioProcessor` class
- [ ] Implement full pipeline:
  1. Validate & decode audio
  2. Normalize audio (resample, convert to mono)
  3. Run separation (if enabled)
  4. Apply VAD (optional)
  5. Transcribe audio
  6. Apply diarization (if enabled)
  7. Format response
- [ ] Add comprehensive error handling
- [ ] Track timing for each stage
- [ ] Singleton pattern for model loading

---

### Phase 4: API Layer (Day 3-4)

#### ✅ Step 4.1: Data Models
**File: `src/api/models.py`**
- [ ] `TranscribeConfig` (request config)
- [ ] `TranscriptSegment` (segment with timestamps)
- [ ] `PipelineInfo` (separation & transcription metadata)
- [ ] `TimingMetrics` (performance metrics)
- [ ] `TranscribeResponse` (complete response model)
- [ ] `ErrorResponse` (error details)

#### ✅ Step 4.2: API Routes
**File: `src/api/routes.py`**
- [ ] `POST /v1/transcribe` endpoint
- [ ] Multipart form handling (file + config JSON)
- [ ] File size validation (413 for oversized)
- [ ] Format validation (400 for invalid)
- [ ] Request ID generation
- [ ] Async processing
- [ ] Response formatting

#### ✅ Step 4.3: FastAPI Application
**File: `src/api/main.py`**
- [ ] Create FastAPI app with metadata
- [ ] Include routes
- [ ] Add middleware (CORS, timing, logging)
- [ ] Exception handlers (400, 413, 422, 500)
- [ ] Health check endpoint: `GET /health`
- [ ] Metrics endpoint: `GET /metrics` (optional)
- [ ] Auto-generate OpenAPI docs

#### ✅ Step 4.4: Middleware & Error Handling
**File: `src/api/middleware.py`**
- [ ] Request logging middleware
- [ ] Timing middleware
- [ ] Error standardization
- [ ] Request ID propagation

---

### Phase 5: CLI Utility (Day 4)

#### ✅ Step 5.1: CLI Client
**File: `src/cli/client.py`**
- [ ] Argument parsing (file path, API URL, config options)
- [ ] File upload via multipart/form-data
- [ ] Response formatting (pretty print JSON)
- [ ] Progress indicator
- [ ] Error handling
- [ ] Make executable: `python -m src.cli.client`

**Usage Example**:
```bash
python -m src.cli.client \
  --file sample_audio/noisy_speech.wav \
  --url http://localhost:8000 \
  --enable-separation \
  --model-size small
```

---

### Phase 6: Testing (Day 4-5)

#### ✅ Step 6.1: Unit Tests
- [ ] `test_audio_utils.py`: Format conversion, resampling
- [ ] `test_separator.py`: Vocal separation logic
- [ ] `test_transcriber.py`: Transcription accuracy
- [ ] `test_vad.py`: Voice activity detection

#### ✅ Step 6.2: Integration Tests
- [ ] `test_api.py`: API endpoint behavior
  - Valid requests
  - Invalid file formats
  - File size limits
  - Missing files
- [ ] `test_integration.py`: Full pipeline with sample audio

#### ✅ Step 6.3: Sample Audio Preparation
**Directory: `sample_audio/`**
- [ ] Create/download 3-5 test clips (10-30s)
- [ ] Document expected behavior in `sample_audio/README.md`:
  - Clean speech → accurate transcription
  - Noisy speech → improved after separation
  - Music with vocals → vocal extraction works
  - Multi-speaker → diarization (if implemented)
  - Very noisy → graceful degradation

---

### Phase 7: Containerization (Day 5)

#### ✅ Step 7.1: Dockerfile Optimization
- [ ] Multi-stage build (builder + runtime)
- [ ] Layer caching for dependencies
- [ ] Pre-download models in build stage
- [ ] Non-root user
- [ ] Health check
- [ ] Minimize image size

**Example**:
```dockerfile
FROM python:3.10-slim as builder
# Install build dependencies, download models
FROM python:3.10-slim as runtime
# Copy only necessary artifacts
```

#### ✅ Step 7.2: Docker Compose
- [ ] Service definition
- [ ] Volume mounts for models (optional)
- [ ] Environment variables
- [ ] Port mapping
- [ ] Resource limits (optional)

#### ✅ Step 7.3: Docker Testing
- [ ] Build image: `docker build -t audio-transcribe:latest .`
- [ ] Run container: `docker-compose up`
- [ ] Test API from host
- [ ] Verify model loading
- [ ] Check performance (cold start, per-request)

---

### Phase 8: Documentation (Day 5-6)

#### ✅ Step 8.1: Architecture Decision Record
**File: `docs/ADR.md`**
- [ ] Pipeline stages diagram
- [ ] Model selection rationale
  - Whisper Small: 244M params, ~5x faster than Medium
  - Demucs htdemucs: Hybrid transformer, SOTA separation
- [ ] GPU/CPU strategy
- [ ] Memory management (singleton models)
- [ ] Concurrency model (uvicorn workers)
- [ ] Observability approach
- [ ] Failure modes & fallbacks

#### ✅ Step 8.2: API Documentation
**File: `docs/API.md`**
- [ ] Endpoint details
- [ ] Request/response examples
- [ ] Error codes
- [ ] Rate limits (if any)
- [ ] Performance expectations

#### ✅ Step 8.3: OpenAPI Specification
**File: `docs/openapi.yaml`**
- [ ] Export from FastAPI: `/openapi.json`
- [ ] Save as YAML
- [ ] Validate schema

#### ✅ Step 8.4: README.md
- [ ] Overview & features
- [ ] Quick start guide
- [ ] Setup instructions (local & Docker)
- [ ] API usage examples
- [ ] CLI usage examples
- [ ] Architecture overview
- [ ] Performance benchmarks
- [ ] Known limitations
- [ ] Contributing guidelines
- [ ] License

---

### Phase 9: Quality & Polish (Day 6)

#### ✅ Step 9.1: Code Quality
- [ ] Run `ruff check .` and fix issues
- [ ] Run `black .` for formatting
- [ ] Run `mypy src/` for type checking
- [ ] Add docstrings to public methods
- [ ] Remove debug code

#### ✅ Step 9.2: Performance Optimization
- [ ] Profile cold start time
- [ ] Profile per-request latency
- [ ] Optimize model loading (lazy vs eager)
- [ ] Add model caching
- [ ] Consider quantization (int8) for faster inference

#### ✅ Step 9.3: Final Testing
- [ ] Run full test suite: `pytest tests/ -v --cov=src`
- [ ] Test with all sample audio files
- [ ] Test both with/without GPU
- [ ] Load testing (optional): `locust` or `hey`
- [ ] Docker build and run test

#### ✅ Step 9.4: Security Review
- [ ] Input validation completeness
- [ ] File size limits enforced
- [ ] Temporary file cleanup
- [ ] No secrets in code
- [ ] Docker security best practices

---

## 📊 Key Decisions & Trade-offs

### Model Selection
| Component | Model | Size | Speed | Accuracy | Rationale |
|-----------|-------|------|-------|----------|-----------|
| ASR | Whisper Small | 244M | Medium | Good | Best accuracy/speed for production |
| Separation | Demucs htdemucs | 83M | Fast | Excellent | Hybrid transformer, CPU-friendly |
| VAD | Silero VAD | 1M | Very Fast | Good | Lightweight, no external deps |

### Performance Targets
- **Cold Start**: < 10s (model loading)
- **Per-Request (30s audio)**: < 15s on CPU, < 5s on GPU
- **Memory**: < 2GB RAM without GPU, < 4GB with GPU

### Fallback Strategy
1. Separation fails → Use raw audio + log warning
2. GPU unavailable → CPU inference (slower)
3. VAD fails → Process full audio
4. Diarization fails → Skip speaker labels

---

## 🚀 Running the Service

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run service
uvicorn src.api.main:app --reload --port 8000

# Run tests
pytest tests/ -v --cov=src
```

### Docker
```bash
# Build
docker build -f docker/Dockerfile -t audio-transcribe:latest .

# Run
docker-compose -f docker/docker-compose.yml up

# Test
curl -X POST "http://localhost:8000/v1/transcribe" \
  -F "file=@sample_audio/noisy_speech.wav" \
  -F 'config={"enable_separation": true}'
```

---

## 📝 Progress Tracking

Use this checklist to track implementation progress. Mark items as complete with `[x]`.

**Estimated Timeline**: 5-6 days for full implementation with testing and documentation.

**Priority Order**:
1. Core infrastructure (config, logging, utils)
2. Audio processing pipeline (transcriber → separator → vad)
3. API layer
4. Testing
5. Docker & documentation
6. CLI & polish

---

## 🎓 Learning Resources

- [Whisper Paper](https://arxiv.org/abs/2212.04356)
- [Demucs Documentation](https://github.com/facebookresearch/demucs)
- [FastAPI Best Practices](https://fastapi.tiangolo.com/tutorial/)
- [Docker Multi-stage Builds](https://docs.docker.com/build/building/multi-stage/)

---

**Good luck with your implementation! 🚀**