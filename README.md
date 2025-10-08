# Audio Transcription Microservice

A high-performance, production-ready microservice for audio transcription with advanced features including speaker diarization, vocal separation, voice activity detection, and real-time processing capabilities.

## 🚀 Features

### Core Functionality
- **Automatic Speech Recognition (ASR)** using OpenAI Whisper models
- **Speaker Diarization** to identify and separate different speakers
- **Vocal Separation** using Demucs for improved transcription quality
- **Voice Activity Detection (VAD)** to remove silence and noise
- **Real-time Processing** with streaming support
- **Multiple Audio Formats** support (WAV, MP3, FLAC, M4A, etc.)

### Advanced Features
- **Language Detection** with confidence scoring
- **Word-level Timestamps** for precise timing information
- **Chunked Processing** for long audio files
- **Performance Monitoring** with detailed metrics
- **Quality Assessment** with audio quality metrics
- **RESTful API** with OpenAPI documentation
- **Docker Support** for easy deployment

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Architecture](#architecture)
- [Development](#development)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🛠 Installation

### Prerequisites

- Python 3.8+
- FFmpeg (for audio format conversion)
- CUDA (optional, for GPU acceleration)

### System Requirements

- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 8GB+ RAM, 4+ CPU cores, GPU with CUDA support
- **Storage**: 2GB+ free space for models and temporary files

### Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd Audio_Transcription_Microservice

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install FFmpeg (Ubuntu/Debian)
sudo apt update
sudo apt install ffmpeg

# Install FFmpeg (macOS)
brew install ffmpeg

# Install FFmpeg (Windows)
# Download from https://ffmpeg.org/download.html
```

### Optional: GPU Support

For GPU acceleration, install PyTorch with CUDA support:

```bash
# For CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 🚀 Quick Start

### 1. Start the Service

```bash
# Development mode with auto-reload
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Production mode
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 2. Test the Service

```bash
# Health check
curl -X GET "http://localhost:8000/v1/health"

# Transcribe audio file
curl -X POST "http://localhost:8000/v1/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/audio.wav"
```

### 3. Access API Documentation

Open your browser and navigate to:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📚 API Documentation

### Base URL
```
http://localhost:8000/v1
```

### Endpoints

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-09T10:30:00.000Z",
  "version": "1.0.0",
  "uptime": 3600.0,
  "services": {
    "audio_processor": {
      "status": "healthy",
      "device": "cpu",
      "services_available": {
        "vad": true,
        "separation": true,
        "transcription": true,
        "diarization": false
      }
    },
    "system_resources": {
      "status": "healthy",
      "memory_usage_percent": 45.2
    }
  }
}
```

#### Transcribe Audio
```http
POST /transcribe
```

**Request:**
- **Content-Type**: `multipart/form-data`
- **Body**: Audio file (WAV, MP3, FLAC, M4A, etc.)

**Optional Parameters:**
```json
{
  "config": {
    "enable_separation": true,
    "enable_vad": true,
    "enable_diarization": false,
    "language": "en",
    "task": "transcribe",
    "word_timestamps": true,
    "min_speakers": 1,
    "max_speakers": 5,
    "num_speakers": null
  }
}
```

**Response:**
```json
{
  "text": "Hello, this is a test transcription.",
  "language": "en",
  "language_probability": 0.95,
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.5,
      "text": "Hello, this is a test transcription.",
      "confidence": 0.92,
      "words": [
        {
          "word": "Hello",
          "start": 0.0,
          "end": 0.5,
          "probability": 0.98
        }
      ]
    }
  ],
  "voice_segments": [
    {
      "start_time": 0.0,
      "end_time": 3.5,
      "confidence": 0.95,
      "is_speech": true
    }
  ],
  "speaker_segments": null,
  "pipeline_info": {
    "transcription": {
      "model_size": "base",
      "device": "cpu",
      "sample_rate": 16000
    },
    "separation": {
      "method": "demucs",
      "enabled": true,
      "device": "cpu"
    },
    "vad": {
      "enabled": true
    },
    "diarization": null
  },
  "processing_time": 15.2,
  "device_info": {
    "device_type": "cpu",
    "device_name": "CPU",
    "memory_total_gb": 16.0,
    "memory_available_gb": 12.0
  },
  "quality_metrics": {
    "transcription": {
      "confidence": 0.92,
      "word_count": 6,
      "duration": 3.5
    }
  },
  "timing_metrics": [
    {
      "operation_name": "load_audio",
      "duration_ms": 150.0,
      "success": true,
      "metadata": {}
    }
  ],
  "request_id": "uuid-string",
  "timestamp": "2025-01-09T10:30:15.000Z",
  "config": {
    "enable_separation": true,
    "enable_vad": true,
    "enable_diarization": false
  }
}
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Audio Processing
TARGET_SAMPLE_RATE=16000
MAX_AUDIO_DURATION=3600
CHUNK_SIZE=30
OVERLAP=2

# Model Configuration
WHISPER_MODEL_SIZE=base
DEMUCS_MODEL=htdemucs
VAD_MODEL=silero_vad

# Device Configuration
DEVICE=auto  # auto, cpu, cuda, cuda:0
FORCE_CPU=false

# Performance
MAX_WORKERS=4
BATCH_SIZE=1
ENABLE_QUANTIZATION=true

# HuggingFace (for diarization)
HUGGINGFACE_TOKEN=your_token_here

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Configuration File

You can also use a YAML configuration file:

```yaml
# config.yaml
server:
  host: "0.0.0.0"
  port: 8000
  debug: false

audio:
  target_sample_rate: 16000
  max_duration: 3600
  chunk_size: 30
  overlap: 2

models:
  whisper:
    size: "base"
    device: "auto"
  demucs:
    model: "htdemucs"
  vad:
    model: "silero_vad"

performance:
  max_workers: 4
  batch_size: 1
  enable_quantization: true
```

## 💡 Usage Examples

### Python Client

```python
import requests
import json

# Transcribe audio file
def transcribe_audio(file_path, config=None):
    url = "http://localhost:8000/v1/transcribe"
    
    with open(file_path, 'rb') as f:
        files = {'file': f}
        data = {'config': json.dumps(config)} if config else {}
        
        response = requests.post(url, files=files, data=data)
        return response.json()

# Example usage
result = transcribe_audio(
    "audio.wav",
    config={
        "enable_separation": True,
        "enable_diarization": True,
        "language": "en"
    }
)

print(f"Transcription: {result['text']}")
print(f"Language: {result['language']}")
print(f"Processing time: {result['processing_time']}s")
```

### JavaScript/Node.js Client

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function transcribeAudio(filePath, config = {}) {
    const form = new FormData();
    form.append('file', fs.createReadStream(filePath));
    
    if (Object.keys(config).length > 0) {
        form.append('config', JSON.stringify(config));
    }
    
    try {
        const response = await axios.post(
            'http://localhost:8000/v1/transcribe',
            form,
            {
                headers: form.getHeaders(),
                timeout: 300000 // 5 minutes
            }
        );
        
        return response.data;
    } catch (error) {
        console.error('Transcription failed:', error.response?.data || error.message);
        throw error;
    }
}

// Example usage
transcribeAudio('audio.wav', {
    enable_separation: true,
    enable_diarization: true,
    language: 'en'
}).then(result => {
    console.log('Transcription:', result.text);
    console.log('Language:', result.language);
    console.log('Processing time:', result.processing_time + 's');
});
```

### Command Line Interface

```bash
# Using curl
curl -X POST "http://localhost:8000/v1/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav" \
  -F 'config={"enable_separation": true, "language": "en"}'

# Using the built-in CLI
python -m src.cli.client transcribe audio.wav --enable-separation --language en

# Advanced CLI usage
python -m src.cli.advanced_client \
  --input audio.wav \
  --output transcription.json \
  --enable-diarization \
  --min-speakers 2 \
  --max-speakers 5 \
  --language en
```

## 🏗 Architecture

### System Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client        │    │   Load Balancer │    │   API Gateway   │
│   (Web/Mobile)  │───▶│   (Optional)    │───▶│   (FastAPI)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Audio         │    │   Processing    │    │   Response      │
│   Storage       │◀───│   Pipeline      │───▶│   Formatter     │
│   (Temporary)   │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   AI Models     │
                       │   • Whisper     │
                       │   • Demucs      │
                       │   • PyAnnote    │
                       │   • Silero VAD  │
                       └─────────────────┘
```

### Processing Pipeline

1. **Audio Input Validation**
   - File format detection
   - Audio quality assessment
   - Duration and size validation

2. **Preprocessing**
   - Format conversion to WAV
   - Resampling to target sample rate
   - Normalization and noise reduction

3. **Vocal Separation** (Optional)
   - Source separation using Demucs
   - Vocal track extraction
   - Quality enhancement

4. **Voice Activity Detection** (Optional)
   - Silence removal
   - Speech segment identification
   - Noise filtering

5. **Transcription**
   - Whisper model inference
   - Language detection
   - Word-level timestamps

6. **Speaker Diarization** (Optional)
   - Speaker identification
   - Speaker change detection
   - Speaker clustering

7. **Post-processing**
   - Result aggregation
   - Quality metrics calculation
   - Response formatting

### Component Architecture

```
src/
├── api/                    # FastAPI application
│   ├── main.py            # Application entry point
│   ├── routes.py          # API endpoints
│   ├── models.py          # Pydantic models
│   └── middleware.py      # Custom middleware
├── services/              # Core services
│   ├── audio_processor.py # Main processing orchestrator
│   ├── transcriber.py     # Whisper transcription
│   ├── separator.py       # Vocal separation
│   ├── diarization.py     # Speaker diarization
│   └── vad.py            # Voice activity detection
├── utils/                 # Utility functions
│   ├── audio_utils.py     # Audio processing utilities
│   ├── timing.py          # Performance monitoring
│   └── config.py          # Configuration management
├── cli/                   # Command-line interfaces
│   ├── client.py          # Basic CLI client
│   └── advanced_client.py # Advanced CLI client
└── tests/                 # Test suite
    ├── test_integration.py
    ├── test_transcriber.py
    └── test_separator.py
```

## 🔧 Development

### Setting Up Development Environment

```bash
# Clone and setup
git clone <repository-url>
cd Audio_Transcription_Microservice
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v

# Run linting
flake8 src/
black src/
mypy src/
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_transcriber.py

# Run with coverage
pytest --cov=src tests/

# Run integration tests
pytest tests/test_integration.py -v
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/

# Security scan
bandit -r src/
```

## 🚀 Deployment

### Docker Deployment

```bash
# Build image
docker build -t audio-transcription-service .

# Run container
docker run -p 8000:8000 \
  -e HUGGINGFACE_TOKEN=your_token \
  -v /path/to/models:/app/models \
  audio-transcription-service

# Using docker-compose
docker-compose up -d
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  audio-transcription:
    build: .
    ports:
      - "8000:8000"
    environment:
      - HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}
      - DEVICE=cpu
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - audio-transcription
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: audio-transcription
spec:
  replicas: 3
  selector:
    matchLabels:
      app: audio-transcription
  template:
    metadata:
      labels:
        app: audio-transcription
    spec:
      containers:
      - name: audio-transcription
        image: audio-transcription-service:latest
        ports:
        - containerPort: 8000
        env:
        - name: HUGGINGFACE_TOKEN
          valueFrom:
            secretKeyRef:
              name: huggingface-secret
              key: token
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: audio-transcription-service
spec:
  selector:
    app: audio-transcription
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Production Considerations

1. **Resource Management**
   - Set appropriate memory limits
   - Configure CPU limits based on workload
   - Monitor GPU memory usage

2. **Security**
   - Use HTTPS in production
   - Implement authentication/authorization
   - Validate file uploads
   - Rate limiting

3. **Monitoring**
   - Health checks
   - Metrics collection
   - Log aggregation
   - Error tracking

4. **Scaling**
   - Horizontal scaling with load balancer
   - Model caching
   - Queue-based processing for high load

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```

**Solution:**
- Reduce batch size
- Use smaller model
- Enable CPU fallback
- Increase GPU memory

#### 2. Audio Format Not Supported
```
AudioFormatError: Unsupported audio format
```

**Solution:**
- Install FFmpeg
- Check file format
- Convert to supported format

#### 3. Model Download Issues
```
OSError: Model not found
```

**Solution:**
- Check internet connection
- Verify HuggingFace token
- Clear model cache
- Download models manually

#### 4. Performance Issues
```
Slow transcription performance
```

**Solution:**
- Use GPU acceleration
- Enable quantization
- Optimize audio preprocessing
- Use smaller models

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
python -m uvicorn src.api.main:app --reload
```

### Health Check Issues

```bash
# Check service status
curl -X GET "http://localhost:8000/v1/health"

# Check system resources
curl -X GET "http://localhost:8000/v1/health" | jq '.services.system_resources'
```

### Log Analysis

```bash
# View logs
tail -f logs/audio_transcription.log

# Filter error logs
grep "ERROR" logs/audio_transcription.log

# Monitor performance
grep "processing_time" logs/audio_transcription.log
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Add tests** for new functionality
5. **Run tests** and ensure they pass
6. **Commit your changes**
   ```bash
   git commit -m "Add your feature"
   ```
7. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```
8. **Create a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write comprehensive tests
- Update documentation
- Use meaningful commit messages

### Code Review Process

1. Automated tests must pass
2. Code quality checks must pass
3. Manual review by maintainers
4. Performance impact assessment
5. Security review for sensitive changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [Demucs](https://github.com/facebookresearch/demucs) for source separation
- [PyAnnote](https://github.com/pyannote/pyannote-audio) for speaker diarization
- [Silero VAD](https://github.com/snakers4/silero-vad) for voice activity detection
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework

## 📞 Support

- **Documentation**: [Wiki](link-to-wiki)
- **Issues**: [GitHub Issues](link-to-issues)
- **Discussions**: [GitHub Discussions](link-to-discussions)
- **Email**: support@example.com

## 🔄 Changelog

### Version 1.0.0
- Initial release
- Core transcription functionality
- Speaker diarization support
- Vocal separation capabilities
- Voice activity detection
- RESTful API with OpenAPI documentation
- Docker support
- Comprehensive test suite

---

**Made with ❤️ for the audio processing community**
