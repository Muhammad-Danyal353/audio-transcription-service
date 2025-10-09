Audio Transcription Minimal Microservice

Overview
- POST /v1/transcribe: accepts multipart/form-data with file and optional JSON config.
- Pipeline: decode -> spectral separation (noise reduction) -> ASR (faster-whisper) -> format.
- Dockerized with Dockerfile and docker-compose.
- CLI utility: Minimal_setup/cli.py

Why minimal choices
- Separation: spectral subtraction (librosa). It is lightweight and works CPU-only. You can enable a Demucs-based method later if needed. The response marks separation.enabled and method.
- ASR: faster-whisper for good CPU performance; model size configurable via config.model_size.
- Audio I/O: librosa + ffmpeg in container to support mp3/m4a/flac/ogg.

API
- Endpoint: POST /v1/transcribe
- Request fields:
  - file: audio binary (wav/mp3/m4a/flac/ogg)
  - config (optional JSON):
    {
      "language_hint": "en",
      "enable_separation": true,
      "diarize": false,
      "model_size": "small",
      "target_sr": 16000
    }
- Response example:
  {
    "request_id": "uuid",
    "duration_sec": 31.2,
    "sample_rate": 16000,
    "pipeline": {
      "separation": {"enabled": true, "method": "spectral"},
      "transcription": {"model": "faster-whisper-small"}
    },
    "segments": [{"start": 0.0, "end": 3.1, "text": "hello world"}],
    "text": "hello world",
    "language": "en",
    "timings_ms": {"load": 420, "separation": 1800, "transcription": 4100, "total": 6400}
  }

Run locally (without Docker)
- Python 3.10+
- Install ffmpeg on your system if not using Docker.
- pip install -r Minimal_setup/requirements.txt
- uvicorn Minimal_setup.src.app:app --reload

Docker
- docker build -t audio-transcribe-minimal -f Minimal_setup/Dockerfile .
- docker compose -f Minimal_setup/docker-compose.yml up --build

CLI
- python Minimal_setup/cli.py sample_audio/121123/84-121123-0000.flac --enable_separation --model_size small

Notes
- Sample audio: use existing ../sample_audio clips in this repo for testing.
- Diarization and VAD are out-of-scope for this minimal implementation; the API accepts the flags but they’re ignored.
- GPU: faster-whisper auto-detects; this container runs fine on CPU.
- Long files: faster-whisper streams segments; for truly long files chunking can be added later.

ADR (short)
- Pipeline: decode (librosa) -> spectral separation (librosa STFT) -> ASR (faster-whisper) -> merge -> format
- Model: faster-whisper small by default (better latency vs accuracy tradeoff). Configurable.
- CPU-first: works without GPU; if GPU is present, faster-whisper will use it automatically.
- Concurrency: uvicorn single worker in compose; scale workers via compose or uvicorn args as needed.
- Observability: timings exposed in response. Can be extended with structured logs.
- Fallbacks: if separation is disabled or fails, raw audio is used; the response flag indicates the method.