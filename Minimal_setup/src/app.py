import json
import os
import tempfile
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import TranscribeConfig, TranscribeResponse
from .pipeline import run_pipeline

app = FastAPI(title="Audio Transcription Minimal Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/v1/transcribe", response_model=TranscribeResponse)
async def transcribe(
    file: UploadFile = File(...),
    config: Optional[str] = Form(None),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # parse config
    cfg = TranscribeConfig()
    if config:
        try:
            cfg = TranscribeConfig(**json.loads(config))
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid config JSON: {e}")

    # save temp
    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        data = await file.read()
        tmp.write(data)
        path = tmp.name

    try:
        return run_pipeline(path, cfg)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run('src.app:app', host="localhost", port=8001, reload=True)