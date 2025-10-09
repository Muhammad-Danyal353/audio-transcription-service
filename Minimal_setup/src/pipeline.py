import time
import uuid
from typing import Dict, Any, Tuple, List

import numpy as np
import librosa

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - environment without faster-whisper
    WhisperModel = None  # type: ignore

from .models import TranscribeConfig, TranscribeResponse, Segment, PipelineInfo


def load_audio(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int, float]:
    """Load and decode audio file."""
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    y = y.astype(np.float32)
    duration = float(len(y) / sr)
    return y, sr, duration


def normalize_audio(y: np.ndarray) -> np.ndarray:
    """Normalize audio to prevent clipping and improve processing."""
    # Normalize to [-1, 1] range
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val * 0.95  # Leave some headroom
    return y.astype(np.float32)


def detect_voice_activity(y: np.ndarray, sr: int, threshold: float = 0.01) -> List[Tuple[float, float]]:
    """Simple VAD using energy-based detection."""
    # Calculate frame energy
    frame_length = int(0.025 * sr)  # 25ms frames
    hop_length = int(0.010 * sr)    # 10ms hop
    
    energy = []
    for i in range(0, len(y) - frame_length, hop_length):
        frame = y[i:i + frame_length]
        energy.append(np.mean(frame ** 2))
    
    # Find voice segments
    voice_segments = []
    in_voice = False
    start_time = 0
    
    for i, e in enumerate(energy):
        time = i * hop_length / sr
        
        if e > threshold and not in_voice:
            start_time = time
            in_voice = True
        elif e <= threshold and in_voice:
            voice_segments.append((start_time, time))
            in_voice = False
    
    # Handle case where audio ends during voice
    if in_voice:
        voice_segments.append((start_time, len(y) / sr))
    
    return voice_segments


def chunk_audio(y: np.ndarray, sr: int, chunk_duration: float = 30.0, overlap: float = 2.0) -> List[Tuple[np.ndarray, float, float]]:
    """Chunk audio into overlapping segments for processing long files."""
    chunk_samples = int(chunk_duration * sr)
    overlap_samples = int(overlap * sr)
    hop_samples = chunk_samples - overlap_samples
    
    chunks = []
    for start in range(0, len(y), hop_samples):
        end = min(start + chunk_samples, len(y))
        chunk = y[start:end]
        start_time = start / sr
        end_time = end / sr
        chunks.append((chunk, start_time, end_time))
    
    return chunks


def spectral_separation(y: np.ndarray, sr: int) -> np.ndarray:
    # Simple spectral subtraction: estimate noise from first 0.5s
    if len(y) < sr // 2:
        return y
    n = y[: min(len(y), int(0.5 * sr))]
    S = librosa.stft(y, n_fft=1024, hop_length=256)
    N = librosa.stft(n, n_fft=1024, hop_length=256)
    mag = np.abs(S)
    phase = np.angle(S)
    noise_mag = np.maximum(np.mean(np.abs(N), axis=1, keepdims=True), 1e-6)
    alpha, beta = 2.0, 0.01
    gain = np.maximum(1.0 - alpha * (noise_mag / (mag + 1e-6)), beta)
    enhanced = (mag * gain) * np.exp(1j * phase)
    y_hat = librosa.istft(enhanced, hop_length=256)
    if len(y_hat) != len(y):
        # pad or trim to match length
        if len(y_hat) < len(y):
            y_hat = np.pad(y_hat, (0, len(y) - len(y_hat)))
        else:
            y_hat = y_hat[: len(y)]
    return y_hat.astype(np.float32)


def transcribe_whisper(y: np.ndarray, sr: int, cfg: TranscribeConfig) -> Tuple[List[Segment], str]:
    language = cfg.language_hint
    model_size = cfg.model_size

    if WhisperModel is None:
        raise RuntimeError("faster-whisper not installed. Install 'faster-whisper'.")

    model = WhisperModel(model_size, device="auto", compute_type="auto")
    segments_iter, info = model.transcribe(y, language=language)

    segments: List[Segment] = []
    texts: List[str] = []
    for s in segments_iter:
        seg = Segment(start=float(s.start), end=float(s.end), text=s.text.strip())
        segments.append(seg)
        if seg.text:
            texts.append(seg.text)

    lang = getattr(info, "language", language or "unknown")
    return segments, lang or "unknown"


def merge_segments(segments: List[Segment], overlap_threshold: float = 0.5) -> List[Segment]:
    """Merge overlapping or adjacent segments."""
    if not segments:
        return segments
    
    merged = []
    current = segments[0]
    
    for next_seg in segments[1:]:
        # If segments overlap or are very close, merge them
        if next_seg.start - current.end <= overlap_threshold:
            # Merge text
            current = Segment(
                start=current.start,
                end=next_seg.end,
                text=f"{current.text} {next_seg.text}".strip()
            )
        else:
            merged.append(current)
            current = next_seg
    
    merged.append(current)
    return merged


def run_pipeline(filepath: str, cfg: TranscribeConfig) -> TranscribeResponse:
    """Run the complete audio processing pipeline following the required sequence:
    decode → normalize → separation → (VAD/chunk) → ASR → (diarization) → merge → format
    """
    t0 = time.time()

    # 1. DECODE: Load audio file
    t_load0 = time.time()
    y, sr, duration = load_audio(filepath, cfg.target_sr)
    t_load = (time.time() - t_load0) * 1000

    # 2. NORMALIZE: Normalize audio levels
    t_norm0 = time.time()
    y = normalize_audio(y)
    t_norm = (time.time() - t_norm0) * 1000

    # 3. SEPARATION: Apply vocal separation if enabled
    t_sep0 = time.time()
    if cfg.enable_separation:
        y_proc = spectral_separation(y, sr)
        sep_info = {"enabled": True, "method": "spectral"}
    else:
        y_proc = y
        sep_info = {"enabled": False, "method": "none"}
    t_sep = (time.time() - t_sep0) * 1000

    # 4. VAD/CHUNK: Apply voice activity detection and chunking for long files
    t_vad0 = time.time()
    voice_segments = detect_voice_activity(y_proc, sr)
    chunks = chunk_audio(y_proc, sr) if duration > 30 else [(y_proc, 0.0, duration)]
    t_vad = (time.time() - t_vad0) * 1000

    # 5. ASR: Transcribe each chunk
    t_asr0 = time.time()
    all_segments = []
    language = "unknown"
    
    for chunk, start_time, end_time in chunks:
        chunk_segments, chunk_lang = transcribe_whisper(chunk, sr, cfg)
        # Adjust timestamps for chunk offset
        for seg in chunk_segments:
            seg.start += start_time
            seg.end += start_time
        all_segments.extend(chunk_segments)
        if chunk_lang != "unknown":
            language = chunk_lang
    
    t_asr = (time.time() - t_asr0) * 1000

    # 6. DIARIZATION: (Not implemented in minimal setup - placeholder)
    t_diar0 = time.time()
    # TODO: Implement speaker diarization if needed
    diarization_info = {"enabled": False, "method": "none"}
    t_diar = (time.time() - t_diar0) * 1000

    # 7. MERGE: Merge overlapping segments
    t_merge0 = time.time()
    merged_segments = merge_segments(all_segments)
    t_merge = (time.time() - t_merge0) * 1000

    # 8. FORMAT: Create final response
    t_format0 = time.time()
    text = " ".join(s.text for s in merged_segments).strip()
    t_format = (time.time() - t_format0) * 1000

    total_ms = (time.time() - t0) * 1000

    return TranscribeResponse(
        request_id=str(uuid.uuid4()),
        duration_sec=duration,
        sample_rate=sr,
        pipeline=PipelineInfo(
            separation=sep_info,
            transcription={"model": f"faster-whisper-{cfg.model_size}"},
        ),
        segments=merged_segments,
        text=text,
        language=language,
        timings_ms={
            "load": t_load, 
            "normalize": t_norm,
            "separation": t_sep, 
            "vad_chunk": t_vad,
            "transcription": t_asr,
            "diarization": t_diar,
            "merge": t_merge,
            "format": t_format,
            "total": total_ms
        },
    )