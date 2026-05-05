"""
FastAPI server for ASR — REST-only, optimized for fast file transcription.

Endpoints:
  POST /transcribe          — single audio file transcription
  POST /transcribe/batch    — batch transcription (multiple files)
  GET  /health              — health check
  GET  /                    — serves the test UI
"""

from __future__ import annotations

import asyncio
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import torch
import torchaudio
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from server.config import ServerConfig
from server.engine import ASREngine

logger = logging.getLogger("asr.api")

# ── Globals (set during lifespan) ──────────────────────────────────────────
_engine: Optional[ASREngine] = None
_config: Optional[ServerConfig] = None
_executor: Optional[ThreadPoolExecutor] = None


def get_engine() -> ASREngine:
    assert _engine is not None, "Engine not initialized"
    return _engine


# ── Lifespan ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _config, _executor
    _config = ServerConfig()
    logger.info("Starting ASR server...")

    _engine = ASREngine(_config)
    _executor = ThreadPoolExecutor(max_workers=_config.max_workers)
    logger.info("ASR server ready on %s:%d", _config.host, _config.port)
    yield

    _executor.shutdown(wait=False)
    logger.info("ASR server shut down")


# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Unified ASR API",
    version="2.0.0",
    description="Fast REST ASR for English, Hindi, and Telugu",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (UI)
STATIC_DIR = Path(__file__).parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Helpers ────────────────────────────────────────────────────────────────
def _decode_audio_bytes(data: bytes, target_sr: int = 16000) -> torch.Tensor:
    """Decode audio bytes (wav/mp3/webm/etc) to (1, N) float32 tensor at target_sr."""
    buf = io.BytesIO(data)

    # Try loading with torchaudio — try multiple format hints for robustness
    wav, sr = None, None
    for fmt in [None, "wav", "mp3", "ogg", "flac", "webm", "mp4"]:
        try:
            buf.seek(0)
            wav, sr = torchaudio.load(buf, format=fmt)
            break
        except Exception:
            continue

    if wav is None:
        # Last resort: treat as raw PCM int16 mono 16kHz
        try:
            pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            wav = torch.from_numpy(pcm).unsqueeze(0)
            sr = target_sr
        except Exception:
            raise ValueError("Could not decode audio data in any supported format")

    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
    return wav


# ── REST Endpoints ─────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "models": ["parakeet-tdt-0.6b-v3", "indic-conformer-600m"]}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    lang: str = Form("en"),
    backend: str = Form("indic"),
):
    """
    Transcribe an audio file. Upload audio and get the transcript back.
    Supported languages: en, hi, te
    """
    if lang not in ("en", "hi", "te"):
        return JSONResponse(status_code=400, content={"error": f"Unsupported language: {lang}"})
    if backend not in ("indic", "whisper"):
        return JSONResponse(status_code=400, content={"error": f"Unsupported backend: {backend}"})

    data = await file.read()
    if not data:
        return JSONResponse(status_code=400, content={"error": "Empty audio file"})

    # Debug: compare upload vs record
    file_name = getattr(file, 'filename', 'unknown')
    logger.info("[DEBUG] Received file: %s size=%d bytes", file_name, len(data))
    
    # Save recorded audio for debugging
    if file_name == 'recording.wav':
        import time
        debug_path = f"debug_recording_{int(time.time())}.wav"
        with open(debug_path, 'wb') as f:
            f.write(data)
        logger.info("[DEBUG] Saved recording to %s", debug_path)
    
    if len(data) >= 16:
        header = data[:16].hex()
        logger.info("[DEBUG] Header: %s", header)
    if len(data) >= 44:
        # Check WAV header fields (fmt chunk is 16 bytes at offset 20)
        import struct
        fmt = struct.unpack('<IHHIIHH', data[20:40])  # 20 bytes: chunkSize, audioFormat, numChannels, sampleRate, byteRate, blockAlign, bitsPerSample
        logger.info("[DEBUG] WAV fmt: fmt=%d channels=%d rate=%d", fmt[1], fmt[2], fmt[3])

    try:
        audio = _decode_audio_bytes(data)
        logger.info("[DEBUG] Decoded tensor: shape=%s dtype=%s min=%.3f max=%.3f", audio.shape, audio.dtype, audio.min().item(), audio.max().item())
        # Check if audio is completely silent
        import torch
        if torch.abs(audio).max().item() < 1e-6:
            logger.warning("[DEBUG] Audio appears to be completely silent!")
    except Exception as e:
        logger.error("Audio decode failed: %s (size=%d bytes file=%s)", e, len(data), file_name, exc_info=True)
        return JSONResponse(status_code=400, content={"error": f"Could not decode audio: {e}"})

    duration_s = audio.shape[1] / 16000
    if duration_s < 0.1:
        return JSONResponse(status_code=400, content={"error": f"Audio too short: {duration_s:.2f}s"})

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(_executor, get_engine().transcribe, audio, lang, backend)
    except Exception as e:
        logger.error("Transcription failed: %s", e, exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Transcription failed: {e}"})

    return {
        "text": result["text"],
        "lang": lang,
        "backend": backend,
        "duration_s": round(duration_s, 2),
        "latency_ms": round(result["latency_ms"], 2),
    }


@app.post("/transcribe/batch")
async def transcribe_batch(
    files: list[UploadFile] = File(...),
    lang: str = Form("en"),
    backend: str = Form("indic"),
):
    """
    Batch transcription — submit multiple files at once for maximum throughput.
    All files are decoded concurrently, then batched through the model together.
    """
    if lang not in ("en", "hi", "te"):
        return JSONResponse(status_code=400, content={"error": f"Unsupported language: {lang}"})
    if backend not in ("indic", "whisper"):
        return JSONResponse(status_code=400, content={"error": f"Unsupported backend: {backend}"})

    audios = []
    for f in files:
        data = await f.read()
        audios.append(_decode_audio_bytes(data))

    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(
        _executor, get_engine().transcribe_batch, audios, lang, backend
    )

    return {
        "results": [
            {"text": r["text"], "latency_ms": round(r["latency_ms"], 2)}
            for r in results
        ],
        "lang": lang,
        "backend": backend,
        "count": len(results),
    }


# ── UI Serving ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(), status_code=200)
    return HTMLResponse(content="<h1>ASR API is running. No UI found.</h1>", status_code=200)
