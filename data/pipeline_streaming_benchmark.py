"""
Unified ASR Pipeline Streaming Benchmark
========================================
Measures:
1) Non-streaming (full audio) latency
2) Streaming chunk latency (first chunk, first non-empty output)
3) End-to-end completion time and RTF
"""
import time
from pathlib import Path
import sys

import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import UnifiedASRPipeline

TARGET_SAMPLE_RATE = 16000
STREAM_CHUNK_MS = 500

TEST_FILES = [
    ("te", "data/test_audio/ai4bharat_Kathbath_te/sample_000000.wav"),
    ("en", "data/test_audio/english/OSR_us_000_0010_8k.wav"),
    ("hi", "data/test_audio/hindi/OSR_in_000_0063_8k.wav"),
]


def load_audio(audio_path: str, target_sr: int = 16000) -> tuple[torch.Tensor, float]:
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        wav = resampler(wav)
        sr = target_sr
    duration_s = wav.shape[1] / sr
    return wav, duration_s


def transcribe_full(pipeline: UnifiedASRPipeline, wav: torch.Tensor, lang: str) -> tuple[str, float]:
    start = time.perf_counter()
    text = pipeline.transcribe(wav, lang)
    latency_ms = (time.perf_counter() - start) * 1000
    return text, latency_ms


def transcribe_streaming(
    pipeline: UnifiedASRPipeline,
    wav: torch.Tensor,
    lang: str,
    chunk_ms: int,
    sample_rate: int,
) -> dict:
    chunk_samples = int(sample_rate * chunk_ms / 1000)
    total_samples = wav.shape[1]
    num_chunks = (total_samples + chunk_samples - 1) // chunk_samples

    first_chunk_latency_ms = None
    first_output_latency_ms = None
    start_all = time.perf_counter()
    outputs = []

    pipeline.reset_stream(lang)

    for idx in range(num_chunks):
        start_sample = idx * chunk_samples
        end_sample = min(start_sample + chunk_samples, total_samples)
        chunk = wav[:, start_sample:end_sample]

        if chunk.shape[1] < int(sample_rate * 0.1):
            continue

        start = time.perf_counter()
        out = pipeline.transcribe_stream(chunk, lang_tag=lang)
        latency_ms = (time.perf_counter() - start) * 1000

        if first_chunk_latency_ms is None:
            first_chunk_latency_ms = latency_ms

        if first_output_latency_ms is None and isinstance(out, str) and out.strip():
            first_output_latency_ms = (time.perf_counter() - start_all) * 1000

        if isinstance(out, str) and out.strip():
            outputs.append(out.strip())

    total_ms = (time.perf_counter() - start_all) * 1000
    return {
        "num_chunks": num_chunks,
        "first_chunk_latency_ms": first_chunk_latency_ms or 0.0,
        "first_output_latency_ms": first_output_latency_ms or 0.0,
        "total_stream_time_ms": total_ms,
        "text": " ".join(outputs),
    }


def main():
    pipeline = UnifiedASRPipeline(device="cuda", debug=True)
    for lang, path in TEST_FILES:
        if not Path(path).exists():
            print(f"Missing file: {path}")
            continue

        print("\n" + "=" * 72)
        print(f"Language: {lang} | File: {path}")

        wav, duration_s = load_audio(path, TARGET_SAMPLE_RATE)
        print(f"Audio duration: {duration_s:.2f}s")

        full_text, full_latency_ms = transcribe_full(pipeline, wav, lang)
        rtf_full = (full_latency_ms / 1000) / max(duration_s, 1e-6)
        print(f"Full transcription latency: {full_latency_ms:.2f}ms (RTF={rtf_full:.3f}x)")
        print(f"Full transcription text: {str(full_text)[:160]}")

        stream = transcribe_streaming(pipeline, wav, lang, STREAM_CHUNK_MS, TARGET_SAMPLE_RATE)
        rtf_stream = (stream["total_stream_time_ms"] / 1000) / max(duration_s, 1e-6)
        print(f"Streaming chunk size: {STREAM_CHUNK_MS}ms")
        print(f"First chunk latency: {stream['first_chunk_latency_ms']:.2f}ms")
        print(f"First output latency: {stream['first_output_latency_ms']:.2f}ms")
        print(f"Total streaming time: {stream['total_stream_time_ms']:.2f}ms (RTF={rtf_stream:.3f}x)")
        print(f"Streaming text: {stream['text'][:160]}")

    pipeline.close()


if __name__ == "__main__":
    main()
