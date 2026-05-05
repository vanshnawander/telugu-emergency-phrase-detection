import os
import torch
import torchaudio
from pipeline import UnifiedASRPipeline
import warnings
import time

warnings.filterwarnings("ignore")

try:
    from jiwer import wer, cer
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False

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

def stream_audio(pipeline: UnifiedASRPipeline, wav: torch.Tensor, lang_tag: str, chunk_ms: int, sample_rate: int):
    chunk_samples = int(sample_rate * chunk_ms / 1000)
    total_samples = wav.shape[1]
    num_chunks = (total_samples + chunk_samples - 1) // chunk_samples

    pipeline.reset_stream(lang_tag)

    first_chunk_latency_ms = None
    first_output_latency_ms = None
    start_all = time.perf_counter()
    outputs = []

    for idx in range(num_chunks):
        start_sample = idx * chunk_samples
        end_sample = min(start_sample + chunk_samples, total_samples)
        chunk = wav[:, start_sample:end_sample]

        if chunk.shape[1] < int(sample_rate * 0.1):
            continue

        start = time.perf_counter()
        out = pipeline.transcribe_stream(chunk, lang_tag=lang_tag)
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
    pipeline = UnifiedASRPipeline(device='cuda', debug=True)

    test_files = [
        ("te", "data/test_audio/ai4bharat_Kathbath_te/sample_000000.wav"),
        ("en", "data/test_audio/english/OSR_us_000_0010_8k.wav"),
        ("hi", "data/test_audio/hindi/OSR_in_000_0063_8k.wav"),
    ]

    for lang, path in test_files:
        if not os.path.exists(path):
            print(f"Test file not found: {path}")
            continue

        print("\n" + "=" * 72)
        print(f"Language: {lang} | File: {path}")

        wav, duration_s = load_audio(path)
        print(f"Audio duration: {duration_s:.2f}s")

        print("\nFull transcription...")
        start = time.perf_counter()
        text_full = pipeline.transcribe(wav, lang_tag=lang)
        latency_ms = (time.perf_counter() - start) * 1000
        rtf = (latency_ms / 1000) / max(duration_s, 1e-6)
        print(f"Latency: {latency_ms:.2f}ms (RTF={rtf:.3f}x)")
        print(f"Result: {str(text_full)[:200]}")

        print("\nStreaming transcription...")
        stream = stream_audio(pipeline, wav, lang, chunk_ms=500, sample_rate=16000)
        rtf_stream = (stream["total_stream_time_ms"] / 1000) / max(duration_s, 1e-6)
        print(f"Chunks: {stream['num_chunks']} | First chunk latency: {stream['first_chunk_latency_ms']:.2f}ms")
        print(f"First output latency: {stream['first_output_latency_ms']:.2f}ms")
        print(f"Total stream time: {stream['total_stream_time_ms']:.2f}ms (RTF={rtf_stream:.3f}x)")
        print(f"Result: {stream['text'][:200]}")

        print("\nComparison (Full vs Streaming)...")
        full_text = str(text_full).strip()
        stream_text = str(stream["text"]).strip()
        if HAS_JIWER:
            try:
                w = wer(full_text, stream_text)
                c = cer(full_text, stream_text)
                print(f"WER: {w:.2%} | CER: {c:.2%}")
            except Exception as e:
                print(f"WER/CER failed: {e}")
        else:
            print("jiwer not installed. Run: uv pip install jiwer")
        full_words = full_text.split()
        stream_words = stream_text.split()
        full_set = set(full_words)
        stream_set = set(stream_words)
        missing = list(full_set - stream_set)
        extra = list(stream_set - full_set)
        print(f"Missing words (stream vs full): {missing[:20]}")
        print(f"Extra words (stream vs full): {extra[:20]}")

if __name__ == "__main__":
    main()
