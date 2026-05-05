"""
Indic Conformer ASR Latency Benchmark
=====================================
Measures ASR inference latency for ai4bharat/indic-conformer-600m-multilingual

This model uses ONNX Runtime internally, so we measure latency via wall-clock time.
Supports both CTC and RNNT decoders, with streaming simulation for RNNT.
"""
import time
import statistics
from pathlib import Path

import torch
import torchaudio
from transformers import AutoModel

# Configuration
NUM_WARMUP = 3
NUM_ITERATIONS = 10
AUDIO_PATH = "data/test_audio/ai4bharat_Kathbath_te/sample_000000.wav"
TARGET_SAMPLE_RATE = 16000
LANGUAGE = "te"  # Telugu

# Streaming configuration
STREAMING_CHUNK_MS = 500  # Chunk size for streaming simulation in milliseconds


def load_audio(audio_path: str, target_sr: int = 16000) -> tuple[torch.Tensor, float]:
    """Load and preprocess audio, returns (audio_tensor, audio_duration_seconds)."""
    wav, sr = torchaudio.load(audio_path)
    wav = torch.mean(wav, dim=0, keepdim=True)  # Convert to mono
    
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        wav = resampler(wav)
        sr = target_sr
    
    audio_duration = wav.shape[1] / sr
    return wav, audio_duration


def benchmark_inference(model, wav, language: str, decoder: str, num_warmup: int, num_iterations: int) -> dict:
    """Benchmark inference latency with warmup and multiple iterations.
    
    Note: This model uses ONNX Runtime internally, so we use wall-clock timing.
    """
    
    # Warmup runs
    print(f"  Running {num_warmup} warmup iterations...")
    for _ in range(num_warmup):
        _ = model(wav, language, decoder)
    
    # Timed runs
    latencies = []
    print(f"  Running {num_iterations} timed iterations...")
    
    for i in range(num_iterations):
        start = time.perf_counter()
        result = model(wav, language, decoder)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
    
    return {
        "latencies": latencies,
        "mean_ms": statistics.mean(latencies),
        "std_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "median_ms": statistics.median(latencies),
        "last_result": result,
    }


def benchmark_streaming_simulation(model, wav, language: str, decoder: str, 
                                    chunk_ms: int, num_iterations: int) -> dict:
    """Simulate streaming ASR by processing audio in chunks.
    
    This tests how the model performs on shorter audio segments,
    which is important for real-time streaming applications.
    
    Note: True streaming would require incremental state, but this
    simulates the latency of processing small chunks independently.
    """
    sample_rate = TARGET_SAMPLE_RATE
    chunk_samples = int(chunk_ms * sample_rate / 1000)
    total_samples = wav.shape[1]
    num_chunks = (total_samples + chunk_samples - 1) // chunk_samples
    
    print(f"  Simulating streaming with {chunk_ms}ms chunks ({num_chunks} chunks total)")
    
    all_latencies = []
    chunk_results = []
    
    for iteration in range(num_iterations):
        iteration_latencies = []
        transcriptions = []
        
        for chunk_idx in range(num_chunks):
            start_sample = chunk_idx * chunk_samples
            end_sample = min(start_sample + chunk_samples, total_samples)
            chunk = wav[:, start_sample:end_sample]
            
            # Skip very short final chunks
            if chunk.shape[1] < sample_rate * 0.1:  # Less than 100ms
                continue
            
            start = time.perf_counter()
            result = model(chunk, language, decoder)
            latency_ms = (time.perf_counter() - start) * 1000
            
            iteration_latencies.append(latency_ms)
            transcriptions.append(result)
        
        all_latencies.extend(iteration_latencies)
        chunk_results = transcriptions  # Keep last iteration's results
    
    return {
        "latencies": all_latencies,
        "mean_ms": statistics.mean(all_latencies),
        "std_ms": statistics.stdev(all_latencies) if len(all_latencies) > 1 else 0,
        "min_ms": min(all_latencies),
        "max_ms": max(all_latencies),
        "median_ms": statistics.median(all_latencies),
        "num_chunks": num_chunks,
        "chunk_ms": chunk_ms,
        "chunk_results": chunk_results,
    }


def check_streaming_support(model) -> dict:
    """Check what streaming-related capabilities the model has."""
    
    streaming_info = {
        "has_streaming_api": False,
        "native_streaming": False,
        "uses_onnx": False,
        "decoder_types": ["ctc", "rnnt"],
        "notes": [],
    }
    
    # Check for ONNX sessions (this model uses ONNX Runtime)
    if hasattr(model, 'encoder_session') or hasattr(model, 'onnx_session'):
        streaming_info["uses_onnx"] = True
        streaming_info["notes"].append("Model uses ONNX Runtime for inference")
    
    # Check for streaming methods
    streaming_methods = ['transcribe_streaming', 'forward_streaming', 
                         'encode_streaming', 'process_chunk', 'streaming_forward']
    for method in streaming_methods:
        if hasattr(model, method):
            streaming_info["has_streaming_api"] = True
            streaming_info["notes"].append(f"Has {method}() method")
    
    # Check model type
    model_type = type(model).__name__
    streaming_info["model_class"] = model_type
    
    # RNNT is inherently more suitable for streaming than CTC
    streaming_info["notes"].append(
        "RNNT decoder is more suitable for streaming (autoregressive token emission)"
    )
    streaming_info["notes"].append(
        "CTC decoder works on full sequence, less ideal for streaming"
    )
    
    # Check internal components
    if hasattr(model, '__dict__'):
        for key in model.__dict__:
            if 'stream' in key.lower():
                streaming_info["notes"].append(f"Has attribute: {key}")
    
    return streaming_info


def main():
    print("=" * 70)
    print("Indic Conformer ASR Latency Benchmark")
    print("=" * 70)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n📍 Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
    
    # Load model
    print("\n📦 Loading model: ai4bharat/indic-conformer-600m-multilingual")
    model_start = time.perf_counter()
    model = AutoModel.from_pretrained(
        "ai4bharat/indic-conformer-600m-multilingual",
        trust_remote_code=True,
    )
    model_load_time = time.perf_counter() - model_start
    print(f"   Model loaded in {model_load_time:.2f}s")
    print(f"   Model type: {type(model).__name__}")
    
    # Note: This model uses ONNX Runtime, parameters are managed by ONNX
    print("   Note: Model uses ONNX Runtime (parameters managed by ONNX)")
    
    # Check streaming support
    print("\n🔄 Checking Streaming Support...")
    streaming_info = check_streaming_support(model)
    print(f"   Uses ONNX: {streaming_info['uses_onnx']}")
    print(f"   Native Streaming API: {streaming_info['has_streaming_api']}")
    print(f"   Available Decoders: {streaming_info['decoder_types']}")
    for note in streaming_info['notes']:
        print(f"   • {note}")
    
    # Load audio
    print(f"\n🎵 Loading audio: {AUDIO_PATH}")
    preprocess_start = time.perf_counter()
    wav, audio_duration = load_audio(AUDIO_PATH, TARGET_SAMPLE_RATE)
    preprocess_time = (time.perf_counter() - preprocess_start) * 1000
    print(f"   Audio duration: {audio_duration:.2f}s")
    print(f"   Preprocessing time: {preprocess_time:.2f}ms")
    print(f"   Audio shape: {wav.shape}")
    
    # Benchmark CTC decoder (full audio)
    print(f"\n⏱️  Benchmarking CTC Decoder (warmup={NUM_WARMUP}, iterations={NUM_ITERATIONS})")
    ctc_results = benchmark_inference(model, wav, LANGUAGE, "ctc", NUM_WARMUP, NUM_ITERATIONS)
    
    # Benchmark RNNT decoder (full audio)
    print(f"\n⏱️  Benchmarking RNNT Decoder (warmup={NUM_WARMUP}, iterations={NUM_ITERATIONS})")
    rnnt_results = benchmark_inference(model, wav, LANGUAGE, "rnnt", NUM_WARMUP, NUM_ITERATIONS)
    
    # Benchmark streaming simulation (RNNT with chunks)
    print(f"\n⏱️  Streaming Simulation (RNNT, {STREAMING_CHUNK_MS}ms chunks)")
    streaming_results = benchmark_streaming_simulation(
        model, wav, LANGUAGE, "rnnt", STREAMING_CHUNK_MS, num_iterations=3
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("📊 Results Summary")
    print("=" * 70)
    
    for name, results in [("CTC (Full Audio)", ctc_results), ("RNNT (Full Audio)", rnnt_results)]:
        print(f"\n{name}:")
        print(f"  Mean latency:   {results['mean_ms']:.2f}ms ± {results['std_ms']:.2f}ms")
        print(f"  Median latency: {results['median_ms']:.2f}ms")
        print(f"  Min/Max:        {results['min_ms']:.2f}ms / {results['max_ms']:.2f}ms")
        
        # Real-time factor (RTF)
        rtf = (results['mean_ms'] / 1000) / audio_duration
        print(f"  Real-Time Factor (RTF): {rtf:.4f}x")
        print(f"  {'✅ FASTER than real-time' if rtf < 1 else '❌ SLOWER than real-time'}")
        if results.get('last_result'):
            text = str(results['last_result'])
            print(f"  Transcription: {text[:80]}{'...' if len(text) > 80 else ''}")
    
    # Streaming results
    print(f"\nRNNT Streaming Simulation ({streaming_results['chunk_ms']}ms chunks):")
    print(f"  Chunks processed: {streaming_results['num_chunks']}")
    print(f"  Per-chunk latency: {streaming_results['mean_ms']:.2f}ms ± {streaming_results['std_ms']:.2f}ms")
    print(f"  Min/Max per chunk: {streaming_results['min_ms']:.2f}ms / {streaming_results['max_ms']:.2f}ms")
    
    # Check if streaming is real-time viable
    chunk_duration_ms = streaming_results['chunk_ms']
    if streaming_results['mean_ms'] < chunk_duration_ms:
        print(f"  ✅ Can process {chunk_duration_ms}ms chunk in {streaming_results['mean_ms']:.0f}ms - STREAMING VIABLE")
    else:
        print(f"  ❌ Chunk processing ({streaming_results['mean_ms']:.0f}ms) > chunk duration ({chunk_duration_ms}ms)")
        print(f"     Need ~{streaming_results['mean_ms']/chunk_duration_ms:.1f}x speedup for real-time streaming")
    
    # Performance summary
    print("\n" + "=" * 70)
    print("🔍 Analysis & Recommendations")
    print("=" * 70)
    
    print("\n📈 Performance Summary:")
    print(f"   • CTC is {'faster' if ctc_results['mean_ms'] < rnnt_results['mean_ms'] else 'slower'} than RNNT")
    speed_diff = abs(ctc_results['mean_ms'] - rnnt_results['mean_ms']) / min(ctc_results['mean_ms'], rnnt_results['mean_ms']) * 100
    print(f"   • Speed difference: {speed_diff:.1f}%")
    
    print("\n🎯 For Streaming ASR:")
    print("   • RNNT is preferred (autoregressive, emits tokens incrementally)")
    print("   • CTC works on full sequence (less suitable for streaming)")
    print("   • Consider chunked processing with cache/state for true streaming")
    
    print("\n🚀 Optimization Opportunities:")
    if not streaming_info['uses_onnx']:
        print("   1. Convert to ONNX for faster inference")
    else:
        print("   1. ✓ Already using ONNX Runtime")
    
    if device == "cpu":
        print("   2. Use GPU with ONNX CUDA/TensorRT execution provider")
    else:
        print("   2. Enable ONNX CUDA execution provider (may need onnxruntime-gpu)")
    
    print("   3. Quantization (INT8) for CPU deployment")
    print("   4. Smaller model variant (30M params) for mobile/edge")
    print("   5. Audio chunking with state caching for true streaming")
    
    # Return for programmatic use
    return {
        "device": device,
        "audio_duration_s": audio_duration,
        "ctc": ctc_results,
        "rnnt": rnnt_results,
        "streaming": streaming_results,
        "streaming_support": streaming_info,
    }


if __name__ == "__main__":
    main()
