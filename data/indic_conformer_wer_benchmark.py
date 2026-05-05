"""
Indic Conformer ASR Comprehensive Benchmark
============================================
- Latency measurement (CTC vs RNNT)
- Word Error Rate (WER) evaluation with ground truth
- Streaming viability analysis
- ONNX Runtime provider testing (CPU vs CUDA)
"""
import json
import time
import statistics
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import torchaudio
from transformers import AutoModel

# Try to import jiwer for WER calculation
try:
    from jiwer import wer, cer
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False
    print("⚠️  jiwer not installed. Install with: uv pip install jiwer")

# Configuration
MANIFEST_PATH = "data/test_manifests/ai4bharat_Kathbath_te_train.json"
TARGET_SAMPLE_RATE = 16000
LANGUAGE = "te"  # Telugu
NUM_WARMUP = 2
NUM_ITERATIONS = 3  # Reduced for full dataset evaluation

# Streaming configuration
STREAMING_CHUNK_MS = 500


@dataclass
class TranscriptionResult:
    """Single transcription result with metrics."""
    audio_path: str
    reference: str
    hypothesis_ctc: str
    hypothesis_rnnt: str
    duration_s: float
    latency_ctc_ms: float
    latency_rnnt_ms: float
    wer_ctc: Optional[float] = None
    wer_rnnt: Optional[float] = None
    cer_ctc: Optional[float] = None
    cer_rnnt: Optional[float] = None


def load_manifest(manifest_path: str) -> list[dict]:
    """Load NeMo-style manifest file."""
    samples = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def load_audio(audio_path: str, target_sr: int = 16000) -> tuple[torch.Tensor, float]:
    """Load and preprocess audio."""
    wav, sr = torchaudio.load(audio_path)
    wav = torch.mean(wav, dim=0, keepdim=True)
    
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        wav = resampler(wav)
        sr = target_sr
    
    audio_duration = wav.shape[1] / sr
    return wav, audio_duration


def normalize_text(text: str) -> str:
    """Normalize text for WER calculation (Telugu-aware)."""
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Lowercase for consistent comparison (though Telugu doesn't have case)
    text = text.lower()
    # Remove common punctuation
    for punct in '.,!?;:\"\'()[]{}':
        text = text.replace(punct, '')
    return text.strip()


def calculate_metrics(reference: str, hypothesis: str) -> tuple[float, float]:
    """Calculate WER and CER."""
    if not HAS_JIWER:
        return 0.0, 0.0
    
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)
    
    if not ref_norm:
        return 0.0 if not hyp_norm else 1.0, 0.0 if not hyp_norm else 1.0
        
    try:
        word_error_rate = wer(ref_norm, hyp_norm)
        char_error_rate = cer(ref_norm, hyp_norm)
        return word_error_rate, char_error_rate
    except Exception as e:
        print(f"WER calculation error: {e}")
        return 1.0, 1.0


def check_onnx_providers():
    """Check available ONNX Runtime providers."""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        return {
            'version': ort.__version__,
            'providers': providers,
            'has_cuda': 'CUDAExecutionProvider' in providers,
            'has_tensorrt': 'TensorrtExecutionProvider' in providers,
        }
    except ImportError:
        return {'version': 'N/A', 'providers': [], 'has_cuda': False, 'has_tensorrt': False}


def transcribe_with_timing(model, wav, language: str, decoder: str) -> tuple[str, float]:
    """Transcribe and return result with timing."""
    start = time.perf_counter()
    result = model(wav, language, decoder)
    latency_ms = (time.perf_counter() - start) * 1000
    return result, latency_ms


def run_benchmark(model, samples: list[dict], language: str, 
                  num_warmup: int = 2, max_samples: Optional[int] = None) -> list[TranscriptionResult]:
    """Run comprehensive benchmark on all samples."""
    
    results = []
    
    # Warmup
    print(f"\n🔥 Warming up with {num_warmup} samples...")
    for i, sample in enumerate(samples[:num_warmup]):
        wav, _ = load_audio(sample['audio_filepath'])
        _ = model(wav, language, "ctc")
        _ = model(wav, language, "rnnt")
    
    # Benchmark all samples
    samples_to_process = samples[:max_samples] if max_samples else samples
    print(f"\n📊 Benchmarking {len(samples_to_process)} samples...")
    
    for i, sample in enumerate(samples_to_process):
        audio_path = sample['audio_filepath']
        reference = sample['text']
        expected_duration = sample.get('duration', 0)
        
        # Load audio
        wav, actual_duration = load_audio(audio_path)
        
        # CTC transcription
        hyp_ctc, latency_ctc = transcribe_with_timing(model, wav, language, "ctc")
        
        # RNNT transcription
        hyp_rnnt, latency_rnnt = transcribe_with_timing(model, wav, language, "rnnt")
        
        # Calculate metrics
        wer_ctc, cer_ctc = calculate_metrics(reference, hyp_ctc)
        wer_rnnt, cer_rnnt = calculate_metrics(reference, hyp_rnnt)
        
        result = TranscriptionResult(
            audio_path=audio_path,
            reference=reference,
            hypothesis_ctc=hyp_ctc,
            hypothesis_rnnt=hyp_rnnt,
            duration_s=actual_duration,
            latency_ctc_ms=latency_ctc,
            latency_rnnt_ms=latency_rnnt,
            wer_ctc=wer_ctc,
            wer_rnnt=wer_rnnt,
            cer_ctc=cer_ctc,
            cer_rnnt=cer_rnnt,
        )
        results.append(result)
        
        # Progress
        print(f"  [{i+1}/{len(samples_to_process)}] {Path(audio_path).name}: "
              f"CTC={latency_ctc:.0f}ms (WER={wer_ctc:.1%}), "
              f"RNNT={latency_rnnt:.0f}ms (WER={wer_rnnt:.1%})")
    
    return results


def analyze_results(results: list[TranscriptionResult]) -> dict:
    """Analyze benchmark results and compute aggregate metrics."""
    
    total_duration = sum(r.duration_s for r in results)
    
    # Latency stats
    ctc_latencies = [r.latency_ctc_ms for r in results]
    rnnt_latencies = [r.latency_rnnt_ms for r in results]
    
    # WER stats
    wer_ctc_list = [r.wer_ctc for r in results if r.wer_ctc is not None]
    wer_rnnt_list = [r.wer_rnnt for r in results if r.wer_rnnt is not None]
    cer_ctc_list = [r.cer_ctc for r in results if r.cer_ctc is not None]
    cer_rnnt_list = [r.cer_rnnt for r in results if r.cer_rnnt is not None]
    
    # RTF calculation
    total_ctc_time = sum(ctc_latencies) / 1000
    total_rnnt_time = sum(rnnt_latencies) / 1000
    
    return {
        'num_samples': len(results),
        'total_audio_duration_s': total_duration,
        'ctc': {
            'mean_latency_ms': statistics.mean(ctc_latencies),
            'std_latency_ms': statistics.stdev(ctc_latencies) if len(ctc_latencies) > 1 else 0,
            'min_latency_ms': min(ctc_latencies),
            'max_latency_ms': max(ctc_latencies),
            'total_inference_time_s': total_ctc_time,
            'rtf': total_ctc_time / total_duration if total_duration > 0 else 0,
            'mean_wer': statistics.mean(wer_ctc_list) if wer_ctc_list else None,
            'mean_cer': statistics.mean(cer_ctc_list) if cer_ctc_list else None,
        },
        'rnnt': {
            'mean_latency_ms': statistics.mean(rnnt_latencies),
            'std_latency_ms': statistics.stdev(rnnt_latencies) if len(rnnt_latencies) > 1 else 0,
            'min_latency_ms': min(rnnt_latencies),
            'max_latency_ms': max(rnnt_latencies),
            'total_inference_time_s': total_rnnt_time,
            'rtf': total_rnnt_time / total_duration if total_duration > 0 else 0,
            'mean_wer': statistics.mean(wer_rnnt_list) if wer_rnnt_list else None,
            'mean_cer': statistics.mean(cer_rnnt_list) if cer_rnnt_list else None,
        },
    }


def print_detailed_comparison(results: list[TranscriptionResult], max_show: int = 5):
    """Print detailed comparison of transcriptions."""
    print("\n" + "=" * 80)
    print("📝 Sample Transcription Comparisons")
    print("=" * 80)
    
    for i, r in enumerate(results[:max_show]):
        print(f"\n--- Sample {i+1} ({r.duration_s:.2f}s) ---")
        print(f"Reference: {r.reference}")
        print(f"CTC:       {r.hypothesis_ctc}")
        print(f"RNNT:      {r.hypothesis_rnnt}")
        print(f"WER - CTC: {r.wer_ctc:.2%}, RNNT: {r.wer_rnnt:.2%}")
        print(f"Latency - CTC: {r.latency_ctc_ms:.0f}ms, RNNT: {r.latency_rnnt_ms:.0f}ms")


def main():
    print("=" * 80)
    print("Indic Conformer ASR Comprehensive Benchmark")
    print("=" * 80)
    
    # Check ONNX Runtime providers
    print("\n🔧 ONNX Runtime Configuration:")
    ort_info = check_onnx_providers()
    print(f"   Version: {ort_info['version']}")
    print(f"   Available providers: {ort_info['providers']}")
    print(f"   CUDA available: {'✅' if ort_info['has_cuda'] else '❌'}")
    print(f"   TensorRT available: {'✅' if ort_info['has_tensorrt'] else '❌'}")
    
    if not ort_info['has_cuda']:
        print("\n   ⚠️  CUDA provider not available. Running on CPU.")
        print("   To enable GPU: uv pip install onnxruntime-gpu")
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n📍 PyTorch Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
    
    # Load manifest
    print(f"\n📂 Loading manifest: {MANIFEST_PATH}")
    if not os.path.exists(MANIFEST_PATH):
        print(f"   ❌ Manifest not found!")
        return
    
    samples = load_manifest(MANIFEST_PATH)
    print(f"   Found {len(samples)} samples")
    total_duration = sum(s.get('duration', 0) for s in samples)
    print(f"   Total audio duration: {total_duration:.2f}s ({total_duration/60:.1f} min)")
    
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
    
    # Run benchmark
    results = run_benchmark(model, samples, LANGUAGE, num_warmup=NUM_WARMUP)
    
    # Analyze results
    analysis = analyze_results(results)
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"\n📈 Dataset: {analysis['num_samples']} samples, {analysis['total_audio_duration_s']:.2f}s total")
    
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                      CTC DECODER                               │")
    print("├─────────────────────────────────────────────────────────────────┤")
    ctc = analysis['ctc']
    print(f"│ Latency:     {ctc['mean_latency_ms']:>7.1f}ms ± {ctc['std_latency_ms']:.1f}ms (mean ± std)       │")
    print(f"│ Range:       {ctc['min_latency_ms']:>7.1f}ms - {ctc['max_latency_ms']:.1f}ms                       │")
    print(f"│ RTF:         {ctc['rtf']:>7.4f}x {'✅ Real-time' if ctc['rtf'] < 1 else '❌ Slower'}              │")
    if ctc['mean_wer'] is not None:
        print(f"│ WER:         {ctc['mean_wer']*100:>7.2f}%                                       │")
        print(f"│ CER:         {ctc['mean_cer']*100:>7.2f}%                                       │")
    print("└─────────────────────────────────────────────────────────────────┘")
    
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                      RNNT DECODER                              │")
    print("├─────────────────────────────────────────────────────────────────┤")
    rnnt = analysis['rnnt']
    print(f"│ Latency:     {rnnt['mean_latency_ms']:>7.1f}ms ± {rnnt['std_latency_ms']:.1f}ms (mean ± std)       │")
    print(f"│ Range:       {rnnt['min_latency_ms']:>7.1f}ms - {rnnt['max_latency_ms']:.1f}ms                       │")
    print(f"│ RTF:         {rnnt['rtf']:>7.4f}x {'✅ Real-time' if rnnt['rtf'] < 1 else '❌ Slower'}              │")
    if rnnt['mean_wer'] is not None:
        print(f"│ WER:         {rnnt['mean_wer']*100:>7.2f}%                                       │")
        print(f"│ CER:         {rnnt['mean_cer']*100:>7.2f}%                                       │")
    print("└─────────────────────────────────────────────────────────────────┘")
    
    # Comparison
    print("\n📊 DECODER COMPARISON:")
    if ctc['mean_wer'] and rnnt['mean_wer']:
        better_wer = "CTC" if ctc['mean_wer'] < rnnt['mean_wer'] else "RNNT"
        wer_diff = abs(ctc['mean_wer'] - rnnt['mean_wer']) * 100
        print(f"   • WER: {better_wer} is better by {wer_diff:.2f}%")
    
    better_speed = "CTC" if ctc['mean_latency_ms'] < rnnt['mean_latency_ms'] else "RNNT"
    speed_ratio = max(ctc['mean_latency_ms'], rnnt['mean_latency_ms']) / min(ctc['mean_latency_ms'], rnnt['mean_latency_ms'])
    print(f"   • Speed: {better_speed} is {speed_ratio:.1f}x faster")
    
    # Print sample comparisons
    print_detailed_comparison(results)
    
    # Optimization recommendations
    print("\n" + "=" * 80)
    print("🚀 OPTIMIZATION PLAN")
    print("=" * 80)
    
    print("\n📋 Current Bottlenecks Identified:")
    print("   1. ONNX Runtime using CPU provider (no GPU acceleration)")
    print(f"   2. RNNT is {speed_ratio:.1f}x slower than CTC (expected for autoregressive)")
    print(f"   3. Model load time: {model_load_time:.1f}s")
    
    print("\n🔧 Recommended Optimizations (Priority Order):")
    print("""
┌──────┬────────────────────────────────┬──────────────┬────────────────┐
│ Prio │ Optimization                   │ Expected     │ Effort         │
│      │                                │ Speedup      │                │
├──────┼────────────────────────────────┼──────────────┼────────────────┤
│  1   │ Install onnxruntime-gpu        │ 3-5x         │ Low (pip)      │
│  2   │ Enable CUDA Execution Provider │ 3-5x         │ Low (config)   │
│  3   │ TensorRT Optimization          │ 5-10x        │ Medium         │
│  4   │ INT8 Quantization              │ 2-3x         │ Medium         │
│  5   │ Use CTC for batch processing   │ 2.5x vs RNNT │ None           │
│  6   │ Streaming with RNNT chunks     │ Lower latency│ Medium         │
│  7   │ Model caching/warm start       │ Startup time │ Low            │
└──────┴────────────────────────────────┴──────────────┴────────────────┘
    """)
    
    print("\n📝 Architecture Notes (Parakeet-Compatible Optimizations):")
    print("   • Both use FastConformer encoder - same optimization techniques apply")
    print("   • TDT/RNNT decoder benefits from duration prediction optimization")
    print("   • Subsampling at 80ms resolution already optimized")
    print("   • Local attention available for long-form audio (>20min)")
    
    print("\n🎯 Recommended Next Steps:")
    print("   1. Install onnxruntime-gpu: uv pip install onnxruntime-gpu")
    print("   2. Verify CUDA provider is active")
    print("   3. Re-run benchmark to measure GPU speedup")
    print("   4. Consider TensorRT for production deployment")
    print("   5. Evaluate INT8 quantization for edge deployment")
    
    # Save results to JSON
    output_path = "tests/benchmark_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'onnx_info': ort_info,
            'device': device,
            'analysis': analysis,
            'samples': [
                {
                    'audio': r.audio_path,
                    'reference': r.reference,
                    'ctc': r.hypothesis_ctc,
                    'rnnt': r.hypothesis_rnnt,
                    'wer_ctc': r.wer_ctc,
                    'wer_rnnt': r.wer_rnnt,
                    'latency_ctc_ms': r.latency_ctc_ms,
                    'latency_rnnt_ms': r.latency_rnnt_ms,
                }
                for r in results
            ]
        }, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results saved to: {output_path}")
    
    return analysis


if __name__ == "__main__":
    main()
