import os
import torch
import torchaudio
import time
from pathlib import Path
from transformers import AutoModel
import logging
import warnings

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

class IndicConformerASR:
    def __init__(self, device='cuda', debug=False, stream_chunk_ms=800, stream_stride_ms=200, stream_max_buffer_ms=6000):
        self.device = device
        self.debug = debug
        if self.debug:
            print(f"Loading Indic Conformer on {device}...")
        
        self.model = AutoModel.from_pretrained(
            "ai4bharat/indic-conformer-600m-multilingual",
            trust_remote_code=True
        )
        
        self.target_sample_rate = 16000
        self.stream_chunk_ms = stream_chunk_ms
        self.stream_stride_ms = stream_stride_ms
        self.stream_max_buffer_ms = stream_max_buffer_ms
        self._stream_reset()
        self._warmup()
        self._log_ort_providers()

    def _warmup(self):
        if self.debug:
            print("Warming up Indic Conformer...")
        dummy_input = torch.zeros(1, 16000)
        with torch.no_grad():
            _ = self.model(dummy_input, "hi", "ctc")

    def _log_ort_providers(self):
        if not self.debug:
            return
        try:
            import onnxruntime as ort
        except Exception:
            print("onnxruntime not available; Indic Conformer will run on CPU.")
            return
        providers = ort.get_available_providers()
        print(f"onnxruntime providers: {providers}")
        if hasattr(self.model, "models") and isinstance(self.model.models, dict):
            enc = self.model.models.get("encoder")
            if enc is not None and hasattr(enc, "get_providers"):
                print(f"Indic Conformer encoder providers: {enc.get_providers()}")

    def _stream_reset(self):
        self._stream_buffer = None
        self._stream_text = ""
        self._stream_lang = None

    def preprocess(self, audio):
        if isinstance(audio, (str, Path)):
            wav, sr = torchaudio.load(audio)
        else:
            wav = audio
            sr = 16000

        if not torch.is_tensor(wav):
            wav = torch.tensor(wav)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
            
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sample_rate)
            wav = resampler(wav)
            
        return wav

    def transcribe(self, audio, lang_tag, decoder='ctc'):
        wav = self.preprocess(audio)
        
        start_time = time.perf_counter()
        with torch.no_grad():
            transcription = self.model(wav, lang_tag, decoder)
        
        if self.debug:
            latency = (time.perf_counter() - start_time) * 1000
            print(f"Indic Conformer ({lang_tag}, {decoder}) Latency: {latency:.2f}ms")
            
        return transcription

    def transcribe_stream(self, audio_chunk, lang_tag):
        wav = self.preprocess(audio_chunk)
        if self._stream_lang != lang_tag:
            self._stream_reset()
            self._stream_lang = lang_tag

        if self._stream_buffer is None:
            self._stream_buffer = wav
        else:
            self._stream_buffer = torch.cat([self._stream_buffer, wav], dim=1)

        chunk_samples = int(self.target_sample_rate * self.stream_chunk_ms / 1000)
        stride_samples = int(self.target_sample_rate * self.stream_stride_ms / 1000)
        max_buffer_samples = int(self.target_sample_rate * self.stream_max_buffer_ms / 1000)

        if self._stream_buffer.shape[1] < chunk_samples:
            return ""

        if self._stream_buffer.shape[1] > max_buffer_samples:
            self._stream_buffer = self._stream_buffer[:, -max_buffer_samples:]

        window_samples = min(self._stream_buffer.shape[1], chunk_samples + stride_samples)
        window = self._stream_buffer[:, -window_samples:]

        text = self.transcribe(window, lang_tag, decoder='ctc')
        if isinstance(text, (list, tuple)):
            text = text[0]

        if text.startswith(self._stream_text):
            incremental = text[len(self._stream_text):].lstrip()
        else:
            incremental = text

        self._stream_text = text
        return incremental

    def reset_stream(self):
        self._stream_reset()
