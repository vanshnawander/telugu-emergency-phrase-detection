import os
import torch
import torchaudio
import time
from pathlib import Path
import logging
import warnings
import threading
from typing import List
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# These are the server-specific imports your main script expects
from server.config import ServerConfig

os.environ["NEMO_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")

try:
    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict
except ImportError:
    pass

logger = logging.getLogger("asr.engine")

class ASREngine:
    def __init__(self, config: ServerConfig):
        self.config = config
        # Logic strictly from your snippet
        self.device = 'cuda' if torch.cuda.is_available() and config.device == 'cuda' else 'cpu'
        self._lock = threading.Lock()
        self.target_sample_rate = 16000
        
        logger.info(f"Loading Models on {self.device}...")
        
        # Load Parakeet (English) using your logic
        # self._load_parakeet(config)
        
        # Load IndicConformer (Hindi/Telugu)
        self._load_indic(config)

        # Load Whisper Tiny Telugu
        self._load_whisper(config)

    def _load_parakeet(self, config: ServerConfig):
        # Logic strictly from your snippet
        self.parakeet = nemo_asr.models.ASRModel.from_pretrained(config.parakeet_model)
        self.parakeet.to(self.device)
        self.parakeet.eval()
        self._configure_decoding()
        self._warmup_parakeet()

    def _configure_decoding(self):
        # Logic strictly from your snippet
        try:
            from omegaconf import open_dict
        except Exception:
            open_dict = None

        decoding_cfg = None
        if hasattr(self.parakeet, "cfg") and "decoding" in self.parakeet.cfg:
            decoding_cfg = self.parakeet.cfg.decoding

        if decoding_cfg is not None and open_dict is not None:
            with open_dict(decoding_cfg):
                if "use_cuda_graph_decoder" in decoding_cfg:
                    decoding_cfg.use_cuda_graph_decoder = False
                if "loop_labels" in decoding_cfg:
                    decoding_cfg.loop_labels = False
            try:
                self.parakeet.change_decoding_strategy(decoding_cfg)
            except Exception:
                pass

        if hasattr(self.parakeet, "decoding") and hasattr(self.parakeet.decoding, "disable_cuda_graphs"):
            try:
                self.parakeet.decoding.disable_cuda_graphs()
            except Exception:
                pass

    def _disable_cuda_graphs_runtime(self):
        # Logic strictly from your snippet
        decoding = getattr(self.parakeet, "decoding", None)
        if decoding is None:
            return

        for attr in ["use_cuda_graphs", "cuda_graphs", "enable_cuda_graphs", "use_cuda_graph_decoder"]:
            if hasattr(decoding, attr):
                try:
                    setattr(decoding, attr, False)
                except Exception:
                    pass

        for child_name in ["decoding", "_decoding", "decoding_computer", "_decoding_computer"]:
            child = getattr(decoding, child_name, None)
            if child is None:
                continue
            for attr in ["use_cuda_graphs", "cuda_graphs", "enable_cuda_graphs", "_use_cuda_graphs"]:
                if hasattr(child, attr):
                    try:
                        setattr(child, attr, False)
                    except Exception:
                        pass
            if hasattr(child, "disable_cuda_graphs"):
                try:
                    child.disable_cuda_graphs()
                except Exception:
                    pass

    def _warmup_parakeet(self):
        # Logic strictly from your snippet
        dummy_input = torch.zeros(1, 16000).to(self.device)
        with torch.no_grad():
            processed_signal, processed_signal_len = self.parakeet.preprocessor(
                input_signal=dummy_input,
                length=torch.tensor([16000]).to(self.device)
            )
            _ = self.parakeet.encoder(audio_signal=processed_signal, length=processed_signal_len)

    def _load_indic(self, config: ServerConfig):
        from transformers import AutoModel
        self.indic = AutoModel.from_pretrained(config.indic_model, trust_remote_code=True)
        self.indic.to(self.device)
        self.indic.eval()
        
        # Warmup Indic
        dummy = torch.zeros(1, 16000, device=self.device)
        with torch.no_grad():
            _ = self.indic(dummy, "hi", config.indic_decoder)

    def _load_whisper(self, config: ServerConfig):
        self.whisper_processor = WhisperProcessor.from_pretrained(config.whisper_model_path)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(config.whisper_model_path)
        self.whisper_model.to(self.device)
        self.whisper_model.eval()

        forced_ids = self.whisper_processor.get_decoder_prompt_ids(
            language=config.whisper_language,
            task="transcribe"
        )
        self.whisper_model.config.forced_decoder_ids = forced_ids

    def preprocess(self, audio):
        # Logic strictly from your snippet
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

    def transcribe(self, audio: torch.Tensor, lang: str, backend: str = "indic") -> dict:
        # Integrated Entry Point
        wav = self.preprocess(audio)
        start_time = time.perf_counter()

        with self._lock:
            if backend == "whisper":
                text = self._transcribe_whisper(wav)
            elif lang in ("hi", "te"):
                text = self._transcribe_indic(wav, lang)
            else:
                text = ""

        latency = (time.perf_counter() - start_time) * 1000
        return {"text": text, "latency_ms": latency}

    @torch.no_grad()
    def _transcribe_parakeet(self, wav: torch.Tensor) -> str:
        # Logic strictly from your snippet
        wav_gpu = wav.to(self.device)
        processed_signal, processed_signal_len = self.parakeet.preprocessor(
            input_signal=wav_gpu,
            length=torch.tensor([wav_gpu.shape[1]]).to(self.device)
        )
        encoded, encoded_len = self.parakeet.encoder(
            audio_signal=processed_signal,
            length=processed_signal_len
        )
        self._disable_cuda_graphs_runtime()
        hypotheses = self.parakeet.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded,
            encoded_lengths=encoded_len,
            return_hypotheses=True
        )
        return self._extract_text(hypotheses)

    @torch.no_grad()
    def _transcribe_indic(self, wav: torch.Tensor, lang: str) -> str:
        res = self.indic(wav.to(self.device), lang, self.config.indic_decoder)
        if isinstance(res, (list, tuple)):
            res = res[0] if res else ""
        return str(res)

    @torch.no_grad()
    def _transcribe_whisper(self, wav: torch.Tensor) -> str:
        inputs = self.whisper_processor(wav.squeeze(0), sampling_rate=self.target_sample_rate, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)

        predicted_ids = self.whisper_model.generate(input_features, max_new_tokens=128)
        text = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return text.strip()

    def _extract_text(self, hypotheses):
        # Logic strictly from your snippet
        if hypotheses is None:
            return ""
        if hasattr(hypotheses, "text"):
            return hypotheses.text or ""
        if isinstance(hypotheses, (list, tuple)):
            if len(hypotheses) == 0:
                return ""
            first = hypotheses[0]
            if hasattr(first, "text"):
                return first.text or ""
            if isinstance(first, (list, tuple)):
                if len(first) == 0:
                    return ""
                second = first[0]
                if hasattr(second, "text"):
                    return second.text or ""
        return ""

    def transcribe_batch(self, audios: List[torch.Tensor], lang: str, backend: str = "indic") -> List[dict]:
        return [self.transcribe(a, lang, backend=backend) for a in audios]