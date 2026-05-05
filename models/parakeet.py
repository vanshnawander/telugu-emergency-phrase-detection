import os
import torch
import torchaudio
import time
from pathlib import Path
import logging
import warnings

os.environ["NEMO_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")

try:
    import nemo.collections.asr as nemo_asr
except ImportError:
    pass

class ParakeetASR:
    def __init__(self, device='cuda', debug=False):
        self.device = 'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.debug = debug
        if self.debug:
            print(f"Loading Parakeet on {self.device}...")
            
        try:
            self.model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
        except Exception:
            self.model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
        self.model.to(self.device)
        self.model.eval()
        self._configure_decoding()
        
        self.target_sample_rate = 16000
        self._warmup()

    def _configure_decoding(self):
        # Disable CUDA graphs (unstable with multi-model GPU sharing),
        # keep loop_labels for speed
        try:
            from omegaconf import open_dict
        except Exception:
            return

        if not (hasattr(self.model, "cfg") and "decoding" in self.model.cfg):
            return

        decoding_cfg = self.model.cfg.decoding
        with open_dict(decoding_cfg):
            if "use_cuda_graph_decoder" in decoding_cfg:
                decoding_cfg.use_cuda_graph_decoder = False
            if "loop_labels" in decoding_cfg:
                decoding_cfg.loop_labels = True
        try:
            self.model.change_decoding_strategy(decoding_cfg)
        except Exception:
            pass

    def _warmup(self):
        if self.debug:
            print("Warming up Parakeet...")
        dummy_input = torch.zeros(1, 16000).to(self.device)
        with torch.no_grad():
            processed_signal, processed_signal_len = self.model.preprocessor(
                input_signal=dummy_input,
                length=torch.tensor([16000]).to(self.device)
            )
            _ = self.model.encoder(audio_signal=processed_signal, length=processed_signal_len)

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

    @torch.no_grad()
    def transcribe(self, audio):
        wav = self.preprocess(audio)
        start_time = time.perf_counter()
        wav_gpu = wav.to(self.device)
        processed_signal, processed_signal_len = self.model.preprocessor(
            input_signal=wav_gpu,
            length=torch.tensor([wav_gpu.shape[1]]).to(self.device)
        )
        encoded, encoded_len = self.model.encoder(
            audio_signal=processed_signal,
            length=processed_signal_len
        )
        hypotheses = self.model.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded,
            encoded_lengths=encoded_len,
            return_hypotheses=True
        )
        text = self._extract_text(hypotheses)

        if self.debug:
            latency = (time.perf_counter() - start_time) * 1000
            print(f"Parakeet Latency: {latency:.2f}ms")

        return text

    def transcribe_stream(self, audio_chunk):
        return self.transcribe(audio_chunk)

    def _extract_text(self, hypotheses):
        try:
            # Standard NeMo: (best_hyps_list, all_hyps)
            if isinstance(hypotheses, tuple) and len(hypotheses) >= 1:
                best = hypotheses[0]
            else:
                best = hypotheses
            if isinstance(best, list) and len(best) > 0:
                hyp = best[0]
                if hasattr(hyp, "text"):
                    return hyp.text or ""
                return str(hyp) if hyp else ""
            if hasattr(best, "text"):
                return best.text or ""
            return str(best) if best else ""
        except Exception:
            return ""

    def reset_stream(self):
        # Stateless decoder today; keep for API symmetry.
        return None
