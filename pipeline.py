from models.indic_conformer import IndicConformerASR
from models.parakeet import ParakeetASR
import torch

class UnifiedASRPipeline:
    def __init__(self, device='cuda', debug=False):
        self.device = 'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.debug = debug
        
        self.indic_model = IndicConformerASR(device=self.device, debug=self.debug)
        self.parakeet_model = ParakeetASR(device=self.device, debug=self.debug)
        
        if self.debug:
            print(f"Unified ASR Pipeline Ready (Device: {self.device})")

    def transcribe(self, audio, lang_tag):
        if lang_tag in ['hi', 'te']:
            return self.indic_model.transcribe(audio, lang_tag)
        elif lang_tag == 'en':
            return self.parakeet_model.transcribe(audio)
        else:
            raise ValueError(f"Unsupported language: {lang_tag}")

    def transcribe_stream(self, audio_chunk, lang_tag):
        if lang_tag in ['hi', 'te']:
            return self.indic_model.transcribe_stream(audio_chunk, lang_tag)
        elif lang_tag == 'en':
            return self.parakeet_model.transcribe_stream(audio_chunk)
        else:
            raise ValueError(f"Unsupported language: {lang_tag}")

    def close(self):
        self.indic_model = None
        self.parakeet_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reset_stream(self, lang_tag=None):
        if lang_tag is None:
            if hasattr(self.indic_model, "reset_stream"):
                self.indic_model.reset_stream()
            if hasattr(self.parakeet_model, "reset_stream"):
                self.parakeet_model.reset_stream()
            return
        if lang_tag in ['hi', 'te'] and hasattr(self.indic_model, "reset_stream"):
            self.indic_model.reset_stream()
        if lang_tag == 'en' and hasattr(self.parakeet_model, "reset_stream"):
            self.parakeet_model.reset_stream()
