from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    device: str = "cuda"
    sample_rate: int = 16000

    # Concurrency — max parallel inference requests dispatched to thread pool
    max_workers: int = 2

    # Model choices
    # parakeet_model: str = "nvidia/parakeet-tdt-0.6b-v3"
    indic_model: str = "ai4bharat/indic-conformer-600m-multilingual"
    whisper_model_path: str = str(Path(__file__).resolve().parent.parent.parent / "whisper-tiny-telugu")
    whisper_language: str = "telugu"

    # Decoding strategy
    indic_decoder: str = "ctc"
    # parakeet_use_tdt: bool = True

    debug: bool = False
