import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch


def load_dotenv_file(env_path: str = ".env") -> None:
    path = Path(env_path)
    if not path.exists() or not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_hf_token() -> Optional[str]:
    load_dotenv_file()
    token = os.environ.get("HF_TOKEN")
    return token.strip() if token else None


def require_hf_token(token: Optional[str] = None) -> str:
    resolved = token.strip() if token else get_hf_token()
    if not resolved or resolved == "YOUR_HF_TOKEN":
        raise ValueError(
            "HF_TOKEN is missing or still set to 'YOUR_HF_TOKEN'. "
            "Set the HF_TOKEN environment variable or add HF_TOKEN=... to your .env file."
        )
    return resolved


def resolve_device(preferred_device: str | None = None) -> str:
    if preferred_device:
        if preferred_device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return preferred_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(device: str) -> torch.dtype:
    return torch.float16 if device.startswith("cuda") else torch.float32


@dataclass
class LailaConfig:
    base_model: str = "ai4bharat/indicwhisper-hi"
    fallback_base_model: str = "openai/whisper-small"
    hub_model_id: str = "techindro/Laila"
    language: str = "hi"
    task: str = "transcribe"

    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    target_modules: Tuple[str, ...] = ("q_proj", "v_proj")

    sample_rate: int = 16000
    n_mels: int = 80
    max_audio_seconds: float = 30.0
    max_label_length: int = 225

    snr_min_db: float = 5.0
    snr_max_db: float = 20.0
    noise_prob: float = 0.5
    noise_dir: str = "data/noise"

    output_dir: str = "outputs/laila-indicwhisper-lora"
    adapters_dir: str = "adapters"
    default_adapter: str = "hinglish"
    adapter_registry: Dict[str, str] = field(
        default_factory=lambda: {
            "hinglish": "adapters/hinglish",
            "bhojpuri": "adapters/bhojpuri",
            "indian_english": "adapters/indian_english",
        }
    )

    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_steps: int = 4000
    save_steps: int = 250
    eval_steps: int = 250
    logging_steps: int = 25
    save_total_limit: int = 3
    fp16: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 2
    seed: int = 42

    indicvoices_dataset: str = "ai4bharat/IndicVoices"
    indicvoices_config: str = "hindi"
    common_voice_dataset: str = "mozilla-foundation/common_voice_17_0"
    local_hinglish_manifest: str = "data/hinglish/manifest.jsonl"
    hf_token: Optional[str] = field(default_factory=get_hf_token)

    def validate_hf_token(self) -> str:
        self.hf_token = require_hf_token(self.hf_token)
        return self.hf_token

    def ensure_directories(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.adapters_dir).mkdir(parents=True, exist_ok=True)
        for path in self.adapter_registry.values():
            Path(path).mkdir(parents=True, exist_ok=True)
