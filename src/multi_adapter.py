from pathlib import Path
from typing import Dict, List, Optional

from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.config import LailaConfig, resolve_device, resolve_dtype


class MultiAdapterManager:
    def __init__(
        self,
        config: Optional[LailaConfig] = None,
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
    ) -> None:
        self.config = config or LailaConfig()
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(self.device)
        self.hf_token = hf_token
        self.loaded_adapters: Dict[str, str] = {}
        self.active_adapter: Optional[str] = None
        self.is_peft_wrapped: bool = False

        self.processor = self._load_processor()
        self.model = self._load_base_model()
        self._load_initial_adapters()

    def _load_processor(self) -> WhisperProcessor:
        try:
            return WhisperProcessor.from_pretrained(self.config.base_model, token=self.hf_token)
        except Exception:
            return WhisperProcessor.from_pretrained(self.config.fallback_base_model, token=self.hf_token)

    def _load_base_model(self) -> WhisperForConditionalGeneration:
        try:
            model = WhisperForConditionalGeneration.from_pretrained(
                self.config.base_model,
                torch_dtype=self.dtype,
                token=self.hf_token,
            )
        except Exception:
            model = WhisperForConditionalGeneration.from_pretrained(
                self.config.fallback_base_model,
                torch_dtype=self.dtype,
                token=self.hf_token,
            )

        for param in model.parameters():
            param.requires_grad = False
        model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def _has_adapter_weights(path: Path) -> bool:
        if not path.exists() or not path.is_dir():
            return False
        has_config = (path / "adapter_config.json").exists()
        has_weights = (path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists()
        return has_config and has_weights

    def _load_initial_adapters(self) -> None:
        for name, rel_path in self.config.adapter_registry.items():
            path = Path(rel_path)
            if self._has_adapter_weights(path):
                self.load_adapter(name, str(path))

        if self.config.default_adapter in self.loaded_adapters:
            self.set_adapter(self.config.default_adapter)
        elif self.loaded_adapters:
            first_adapter = next(iter(self.loaded_adapters))
            self.set_adapter(first_adapter)

    def load_adapter(self, adapter_name: str, adapter_path: str) -> bool:
        path = Path(adapter_path)
        if not self._has_adapter_weights(path):
            fallback_path = Path(self.config.adapter_registry.get(self.config.default_adapter, ""))
            if adapter_name != self.config.default_adapter and self._has_adapter_weights(fallback_path):
                path = fallback_path
                adapter_name = self.config.default_adapter
            else:
                return False

        if adapter_name in self.loaded_adapters:
            return True

        if not self.is_peft_wrapped:
            self.model = PeftModel.from_pretrained(
                self.model,
                str(path),
                adapter_name=adapter_name,
            )
            self.model.to(self.device)
            self.model.eval()
            self.is_peft_wrapped = True
        else:
            self.model.load_adapter(str(path), adapter_name=adapter_name)

        self.loaded_adapters[adapter_name] = str(path)
        if self.active_adapter is None:
            self.active_adapter = adapter_name
        return True

    def set_adapter(self, adapter_name: str) -> str:
        if not self.is_peft_wrapped:
            self.active_adapter = None
            return "base"

        if adapter_name not in self.loaded_adapters:
            reg_path = self.config.adapter_registry.get(adapter_name)
            if reg_path:
                self.load_adapter(adapter_name, reg_path)

        if adapter_name in self.loaded_adapters:
            self.model.set_adapter(adapter_name)
            self.active_adapter = adapter_name
            return adapter_name

        if self.config.default_adapter in self.loaded_adapters:
            self.model.set_adapter(self.config.default_adapter)
            self.active_adapter = self.config.default_adapter
            return self.config.default_adapter

        return self.active_adapter or "base"

    def available_adapters(self) -> List[str]:
        if self.loaded_adapters:
            return list(self.loaded_adapters.keys())
        return list(self.config.adapter_registry.keys())
