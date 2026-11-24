from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from src.config import LailaConfig
from src.dataset import load_audio_16k_mono
from src.multi_adapter import MultiAdapterManager
from src.script_toggle import ScriptMode, ScriptToggler
from src.streaming_vad import SileroVADStream


class LailaASR:
    def __init__(
        self,
        config: Optional[LailaConfig] = None,
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
        use_vad: bool = True,
    ) -> None:
        self.config = config or LailaConfig()
        self.adapter_manager = MultiAdapterManager(
            config=self.config,
            device=device,
            hf_token=hf_token,
        )
        self.vad = SileroVADStream(sample_rate=self.config.sample_rate) if use_vad else None
        self.script_toggler = ScriptToggler()

    def _transcribe_array(
        self,
        waveform: np.ndarray,
        language: Optional[str] = None,
    ) -> str:
        if waveform.size == 0:
            return ""

        processor = self.adapter_manager.processor
        model = self.adapter_manager.model
        device = self.adapter_manager.device
        dtype = self.adapter_manager.dtype

        inputs = processor.feature_extractor(
            waveform,
            sampling_rate=self.config.sample_rate,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(device=device, dtype=dtype)

        forced_decoder_ids = None
        if language:
            forced_decoder_ids = processor.get_decoder_prompt_ids(
                language=language,
                task=self.config.task,
            )

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=self.config.max_label_length,
            )

        decoded = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return decoded.strip()

    def transcribe(
        self,
        audio_input: Union[str, Path, np.ndarray, tuple],
        adapter: Optional[str] = None,
        target_script: ScriptMode = "auto",
        language: Optional[str] = None,
        use_vad: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if isinstance(audio_input, (str, Path)) and not Path(audio_input).exists():
            raise FileNotFoundError(f"Audio path does not exist: {audio_input}")

        active_adapter = self.adapter_manager.set_adapter(adapter or self.config.default_adapter)
        waveform = load_audio_16k_mono(audio_input, target_sr=self.config.sample_rate)

        if waveform.size == 0:
            return {
                "text": "",
                "raw_text": "",
                "segments": [],
                "adapter": active_adapter,
                "script": target_script,
            }

        apply_vad = self.vad is not None if use_vad is None else (use_vad and self.vad is not None)
        segment_results: List[Dict[str, Any]] = []

        if apply_vad and self.vad is not None:
            segments = self.vad.extract_speech_segments(waveform)
            if not segments:
                return {
                    "text": "",
                    "raw_text": "",
                    "segments": [],
                    "adapter": active_adapter,
                    "script": target_script,
                }
            for seg in segments:
                seg_text = self._transcribe_array(seg.audio, language=language)
                if seg_text:
                    converted = self.script_toggler.convert(seg_text, target_script=target_script)
                    segment_results.append(
                        {
                            "start": round(seg.start_sec, 2),
                            "end": round(seg.end_sec, 2),
                            "raw_text": seg_text,
                            "text": converted,
                        }
                    )
            raw_text = " ".join(item["raw_text"] for item in segment_results).strip()
            final_text = " ".join(item["text"] for item in segment_results).strip()
        else:
            raw_text = self._transcribe_array(waveform, language=language)
            final_text = self.script_toggler.convert(raw_text, target_script=target_script)
            if raw_text:
                duration = round(waveform.size / float(self.config.sample_rate), 2)
                segment_results.append(
                    {
                        "start": 0.0,
                        "end": duration,
                        "raw_text": raw_text,
                        "text": final_text,
                    }
                )

        return {
            "text": final_text,
            "raw_text": raw_text,
            "segments": segment_results,
            "adapter": active_adapter,
            "script": target_script,
        }
