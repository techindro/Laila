import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, Dataset, concatenate_datasets, load_dataset

from src.config import LailaConfig


NORMALIZATION_PATTERN = re.compile(r"[^\u0900-\u097Fa-z0-9\s]|[\u0964\u0965]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_transcript(text: str) -> str:
    if not text:
        return ""
    cleaned = text.lower().strip()
    cleaned = NORMALIZATION_PATTERN.sub(" ", cleaned)
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned


def load_audio_16k_mono(audio_input: Any, target_sr: int = 16000) -> np.ndarray:
    if isinstance(audio_input, (str, Path)):
        path = Path(audio_input)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        waveform, sr = sf.read(str(path), dtype="float32", always_2d=False)
    elif isinstance(audio_input, dict) and "array" in audio_input:
        waveform = np.asarray(audio_input["array"], dtype=np.float32)
        sr = int(audio_input.get("sampling_rate", target_sr))
    elif isinstance(audio_input, tuple) and len(audio_input) == 2:
        sr = int(audio_input[0])
        waveform = np.asarray(audio_input[1], dtype=np.float32)
    else:
        waveform = np.asarray(audio_input, dtype=np.float32)
        sr = target_sr

    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=-1)

    if sr != target_sr and waveform.size > 0:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)

    max_val = np.max(np.abs(waveform)) if waveform.size > 0 else 0.0
    if max_val > 1.0:
        waveform = waveform / max_val

    return waveform.astype(np.float32)


class NoiseAugmentor:
    def __init__(
        self,
        noise_dir: Optional[str] = None,
        sample_rate: int = 16000,
        snr_min_db: float = 5.0,
        snr_max_db: float = 20.0,
        prob: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.sample_rate = sample_rate
        self.snr_min_db = snr_min_db
        self.snr_max_db = snr_max_db
        self.prob = prob
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.noise_clips: List[np.ndarray] = []

        if noise_dir:
            noise_path = Path(noise_dir)
            if noise_path.exists() and noise_path.is_dir():
                for ext in ("*.wav", "*.flac", "*.mp3"):
                    for file_path in noise_path.rglob(ext):
                        try:
                            clip = load_audio_16k_mono(file_path, target_sr=self.sample_rate)
                            if clip.size > 0:
                                self.noise_clips.append(clip)
                        except Exception:
                            continue

    def _synthesize_environment_noise(self, length: int, profile: str) -> np.ndarray:
        t = np.arange(length, dtype=np.float32) / float(self.sample_rate)
        white = self.np_rng.standard_normal(length).astype(np.float32)
        if profile == "fan":
            hum = 0.35 * np.sin(2.0 * math.pi * 50.0 * t) + 0.15 * np.sin(2.0 * math.pi * 100.0 * t)
            smooth = np.convolve(white, np.ones(16, dtype=np.float32) / 16.0, mode="same")
            noise = 0.6 * smooth + hum
        elif profile == "street":
            low_pass = np.convolve(white, np.ones(32, dtype=np.float32) / 32.0, mode="same")
            rumble = 0.25 * np.sin(2.0 * math.pi * 80.0 * t)
            bursts = (self.np_rng.random(length) > 0.9995).astype(np.float32) * self.np_rng.uniform(0.5, 1.5, length)
            noise = low_pass + rumble + bursts.astype(np.float32)
        else:
            modulation = 0.5 + 0.5 * np.sin(2.0 * math.pi * 2.5 * t)
            mid_band = np.convolve(white, np.ones(8, dtype=np.float32) / 8.0, mode="same")
            noise = mid_band * modulation.astype(np.float32)

        peak = np.max(np.abs(noise))
        if peak > 0:
            noise = noise / peak
        return noise.astype(np.float32)

    def _sample_noise(self, target_length: int) -> np.ndarray:
        if self.noise_clips:
            clip = self.rng.choice(self.noise_clips)
            if clip.size >= target_length:
                start = self.rng.randint(0, clip.size - target_length)
                return clip[start : start + target_length]
            repeats = math.ceil(target_length / max(clip.size, 1))
            return np.tile(clip, repeats)[:target_length]
        profile = self.rng.choice(["fan", "street", "crowd"])
        return self._synthesize_environment_noise(target_length, profile)

    def __call__(self, waveform: np.ndarray, training: bool = True) -> np.ndarray:
        if not training or waveform.size == 0 or self.rng.random() > self.prob:
            return waveform

        noise = self._sample_noise(waveform.size)
        signal_power = float(np.mean(waveform ** 2)) + 1e-10
        noise_power = float(np.mean(noise ** 2)) + 1e-10

        snr_db = self.rng.uniform(self.snr_min_db, self.snr_max_db)
        target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        scale = math.sqrt(target_noise_power / noise_power)
        mixed = waveform + scale * noise

        peak = np.max(np.abs(mixed))
        if peak > 1.0:
            mixed = mixed / peak
        return mixed.astype(np.float32)


def _extract_text_column(example: Dict[str, Any]) -> str:
    for key in ("sentence", "text", "normalized_text", "transcript", "transcription"):
        if key in example and isinstance(example[key], str) and example[key].strip():
            return example[key].strip()
    return ""


def _standardize_record(example: Dict[str, Any], language_tag: str) -> Dict[str, Any]:
    audio_data = example.get("audio")
    if isinstance(audio_data, dict):
        array = audio_data.get("array", np.array([], dtype=np.float32))
        sr = int(audio_data.get("sampling_rate", 16000))
    else:
        array = np.array([], dtype=np.float32)
        sr = 16000
    return {
        "audio": {"array": array, "sampling_rate": sr},
        "transcript": _extract_text_column(example),
        "language_tag": language_tag,
    }


def load_local_hinglish_dataset(manifest_path: str, sample_rate: int = 16000) -> Optional[Dataset]:
    path = Path(manifest_path)
    if not path.exists():
        return None

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            audio_file = Path(item["audio_filepath"])
            if not audio_file.is_absolute():
                audio_file = path.parent / audio_file
            if not audio_file.exists():
                continue
            waveform = load_audio_16k_mono(audio_file, target_sr=sample_rate)
            records.append(
                {
                    "audio": {"array": waveform, "sampling_rate": sample_rate},
                    "transcript": item.get("text", item.get("transcript", "")),
                    "language_tag": item.get("language_tag", "hinglish"),
                }
            )

    if not records:
        return None
    return Dataset.from_list(records)


def _load_indicvoices_safe(
    config: LailaConfig,
    split: str,
    max_samples: Optional[int] = None,
    hf_token: Optional[str] = None,
) -> Optional[Dataset]:
    token = hf_token or config.hf_token
    try:
        stream_ds = load_dataset(
            config.indicvoices_dataset,
            config.indicvoices_config,
            split=split,
            streaming=True,
            token=token,
        )
        stream_ds = stream_ds.cast_column("audio", Audio(sampling_rate=config.sample_rate))
        limit = max_samples or 2000
        rows: List[Dict[str, Any]] = []
        for idx, sample in enumerate(stream_ds):
            if idx >= limit:
                break
            rows.append(_standardize_record(sample, "hi"))
        return Dataset.from_list(rows) if rows else None
    except Exception:
        return None


def _load_common_voice_safe(
    config: LailaConfig,
    lang_code: str,
    split: str,
    language_tag: str,
    max_samples: Optional[int] = None,
    hf_token: Optional[str] = None,
) -> Optional[Dataset]:
    token = hf_token or config.hf_token
    try:
        ds = load_dataset(
            config.common_voice_dataset,
            lang_code,
            split=split,
            streaming=True,
            token=token,
        )
        ds = ds.cast_column("audio", Audio(sampling_rate=config.sample_rate))
        if lang_code == "en":
            ds = ds.filter(
                lambda x: "india" in str(x.get("accent", "")).lower()
                or "indian" in str(x.get("accents", "")).lower()
            )
        limit = max_samples or 2000
        rows: List[Dict[str, Any]] = []
        for idx, sample in enumerate(ds):
            if idx >= limit:
                break
            rows.append(_standardize_record(sample, language_tag))
        return Dataset.from_list(rows) if rows else None
    except Exception:
        return None


def load_laila_datasets(
    config: LailaConfig,
    max_samples_per_source: Optional[int] = 1500,
    hf_token: Optional[str] = None,
) -> Tuple[Dataset, Dataset]:
    token = hf_token or config.hf_token
    train_parts: List[Dataset] = []
    eval_parts: List[Dataset] = []

    iv_train = _load_indicvoices_safe(config, "train", max_samples_per_source, token)
    if iv_train is not None:
        train_parts.append(iv_train)

    iv_val_limit = max_samples_per_source // 4 if max_samples_per_source else 300
    iv_val = _load_indicvoices_safe(config, "valid", iv_val_limit, token)
    if iv_val is not None:
        eval_parts.append(iv_val)

    cv_hi_train = _load_common_voice_safe(config, "hi", "train", "hi", max_samples_per_source, token)
    if cv_hi_train is not None:
        train_parts.append(cv_hi_train)

    cv_hi_limit = max_samples_per_source // 4 if max_samples_per_source else 300
    cv_hi_eval = _load_common_voice_safe(config, "hi", "test", "hi", cv_hi_limit, token)
    if cv_hi_eval is not None:
        eval_parts.append(cv_hi_eval)

    cv_en_train = _load_common_voice_safe(config, "en", "train", "en-IN", max_samples_per_source, token)
    if cv_en_train is not None:
        train_parts.append(cv_en_train)

    cv_en_limit = max_samples_per_source // 4 if max_samples_per_source else 300
    cv_en_eval = _load_common_voice_safe(config, "en", "test", "en-IN", cv_en_limit, token)
    if cv_en_eval is not None:
        eval_parts.append(cv_en_eval)

    local_hinglish = load_local_hinglish_dataset(config.local_hinglish_manifest, config.sample_rate)
    if local_hinglish is not None:
        split_ds = local_hinglish.train_test_split(test_size=0.1, seed=config.seed)
        train_parts.append(split_ds["train"])
        eval_parts.append(split_ds["test"])

    if not train_parts:
        raise RuntimeError(
            "No training datasets could be loaded. "
            "Ensure Hugging Face access to IndicVoices/Common Voice or provide a local Hinglish manifest."
        )

    train_ds = concatenate_datasets(train_parts).shuffle(seed=config.seed)
    eval_ds = concatenate_datasets(eval_parts) if eval_parts else train_ds.select(range(min(200, len(train_ds))))
    return train_ds, eval_ds


def prepare_dataset_features(
    dataset: Dataset,
    processor: Any,
    config: LailaConfig,
    noise_augmentor: Optional[NoiseAugmentor] = None,
    training: bool = True,
) -> Dataset:
    def _prepare_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        waveform = load_audio_16k_mono(batch["audio"], target_sr=config.sample_rate)
        max_samples = int(config.max_audio_seconds * config.sample_rate)
        if waveform.size > max_samples:
            waveform = waveform[:max_samples]

        if training and noise_augmentor is not None:
            waveform = noise_augmentor(waveform, training=True)

        features = processor.feature_extractor(
            waveform,
            sampling_rate=config.sample_rate,
            return_attention_mask=False,
        )
        transcript = normalize_transcript(batch.get("transcript", ""))
        labels = processor.tokenizer(
            transcript,
            truncation=True,
            max_length=config.max_label_length,
        ).input_ids

        return {
            "input_features": features["input_features"][0],
            "labels": labels,
        }

    return dataset.map(
        _prepare_batch,
        remove_columns=dataset.column_names,
        desc="Preparing log-Mel features" if not training else "Preparing augmented training features",
    )


@dataclass
class DataCollatorSpeechSeq2Seq:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if labels.shape[1] > 0 and (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
