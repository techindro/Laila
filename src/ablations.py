import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jiwer
import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model

from src.config import LailaConfig, resolve_device, resolve_dtype
from src.dataset import NoiseAugmentor, load_audio_16k_mono, normalize_transcript
from src.evaluate import compute_wer_cer
from src.script_toggle import ScriptToggler
from src.train import load_whisper_base


def count_trainable_parameters(model: Any) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _resolve_target_modules(
    model: Any,
    modules: Sequence[str],
    decoder_only: bool = False,
) -> List[str]:
    if not decoder_only:
        return list(modules)
    matched: List[str] = []
    for name, _ in model.named_modules():
        if "decoder" in name and any(name.endswith(f".{m}") for m in modules):
            matched.append(name)
    return matched if matched else list(modules)


def build_ablation_lora_model(
    config: LailaConfig,
    rank: int,
    target_modules: Sequence[str],
    decoder_only: bool = False,
    hf_token: Optional[str] = None,
) -> Tuple[Any, Any, int]:
    device = resolve_device()
    model, processor = load_whisper_base(
        config.base_model,
        config.fallback_base_model,
        device=device,
        hf_token=hf_token,
    )
    for param in model.parameters():
        param.requires_grad = False

    resolved_targets = _resolve_target_modules(model, target_modules, decoder_only=decoder_only)
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=2 * rank,
        target_modules=resolved_targets,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    peft_model = get_peft_model(model, lora_config)
    trainable, _ = count_trainable_parameters(peft_model)
    return peft_model, processor, trainable


def run_rank_ablation_parameter_audit(
    config: LailaConfig,
    ranks: Sequence[int] = (8, 16, 32, 64),
    hf_token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for r in ranks:
        model, _, trainable = build_ablation_lora_model(
            config=config,
            rank=r,
            target_modules=config.target_modules,
            decoder_only=False,
            hf_token=hf_token,
        )
        results.append(
            {
                "rank": r,
                "alpha": 2 * r,
                "trainable_params": trainable,
                "trainable_millions": round(trainable / 1e6, 2),
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def get_module_noise_ablation_configs() -> List[Dict[str, Any]]:
    return [
        {
            "name": "q_proj, v_proj (noise off)",
            "target_modules": ("q_proj", "v_proj"),
            "decoder_only": False,
            "noise_enabled": False,
        },
        {
            "name": "q_proj, v_proj (noise on)",
            "target_modules": ("q_proj", "v_proj"),
            "decoder_only": False,
            "noise_enabled": True,
        },
        {
            "name": "q_proj, k_proj, v_proj, out_proj (noise on)",
            "target_modules": ("q_proj", "k_proj", "v_proj", "out_proj"),
            "decoder_only": False,
            "noise_enabled": True,
        },
        {
            "name": "Decoder only q_proj, v_proj (noise on)",
            "target_modules": ("q_proj", "v_proj"),
            "decoder_only": True,
            "noise_enabled": True,
        },
    ]


def add_fixed_snr_noise(
    waveform: np.ndarray,
    snr_db: Optional[float],
    augmentor: NoiseAugmentor,
) -> np.ndarray:
    if snr_db is None or waveform.size == 0:
        return waveform
    noise = augmentor._sample_noise(waveform.size)
    signal_power = float(np.mean(waveform ** 2)) + 1e-10
    noise_power = float(np.mean(noise ** 2)) + 1e-10
    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    scale = math.sqrt(target_noise_power / noise_power)
    mixed = waveform + scale * noise
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32)


def evaluate_noise_robustness(
    model: Any,
    processor: Any,
    dataset: Dataset,
    config: LailaConfig,
    snr_levels_db: Optional[Dict[str, Optional[float]]] = None,
) -> Dict[str, Dict[str, float]]:
    levels = snr_levels_db or {
        "Clean": None,
        "High SNR": 25.0,
        "Medium SNR": 15.0,
        "Low SNR": 5.0,
    }
    device = resolve_device()
    dtype = resolve_dtype(device)
    model.to(device)
    model.eval()
    augmentor = NoiseAugmentor(noise_dir=config.noise_dir, sample_rate=config.sample_rate, prob=1.0, seed=config.seed)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="hi", task="transcribe")

    results: Dict[str, Dict[str, float]] = {}
    for label, snr in levels.items():
        refs: List[str] = []
        hyps: List[str] = []
        for sample in dataset:
            ref = normalize_transcript(sample.get("transcript", ""))
            if not ref:
                continue
            waveform = load_audio_16k_mono(sample["audio"], target_sr=config.sample_rate)
            noisy_wave = add_fixed_snr_noise(waveform, snr, augmentor)
            inputs = processor.feature_extractor(
                noisy_wave,
                sampling_rate=config.sample_rate,
                return_tensors="pt",
            )
            input_features = inputs.input_features.to(device=device, dtype=dtype)
            with torch.no_grad():
                pred_ids = model.generate(
                    input_features,
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=config.max_label_length,
                )
            raw_hyp = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0]
            refs.append(ref)
            hyps.append(normalize_transcript(raw_hyp))
        results[label] = compute_wer_cer(refs, hyps)
    return results


def analyze_code_switched_errors(
    references: Sequence[str],
    hypotheses: Sequence[str],
    max_examples: int = 10,
) -> Dict[str, Any]:
    toggler = ScriptToggler()
    counts = {
        "wrong_script": 0,
        "phonetic_substitution": 0,
        "missed_switch": 0,
    }
    examples: List[Dict[str, str]] = []

    for ref_raw, hyp_raw in zip(references, hypotheses):
        ref = normalize_transcript(ref_raw)
        hyp = normalize_transcript(hyp_raw)
        if not ref or ref == hyp:
            continue

        ref_has_dev = toggler.contains_devanagari(ref)
        hyp_has_dev = toggler.contains_devanagari(hyp)
        ref_has_lat = any("a" <= ch <= "z" for ch in ref)
        hyp_has_lat = any("a" <= ch <= "z" for ch in hyp)

        error_type = "phonetic_substitution"
        if (ref_has_dev != hyp_has_dev) or (ref_has_lat != hyp_has_lat):
            if (ref_has_dev and ref_has_lat) and not (hyp_has_dev and hyp_has_lat):
                error_type = "missed_switch"
            else:
                error_type = "wrong_script"

        counts[error_type] += 1
        if len(examples) < max_examples:
            examples.append(
                {
                    "type": error_type,
                    "reference": ref,
                    "hypothesis": hyp,
                    "wer": f"{jiwer.wer([ref], [hyp]):.3f}",
                }
            )

    return {"counts": counts, "examples": examples}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Laila paper ablation parameter audit and noise robustness checks")
    parser.add_argument("--output_json", type=str, default="outputs/ablation_audit.json")
    parser.add_argument("--max_samples", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = LailaConfig()
    rank_audit = run_rank_ablation_parameter_audit(cfg, hf_token=cfg.hf_token)
    module_configs = get_module_noise_ablation_configs()
    payload = {
        "rank_ablation_audit": rank_audit,
        "module_noise_configs": module_configs,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
