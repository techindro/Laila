import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jiwer
import torch
from datasets import Dataset
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.config import LailaConfig, resolve_device, resolve_dtype
from src.dataset import (
    _load_common_voice_safe,
    _load_indicvoices_safe,
    load_audio_16k_mono,
    load_local_hinglish_dataset,
    normalize_transcript,
)


def _transcribe_dataset(
    model: Any,
    processor: WhisperProcessor,
    dataset: Dataset,
    device: str,
    sample_rate: int = 16000,
    language: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    model.eval()
    dtype = resolve_dtype(device)
    references: List[str] = []
    hypotheses: List[str] = []

    forced_decoder_ids = None
    if language:
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")

    for sample in dataset:
        ref = normalize_transcript(sample.get("transcript", ""))
        if not ref:
            continue

        waveform = load_audio_16k_mono(sample["audio"], target_sr=sample_rate)
        inputs = processor.feature_extractor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(device=device, dtype=dtype)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=225,
            )

        raw_hyp = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        hyp = normalize_transcript(raw_hyp)

        references.append(ref)
        hypotheses.append(hyp)

    return references, hypotheses


def compute_wer_cer(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    if not references:
        return {"wer": 0.0, "cer": 0.0, "samples": 0}
    return {
        "wer": float(jiwer.wer(references, hypotheses)),
        "cer": float(jiwer.cer(references, hypotheses)),
        "samples": len(references),
    }


def load_eval_splits(
    config: LailaConfig,
    max_samples: int = 200,
    hf_token: Optional[str] = None,
) -> Dict[str, Optional[Dataset]]:
    hindi_ds = _load_indicvoices_safe(config, "valid", max_samples=max_samples, hf_token=hf_token)
    if hindi_ds is None:
        hindi_ds = _load_common_voice_safe(config, "hi", "test", "hi", max_samples=max_samples, hf_token=hf_token)

    indian_en_ds = _load_common_voice_safe(config, "en", "test", "en-IN", max_samples=max_samples, hf_token=hf_token)
    hinglish_ds = load_local_hinglish_dataset(config.local_hinglish_manifest, sample_rate=config.sample_rate)
    if hinglish_ds is not None and len(hinglish_ds) > max_samples:
        hinglish_ds = hinglish_ds.select(range(max_samples))

    return {
        "Hindi": hindi_ds,
        "Indian-English": indian_en_ds,
        "Code-Switched Hinglish": hinglish_ds,
    }


def evaluate_model_suite(
    config: LailaConfig,
    adapter_path: str,
    max_samples: int = 200,
    hf_token: Optional[str] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    device = resolve_device()
    dtype = resolve_dtype(device)
    splits = load_eval_splits(config, max_samples=max_samples, hf_token=hf_token)

    model_specs = [
        ("openai/whisper-small", "openai/whisper-small", None),
        ("ai4bharat/indicwhisper-hi", config.base_model, None),
        ("Laila (LoRA)", config.base_model, adapter_path),
    ]

    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for display_name, base_id, lora_dir in model_specs:
        results[display_name] = {}
        try:
            processor = WhisperProcessor.from_pretrained(base_id, token=hf_token)
            model = WhisperForConditionalGeneration.from_pretrained(base_id, torch_dtype=dtype, token=hf_token)
            if lora_dir and Path(lora_dir).exists():
                model = PeftModel.from_pretrained(model, lora_dir)
            model.to(device)
        except Exception as exc:
            for split_name in splits:
                results[display_name][split_name] = {"wer": -1.0, "cer": -1.0, "samples": 0}
            print(f"Skipping {display_name}: {exc}")
            continue

        for split_name, ds in splits.items():
            if ds is None or len(ds) == 0:
                results[display_name][split_name] = {"wer": -1.0, "cer": -1.0, "samples": 0}
                continue
            lang_hint = "en" if split_name == "Indian-English" else "hi"
            refs, hyps = _transcribe_dataset(
                model=model,
                processor=processor,
                dataset=ds,
                device=device,
                sample_rate=config.sample_rate,
                language=lang_hint,
            )
            results[display_name][split_name] = compute_wer_cer(refs, hyps)

        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return results


def format_benchmark_table(results: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    headers = [
        "Model",
        "Hindi WER",
        "Hindi CER",
        "Indian-En WER",
        "Indian-En CER",
        "Hinglish WER",
        "Hinglish CER",
    ]
    rows: List[List[str]] = []

    def _fmt(val: float) -> str:
        return "N/A" if val < 0 else f"{val * 100:.2f}%"

    for model_name, split_metrics in results.items():
        hi = split_metrics.get("Hindi", {})
        en = split_metrics.get("Indian-English", {})
        hg = split_metrics.get("Code-Switched Hinglish", {})
        rows.append(
            [
                model_name,
                _fmt(hi.get("wer", -1.0)),
                _fmt(hi.get("cer", -1.0)),
                _fmt(en.get("wer", -1.0)),
                _fmt(en.get("cer", -1.0)),
                _fmt(hg.get("wer", -1.0)),
                _fmt(hg.get("cer", -1.0)),
            ]
        )

    col_widths = [max(len(r[i]) for r in [headers] + rows) for i in range(len(headers))]
    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    header_line = "| " + " | ".join(headers[i].ljust(col_widths[i]) for i in range(len(headers))) + " |"
    body_lines = [
        "| " + " | ".join(row[i].ljust(col_widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep] + body_lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Laila against Whisper-Small and IndicWhisper")
    parser.add_argument("--adapter_path", type=str, default="adapters/hinglish")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--output_json", type=str, default="outputs/benchmark_results.json")
    parser.add_argument("--hf_token", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = LailaConfig()
    benchmark_results = evaluate_model_suite(
        config=cfg,
        adapter_path=args.adapter_path,
        max_samples=args.max_samples,
        hf_token=args.hf_token,
    )
    table = format_benchmark_table(benchmark_results)
    print(table)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(benchmark_results, indent=2), encoding="utf-8")
