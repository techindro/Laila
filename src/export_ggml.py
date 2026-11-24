import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.config import LailaConfig


def merge_lora_weights(
    base_model_id: str,
    fallback_model_id: str,
    adapter_path: str,
    merged_output_dir: str,
    hf_token: Optional[str] = None,
) -> Path:
    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    out_dir = Path(merged_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        processor = WhisperProcessor.from_pretrained(base_model_id, token=hf_token)
        base_model = WhisperForConditionalGeneration.from_pretrained(
            base_model_id,
            torch_dtype=torch.float32,
            token=hf_token,
        )
    except Exception:
        processor = WhisperProcessor.from_pretrained(fallback_model_id, token=hf_token)
        base_model = WhisperForConditionalGeneration.from_pretrained(
            fallback_model_id,
            torch_dtype=torch.float32,
            token=hf_token,
        )

    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(str(out_dir), safe_serialization=True)
    processor.save_pretrained(str(out_dir))
    return out_dir


def convert_to_ggml_q4(
    merged_dir: Path,
    ggml_output_dir: str,
    whisper_cpp_dir: str,
    quantization: str = "q4_0",
) -> Path:
    cpp_root = Path(whisper_cpp_dir)
    if not cpp_root.exists():
        raise FileNotFoundError(f"whisper.cpp directory not found: {cpp_root}")

    out_dir = Path(ggml_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    convert_script = cpp_root / "models" / "convert-h5-to-ggml.py"
    if not convert_script.exists():
        raise FileNotFoundError(f"Conversion script not found: {convert_script}")

    f16_bin = out_dir / "ggml-model-f16.bin"
    subprocess.run(
        [
            "python",
            str(convert_script),
            str(merged_dir),
            str(cpp_root),
            str(out_dir),
        ],
        check=True,
    )

    generated_bin = out_dir / "ggml-model.bin"
    if generated_bin.exists() and not f16_bin.exists():
        generated_bin.rename(f16_bin)

    quantize_bin = cpp_root / "build" / "bin" / "quantize"
    if not quantize_bin.exists():
        quantize_bin = cpp_root / "quantize"
    if not quantize_bin.exists():
        exe_path = shutil.which("whisper-quantize") or shutil.which("quantize")
        if exe_path:
            quantize_bin = Path(exe_path)
        else:
            raise FileNotFoundError("whisper.cpp quantize binary not found.")

    quantized_bin = out_dir / f"laila-whisper-small-{quantization}.bin"
    subprocess.run(
        [
            str(quantize_bin),
            str(f16_bin),
            str(quantized_bin),
            quantization,
        ],
        check=True,
    )
    return quantized_bin


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Laila LoRA and quantize to 4-bit GGML for Android")
    parser.add_argument("--adapter_path", type=str, default="adapters/hinglish")
    parser.add_argument("--merged_dir", type=str, default="exports/merged")
    parser.add_argument("--output_dir", type=str, default="exports/ggml")
    parser.add_argument("--whisper_cpp_dir", type=str, default="whisper.cpp")
    parser.add_argument("--quantization", type=str, default="q4_0")
    parser.add_argument("--merge_only", action="store_true")
    parser.add_argument("--hf_token", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = LailaConfig()
    merged_path = merge_lora_weights(
        base_model_id=cfg.base_model,
        fallback_model_id=cfg.fallback_base_model,
        adapter_path=args.adapter_path,
        merged_output_dir=args.merged_dir,
        hf_token=args.hf_token,
    )
    print(f"Merged model saved to: {merged_path}")

    if not args.merge_only:
        ggml_file = convert_to_ggml_q4(
            merged_dir=merged_path,
            ggml_output_dir=args.output_dir,
            whisper_cpp_dir=args.whisper_cpp_dir,
            quantization=args.quantization,
        )
        print(f"Quantized 4-bit GGML model saved to: {ggml_file}")
