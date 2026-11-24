import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import jiwer
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from src.config import LailaConfig, require_hf_token, resolve_device, resolve_dtype
from src.dataset import (
    DataCollatorSpeechSeq2Seq,
    NoiseAugmentor,
    load_laila_datasets,
    normalize_transcript,
    prepare_dataset_features,
)


def load_whisper_base(
    model_name: str,
    fallback_model_name: str,
    device: str,
    hf_token: Optional[str] = None,
) -> Tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    dtype = resolve_dtype(device)
    try:
        processor = WhisperProcessor.from_pretrained(model_name, token=hf_token)
        model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            token=hf_token,
        )
        return model, processor
    except Exception:
        processor = WhisperProcessor.from_pretrained(fallback_model_name, token=hf_token)
        model = WhisperForConditionalGeneration.from_pretrained(
            fallback_model_name,
            torch_dtype=dtype,
            token=hf_token,
        )
        return model, processor


def apply_laila_lora(model: WhisperForConditionalGeneration, config: LailaConfig) -> PeftModel:
    for param in model.parameters():
        param.requires_grad = False

    if config.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.target_modules),
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


class MetricsLoggingCallback(TrainerCallback):
    def on_evaluate(self, args: Any, state: Any, control: Any, metrics: Optional[Dict[str, float]] = None, **kwargs: Any) -> None:
        if not metrics:
            return
        wer_val = metrics.get("eval_wer")
        cer_val = metrics.get("eval_cer")
        if wer_val is not None and cer_val is not None:
            print(f"Step {state.global_step} | eval_wer={wer_val:.4f} | eval_cer={cer_val:.4f}")


def build_compute_metrics(processor: WhisperProcessor):
    def compute_metrics(pred: Any) -> Dict[str, float]:
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        norm_preds = [normalize_transcript(text) for text in pred_str]
        norm_labels = [normalize_transcript(text) for text in label_str]

        paired = [(ref, hyp) for ref, hyp in zip(norm_labels, norm_preds) if ref]
        if not paired:
            return {"wer": 0.0, "cer": 0.0}

        refs = [item[0] for item in paired]
        hyps = [item[1] for item in paired]

        wer_score = float(jiwer.wer(refs, hyps))
        cer_score = float(jiwer.cer(refs, hyps))
        return {"wer": wer_score, "cer": cer_score}

    return compute_metrics


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    root = Path(output_dir)
    if not root.exists():
        return None
    checkpoints = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    return str(checkpoints[-1]) if checkpoints else None


def train_laila(
    config: LailaConfig,
    adapter_name: str = "hinglish",
    max_samples_per_source: Optional[int] = 1500,
    push_to_hub: bool = False,
    hf_token: Optional[str] = None,
) -> Path:
    config.ensure_directories()
    token = hf_token or config.hf_token
    if push_to_hub:
        token = require_hf_token(token)
    device = resolve_device()

    model, processor = load_whisper_base(
        config.base_model,
        config.fallback_base_model,
        device=device,
        hf_token=token,
    )
    processor.tokenizer.set_prefix_tokens(language=config.language, task=config.task)
    model.generation_config.language = config.language
    model.generation_config.task = config.task
    model.generation_config.forced_decoder_ids = None

    peft_model = apply_laila_lora(model, config)

    raw_train, raw_eval = load_laila_datasets(
        config,
        max_samples_per_source=max_samples_per_source,
        hf_token=token,
    )

    noise_augmentor = NoiseAugmentor(
        noise_dir=config.noise_dir,
        sample_rate=config.sample_rate,
        snr_min_db=config.snr_min_db,
        snr_max_db=config.snr_max_db,
        prob=config.noise_prob,
        seed=config.seed,
    )

    train_dataset = prepare_dataset_features(
        raw_train,
        processor=processor,
        config=config,
        noise_augmentor=noise_augmentor,
        training=True,
    )
    eval_dataset = prepare_dataset_features(
        raw_eval,
        processor=processor,
        config=config,
        noise_augmentor=None,
        training=False,
    )

    data_collator = DataCollatorSpeechSeq2Seq(
        processor=processor,
        decoder_start_token_id=peft_model.config.decoder_start_token_id,
    )

    use_fp16 = config.fp16 and device.startswith("cuda")
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        fp16=use_fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=config.max_label_length,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        dataloader_num_workers=config.dataloader_num_workers,
        seed=config.seed,
        push_to_hub=push_to_hub,
        hub_model_id=config.hub_model_id if push_to_hub else None,
        hub_token=token if push_to_hub else None,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=peft_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(processor),
        tokenizer=processor.feature_extractor,
        callbacks=[MetricsLoggingCallback()],
    )

    resume_checkpoint = find_latest_checkpoint(config.output_dir)
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    adapter_save_dir = Path(config.adapter_registry.get(adapter_name, Path(config.adapters_dir) / adapter_name))
    adapter_save_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(adapter_save_dir))
    processor.save_pretrained(str(adapter_save_dir))

    if push_to_hub:
        peft_model.push_to_hub(config.hub_model_id, token=token)
        processor.push_to_hub(config.hub_model_id, token=token)

    return adapter_save_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Laila LoRA ASR adapter")
    parser.add_argument("--adapter_name", type=str, default="hinglish")
    parser.add_argument("--max_steps", type=int, default=4000)
    parser.add_argument("--max_samples", type=int, default=1500)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hf_token", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = LailaConfig(max_steps=args.max_steps)
    saved_path = train_laila(
        config=cfg,
        adapter_name=args.adapter_name,
        max_samples_per_source=args.max_samples,
        push_to_hub=args.push_to_hub,
        hf_token=args.hf_token,
    )
    print(f"Saved trained adapter to: {saved_path}")
