import json
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import gradio as gr

_ASR_INSTANCE: Optional[Any] = None


def _ensure_repo_root() -> None:
    root_dir = Path(__file__).resolve().parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))


def get_asr_engine() -> Any:
    global _ASR_INSTANCE
    if _ASR_INSTANCE is None:
        _ensure_repo_root()
        from src.config import LailaConfig
        from src.inference import LailaASR

        config = LailaConfig()
        _ASR_INSTANCE = LailaASR(config=config, use_vad=True)
    return _ASR_INSTANCE


def run_transcription(
    audio_path: Optional[str],
    adapter_name: str,
    script_mode: str,
    use_vad: bool,
) -> Tuple[str, str]:
    if not audio_path:
        return "", json.dumps({"status": "No audio provided"}, indent=2)

    engine = get_asr_engine()
    result = engine.transcribe(
        audio_input=audio_path,
        adapter=adapter_name,
        target_script=script_mode,
        use_vad=use_vad,
    )
    metadata = {
        "active_adapter": result["adapter"],
        "script": result["script"],
        "segments": result["segments"],
    }
    return result["text"], json.dumps(metadata, ensure_ascii=False, indent=2)


def build_interface() -> gr.Blocks:
    _ensure_repo_root()
    from src.config import LailaConfig

    config = LailaConfig()
    adapter_choices = list(config.adapter_registry.keys())

    with gr.Blocks(title="Laila - Indian ASR") as demo:
        gr.Markdown("# Laila (`techindro/Laila`) - Indian Multilingual & Code-Switched ASR")
        gr.Markdown(
            "Parameter-efficient `ai4bharat/indicwhisper-hi` + LoRA with Silero VAD, "
            "hot-swappable regional adapters, and Devanagari/Roman script toggle."
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Audio Input (16 kHz Mono)",
                )
                adapter_dropdown = gr.Dropdown(
                    choices=adapter_choices,
                    value=config.default_adapter,
                    label="Regional LoRA Adapter",
                )
                script_radio = gr.Radio(
                    choices=["auto", "devanagari", "roman"],
                    value="auto",
                    label="Output Script",
                )
                vad_checkbox = gr.Checkbox(
                    value=True,
                    label="Enable Silero VAD Silence Filtering",
                )
                submit_btn = gr.Button("Transcribe", variant="primary")

            with gr.Column(scale=1):
                transcript_output = gr.Textbox(
                    label="Transcription",
                    lines=6,
                )
                segments_output = gr.Code(
                    label="VAD Segments & Adapter Metadata",
                    language="json",
                )

        submit_btn.click(
            fn=run_transcription,
            inputs=[audio_input, adapter_dropdown, script_radio, vad_checkbox],
            outputs=[transcript_output, segments_output],
        )

    return demo


if __name__ == "__main__":
    app = build_interface()
    app.launch()
