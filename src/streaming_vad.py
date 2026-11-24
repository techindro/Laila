from dataclasses import dataclass
from typing import Generator, Iterable, List, Tuple

import numpy as np
import torch

from src.dataset import load_audio_16k_mono


@dataclass
class SpeechSegment:
    start_sample: int
    end_sample: int
    start_sec: float
    end_sec: float
    audio: np.ndarray


class SileroVADStream:
    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 300,
        max_speech_duration_s: float = 30.0,
        speech_pad_ms: int = 100,
    ) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.max_speech_duration_s = max_speech_duration_s
        self.speech_pad_ms = speech_pad_ms
        self.model = None
        self.get_speech_timestamps = None
        self._load_silero_vad()

    def _load_silero_vad(self) -> None:
        try:
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self.model = model
            self.get_speech_timestamps = utils[0]
        except Exception:
            self.model = None
            self.get_speech_timestamps = None

    def _energy_vad_fallback(self, waveform: np.ndarray, rms_threshold: float = 0.015) -> List[dict]:
        frame_len = int(0.03 * self.sample_rate)
        hop_len = int(0.01 * self.sample_rate)
        if waveform.size < frame_len:
            rms = float(np.sqrt(np.mean(waveform ** 2))) if waveform.size > 0 else 0.0
            if rms >= rms_threshold:
                return [{"start": 0, "end": waveform.size}]
            return []

        speech_frames: List[Tuple[int, int]] = []
        in_speech = False
        start_idx = 0
        for pos in range(0, waveform.size - frame_len + 1, hop_len):
            frame = waveform[pos : pos + frame_len]
            rms = float(np.sqrt(np.mean(frame ** 2)))
            if rms >= rms_threshold and not in_speech:
                in_speech = True
                start_idx = pos
            elif rms < rms_threshold and in_speech:
                in_speech = False
                speech_frames.append((start_idx, pos + frame_len))

        if in_speech:
            speech_frames.append((start_idx, waveform.size))

        min_samples = int((self.min_speech_duration_ms / 1000.0) * self.sample_rate)
        return [
            {"start": s, "end": e}
            for s, e in speech_frames
            if (e - s) >= min_samples
        ]

    def extract_speech_segments(self, audio_input: object) -> List[SpeechSegment]:
        waveform = load_audio_16k_mono(audio_input, target_sr=self.sample_rate)
        if waveform.size == 0 or float(np.max(np.abs(waveform))) < 1e-6:
            return []

        if self.model is not None and self.get_speech_timestamps is not None:
            tensor_audio = torch.from_numpy(waveform)
            timestamps = self.get_speech_timestamps(
                tensor_audio,
                self.model,
                threshold=self.threshold,
                sampling_rate=self.sample_rate,
                min_speech_duration_ms=self.min_speech_duration_ms,
                min_silence_duration_ms=self.min_silence_duration_ms,
                max_speech_duration_s=self.max_speech_duration_s,
                speech_pad_ms=self.speech_pad_ms,
            )
        else:
            timestamps = self._energy_vad_fallback(waveform)

        if not timestamps:
            return []

        segments: List[SpeechSegment] = []
        for ts in timestamps:
            start_sample = max(0, int(ts["start"]))
            end_sample = min(waveform.size, int(ts["end"]))
            if end_sample <= start_sample:
                continue
            chunk = waveform[start_sample:end_sample]
            segments.append(
                SpeechSegment(
                    start_sample=start_sample,
                    end_sample=end_sample,
                    start_sec=start_sample / float(self.sample_rate),
                    end_sec=end_sample / float(self.sample_rate),
                    audio=chunk,
                )
            )
        return segments

    def stream_chunks(
        self,
        audio_chunks: Iterable[np.ndarray],
        chunk_seconds: float = 4.0,
    ) -> Generator[SpeechSegment, None, None]:
        buffer = np.array([], dtype=np.float32)
        target_samples = int(chunk_seconds * self.sample_rate)

        for chunk in audio_chunks:
            mono = load_audio_16k_mono(chunk, target_sr=self.sample_rate)
            if mono.size == 0:
                continue
            buffer = np.concatenate([buffer, mono])
            if buffer.size >= target_samples:
                for segment in self.extract_speech_segments(buffer):
                    yield segment
                buffer = np.array([], dtype=np.float32)

        if buffer.size > 0:
            for segment in self.extract_speech_segments(buffer):
                yield segment
