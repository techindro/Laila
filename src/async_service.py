import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import Any, Dict, Optional, Union

import numpy as np

from src.config import LailaConfig
from src.dataset import load_audio_16k_mono
from src.inference import LailaASR
from src.script_toggle import ScriptMode


@dataclass
class TranscriptionJob:
    job_id: str
    audio_input: Union[str, Path, np.ndarray, tuple]
    adapter: str = "hinglish"
    target_script: ScriptMode = "auto"
    use_vad: bool = True
    status: str = "queued"
    retries: int = 0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    rtf: Optional[float] = None
    created_at: float = field(default_factory=time.time)


class AsyncInferenceService:
    def __init__(
        self,
        config: Optional[LailaConfig] = None,
        max_retries: int = 2,
    ) -> None:
        self.config = config or LailaConfig()
        self.max_retries = max_retries
        self.queue: Queue[str] = Queue()
        self.jobs: Dict[str, TranscriptionJob] = {}
        self._worker_engine: Optional[LailaASR] = None

    def _get_worker(self) -> LailaASR:
        if self._worker_engine is None:
            self._worker_engine = LailaASR(config=self.config, use_vad=True)
        return self._worker_engine

    def submit_job(
        self,
        audio_input: Union[str, Path, np.ndarray, tuple],
        adapter: Optional[str] = None,
        target_script: ScriptMode = "auto",
        use_vad: bool = True,
    ) -> str:
        job_id = str(uuid.uuid4())
        job = TranscriptionJob(
            job_id=job_id,
            audio_input=audio_input,
            adapter=adapter or self.config.default_adapter,
            target_script=target_script,
            use_vad=use_vad,
            status="queued",
        )
        self.jobs[job_id] = job
        self.queue.put(job_id)
        return job_id

    def process_next_job(self) -> Optional[TranscriptionJob]:
        if self.queue.empty():
            return None
        job_id = self.queue.get()
        job = self.jobs[job_id]
        job.status = "processing"

        while job.retries <= self.max_retries:
            try:
                waveform = load_audio_16k_mono(job.audio_input, target_sr=self.config.sample_rate)
                audio_duration_sec = max(waveform.size / float(self.config.sample_rate), 1e-6)
                worker = self._get_worker()

                start_time = time.perf_counter()
                result = worker.transcribe(
                    audio_input=waveform,
                    adapter=job.adapter,
                    target_script=job.target_script,
                    use_vad=job.use_vad,
                )
                elapsed_sec = time.perf_counter() - start_time

                job.rtf = round(elapsed_sec / audio_duration_sec, 4)
                result["rtf"] = job.rtf
                result["audio_duration_sec"] = round(audio_duration_sec, 2)
                job.result = result
                job.status = "completed"
                return job
            except Exception as exc:
                job.retries += 1
                job.error = str(exc)
                if job.retries > self.max_retries:
                    job.status = "failed"
                    return job
        return job

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        if job_id not in self.jobs:
            raise KeyError(f"Unknown job_id: {job_id}")
        job = self.jobs[job_id]
        return {
            "job_id": job.job_id,
            "status": job.status,
            "retries": job.retries,
            "rtf": job.rtf,
            "result": job.result,
            "error": job.error,
        }
