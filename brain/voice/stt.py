"""Speech-to-text behind a one-method protocol. faster-whisper (CTranslate2) is imported lazily: GPU int8_float16 at run
time (design 5.D), CPU int8 as the fallback / for tests. Models are local paths; nothing is downloaded here."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .audio import SAMPLE_RATE


@dataclass
class Transcript:
    text: str
    confidence: float | None = None     # 0..1 from mean segment log-prob, None if unknown
    seconds: float = 0.0                # STT wall time (reported separately from the robot's reaction time)


class STT(Protocol):
    def transcribe(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> Transcript: ...


class FasterWhisperSTT:
    def __init__(self, model_path: str, device: str = "cpu", compute_type: str | None = None, beam_size: int = 1, language: str = "en",
                 places=None, bias: bool = True):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_path, device=device, compute_type=compute_type or ("int8_float16" if device == "cuda" else "int8"))
        self.beam, self.language = beam_size, language
        self.set_vocabulary(places, bias)

    def set_vocabulary(self, places=None, bias: bool = True) -> None:
        """Bias decoding towards the current place names / object classes / colours (call again when places change)."""
        from .vocab import hotwords, initial_prompt
        self.prompt, self.hotwords = (initial_prompt(places), hotwords(places)) if bias else (None, None)

    def transcribe(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> Transcript:
        if sr != SAMPLE_RATE:
            from .audio import resample
            audio = resample(audio, sr)
        t0 = time.perf_counter()
        segs, _ = self.model.transcribe(np.asarray(audio, np.float32), language=self.language, beam_size=self.beam,
                                        condition_on_previous_text=False, vad_filter=False, without_timestamps=True,
                                        initial_prompt=self.prompt, hotwords=self.hotwords)
        segs = list(segs)
        text = "".join(s.text for s in segs).strip()
        conf = float(np.exp(np.mean([s.avg_logprob for s in segs]))) if segs else None
        return Transcript(text, conf, time.perf_counter() - t0)


class ScriptedSTT:
    """Test double: returns queued texts, or `fn(audio)`. Records the audio lengths it was asked to transcribe."""

    def __init__(self, texts=None, fn=None):
        self.texts, self.fn, self.calls = list(texts or []), fn, []

    def transcribe(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> Transcript:
        self.calls.append(len(audio) / sr)
        text = self.fn(audio) if self.fn else (self.texts.pop(0) if self.texts else "")
        return Transcript(text, 0.9, 0.0)
