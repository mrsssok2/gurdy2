"""Speech-to-text behind a one-method protocol. faster-whisper (CTranslate2) is imported lazily: GPU int8_float16 at run
time (design 5.D), CPU int8 as the fallback / for tests. Models are local paths; nothing is downloaded here."""
from __future__ import annotations

import re
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
    no_speech_prob: float | None = None  # Whisper's own "this is not speech" estimate (max over segments)
    compression_ratio: float | None = None   # high = repetitive text (a hallucination loop)


from .transcript_filter import MIN_RMS, repeated_tokens, reject_values      # noqa: F401  (the rules live there so the browser demo shares them; names re-exported)


def reject_reason(tr: "Transcript", audio=None) -> str | None:
    """Why a transcript is treated as "I didn't catch that" instead of a command (None = fine)."""
    rms = float(np.sqrt(np.mean(np.asarray(audio, np.float32) ** 2))) if audio is not None and len(audio) else None
    return reject_values(tr.text, rms, compression_ratio=tr.compression_ratio, no_speech_prob=tr.no_speech_prob, confidence=tr.confidence)


class STT(Protocol):
    def transcribe(self, audio: np.ndarray, sr: int = SAMPLE_RATE, prompt: str | None = None) -> Transcript: ...


def cuda_available() -> bool:
    """True when CTranslate2 (faster-whisper's engine) was built with CUDA and sees a GPU."""
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:                                  # noqa: BLE001  (no ctranslate2 / no driver)
        return False


def load_whisper(model_path: str, device: str = "auto", **kw) -> "FasterWhisperSTT":
    """Whisper on the GPU when there is one (int8_float16, about 1 GiB VRAM, ~0.3 s per utterance), CPU int8 otherwise (6-10 s).
    device: auto | cuda | cpu. "auto" and "cuda" fall back to the CPU when the GPU model cannot be loaded (no CUDA build, VRAM full)."""
    if device not in ("auto", "cuda", "cpu"):
        raise ValueError("device: auto | cuda | cpu")
    if device != "cpu" and cuda_available():
        try:
            stt = FasterWhisperSTT(model_path, "cuda", **kw)
            stt.device = "cuda"
            return stt
        except Exception as e:                         # noqa: BLE001  (CUDA out of memory, cuDNN missing ...)
            print(f"[stt] GPU load failed ({type(e).__name__}: {str(e)[:120]}); using the CPU", flush=True)
    elif device == "cuda":
        print("[stt] no CUDA device visible to ctranslate2; using the CPU", flush=True)
    stt = FasterWhisperSTT(model_path, "cpu", **kw)
    stt.device = "cpu"
    return stt


class FasterWhisperSTT:
    def __init__(self, model_path: str, device: str = "cpu", compute_type: str | None = None, beam_size: int = 1, language: str = "en",
                 places=None, bias: "bool | str" = "prompt",
                 vad_filter: bool = True):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_path, device=device, compute_type=compute_type or ("int8_float16" if device == "cuda" else "int8"))
        self.beam, self.language, self.vad_filter = beam_size, language, vad_filter
        self.set_vocabulary(places, bias)

    def set_vocabulary(self, places=None, bias: "bool | str" = "prompt") -> None:
        """Bias decoding towards the current place names / object classes / colours (call again when places change).
        bias: "none" | "prompt" (initial_prompt only: default, best confidence on the 5 real clips of 2026-11-18)
              | "full" (prompt + hotwords: made short phrases WORSE there; kept for room letters, see scripts/voice_ab.py).
        True = "full", False = "none" (old calls)."""
        from .vocab import hotwords, initial_prompt
        bias = {True: "full", False: "none"}.get(bias, bias)
        if bias not in ("none", "prompt", "full"):
            raise ValueError("bias: none | prompt | full")
        self.bias = bias
        self.prompt = initial_prompt(places) if bias in ("prompt", "full") else None
        self.hotwords = hotwords(places) if bias == "full" else None

    def transcribe(self, audio: np.ndarray, sr: int = SAMPLE_RATE, prompt: str | None = None) -> Transcript:
        """`prompt`: question-specific initial_prompt (the expected answers); replaces the command prompt, hotwords off."""
        if sr != SAMPLE_RATE:
            from .audio import resample
            audio = resample(audio, sr)
        t0 = time.perf_counter()
        # Whisper's own safeguards against hallucinated text on quiet / unclear audio: no repetition carried between segments,
        # silence-model VAD, no-speech / compression-ratio / log-prob thresholds (a failed decode is retried at a higher temperature)
        segs, _ = self.model.transcribe(np.asarray(audio, np.float32), language=self.language, beam_size=self.beam,
                                        condition_on_previous_text=False, vad_filter=self.vad_filter,
                                        vad_parameters={"threshold": 0.4, "min_silence_duration_ms": 500},
                                        no_speech_threshold=0.6, compression_ratio_threshold=2.4, log_prob_threshold=-1.0,
                                        without_timestamps=True, initial_prompt=prompt or self.prompt, hotwords=None if prompt else self.hotwords)
        segs = list(segs)
        text = "".join(s.text for s in segs).strip()
        conf = float(np.exp(np.mean([s.avg_logprob for s in segs]))) if segs else None
        return Transcript(text, conf, time.perf_counter() - t0,
                          max((s.no_speech_prob for s in segs), default=None), max((s.compression_ratio for s in segs), default=None))


class ScriptedSTT:
    """Test double: returns queued texts, or `fn(audio)`. Records the audio lengths it was asked to transcribe."""

    def __init__(self, texts=None, fn=None):
        self.texts, self.fn, self.calls = list(texts or []), fn, []

    def transcribe(self, audio: np.ndarray, sr: int = SAMPLE_RATE, prompt: str | None = None) -> Transcript:
        self.calls.append(len(audio) / sr)
        self.prompts = getattr(self, "prompts", []) + [prompt]
        text = self.fn(audio) if self.fn else (self.texts.pop(0) if self.texts else "")
        return Transcript(text, 0.9, 0.0)
