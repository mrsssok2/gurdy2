"""Text-to-speech + interruptible playback.

PiperTTS: Piper ONNX voice, CPU, imported lazily (design 5.D; Piper's licence stays outside our MIT code like Ultralytics).
Speaker: plays synthesized audio chunk by chunk (one chunk per mic frame, so the mic clock paces playback and a barge-in
stop() takes effect within one chunk). `say()` is non-blocking; call `tick()` once per mic frame."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .audio import FRAME_S, NullSink


class TTS(Protocol):
    sample_rate: int

    def synthesize(self, text: str) -> np.ndarray: ...


class PiperTTS:
    def __init__(self, model_path: str, use_cuda: bool = False, length_scale: float = 1.0):
        from piper import PiperVoice, SynthesisConfig
        self.voice = PiperVoice.load(model_path, use_cuda=use_cuda)
        self.sample_rate = int(self.voice.config.sample_rate)
        self.cfg = SynthesisConfig(length_scale=length_scale)

    def synthesize(self, text: str) -> np.ndarray:
        parts = [c.audio_float_array for c in self.voice.synthesize(text, syn_config=self.cfg)]
        return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, np.float32)


class Speaker:
    def __init__(self, tts: TTS, sink=None):
        self.tts, self.sink = tts, sink or NullSink()
        self.queue: list[np.ndarray] = []       # whole sentences; each is synthesized when it is reached (first audio sooner)
        self.texts: list[str] = []
        self.cur: np.ndarray | None = None
        self.pos = 0
        self.spoken: list[str] = []             # fully played sentences
        self.interrupted: list[str] = []        # sentences cut off by stop()
        self.cur_text = ""

    @property
    def speaking(self) -> bool:
        return self.cur is not None or bool(self.texts)

    def say(self, text: str) -> None:
        for s in _sentences(text):
            self.texts.append(s)

    def tick(self) -> None:
        """Play one frame of audio (call once per mic frame)."""
        if self.cur is None:
            if not self.texts:
                return
            self.cur_text = self.texts.pop(0)
            self.cur = self.tts.synthesize(self.cur_text)
            self.pos = 0
            self.sink.start(self.tts.sample_rate)
        n = int(self.tts.sample_rate * FRAME_S)
        self.sink.write(self.cur[self.pos:self.pos + n])
        self.pos += n
        if self.pos >= len(self.cur):
            self.spoken.append(self.cur_text)
            self.cur = None

    def stop(self) -> bool:
        """Barge-in: drop everything not yet played. Returns True if something was cut off."""
        was = self.speaking
        if self.cur is not None:
            self.interrupted.append(self.cur_text)
        self.interrupted.extend(self.texts)
        self.cur, self.texts = None, []
        if was:
            self.sink.stop()
        return was


def _sentences(text: str) -> list[str]:
    import re
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
