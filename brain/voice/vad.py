"""Voice activity detection + utterance segmentation.

EnergyVad: per-frame speech / non-speech from short-time energy against an adaptive noise floor (CPU, no model, no
dependency). Silero VAD (design 5.D) can replace it behind the same `is_speech(frame) -> bool` call; the energy VAD is
the fallback and the test implementation. LIMITATION: energy VAD cannot tell speech from other loud sounds.

Segmenter: turns frames into utterances. Two modes:
  * "vad"  - always listening: an utterance starts after `start_s` of speech, ends after `hangover_s` of silence.
  * "ptt"  - push-to-talk: the utterance is everything between press() and release(); leading/trailing silence is trimmed.
Events are returned from feed(): ("start", None) / ("end", audio) / ("partial", audio_so_far) every `partial_every_s`."""
from __future__ import annotations

from collections import deque

import numpy as np

from .audio import FRAME_S


class EnergyVad:
    def __init__(self, margin_db: float = 9.0, floor_min_db: float = -62.0, floor_max_db: float = -35.0, adapt: float = 0.05, creep: float = 0.004):
        self.margin_db, self.adapt, self.creep = margin_db, adapt, creep
        self.floor_min, self.floor_max = floor_min_db, floor_max_db
        self.floor = floor_min_db + 6.0            # dB; start low, adapts upward in noise
        self.boost_db = 0.0                        # raised while the robot talks (barge-in, echo)

    @staticmethod
    def level_db(frame: np.ndarray) -> float:
        return 10 * np.log10(float(np.mean(np.square(frame, dtype=np.float64))) + 1e-12)

    def is_speech(self, frame: np.ndarray) -> bool:
        lv = self.level_db(frame)
        speech = lv > self.floor + self.margin_db + self.boost_db
        # learn the floor quickly from non-speech frames; during "speech" only creep upward, so a steady loud noise
        # (fan, motors) stops counting as speech after some seconds, while a 2 s utterance barely moves it
        rate = self.adapt if not speech else self.creep
        self.floor = float(np.clip(self.floor + rate * (lv - self.floor), self.floor_min, self.floor_max))
        return speech


class Segmenter:
    def __init__(self, vad=None, mode: str = "vad", start_s: float = 0.12, hangover_s: float = 0.5, preroll_s: float = 0.4, ptt_preroll_s: float = 0.3,
                 max_s: float = 15.0, min_s: float = 0.2, partial_every_s: float | None = None):
        if mode not in ("vad", "ptt"):
            raise ValueError("mode: vad | ptt")
        self.vad = vad or EnergyVad()
        self.mode = mode
        self.n_start = max(1, round(start_s / FRAME_S))
        self.lead_pad, self.tail_pad = 12, 8          # frames kept before the first / after the last loud frame (0.24 s / 0.16 s)
        self.n_hang = max(1, round(hangover_s / FRAME_S))
        self.n_max = round(max_s / FRAME_S)
        self.n_min = max(1, round(min_s / FRAME_S))
        self.n_partial = round(partial_every_s / FRAME_S) if partial_every_s else 0
        self.preroll: deque = deque(maxlen=max(1, round(preroll_s / FRAME_S)))
        self.ptt_preroll: deque = deque(maxlen=max(1, round(ptt_preroll_s / FRAME_S)))   # audio just before the button press
        self.buf: list[np.ndarray] = []
        self.active = False
        self.ptt_down = False
        self.run = 0            # consecutive speech frames (before start) / silence frames (after)
        self._since_partial = 0

    # ---- push-to-talk
    def press(self):
        self.ptt_down = True
        self.buf, self.active, self.run, self._since_partial = [], False, 0, 0

    def release(self) -> list:
        self.ptt_down = False
        return self._finish() if self.buf else []

    # ---- frames
    def feed(self, frame: np.ndarray) -> list[tuple[str, np.ndarray | None]]:
        sp = self.vad.is_speech(frame)
        ev: list = []
        if self.mode == "ptt":
            if not self.ptt_down:
                self.ptt_preroll.append(frame)       # people start talking a moment before / while pressing
                return ev
            if not self.buf:
                ev.append(("start", None))
                self.buf = list(self.ptt_preroll)
                self.ptt_preroll.clear()
            self.buf.append(frame)
            return ev + self._partial()
        if not self.active:
            self.preroll.append(frame)
            self.run = self.run + 1 if sp else 0
            if self.run >= self.n_start:
                self.active, self.run, self._since_partial = True, 0, 0
                self.buf = list(self.preroll)
                self.preroll.clear()
                ev.append(("start", None))
            return ev
        self.buf.append(frame)
        self.run = 0 if sp else self.run + 1
        if self.run >= self.n_hang or len(self.buf) >= self.n_max:
            return ev + self._finish()
        return ev + self._partial()

    def _partial(self) -> list:
        self._since_partial += 1
        if self.n_partial and self._since_partial >= self.n_partial:
            self._since_partial = 0
            return [("partial", np.concatenate(self.buf))]
        return []

    def _finish(self) -> list:
        audio = np.concatenate(self.buf) if self.buf else np.zeros(0, np.float32)
        self.buf, self.active, self.run = [], False, 0
        audio = self._trim(audio)
        return [("end", audio if len(audio) >= self.n_min * round(FRAME_S * 16000) else None)]

    def _trim(self, audio: np.ndarray) -> np.ndarray:
        """Cut leading silence and the trailing hangover (keep 0.1 s), so Whisper sees speech, not a long silence."""
        n = round(FRAME_S * 16000)
        lv = [self.vad.level_db(audio[i:i + n]) for i in range(0, len(audio) - n + 1, n)]
        if not lv:
            return audio
        thr = max(self.vad.floor + self.vad.margin_db, max(lv) - 35.0)
        idx = [i for i, v in enumerate(lv) if v > thr]
        if not idx:
            return audio[:0]
        a, b = max(0, idx[0] - self.lead_pad), min(len(lv), idx[-1] + 1 + self.tail_pad)   # soft onsets ("Wh-", "Where") must survive
        return audio[a * n:b * n]
