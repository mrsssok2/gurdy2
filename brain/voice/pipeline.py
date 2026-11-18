"""Voice pipeline: mic frames -> segmenter (VAD / push-to-talk) -> STT -> stop fast path -> agent -> TTS, with barge-in.

Order of reaction to speech (design 4.3 level 2):
  1. speech starts while the robot is talking          -> TTS stops at once (barge-in), before anything is transcribed
  2. while the robot MOVES, a short tail window of the running utterance is transcribed every `partial_every_s`;
     a stop phrase there fires `on_estop` without waiting for the end of the utterance
  3. at the end of the utterance the transcript is checked for a stop phrase BEFORE the agent (no LLM on that path);
     `on_estop` fires first, then the agent still gets the text (its router answers "stopped" from a template, no LLM)
`on_estop` should publish /estop (RosBackend.estop()); it is idempotent on the receiving side.

Single thread: feed() is called once per 20 ms mic frame and also plays one frame of TTS, so the mic is the clock.
LIMITATION: without echo cancellation the mic hears the robot's own voice; `speaking_boost_db` raises the VAD threshold
while it speaks, a headset avoids the problem."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .audio import FRAME_S, SAMPLE_RATE
from .stt import reject_reason, STT
from .tts import Speaker
from .vad import Segmenter


@dataclass
class VoiceConfig:
    mode: str = "vad"                       # vad | ptt
    partial_every_s: float = 0.6            # stop-word check cadence while moving
    partial_window_s: float = 1.5           # transcribe only the last N seconds of the running utterance
    speaking_boost_db: float = 8.0          # VAD threshold raise while the robot talks
    ptt_postroll_s: float = 0.5             # push-to-talk keeps recording this long after Enter (the last word is often still being said)
    min_confidence: float = 0.0             # below this the agent gets None confidence-flagged text (it may ask again)


class VoicePipeline:
    def __init__(self, stt: STT, speaker: Speaker, on_text: Callable[[str, float | None], str | None],
                 on_estop: Callable[[], None], is_moving: Callable[[], bool] = lambda: False,
                 cfg: VoiceConfig | None = None, segmenter: Segmenter | None = None, is_stop_phrase=None):
        self.cfg = cfg or VoiceConfig()
        self.stt, self.speaker, self.on_text, self.on_estop, self.is_moving = stt, speaker, on_text, on_estop, is_moving
        self.seg = segmenter or Segmenter(mode=self.cfg.mode, partial_every_s=self.cfg.partial_every_s, ptt_postroll_s=self.cfg.ptt_postroll_s)
        if is_stop_phrase is None:
            from ..agent.graph import is_estop_phrase as is_stop_phrase
        self.is_stop = is_stop_phrase
        self.t = 0.0
        self.log: list[dict] = []
        self._speech_t0: float | None = None
        self._estop_this_utt = False

    # ---- push-to-talk (UI button / key)
    def press(self):
        self._barge_in("ptt")
        self.seg.press()
        self._speech_t0, self._estop_this_utt = self.t, False

    def release(self):
        self._handle(self.seg.release())

    # ---- mic
    def feed(self, frame: np.ndarray) -> None:
        self.t += FRAME_S
        self.seg.vad.boost_db = self.cfg.speaking_boost_db if self.speaker.speaking else 0.0
        self._handle(self.seg.feed(frame))
        self.speaker.tick()

    def _handle(self, events: list) -> None:
        for kind, audio in events:
            if kind == "start":
                self._speech_t0, self._estop_this_utt = self.t, False
                self._barge_in("speech")
            elif kind == "partial":
                if self.is_moving() and not self._estop_this_utt:
                    tail = audio[-int(self.cfg.partial_window_s * SAMPLE_RATE):]
                    tr = self.stt.transcribe(tail)
                    if tr.text and self.is_stop(tr.text):
                        self._fire_estop(tr.text, "partial", tr.seconds)
            elif kind == "end" and audio is not None:
                self._on_utterance(audio)

    def _barge_in(self, why: str):
        if self.speaker.stop():
            self.log.append({"t": self.t, "event": "barge_in", "why": why})

    def _on_utterance(self, audio: np.ndarray) -> None:
        tr = self.stt.transcribe(audio)
        self.log.append({"t": self.t, "event": "transcript", "text": tr.text, "stt_s": tr.seconds, "audio_s": len(audio) / SAMPLE_RATE})
        if not tr.text:
            return
        why = reject_reason(tr, audio)
        if why and not self.is_stop(tr.text):
            self.log.append({"t": self.t, "event": "rejected", "text": tr.text, "why": why})
            self.say("Sorry, I didn't catch that.")
            return
        if self.is_stop(tr.text) and not self._estop_this_utt:
            self._fire_estop(tr.text, "final", tr.seconds)
        reply = self.on_text(tr.text, tr.confidence)
        if reply:
            self.say(reply)

    def _fire_estop(self, text: str, where: str, stt_s: float):
        self._estop_this_utt = True
        self.speaker.stop()
        self.on_estop()
        self.log.append({"t": self.t, "event": "estop", "text": text, "where": where, "stt_s": stt_s,
                         "since_speech_start_s": None if self._speech_t0 is None else self.t - self._speech_t0})

    def say(self, text: str) -> None:
        self.log.append({"t": self.t, "event": "say", "text": text})
        self.speaker.say(text)

    def events(self, name: str) -> list[dict]:
        return [e for e in self.log if e["event"] == name]
