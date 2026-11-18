"""Live voice loop: mic stream -> VoicePipeline (threaded) -> host (agent) -> Piper, plus keyboard control.

What runs where
  * main thread   : reads 20 ms mic frames (the clock), feeds the pipeline, plays TTS one frame per tick. It never waits for
                    Whisper, the LLM or the robot, so barge-in (speech start -> TTS stops within one frame) stays instant.
  * worker thread : transcribes finished utterances in order, runs the stop check, then asks the host (agent) and speaks the reply.
  * stop thread   : while the robot moves, transcribes the tail of the running utterance (a stop phrase fires `estop` before the
                    sentence ends). One at a time; a partial that finds it busy is skipped (CPU Whisper is slower than real time).
  * host pump     : alerts / mission results arrive via `host.on_message` and are spoken (not while the user is talking: they are queued).
  * stdin thread  : Enter = start recording, Enter again = send (push-to-talk). A typed line = a command as text (reply is spoken
                    too, handy without a mic). `q` quits. In VAD mode Enter does nothing, speak any time.
The yes / no for "Did you mean ...?" and for safety confirmations is just the next utterance: the agent holds the question
(`pending_fix` / confirmation gate), see Agent.say."""
from __future__ import annotations

import queue
import sys
import threading
import time

import numpy as np

from .audio import FRAME_S, SAMPLE_RATE
from .pipeline import VoiceConfig, VoicePipeline
from .stt import reject_reason


class LockedSpeaker:
    """Thread-safe facade over Speaker (say / stop come from worker threads, tick from the mic thread)."""

    def __init__(self, sp):
        self._sp, self._lk = sp, threading.RLock()

    @property
    def speaking(self):
        return self._sp.speaking

    def __getattr__(self, name):            # spoken, interrupted, texts, ...
        return getattr(self._sp, name)

    def say(self, t):
        with self._lk:
            self._sp.say(t)

    def stop(self):
        with self._lk:
            return self._sp.stop()

    def tick(self):
        with self._lk:
            self._sp.tick()


class LivePipeline(VoicePipeline):
    def __init__(self, stt, speaker, host, cfg: VoiceConfig | None = None, say_prefix: str = "", out=print, debug_dir=None, **kw):
        sp = speaker if isinstance(speaker, LockedSpeaker) else LockedSpeaker(speaker)
        super().__init__(stt, sp, lambda t, c, sa=None: host.say(t, c, sa), host.estop, host.is_moving, cfg, **kw)
        self.host = host
        self._out_lock = threading.Lock()
        self._out = out
        self._inflight = 0                            # utterances / typed lines being processed (STT -> agent -> reply)
        self._inflight_lock = threading.Lock()
        self._rec_speaking = False                    # the robot was talking at some point during the current recording
        self.debug_dir, self._n_saved = debug_dir, 0     # every utterance is saved as a WAV (+ index.jsonl) to listen to later
        self.jobs: queue.Queue = queue.Queue()
        self._start_wall: float | None = None          # wall clock when the current utterance started (answer timeouts count to HERE)
        self._was_speaking = False
        self._partial_busy = threading.Event()
        self._held: list[str] = []                    # robot messages waiting until the user stops talking
        self._held_lock = threading.Lock()
        self._alive = True
        threading.Thread(target=self._worker, daemon=True).start()
        host.on_message(self.robot_message)

    def out(self, text: str) -> None:
        with self._out_lock:
            self._out(text)

    @property
    def busy(self) -> bool:
        """An utterance is still being transcribed / answered: push-to-talk waits (one at a time)."""
        return self._inflight > 0

    def _inc(self, d: int) -> None:
        with self._inflight_lock:
            self._inflight = max(0, self._inflight + d)

    # ---- robot -> user (alerts, mission results)
    def robot_message(self, text: str) -> None:
        if self.seg.active or self.seg.ptt_down:     # the user is talking: do not talk over them
            with self._held_lock:
                self._held.append(text)
            return
        self.out(f"Gurdy: {text}")
        self.say(text)

    def _flush_held(self) -> None:
        if self._held and not (self.seg.active or self.seg.ptt_down):
            with self._held_lock:
                msgs, self._held = self._held, []
            for m in msgs:
                self.out(f"Gurdy: {m}")
                self.say(m)

    def feed(self, frame: np.ndarray) -> None:
        super().feed(frame)
        sp = bool(self.speaker.speaking)
        if self._was_speaking and not sp:                # the robot just finished speaking: a question's answer window starts now
            self.host.spoken_done()
        self._was_speaking = sp
        if sp and (self.seg.active or self.seg.ptt_down):
            self._rec_speaking = True                    # logged with the clip: the mic may have heard the robot's own voice
        self._flush_held()

    # ---- user -> robot, off the mic thread
    def press(self):
        self._start_wall = time.monotonic()
        self._rec_speaking = bool(self.speaker.speaking)
        super().press()

    def _on_utterance(self, audio: np.ndarray) -> None:
        self._inc(1)
        rec, self._rec_speaking = self._rec_speaking, bool(self.speaker.speaking)
        self.jobs.put(("audio", (audio, self._start_wall, rec)))

    def submit_text(self, text: str) -> None:
        self._inc(1)
        self.jobs.put(("text", text))

    def _worker(self) -> None:
        while self._alive:
            try:
                kind, payload = self.jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if kind == "audio":
                    self._utterance_job(*payload)
                else:
                    self._text_job(payload, None)
            except Exception as e:                      # never let one bad turn kill the loop
                self.out(f"[voice] error: {type(e).__name__}: {e}")
            finally:
                self._inc(-1)

    def _save_debug(self, audio: np.ndarray, tr, robot_speaking: bool = False, rejected: str | None = None) -> None:
        if not self.debug_dir:
            return
        import json
        from pathlib import Path
        from .audio import write_wav
        d = Path(self.debug_dir)
        d.mkdir(parents=True, exist_ok=True)
        self._n_saved += 1
        name = f"{time.strftime('%Y%m%d_%H%M%S')}_{self._n_saved:03d}.wav"
        write_wav(d / name, audio)
        a = np.asarray(audio, np.float32)
        rec = {"wav": name, "text": tr.text, "confidence": tr.confidence, "audio_s": round(len(a) / SAMPLE_RATE, 2),
               "rms": float(np.sqrt(np.mean(a ** 2))) if len(a) else 0.0, "peak": float(np.abs(a).max()) if len(a) else 0.0,
               "mode": self.cfg.mode, "robot_speaking": bool(robot_speaking), "rejected": rejected}
        with open(d / "index.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _utterance_job(self, audio: np.ndarray, started_wall: float | None = None, robot_speaking: bool = False) -> None:
        hint = getattr(self.host, "answer_hint", lambda: None)()          # a question is pending: bias Whisper to the expected answers
        tr = self.stt.transcribe(audio, prompt=hint) if hint else self.stt.transcribe(audio)
        why = reject_reason(tr, audio) if tr.text or len(audio) else None
        if why and tr.text and self.is_stop(tr.text):
            why = None                                   # a stop is never discarded
        try:
            self._save_debug(audio, tr, robot_speaking, why)
        except Exception as e:                          # noqa: BLE001  debugging aid, never breaks the loop
            self.out(f"[voice] could not save wav: {e}")
        self.log.append({"t": self.t, "event": "transcript", "text": tr.text, "stt_s": tr.seconds, "audio_s": len(audio) / SAMPLE_RATE})
        if not tr.text:
            self.out("(heard nothing)")
            return
        if why:                                          # hallucination on quiet / unclear audio ("Roo. Roo. Roo.")
            self.out(f"(heard noise: {tr.text!r}, {why}; ignored)")
            self.say("Sorry, I didn't catch that.")
            return
        self.out(f"You: {tr.text}   [STT {tr.seconds:.1f} s, confidence {'?' if tr.confidence is None else f'{tr.confidence:.2f}'}]")
        self._text_job(tr.text, tr.confidence, started_wall)

    def _text_job(self, text: str, conf: float | None, started_wall: float | None = None) -> None:
        if self.is_stop(text):                       # idempotent: a second stop after a partial one changes nothing
            self._fire_estop(text, "final", 0.0)
        t0 = time.monotonic()
        reply = self.on_text(text, conf, None if started_wall is None else time.monotonic() - started_wall)
        if reply:
            self.out(f"Gurdy: {reply}   [{time.monotonic() - t0:.1f} s]")
            self.say(reply)

    def _handle(self, events: list) -> None:
        for kind, audio in events:
            if kind == "start":
                self._start_wall = time.monotonic()
                self._rec_speaking = bool(self.speaker.speaking)   # before the barge-in below silences the robot
            if kind == "partial":
                if self.is_moving() and not self._estop_this_utt and not self._partial_busy.is_set():
                    self._partial_busy.set()
                    tail = audio[-int(self.cfg.partial_window_s * SAMPLE_RATE):]
                    threading.Thread(target=self._partial_job, args=(tail,), daemon=True).start()
            else:
                super()._handle([(kind, audio)])

    def _partial_job(self, tail: np.ndarray) -> None:
        try:
            tr = self.stt.transcribe(tail)
            if tr.text and self.is_stop(tr.text) and not self._estop_this_utt:
                self._fire_estop(tr.text, "partial", tr.seconds)
                self.out(f"[stop heard in {tr.seconds:.1f} s: {tr.text!r}]")
        finally:
            self._partial_busy.clear()

    def close(self) -> None:
        self._alive = False


def run_live(pipe: LivePipeline, source, ctl: "queue.Queue", stop: threading.Event | None = None) -> None:
    """Feed `source` (iterable of 20 ms float32 frames) until it ends, `stop` is set or a ("quit",) control arrives.
    Controls: ("press",) ("release",) ("text", str) ("quit",). Real-time pacing comes from the source."""
    stop = stop or threading.Event()
    recording = False
    for frame in source:
        while True:
            try:
                c = ctl.get_nowait()
            except queue.Empty:
                break
            if c[0] == "quit":
                return
            if c[0] == "enter" and pipe.cfg.mode == "ptt":      # push-to-talk key: strictly one recording at a time
                if recording:
                    pipe.release(); recording = False
                    pipe.out("  ■ sent  (thinking...)" if pipe.busy or pipe.seg.finishing else "  ■ (too short, nothing sent)")
                elif pipe.busy or pipe.seg.finishing:
                    pipe.out("  (still thinking - wait for the answer, then press Enter)")
                else:
                    pipe.press(); recording = True
                    pipe.out("  ● recording ... press Enter to send")
            elif c[0] == "press" and pipe.cfg.mode == "ptt":
                pipe.press(); recording = True
            elif c[0] == "release" and recording:
                pipe.release(); recording = False
            elif c[0] == "text":
                pipe.submit_text(c[1])
        pipe.feed(frame)
        if stop.is_set():
            return


def stdin_controls(ctl: "queue.Queue", mode: str, out=print) -> threading.Thread:
    """Keyboard thread: it only forwards keys; run_live decides (and prints) what an Enter means, so terminal lines never interleave."""
    def loop():
        for line in sys.stdin:
            line = line.strip()
            if line in ("q", "quit", "exit"):
                ctl.put(("quit",)); return
            ctl.put(("enter",) if line == "" else ("text", line))
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t
