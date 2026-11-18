"""Audio basics: 16 kHz mono float32 in [-1, 1], 20 ms frames, WAV I/O (stdlib `wave`), resampling, real devices."""
from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
FRAME_S = 0.02
FRAME_N = int(SAMPLE_RATE * FRAME_S)


def resample(a: np.ndarray, sr_in: int, sr_out: int = SAMPLE_RATE) -> np.ndarray:
    a = np.asarray(a, np.float32)
    if sr_in == sr_out or len(a) == 0:
        return a
    n = max(1, int(round(len(a) * sr_out / sr_in)))
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)


def read_wav(path: str | Path, sr_out: int = SAMPLE_RATE) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"{path}: only 16-bit PCM WAV supported (got {8 * width}-bit)")
    a = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return resample(a, sr, sr_out)


def write_wav(path: str | Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    pcm = (np.clip(np.asarray(audio, np.float32), -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def frames(audio: np.ndarray, n: int = FRAME_N):
    """Split into whole frames (a short tail is zero-padded)."""
    for i in range(0, len(audio), n):
        f = audio[i:i + n]
        yield f if len(f) == n else np.pad(f, (0, n - len(f)))


# ---------------------------------------------------------------- real devices (WSLg PulseAudio; not used by the tests)
class ArecordSource:
    """Mic frames from `parecord` (WSLg PulseAudio) (raw s16le 16 kHz mono on stdout). Iterating blocks in real time."""

    def __init__(self, cmd: list[str] | None = None):
        self.cmd = cmd or ["parecord", "--raw", "--format=s16le", f"--rate={SAMPLE_RATE}", "--channels=1", "--latency-msec=20"]
        self.proc: subprocess.Popen | None = None

    def __iter__(self):
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE)
        try:
            while True:
                raw = self.proc.stdout.read(FRAME_N * 2)
                if len(raw) < FRAME_N * 2:
                    return
                yield np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
        finally:
            self.close()

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()


class NullSink:
    """Playback sink that records what was played (tests). `write` takes float32 at `sr`."""

    def __init__(self, sr: int = SAMPLE_RATE):
        self.sr = sr
        self.chunks: list[np.ndarray] = []
        self.stopped = 0

    def start(self, sr: int) -> None:
        self.sr = sr

    def write(self, chunk: np.ndarray) -> None:
        self.chunks.append(np.asarray(chunk, np.float32))

    def stop(self) -> None:
        self.stopped += 1

    @property
    def played_s(self) -> float:
        return sum(len(c) for c in self.chunks) / self.sr


class AplaySink:
    """Plays chunks through `paplay` (raw s16le). stop() kills the process: playback ends within the pipe buffer."""

    def __init__(self):
        self.proc: subprocess.Popen | None = None

    def start(self, sr: int) -> None:
        self.stop()
        self.proc = subprocess.Popen(["paplay", "--raw", "--format=s16le", f"--rate={sr}", "--channels=1", "--latency-msec=100"],
                                     stdin=subprocess.PIPE)

    def write(self, chunk: np.ndarray) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write((np.clip(chunk, -1, 1) * 32767).astype("<i2").tobytes())
                self.proc.stdin.flush()
            except BrokenPipeError:
                pass

    def stop(self) -> None:
        if self.proc:
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait()
            self.proc = None
