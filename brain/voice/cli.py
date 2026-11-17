"""Run a WAV file (or the live mic) through the voice pipeline against the mock world.
  python3 -m brain.voice.cli --wav utterance.wav [--stt-model /path/to/ct2 --device cpu] [--tts-model piper.onnx]
  python3 -m brain.voice.cli --mic --ptt      (WSLg mic via arecord; Enter = press / release)  [live path untested]
Without --stt-model / --tts-model the engines are not loaded (STT prints nothing): give both for a real run."""
from __future__ import annotations

import argparse

import numpy as np

from ..agent.graph import Agent
from ..agent.llm import KeywordLLM
from ..mock import MockBackend
from ..tools import ToolLayer
from .audio import FRAME_N, frames, read_wav
from .pipeline import VoiceConfig, VoicePipeline
from .stt import FasterWhisperSTT
from .tts import PiperTTS, Speaker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav")
    ap.add_argument("--stt-model", required=True)
    ap.add_argument("--tts-model", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    b = MockBackend()
    agent = Agent(ToolLayer(b), KeywordLLM(b.known_places))
    pipe = VoicePipeline(FasterWhisperSTT(a.stt_model, a.device), Speaker(PiperTTS(a.tts_model)),
                         lambda t, c: agent.say(t, c).reply, lambda: agent.layer.call("stop"),
                         lambda: b.t > 0 and agent.executor.state == "running", VoiceConfig())
    audio = np.concatenate([read_wav(a.wav), np.zeros(FRAME_N * 40, np.float32)])
    for f in frames(audio):
        pipe.feed(f)
    for e in pipe.log:
        print(e)


if __name__ == "__main__":
    main()
