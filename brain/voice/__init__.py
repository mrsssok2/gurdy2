"""Voice path (milestone 5.8 prep): mic -> push-to-talk / VAD -> STT -> stop fast path -> agent -> TTS, with barge-in.

Everything is driven by mic frames (single thread, the mic is the clock), so tests run deterministically on CPU
with WAV files. Heavy engines (faster-whisper, Piper) are imported lazily and sit behind small protocols."""
from .audio import FRAME_S, SAMPLE_RATE, read_wav, resample, write_wav
from .pipeline import VoiceConfig, VoicePipeline
from .stt import FasterWhisperSTT, ScriptedSTT, Transcript
from .tts import PiperTTS, Speaker
from .vad import EnergyVad, Segmenter

__all__ = ["FRAME_S", "SAMPLE_RATE", "read_wav", "resample", "write_wav", "VoiceConfig", "VoicePipeline", "FasterWhisperSTT",
           "ScriptedSTT", "Transcript", "PiperTTS", "Speaker", "EnergyVad", "Segmenter"]
