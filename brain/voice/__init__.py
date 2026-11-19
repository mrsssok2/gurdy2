"""Voice path (milestone 5.8 prep): mic -> push-to-talk / VAD -> STT -> stop fast path -> agent -> TTS, with barge-in.

Everything is driven by mic frames (single thread, the mic is the clock), so tests run deterministically on CPU
with WAV files. Heavy engines (faster-whisper, Piper) are imported lazily and sit behind small protocols.

Lazy exports (PEP 562): `import brain.voice.host` must not import numpy / audio code. The browser demo (Pyodide) runs the agent host without any of it."""
_EXPORTS = {"FRAME_S": "audio", "SAMPLE_RATE": "audio", "read_wav": "audio", "resample": "audio", "write_wav": "audio",
            "VoiceConfig": "pipeline", "VoicePipeline": "pipeline", "FasterWhisperSTT": "stt", "ScriptedSTT": "stt", "Transcript": "stt",
            "PiperTTS": "tts", "Speaker": "tts", "EnergyVad": "vad", "Segmenter": "vad"}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        import importlib
        return getattr(importlib.import_module("." + _EXPORTS[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
