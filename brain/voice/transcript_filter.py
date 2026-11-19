"""Hallucination filter for speech-to-text results, as plain functions (no numpy, no audio libraries) so the SAME rules run in the live microphone
pipeline (brain/voice/stt.py), the server's /api/voice and the in-browser demo (Pyodide, web/static_demo/session.py).

Whisper invents text on silence / noise ("Thank you.", "Roo. Roo. Roo."). A transcript is rejected (-> "Sorry, I didn't catch that.") when:
  * the clip was silence / mic noise (RMS below MIN_RMS),
  * it is the same 1-3 word chunk repeated (a decoding loop),
  * Whisper's own numbers say so (compression ratio, no-speech probability with low confidence) - when the engine provides them,
  * `weak_signal_phrases=True` (small browser models, which give no such numbers): a stock silence phrase on a quiet clip.
A stop phrase is never rejected: the callers check `is_estop_phrase` first."""
from __future__ import annotations

import re

MIN_RMS = 0.004             # below this the clip is silence / mic noise (speech here is 0.02 - 0.1)
LOW_RMS = 0.012             # "quiet" for the stock-phrase check
_WORD = re.compile(r"[a-z']+")
# what Whisper-class models typically print for silence or background noise
SILENCE_PHRASES = {"thank you", "thanks", "thank you very much", "thanks for watching", "thank you for watching", "you", "bye", "bye bye", "okay", "ok",
                   "uh", "um", "hmm", "mm", "huh", "oh", "so", "the", "i", "yeah", "music", "silence", "blank audio", "no speech", "applause", "laughter"}


def repeated_tokens(text: str) -> bool:
    """"Roo. Roo. Roo." / "Go, go, go, go." / "go to go to go to" / "go to room b go to room b ...": the same 1-6 word chunk at least 3 times, or one
    word repeated 3+ times in a row covering most of the text. Whisper does this on quiet or unclear audio (a 6-word period was added after the
    in-browser tiny model printed "Go to Room B." ten times for a 2.6 s clip)."""
    t = _WORD.findall(text.lower())
    if len(t) < 3:
        return False
    for p in (1, 2, 3, 4, 5, 6):
        if len(t) // p >= 3 and all(t[i] == t[i % p] for i in range(len(t))):
            return True
    best = run = 1
    for a, b in zip(t, t[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best >= 3 and best / len(t) >= 0.7


def is_silence_phrase(text: str) -> bool:
    words = " ".join(_WORD.findall(re.sub(r"[\[\](){}*♪]", " ", text.lower())))
    return not words or words in SILENCE_PHRASES


def reject_values(text: str, rms: float | None = None, *, compression_ratio: float | None = None, no_speech_prob: float | None = None,
                  confidence: float | None = None, weak_signal_phrases: bool = False) -> str | None:
    """Why a transcript is treated as "I didn't catch that" instead of a command (None = fine)."""
    if rms is not None and rms < MIN_RMS:
        return "too quiet"
    if repeated_tokens(text):
        return "repetition"
    if compression_ratio is not None and compression_ratio > 2.4:
        return "repetition"
    if no_speech_prob is not None and no_speech_prob > 0.8 and (confidence is None or confidence < 0.5):
        return "no speech"
    if weak_signal_phrases and rms is not None and rms < LOW_RMS and is_silence_phrase(text):
        return "no speech"
    return None
