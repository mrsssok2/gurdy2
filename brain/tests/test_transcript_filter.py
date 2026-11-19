"""The hallucination rules are shared by the live pipeline, the server and the browser demo: same decisions everywhere."""
from brain.voice.stt import Transcript, reject_reason
from brain.voice.transcript_filter import MIN_RMS, is_silence_phrase, reject_values, repeated_tokens

import numpy as np


def test_live_pipeline_still_decides_like_before():
    loud = np.full(16000, 0.05, np.float32)
    quiet = np.full(16000, 0.001, np.float32)
    assert reject_reason(Transcript("go to room B"), loud) is None
    assert reject_reason(Transcript("go to room B"), quiet) == "too quiet"
    assert reject_reason(Transcript("Roo. Roo. Roo."), loud) == "repetition"
    assert reject_reason(Transcript("go", compression_ratio=3.0), loud) == "repetition"
    assert reject_reason(Transcript("hmm", confidence=0.2, no_speech_prob=0.95), loud) == "no speech"
    assert reject_reason(Transcript("Thank you."), loud) is None                         # the live pipeline does not use the stock-phrase rule


def test_plain_function_matches_the_numpy_wrapper():
    assert reject_values("go home", MIN_RMS / 2) == "too quiet"
    assert reject_values("go home", 0.05) is None
    assert reject_values("go go go go", 0.05) == "repetition"
    assert reject_values("go home", None) is None                                       # no audio level known: nothing to say about it


def test_weak_signal_rule_is_only_for_browser_models_and_only_on_quiet_clips():
    assert reject_values("Thank you.", 0.006, weak_signal_phrases=True) == "no speech"
    assert reject_values("Thank you.", 0.05, weak_signal_phrases=True) is None          # a clearly spoken "thank you" is kept
    assert reject_values("Thank you.", 0.006) is None
    assert reject_values("go to room B", 0.006, weak_signal_phrases=True) is None
    assert is_silence_phrase("[BLANK_AUDIO]") and is_silence_phrase("  ") and not is_silence_phrase("go home")


def test_repetition_detector():
    assert repeated_tokens("go to go to go to") and repeated_tokens("Roo. Roo. Roo.")
    assert repeated_tokens("Go to Room B. " * 10 + "Go to Room")                      # the 4-word loop the browser's tiny model produced
    assert repeated_tokens("find the red box " * 3)
    assert not repeated_tokens("go to room B and then go home")
    assert not repeated_tokens("go to room B go to room B")                      # said twice is a user, not a loop
    assert not repeated_tokens("go to room A then room B then room A")
