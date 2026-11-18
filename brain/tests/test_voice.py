"""Voice path on CPU with short Piper-synthesised WAVs (brain/tests/data/voice, made by scripts/make_voice_test_wavs.py).
Engines are scripted (no Whisper in the test run); the real-engine check is gated by VOICE_STT_MODEL / VOICE_TTS_MODEL."""
import os
from pathlib import Path

import numpy as np
import pytest

from brain.agent.graph import Agent
from brain.agent.llm import KeywordLLM
from brain.mock import MockBackend
from brain.tools import ToolLayer
from brain.voice import EnergyVad, ScriptedSTT, Segmenter, Speaker, VoiceConfig, VoicePipeline, read_wav, resample, write_wav
from brain.voice.audio import FRAME_N, SAMPLE_RATE, NullSink, frames

DATA = Path(__file__).parent / "data" / "voice"
RNG = np.random.default_rng(0)


def noise(sec, db=-60.0):
    return (RNG.normal(0, 10 ** (db / 20), int(sec * SAMPLE_RATE))).astype(np.float32)


def stream(wav, lead=0.6, tail=1.2, db=-60.0):
    a = read_wav(DATA / f"{wav}.wav")
    return np.concatenate([noise(lead, db), a + noise(len(a) / SAMPLE_RATE, db), noise(tail, db)]), lead, len(a) / SAMPLE_RATE


class SineTTS:
    sample_rate = 22050

    def synthesize(self, text):
        n = int(0.06 * len(text) * self.sample_rate)          # 60 ms per character
        return (0.3 * np.sin(2 * np.pi * 220 * np.arange(n) / self.sample_rate)).astype(np.float32)


def make(stt, moving=False, mode="vad", reply=None, **cfg):
    sink = NullSink()
    est, texts = [], []
    pipe = VoicePipeline(stt, Speaker(SineTTS(), sink), lambda t, c: (texts.append(t), reply)[1], lambda: est.append(pipe.t),
                         lambda: moving, VoiceConfig(mode=mode, **cfg))
    return pipe, sink, est, texts


def run(pipe, audio):
    for f in frames(audio):
        pipe.feed(f)


# ---------------------------------------------------------------- audio / VAD
def test_wav_roundtrip_and_resample(tmp_path):
    a = (0.4 * np.sin(2 * np.pi * 300 * np.arange(8000) / 16000)).astype(np.float32)
    write_wav(tmp_path / "a.wav", a)
    b = read_wav(tmp_path / "a.wav")
    assert len(b) == 8000 and np.max(np.abs(a - b)) < 1e-3
    assert len(resample(a, 16000, 8000)) == 4000


def test_vad_finds_one_utterance_and_trims_silence():
    audio, lead, dur = stream("walk_forward")
    seg, ends, starts = Segmenter(), [], 0
    for f in frames(audio):
        for k, a in seg.feed(f):
            starts += k == "start"
            if k == "end":
                ends.append(a)
    assert starts == 1 and len(ends) == 1
    assert dur - 0.3 < len(ends[0]) / SAMPLE_RATE < dur + 0.5          # speech plus a little margin, hangover trimmed


def test_vad_ignores_pure_noise_even_loud_floor():
    seg = Segmenter()
    evs = [(i, e[0]) for i, f in enumerate(frames(noise(14, -44.0))) for e in seg.feed(f)]
    assert all(i < 9 * 50 for i, _ in evs)                             # steady loud noise: the floor creeps up, no events after ~9 s


def test_ptt_collects_between_press_and_release():
    seg = Segmenter(mode="ptt")
    audio, _, _ = stream("stop", lead=0.0, tail=0.0)
    assert [e for f in frames(audio) for e in seg.feed(f)] == []     # button up: nothing
    seg.press()
    evs = [e for f in frames(audio) for e in seg.feed(f)]
    assert evs == [("start", None)]
    evs = seg.release()
    assert evs[0][0] == "end" and evs[0][1] is not None and 0.3 < len(evs[0][1]) / SAMPLE_RATE < 0.9


# ---------------------------------------------------------------- stop fast path
@pytest.mark.parametrize("wav,text", [("stop", "Stop."), ("halt", "Halt!")])
def test_stop_word_fires_estop_before_the_llm_and_agent_still_answers(wav, text):
    b = MockBackend()
    agent = Agent(ToolLayer(b), KeywordLLM(b.known_places))
    est = []
    pipe = VoicePipeline(ScriptedSTT([text]), Speaker(SineTTS(), NullSink()), lambda t, c: agent.say(t, c).reply,
                         lambda: est.append(pipe.t))
    audio, lead, dur = stream(wav)
    run(pipe, audio)
    assert len(est) == 1                                               # exactly once
    e = pipe.events("estop")[0]
    assert e["where"] == "final" and lead < est[0] < lead + dur + 0.8
    assert pipe.events("say") and pipe.speaker.spoken                  # the agent's answer was spoken (template, no LLM)
    assert any(c["tool"] == "stop" for c in agent.layer.log)


def test_normal_command_does_not_estop_and_reply_is_spoken():
    pipe, sink, est, texts = make(ScriptedSTT(["walk forward two meters"]), reply="Walking forward. Two meters.")
    run(pipe, stream("walk_forward", tail=2.5)[0])
    assert est == [] and texts == ["walk forward two meters"]
    assert pipe.speaker.spoken == ["Walking forward.", "Two meters."] and sink.played_s > 0.5


def test_question_about_stopping_is_not_an_estop():
    pipe, _, est, _ = make(ScriptedSTT(["why did you stop"]))
    run(pipe, stream("question")[0])
    assert est == []


def test_stop_inside_a_long_utterance_fires_before_it_ends_while_moving():
    # the tail window (<= 1.5 s) says "stop"; the full utterance is a long sentence, not a stop phrase
    stt = ScriptedSTT(fn=lambda a: "stop" if len(a) <= 1.6 * SAMPLE_RATE else "walk forward slowly and then turn to the left and stop")
    pipe, _, est, texts = make(stt, moving=True)
    audio, lead, dur = stream("long_then_stop")
    run(pipe, audio)
    e = pipe.events("estop")
    assert len(e) == 1 and e[0]["where"] == "partial"
    assert est[0] < lead + dur - 0.5                                   # before the utterance finished
    assert e[0]["since_speech_start_s"] < 1.5
    assert len(texts) == 1                                             # the full text still reaches the agent


def test_no_partial_checks_while_standing_still():
    stt = ScriptedSTT(fn=lambda a: "stop" if len(a) <= 1.6 * SAMPLE_RATE else "walk forward slowly and then turn to the left and stop")
    pipe, _, est, _ = make(stt, moving=False)
    run(pipe, stream("long_then_stop")[0])
    assert est == [] and len(stt.calls) == 1                            # only the final transcription


# ---------------------------------------------------------------- barge-in
def test_barge_in_stops_tts_within_one_frame_of_speech_start_and_drops_the_rest():
    stt = ScriptedSTT(["stop"])
    pipe, sink, est, _ = make(stt)
    pipe.say("I am walking to the kitchen now. It is about four meters away. I will tell you when I arrive.")
    run(pipe, noise(1.0))                                               # robot talks over room noise
    assert pipe.speaker.speaking and sink.played_s > 0.9
    before = sink.played_s
    t_speech = pipe.t
    run(pipe, np.concatenate([read_wav(DATA / "stop.wav"), noise(1.2)]))
    bi = pipe.events("barge_in")
    assert len(bi) == 1 and bi[0]["t"] - t_speech < 0.3                 # start_s (0.12 s) + a frame
    assert sink.stopped == 1 and not pipe.speaker.speaking
    assert pipe.speaker.interrupted                                     # unplayed sentences dropped
    assert sink.played_s - before < 0.3                                 # at most a few frames after the speech began


def test_ptt_press_also_barges_in():
    pipe, sink, _, _ = make(ScriptedSTT([""]), mode="ptt")
    pipe.say("A long answer that goes on and on.")
    run(pipe, noise(0.3))
    pipe.press()
    assert not pipe.speaker.speaking and sink.stopped == 1


def test_speaking_boost_raises_the_vad_threshold_against_echo():
    pipe, _, _, _ = make(ScriptedSTT([""]), speaking_boost_db=8.0)
    pipe.say("Talking.")
    pipe.feed(noise(0.02, -60.0)[:FRAME_N])
    assert pipe.seg.vad.boost_db == 8.0
    run(pipe, noise(3.0))
    assert pipe.seg.vad.boost_db == 0.0                                 # back to normal when it stops talking


# ---------------------------------------------------------------- real engines (optional)
@pytest.mark.skipif(not (os.environ.get("VOICE_STT_MODEL") and os.environ.get("VOICE_TTS_MODEL")), reason="engines not configured")
def test_real_whisper_and_piper_roundtrip():
    from brain.voice import FasterWhisperSTT, PiperTTS
    stt = FasterWhisperSTT(os.environ["VOICE_STT_MODEL"], "cpu")
    assert "stop" in stt.transcribe(read_wav(DATA / "stop.wav")).text.lower()
    tts = PiperTTS(os.environ["VOICE_TTS_MODEL"])
    a = tts.synthesize("Where are you?")
    assert len(a) / tts.sample_rate > 0.5


@pytest.mark.parametrize("heard", ["top.", "Holt.", "Stop.", "Halt!", "stap"])
def test_real_whisper_outputs_for_stop_and_halt_hit_the_fast_path(heard):
    """'top.' and 'Holt.' are what Whisper turbo (CPU int8) returned for the Piper clips stop.wav / halt.wav."""
    from brain.agent.graph import is_estop_phrase
    assert is_estop_phrase(heard)


# ---------------------------------------------------------------- mic warm-up / pre-roll (WSLg RDPSource wakes in ~1 s)
def _fake_mic_cmd(silent_s, tone_s):
    import sys
    code = ("import sys,numpy as np;sr=16000;"
            f"z=np.zeros(int({silent_s}*sr),'<i2');t=(8000*np.sin(np.arange(int({tone_s}*sr))*0.2)).astype('<i2');"
            "sys.stdout.buffer.write(z.tobytes()+t.tobytes());sys.stdout.buffer.flush()")
    return [sys.executable, "-c", code]


def test_source_discards_warmup_and_stays_open():
    from brain.voice.audio import ArecordSource
    src = ArecordSource(_fake_mic_cmd(1.0, 1.0), warmup_s=1.0, wall_clock=False)
    src.start()
    proc = src.proc
    src.start()                                           # idempotent: the same stream, not a reopen
    assert src.proc is proc and src.warmup_rms == 0.0
    first = next(iter(src))
    assert np.sqrt((first ** 2).mean()) > 0.1             # the silent wake-up is gone, the first frame we get is live audio
    src.close()


def test_source_without_warmup_starts_in_the_silence():
    from brain.voice.audio import ArecordSource
    src = ArecordSource(_fake_mic_cmd(1.0, 1.0), warmup_s=0.0, wall_clock=False)
    assert np.sqrt((next(iter(src)) ** 2).mean()) == 0.0
    src.close()


def test_ptt_preroll_keeps_speech_before_the_press():
    seg = Segmenter(mode="ptt")
    tone = (0.3 * np.sin(np.arange(FRAME_N * 10) * 0.2)).astype(np.float32)
    for f in frames(np.concatenate([noise(1.0), tone])):             # speech starts, the button comes 10 frames later
        seg.feed(f)
    seg.press()
    for f in frames(noise(0.3)):
        seg.feed(f)
    ev = seg.release()
    assert ev and ev[0][0] == "end" and ev[0][1] is not None
    assert np.abs(ev[0][1]).max() > 0.2                               # the first syllable (before the press) is in the segment


def test_vad_preroll_keeps_a_soft_onset():
    seg = Segmenter()
    assert seg.preroll.maxlen * 0.02 >= 0.4 - 1e-9


def test_bias_modes():
    from brain.voice.stt import FasterWhisperSTT
    s = object.__new__(FasterWhisperSTT)
    for mode, has_prompt, has_hot in (("none", False, False), ("prompt", True, False), ("full", True, True), (True, True, True), (False, False, False)):
        s.set_vocabulary(["room A", "room B"], mode)
        assert (s.prompt is not None, s.hotwords is not None) == (has_prompt, has_hot), mode
    import pytest
    with pytest.raises(ValueError):
        s.set_vocabulary(None, "loud")
