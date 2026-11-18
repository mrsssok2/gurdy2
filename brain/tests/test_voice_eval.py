"""Read-aloud voice eval (brain/voice/eval_read.py): prompts, WER, command accuracy, scoring a recorded folder, the recording loop."""
import json
import queue
import re

import numpy as np
import pytest
import yaml

from brain.voice.audio import FRAME_N, SAMPLE_RATE, frames, read_wav
from brain.voice.eval_read import (format_report, load_prompts, normalize_words, parse_command, record, score_folder, score_item, summarize, wer)
from brain.voice.stt import Transcript

RNG = np.random.default_rng(3)


def quiet(s):
    return RNG.normal(0, 10 ** (-60 / 20), int(s * SAMPLE_RATE)).astype(np.float32)


def speech(s):
    return RNG.normal(0, 0.1, int(s * SAMPLE_RATE)).astype(np.float32)


def test_prompts_cover_the_required_kinds_and_are_not_copied_from_ood():
    ps = load_prompts()
    assert 35 <= len(ps) <= 50 and len({p["text"] for p in ps}) == len(ps)
    cats = {p["category"] for p in ps}
    assert {"room", "phonetic", "object", "home", "status", "stop", "yes_no", "long", "casual"} <= cats
    texts = " ".join(p["text"] for p in ps).lower()
    for w in ("room a", "room b", "alpha", "bravo", "delta", "go home", "where are you", "what do you see", "stop", "yes", "no"):
        assert w in texts, w
    norm = lambda t: re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()           # noqa: E731
    ood = {norm(t["say"]) for c in yaml.safe_load(open("brain/eval/testset_ood_v1.yaml"))["cases"] for t in c["turns"]}
    assert [p["text"] for p in ps if norm(p["text"]) in ood] == []
    lens = [len(p["text"].split()) for p in ps]
    assert min(lens) == 1 and max(lens) >= 10                                # short and long


def test_wer():
    assert wer("Go to room A.", "go to room a") == (0, 4)
    assert wer("Go to room A.", "Go to room") == (1, 4)
    assert wer("Go to room A", "So to room B") == (2, 4)
    assert wer("Walk forward 2 metres", "walk forward two metres") == (0, 4)
    assert wer("Stop.", "") == (1, 1)
    assert normalize_words("Okay, let's go.") == ["okay", "let's", "go"]


def test_command_equality_uses_the_agents_room_normalisation():
    assert parse_command("Go to room Alpha.") == parse_command("go to room A")
    assert parse_command("Go to room D as in dog") == parse_command("go to room delta")
    assert parse_command("Yes, go ahead") == "YES" and parse_command("no") == "NO"
    ok = score_item("Go to room Bravo.", "Go to room B")
    assert ok["command_ok"] and ok["cmd_basis"] == "command" and ok["errors"] == 1          # wrong words, right command
    assert not score_item("Go to room B.", "Go to room A.")["command_ok"]
    assert not score_item("Go to room A.", "Go to room")["command_ok"]
    assert not score_item("Stop.", "")["command_ok"]
    t = score_item("Could you check what is in room A?", "Could you check what is in room A")       # keyword parser cannot parse it: words
    assert t["cmd_basis"] == "text" and t["command_ok"]
    assert not score_item("Could you check what is in room A?", "Could you check what is in room B")["command_ok"]


def test_summarize_per_category():
    items = [{"category": "room", "errors": 1, "ref_words": 4, "command_ok": False, "confidence": 0.5},
             {"category": "room", "errors": 0, "ref_words": 4, "command_ok": True, "confidence": 0.9},
             {"category": "stop", "errors": 0, "ref_words": 1, "command_ok": True, "confidence": 0.8}]
    s = summarize(items)
    assert s["overall"] == {"n": 3, "wer_pct": 11.1, "command_acc_pct": 66.7, "mean_confidence": 0.73}
    assert s["by_category"]["room"]["wer_pct"] == 12.5 and s["by_category"]["room"]["command_acc_pct"] == 50.0


class FakeSTT:
    """Answers from a dict wav-length -> text; set .text for the next clips."""
    def __init__(self, mapping): self.mapping, self.bias = mapping, None

    def transcribe(self, audio, sr=SAMPLE_RATE, prompt=None):
        return Transcript(self.mapping(self.bias, len(audio)), 0.8, 0.01)


def _fake_folder(tmp_path):
    from brain.voice.audio import write_wav
    rows = [("room", "Go to room A."), ("stop", "Stop."), ("casual", "Okay, let's go home now.")]
    for i, (c, t) in enumerate(rows, 1):
        write_wav(tmp_path / f"{i:03d}.wav", speech(0.5 + 0.1 * i))
    (tmp_path / "meta.jsonl").write_text("".join(json.dumps({"id": i, "category": c, "text": t, "wav": f"{i:03d}.wav"}) + "\n"
                                                 for i, (c, t) in enumerate(rows, 1)))
    return rows


def test_score_folder_compares_bias_modes_and_drops_hallucinations(tmp_path):
    _fake_folder(tmp_path)
    ref = {0.6: "Go to room A.", 0.7: "Stop.", 0.8: "Okay, let's go home now."}
    seen = []

    def stt_for_mode(mode):
        def mapping(_, n):
            sec = round(n / SAMPLE_RATE, 1)
            if mode == "none":
                return {0.6: "Go to room.", 0.7: "Stop.", 0.8: "Roo. Roo. Roo."}[sec]          # a miss, a hit, a hallucination
            return ref[sec]
        seen.append(mode)
        return FakeSTT(mapping)
    res = score_folder(tmp_path, stt_for_mode, ["none", "prompt"])
    assert seen == ["none", "prompt"]
    assert res["modes"]["prompt"]["overall"]["command_acc_pct"] == 100.0 and res["modes"]["prompt"]["overall"]["wer_pct"] == 0.0
    none = res["modes"]["none"]
    assert none["overall"]["command_acc_pct"] == 33.3 and none["by_category"]["stop"]["command_acc_pct"] == 100.0
    bad = {x["id"]: x for x in res["items"]["none"]}
    assert bad[3]["rejected"] == "repetition" and bad[3]["hyp"] == "Roo. Roo. Roo." and not bad[3]["command_ok"]
    rep = format_report(res)
    assert "none" in rep and "prompt" in rep and "Go to room A." in rep and "casual" in rep


def test_recording_loop_saves_wav_and_reference_one_at_a_time(tmp_path):
    prompts = [{"id": 1, "category": "room", "text": "Go to room A."}, {"id": 2, "category": "stop", "text": "Stop."}]
    lines = []
    ctl = queue.Queue()
    audio = np.concatenate([quiet(0.6), speech(0.8), quiet(0.2), quiet(0.8), speech(0.5), quiet(0.2), quiet(1.0)])
    fr = list(frames(audio))
    sent = []

    def src():
        for i, f in enumerate(fr):
            if i == 5:
                ctl.put(("enter",))                              # start sentence 1
            if i == 45:
                ctl.put(("enter",))                              # send (0.9 s of audio: speech ends at 0.6+0.8 s = frame 70)
            if i == 50:
                ctl.put(("enter",))                              # Enter during the post-roll: ignored
                sent.append(i)
            if i == 110:
                ctl.put(("enter",))                              # start sentence 2
            if i == 140:
                ctl.put(("enter",))                              # send
            yield f
        for _ in range(40):
            yield fr[0] * 0.0
    meta = record(prompts, tmp_path, src(), ctl, out=lines.append)
    assert [m["id"] for m in meta] == [1, 2] and [m["text"] for m in meta] == ["Go to room A.", "Stop."]
    rows = [json.loads(x) for x in (tmp_path / "meta.jsonl").read_text().splitlines()]
    assert [r["wav"] for r in rows] == ["001.wav", "002.wav"] and all((tmp_path / r["wav"]).exists() for r in rows)
    assert len(read_wav(tmp_path / "001.wav")) > 0.5 * SAMPLE_RATE
    text = "\n".join(lines)
    assert text.count("recording") == 2 and "[1/2] (room)" in text and "[2/2] (stop)" in text and "Go to room A." in text


def test_recording_redo_and_quit(tmp_path):
    prompts = [{"id": 1, "category": "a", "text": "One."}, {"id": 2, "category": "a", "text": "Two."}]
    ctl = queue.Queue()
    ctl.put(("skip",)); ctl.put(("redo",)); ctl.put(("quit",))
    lines = []
    assert record(prompts, tmp_path, iter([np.zeros(FRAME_N, np.float32)] * 5), ctl, out=lines.append) == []
    assert sum("[1/2]" in l for l in lines) == 2 and any("[2/2]" in l for l in lines)
