import json
import random

from brain.agent.ollama_llm import build_system
from brain.data_gen.generate import run_dialogue
from brain.data_gen.generate import test_utterances as _test_utterances
from brain.data_gen.lexicon import norm, pick
from brain.data_gen.merge_noise import meaning_kept, table_variant
from brain.tools import TOOLS
from brain.tools.schema import validate

TESTS = _test_utterances()


def _examples(n=60, split="train"):
    out, drops = [], []
    for seed in range(n):
        ex, why, _ = run_dialogue(seed, split, TESTS)
        out += ex
        drops.append(why)
    return out, drops


def test_examples_are_valid_deterministic_and_do_not_leak_test_utterances():
    ex, drops = _examples()
    assert len(ex) > 50 and not any(d and d.startswith("crash") for d in drops)
    again, _ = _examples()
    assert [e["user"] for e in ex] == [e["user"] for e in again]            # seeded -> reproducible
    for e in ex:
        assert norm(e["say"]) not in TESTS
        assert e["user"].startswith("TASK: ")
        if e["kind"] == "plan":
            out = json.loads(e["assistant"])
            for c in out.get("calls", []):
                assert validate(c["args"], TOOLS[c["tool"]].parameters) == []


def test_val_and_train_use_different_templates():
    t = [f"template number {i}" for i in range(60)]
    rng = random.Random(0)
    tr = {pick(rng, t, "train") for _ in range(300)}
    va = {pick(rng, t, "val") for _ in range(300)}
    assert tr and va and not (tr & va)


def test_prompt_prefix_of_examples_matches_the_served_model():
    # the system prompt is stored once; user messages carry only per-turn content, never tool specs
    ex, _ = _examples(10)
    assert all('"name":"' not in e["user"] for e in ex) and len(build_system("json")) > 5000


def test_meaning_kept_rules():
    assert meaning_kept("turn left ninety degrees", "Turn left 90 degrees.")
    assert meaning_kept("turn left ninety degrees", "turn lift ninety degrees")        # known confusion
    assert meaning_kept("go to the red box", "Go to the read box.")
    assert not meaning_kept("turn left ninety degrees", "Turn left 93")
    assert not meaning_kept("walk forward two metres", "What's up with you?")
    assert not meaning_kept("where are we", "There we go.")


def test_table_variant_changes_one_word():
    v = table_variant(random.Random(1), "turn left ninety degrees")
    assert v and v != "turn left ninety degrees" and len(v.split()) == 4
