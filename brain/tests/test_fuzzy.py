"""Speech-recognition repair: Whisper vocabulary bias + "Did you mean room B?" (never executed without a yes)."""
import pytest

from brain.agent.fuzzy import suggest_object, suggest_place
from brain.agent.graph import Agent
from brain.agent.llm import KeywordLLM
from brain.mock import MockBackend
from brain.tools import ToolLayer
from brain.voice.vocab import hotwords, initial_prompt, vocabulary

PLACES = ["home", "room A", "room B", "doorway"]


@pytest.mark.parametrize("text,want", [("Go to room V and then find the red box.", "room B"), ("go to room e", "room B"),
                                       ("go to room D", "room B"), ("go to room P please", "room B"), ("go to dorway", "doorway")])
def test_place_suggestions(text, want):
    s = suggest_place(text, PLACES)
    assert s and s.options == [want]
    assert want.lower() in s.corrected[want].lower()


@pytest.mark.parametrize("text", ["go to room B", "go to room A", "Go to room b.", "go to room 8", "go to the kitchen", "go to home",
                                  "save this place as room V", "find the red box", "what is the battery"])
def test_no_suggestion_for_known_or_unrelated(text):
    assert suggest_place(text, PLACES) is None


def test_correction_keeps_rest_of_sentence():
    s = suggest_place("Go to room V and then find the red box.", PLACES)
    assert s.corrected["room B"] == "Go to room B and then find the red box."


def test_object_suggestion():
    s = suggest_object("find the extinguiser", PLACES)
    assert s and s.options == ["extinguisher"]
    assert suggest_object("find the red box", PLACES) is None
    assert suggest_object("find the table", PLACES) is None


def _agent():
    b = MockBackend()
    return b, Agent(ToolLayer(b), KeywordLLM(b.known_places))


def test_agent_asks_and_does_not_move_without_yes():
    b, a = _agent()
    r = a.say("go to room V")
    assert "Did you mean room B" in r.reply and r.calls == [] and r.llm_calls == 0
    assert a.executor.state == "idle"


def test_agent_runs_corrected_request_after_yes():
    b, a = _agent()
    a.say("go to room V")
    r = a.say("yes")
    assert any(c["tool"] == "go_to" for c in r.calls) and "room B" in str(r.calls)


def test_agent_drops_fix_on_other_answer():
    b, a = _agent()
    a.say("go to room V")
    r = a.say("no, go home")
    assert a.pending_fix is None and "room b" not in str(r.calls).lower()
    r = a.say("yes")                        # a stale yes must not resurrect the old request
    assert not any(c["tool"] == "go_to" for c in r.calls)


def test_stop_beats_pending_fix():
    b, a = _agent()
    a.say("go to room V")
    r = a.say("stop")
    assert r.reply == "Stopped." and a.pending_fix is None


def test_fix_expires():
    b, a = _agent()
    a.say("go to room V")
    a.pending_fix["expires_at"] = -1
    r = a.say("yes")
    assert not any(c["tool"] == "go_to" for c in r.calls)


def test_vocabulary_contains_places_objects_colours():
    v = vocabulary(PLACES)
    for w in ("room A", "room B", "doorway", "home", "barrel", "fire extinguisher", "red", "yellow"):
        assert w in v
    assert "room B" in initial_prompt(PLACES) and "room B" in hotwords(PLACES)


@pytest.mark.parametrize("text", ["go to room B and then come back", "come back", "walk home and halt", "go to the doorway then come home"])
def test_ordinary_words_are_not_places(text):
    assert suggest_place(text, PLACES) is None
