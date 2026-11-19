"""Smart keyword planner: common phrasings -> the right tool call; deterministic; baseline unchanged."""
import pytest

from brain.agent.llm import KeywordLLM
from brain.agent.phrasing import normalize

PLACES = ["home", "room A", "room B", "doorway"]


def plan(text, smart=True):
    return KeywordLLM(lambda: PLACES, smart=smart).plan("", {}, [], text, [])


def calls(text):
    r = plan(text)
    assert "calls" in r, (text, r)
    return [(c["tool"], c["args"]) for c in r["calls"]]


@pytest.mark.parametrize("text", [
    "enter room b", "Enter room B.", "go into room b", "head to room b", "walk to room b", "take me to room B", "move to room b",
    "visit room b", "please go to room b", "can you go to room b please", "could you head over to room B", "bring me to room b",
    "navigate to room b", "proceed to room B", "room b", "hey gurdy, go to room b", "i want you to enter room b"])
def test_go_to_room_b(text):
    assert calls(text) == [("go_to", {"place": "room B"})]


def test_go_back_to_room_a_is_a_place_not_a_backward_walk():
    assert calls("go back to room a") == [("go_to", {"place": "room A"})]
    assert calls("walk back to the doorway") == [("go_to", {"place": "doorway"})]


@pytest.mark.parametrize("text", ["come back", "go back", "return", "return home", "come home", "go home", "head back", "go back to the start",
                                  "go back to base", "take yourself home", "come back here"])
def test_return_home(text):
    assert calls(text) == [("return_home", {})]


def test_explicit_backward_walk_still_works():
    assert calls("walk back 2 meters") == [("walk", {"direction": "backward", "distance_m": 2.0})]
    assert calls("step back") == [("walk", {"direction": "backward", "distance_m": 1.0})]


@pytest.mark.parametrize("text,obj", [
    ("find the red box", {"class": "box", "color": "red"}), ("look for the red box", {"class": "box", "color": "red"}),
    ("search for the red box", {"class": "box", "color": "red"}), ("search the room for the red box", {"class": "box", "color": "red"}),
    ("go find the red box", {"class": "box", "color": "red"}), ("can you find me the red box", {"class": "box", "color": "red"}),
    ("please locate the red box", {"class": "box", "color": "red"}), ("hunt for the green barrel", {"class": "barrel", "color": "green"})])
def test_find(text, obj):
    (tool, args), = calls(text)
    assert tool == "find" and args["object"] == obj


def test_where_is_and_describe():
    assert calls("where is the green barrel")[0][0] == "where_is"
    assert calls("where can i find the green barrel")[0][0] in ("find", "where_is")
    assert calls("what do you see")[0][0] == "describe_view"
    assert calls("what's around you")[0][0] == "whats_nearby"
    assert calls("what is around")[0][0] == "whats_nearby"


def test_explore_phrasings():
    for t in ("look around", "have a look around", "scan the room", "explore", "map the area"):
        assert calls(t) == [("explore", {})], t


def test_rotate_is_turn():
    assert calls("rotate left 90 degrees")[0][0] == "turn"
    assert calls("spin around")[0][0] == "turn"


def test_multi_step():
    assert calls("enter room b and find the green barrel") == [("go_to", {"place": "room B"}),
                                                              ("find", {"object": {"class": "barrel", "color": "green"}, "approach": False})]
    assert [c[0] for c in calls("take me to room b then come back")] == ["go_to", "return_home"]
    assert [c[0] for c in calls("visit the doorway, then head home")] == ["go_to", "return_home"]


def test_stop_and_status_untouched():
    assert calls("stop")[0][0] == "stop"
    assert calls("what's your battery level")[0][0] == "status"


def test_unknown_stays_unknown():
    assert plan("sing me a song") == {"reply": "Sorry, I did not understand that."}
    assert plan("enter room z")["reply"].startswith("Sorry")


def test_deterministic():
    assert plan("take me to room b") == plan("take me to room b")


def test_baseline_unchanged():
    """The evaluation baseline (smart=False) must keep failing on the new phrasings, otherwise its 55/100 number would silently change."""
    assert plan("enter room b", smart=False) == {"reply": "Sorry, I did not understand that."}
    assert plan("head to room b", smart=False)["calls"][0]["args"] == {"place": "room B"}


def test_normalize_keeps_unknown_text():
    assert normalize("???", PLACES) == "???"
