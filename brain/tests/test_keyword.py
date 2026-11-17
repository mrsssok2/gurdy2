import pytest

from brain.eval.keyword_parser import parse
from brain.tools import validate, TOOLS

PLACES = ["home", "room A", "room B", "doorway"]


def calls(text):
    return parse(text, PLACES).get("calls")


@pytest.mark.parametrize("text,tool,args", [
    ("go to room B", "go_to", {"place": "room B"}),
    ("Navigate to the doorway please", "go_to", {"place": "doorway"}),
    ("walk forward 2 metres", "walk", {"direction": "forward", "distance_m": 2.0}),
    ("move back 50 cm", "walk", {"direction": "backward", "distance_m": 0.5}),
    ("go left", "walk", {"direction": "left", "distance_m": 1.0}),
    ("turn right 45 degrees", "turn", {"direction": "right", "angle_deg": 45.0}),
    ("turn around", "turn", {"direction": "left", "angle_deg": 180.0}),
    ("find the red box", "find", {"object": {"class": "box", "color": "red"}, "approach": False}),
    ("look for the barrel", "find", {"object": {"class": "barrel"}, "approach": False}),
    ("go home", "return_home", {}),
    ("come back", "return_home", {}),
    ("explore", "explore", {}),
    ("where is the blue box", "where_is", {"object": {"class": "box", "color": "blue"}}),
    ("where are you", "where_am_i", {}),
    ("what is nearby", "whats_nearby", {}),
    ("how is the battery", "status", {}),
    ("pause", "mission_control", {"action": "pause"}),
    ("resume", "mission_control", {"action": "resume"}),
    ("add a no go zone in room B", "no_go_zone", {"action": "add", "place": "room B"}),
    ("remove the no-go zone in room B", "no_go_zone", {"action": "remove", "place": "room B"}),
    ("save this place as kitchen", "save_place", {"name": "kitchen"}),
    ("stop", "stop", {}),
])
def test_single_commands(text, tool, args):
    c = calls(text)
    assert c and c[0]["tool"] == tool and c[0]["args"] == args


def test_every_parsed_call_is_schema_valid():
    for t in ["go to room B", "walk forward 2 metres", "turn right 45 degrees", "find the red box", "where is the blue box",
              "go home", "add a no go zone in room B", "patrol room A and room B", "inspect the red box", "explore"]:
        for c in calls(t):
            assert validate(c["args"], TOOLS[c["tool"]].parameters) == [], (t, c)


def test_multi_clause_and_unknown():
    c = calls("go to room B and then find the red box then come back")
    assert [x["tool"] for x in c] == ["go_to", "find", "return_home"]
    assert "reply" in parse("sing me a song", PLACES)
    assert "reply" in parse("go there", PLACES)                    # no context: the baseline cannot resolve it


def test_baseline_known_weaknesses_documented_by_tests():
    # it never asks: an ambiguous noun just becomes the first parse (here a box with no colour)
    assert calls("go to the box")[0]["args"] == {"object": {"class": "box"}}
    # "back" always means home, even when the user meant "walk back 1 metre" without a verb of motion
    assert calls("come back")[0]["tool"] == "return_home"
