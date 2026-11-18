"""Fixes for the final-eval failures, tested with DIFFERENT wording than the eval missions:
  - requests the robot cannot do (no arm / mail / photos) are declined honestly before the planner (capabilities.py)
  - "the other X" / "another X" is resolved against what was just discussed (refs.py)
Run in the brain image: pytest brain/tests/test_honest_refs.py"""
import pytest

from brain.agent.capabilities import unsupported_request
from brain.agent.graph import Agent
from brain.agent.llm import ScriptedLLM
from brain.agent.refs import resolve_other
from brain.mock import MockBackend
from brain.tools import ToolLayer


# ------------------------------------------------------------------ capabilities
@pytest.mark.parametrize("text,needle", [
    ("email me a picture of the chair", "emails"),
    ("could you send a text to my brother", "messages"),
    ("call my mom please", "phone calls"),
    ("snap a selfie in room A", "photos"),
    ("record a video of the whole room", "videos"),
    ("pick up the blue box", "carry"),
    ("fetch me the barrel", "carry"),
    ("bring me the cone", "carry"),
    ("order me a pizza", "do that kind of thing"),
    ("open the door for me", "do that kind of thing"),
    ("take a picture of the toolbox and send it to me", "photos"),
])
def test_unsupported_requests_get_an_honest_decline(text, needle):
    reply = unsupported_request(text)
    assert reply and reply.startswith("I can't") and needle in reply
    assert "I can walk to places" in reply          # and says what it CAN do


@pytest.mark.parametrize("text", [
    "go to room B", "bring me to the red box", "bring me back home", "carry on", "in order to reach room B walk forward",
    "call this spot the workshop", "save this place as the dock", "take a left turn", "turn right and walk two metres", "what do you see",
    "where did you last see the blue box", "record this position as the charging corner", "find the chair", "take me to the doorway",
    "describe what is in front of you", "keep out of room A", "tell me your battery level", "lift the no-go zone on room B",
    "lift the restriction on the doorway",
])
def test_normal_commands_are_not_declined(text):
    assert unsupported_request(text) is None


def make_agent(plans=None, responses=None):
    b = MockBackend()
    llm = ScriptedLLM(plans or [], responses or [])
    return Agent(ToolLayer(b), llm), b, llm


def test_decline_happens_before_the_planner_and_moves_nothing():
    agent, b, llm = make_agent()
    r = agent.say("grab the yellow box and carry it to room A")
    assert r.reply.startswith("I can't") and r.calls == [] and llm.plan_calls == []
    assert r.verdicts[0]["rule"] == "unsupported_capability"
    assert b.status().data["mode"] == "idle"


def test_a_pending_question_is_answered_not_declined():
    agent, b, llm = make_agent(plans=[{"calls": [{"tool": "no_go_zone", "args": {"action": "remove", "place": "room B"}}], "continue": False}])
    b.no_go_zone("add", place="room B")
    r1 = agent.say("lift the no-go zone on room B")           # a confirmation question, not an "I can't"
    assert "?" in r1.reply
    r2 = agent.say("no")
    assert r2.reply == "Okay, cancelled."


# ------------------------------------------------------------------ the other X
def recs(*items):
    return lambda cls: [i for i in items if i["class"] == cls]


RED = {"id": 1, "class": "box", "color": "red", "place": "room A"}
BLUE = {"id": 2, "class": "box", "color": "blue", "place": "room A"}
YELLOW = {"id": 3, "class": "box", "color": "yellow", "place": "room B"}
GREEN_BARREL = {"id": 4, "class": "barrel", "color": "green", "place": "room B"}
BLUE_BARREL = {"id": 5, "class": "barrel", "color": "blue", "place": "room A"}


def test_other_with_one_remaining_is_rewritten_to_a_specific_object():
    r = resolve_other("go to the other barrel", {"object": {"id": 4, "class": "barrel", "color": "green"}}, recs(GREEN_BARREL, BLUE_BARREL))
    assert r.rule == "other_resolved" and r.text == "go to the blue barrel"


def test_other_excludes_only_the_last_mentioned_object_of_that_class():
    focus = {"object": {"id": 1, "class": "box", "color": "red"}}
    r = resolve_other("walk towards another box", focus, recs(RED, BLUE, YELLOW))
    assert r.question == "Which one: the blue box or the yellow box?" and set(r.options) == {"blue", "yellow"}
    assert r.options["yellow"] == "walk towards the yellow box"
    focus = {"object": {"id": 2, "class": "box", "color": "blue"}}
    r2 = resolve_other("look at the other crate", focus, recs(RED, BLUE, YELLOW))
    assert set(r2.options) == {"red", "yellow"} and r2.options["red"] == "look at the red crate"


def test_a_focus_of_another_class_does_not_exclude_anything():
    focus = {"object": {"id": 4, "class": "barrel", "color": "green"}}
    r = resolve_other("inspect the other box", focus, recs(RED, BLUE, GREEN_BARREL))
    assert set(r.options) == {"red", "blue"}


def test_nothing_else_known_says_so_and_nothing_seen_says_so():
    r = resolve_other("go to the other barrel", {"object": {"id": 4, "class": "barrel", "color": "green"}}, recs(GREEN_BARREL))
    assert r.reply == "I only know one barrel: the green barrel."
    r = resolve_other("go to another chair", None, recs(RED))
    assert r.reply == "I haven't seen any chair yet."


def test_colour_given_with_other_filters_the_candidates():
    two_red = [RED, {"id": 9, "class": "box", "color": "red", "place": "room B"}, BLUE]
    r = resolve_other("go to the other red box", {"object": {"id": 1, "class": "box", "color": "red"}}, recs(*two_red))
    assert r.rule == "other_which_unclear" and "more than one" in r.question      # two red boxes: a colour cannot name one, so it asks


def test_not_an_object_reference_is_left_alone():
    for t in ("go the other way", "turn to the other side", "what about the other room", "walk forward"):
        assert resolve_other(t, None, recs(RED)) is None


def test_agent_rewrites_before_the_planner_and_asks_when_unsure():
    agent, b, llm = make_agent(plans=[{"calls": [{"tool": "go_to", "args": {"object": {"class": "barrel", "color": "blue"}}}], "continue": False}],
                               responses=["Going to the blue barrel."])
    from brain.mock.world import GtObject
    b.world.objects.append(GtObject(7, "barrel", "blue", 1.0, -1.0, 0.2))      # flat_lab has one barrel (green): add a second one
    b.world._build()
    for o in b.world.objects:
        if o.cls == "barrel":
            b._sight(o)
    green = next(t for t in b.tracks.values() if t["color"] == "green")
    agent.layer.focus["object"] = {"id": green["id"], "class": "barrel", "color": "green"}
    agent.say("please go to the other barrel")
    assert llm.plan_calls and llm.plan_calls[0]["utterance"] == "please go to the blue barrel"


def test_agent_asks_which_one_and_the_answer_runs_the_chosen_object():
    agent, b, llm = make_agent(plans=[{"calls": [{"tool": "where_is", "args": {"object": {"class": "box", "color": "yellow"}}}], "continue": False}],
                               responses=["The yellow box is in room B."])
    for o in b.world.objects:
        if o.cls == "box":
            b._sight(o)
    red = next(t for t in b.tracks.values() if t["color"] == "red")
    agent.layer.focus["object"] = {"id": red["id"], "class": "box", "color": "red"}
    r = agent.say("where is the other box")
    assert r.reply.startswith("Which one:") and "blue box" in r.reply and "yellow box" in r.reply and llm.plan_calls == []
    r2 = agent.say("the yellow one")
    assert llm.plan_calls[0]["utterance"] == "where is the yellow box"
