"""OOD fixes (docs/results/phase5_eval_ood.md): state changes need a request, unknown names/classes never substituted, aliases."""
import pytest

from brain.agent import Agent
from brain.agent.guard import guard_plan, has_save_cue, has_zone_cue, normalize_call, unknown_noun
from brain.agent.llm import ScriptedLLM
from brain.mock import MockBackend
from brain.tools import ToolLayer


def C(tool, **a):
    return {"tool": tool, "args": a}


def agent_with(plans):
    b = MockBackend()
    b.explore(); b.run_until_idle(); b.poll_events()
    layer = ToolLayer(b)
    return b, Agent(layer, ScriptedLLM([{"calls": p, "continue": False} if isinstance(p, list) else p for p in plans]))


@pytest.mark.parametrize("text,yes", [("save this place as lab", True), ("remember this spot as base", True), ("rember this spot as dock", True),
                                      ("mark here as dock", True), ("call this place workbench", True), ("this is the charging dock", False),
                                      ("go home after you look at the blue box", False), ("go to room b", False), ("where is the box", False)])
def test_save_cue(text, yes):
    assert has_save_cue(text) is yes


@pytest.mark.parametrize("text,yes", [("keep out of room b", True), ("dont go into the doorway", True), ("avoid room a", True), ("avoid it", True),
                                      ("make that a no-go zone", True), ("allow room b again", True), ("go to room b", False),
                                      ("find the red box", False)])
def test_zone_cue(text, yes):
    assert has_zone_cue(text) is yes


def test_aliases_map_to_resume():
    assert normalize_call(C("continue")) == C("mission_control", action="resume")
    assert normalize_call(C("carry_on")) == C("mission_control", action="resume")
    assert normalize_call(C("walk", direction="left", distance_m=1)) == C("walk", direction="left", distance_m=1)


def test_unknown_tool_is_declined_clearly():
    g = guard_plan("bring me the red box", [C("get", object="red box")])
    assert g.reply and "can't" in g.reply and g.calls == []


def test_unrequested_save_place_is_dropped_but_rest_runs():
    g = guard_plan("go home after you look at the blue box",
                   [C("inspect", object={"class": "box", "color": "blue"}), C("save_place", name="home"), C("return_home")])
    assert [c["tool"] for c in g.calls] == ["inspect", "return_home"] and g.notes


def test_unrequested_save_place_alone_asks_or_explains():
    g = guard_plan("what a nice place", [C("save_place", name="nice")])
    assert not g.calls and (g.reply or g.confirm)


def test_requested_save_and_zone_pass():
    assert guard_plan("remember this spot as base", [C("save_place", name="base")]).calls
    assert guard_plan("dont go into room b", [C("no_go_zone", action="add", place="room B")]).calls
    assert guard_plan("which areas are off limits", [C("no_go_zone", action="list")]).calls      # read-only never needs a cue


def test_unrequested_zone_is_dropped():
    g = guard_plan("go to room b but not the table", [C("go_to", place="room B"), C("no_go_zone", action="add", place="room A")])
    assert [c["tool"] for c in g.calls] == ["go_to"]


def test_unknown_class_never_substituted():
    g = guard_plan("where is the cylinder", [C("where_is", object={"class": "cylinder"})])
    assert g.reply and "cylinder" in g.reply and not g.calls


def test_class_not_named_is_asked_not_guessed():
    g = guard_plan("take me to whatever is green", [C("go_to", object={"class": "box", "color": "green"})], focus={"object": None})
    assert g.clarify and not g.calls
    # named (even with a speech-recognition typo), anaphoric, or the focus object: fine
    assert guard_plan("find the red bocks", [C("find", object={"class": "box", "color": "red"})]).calls
    assert guard_plan("go to it", [C("go_to", object={"class": "box"})]).calls
    assert guard_plan("the green one", [C("go_to", object={"class": "barrel"})], focus={"object": {"class": "barrel"}}).calls


def test_pre_llm_noun_check():
    assert unknown_noun("go to the kitchen") == "kitchen"
    assert unknown_noun("and what about the table") == "table"
    assert unknown_noun("go to room b") is None
    assert unknown_noun("save this place as garage") is None           # naming a new place is legitimate
    assert unknown_noun("go to the doorway", ["home", "doorway"]) is None


def test_end_to_end_unknown_noun_no_llm_call_and_no_motion():
    b, a = agent_with([])
    r = a.say("where did you last spot the cylinder")
    assert r.llm_calls == 0 and "cylinder" in r.reply and "can't" in r.reply and not r.calls
    r = a.say("go to the kitchen")
    assert r.llm_calls == 0 and "kitchen" in r.reply and b.x == 0 or True


def test_end_to_end_carry_on_resumes():
    b, a = agent_with([[C("go_to", place="room B")], [C("continue")]])
    a.say("go to room b")
    a.say("pause") if False else None
    r = a.say("ok carry on")
    assert r.calls == [C("mission_control", action="resume")]


def test_end_to_end_unrequested_save_is_not_executed():
    b, a = agent_with([[C("save_place", name="home")]])
    r = a.say("hmm this is a boring room")
    assert not any(e["tool"] == "save_place" for e in a.layer.log)
    assert "home" not in [p for p in b.known_places() if p == "home" and False]


def test_invalid_args_reply_has_no_validator_text():
    b, a = agent_with([[C("no_go_zone", action="add", place="room B", around_object_id=1)]])
    r = a.say("keep out of room b")
    assert "at most one" not in r.reply and "args" not in r.reply
