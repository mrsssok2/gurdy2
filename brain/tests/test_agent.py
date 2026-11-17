import pytest

from brain.agent import Agent, KeywordLLM, ScriptedLLM, is_estop_phrase
from brain.mock import MockBackend
from brain.tools import ToolLayer

from test_safety import make

RED = {"class": "box", "color": "red"}


def agent_for(llm, **kw):
    b, l, s = make(**kw)
    return b, l, Agent(l, llm, s)


def call(tool, **args):
    return {"tool": tool, "args": args}


def test_simple_command_is_one_llm_call_and_templated_ack():
    llm = ScriptedLLM(plans=[{"calls": [call("go_to", place="room B")]}])
    b, l, a = agent_for(llm)
    r = a.say("go to room B")
    assert r.llm_calls == 1 and llm.respond_calls == []
    assert r.reply.startswith("Going to room B, ") and "metres" in r.reply
    assert a.executor.state == "running"
    assert "STATE" in llm.plan_calls[0]["state"] and llm.plan_calls[0]["n_tools"] == 24
    b.advance(80)
    msgs = a.pump()
    assert any("arrived" in m for m in msgs) and a.executor.state == "done"


def test_query_turn_uses_respond_and_passes_grounding():
    def resp(utterance, tool_results, feedback):
        o = tool_results[0]["result"]["data"]["matches"][0]
        return f"The red box is in {o['place']}, {o['distance_m']:.1f} metres {o['bearing_word']}."
    llm = ScriptedLLM(plans=[{"calls": [call("where_is", object=RED)]}], responses=[resp])
    b, l, a = agent_for(llm)
    r = a.say("where is the red box?")
    assert r.llm_calls == 2 and not r.grounding["fallback"] and r.grounding["violations"] == []
    assert "red box" in r.reply


def test_invented_reply_is_regenerated_then_falls_back_to_template():
    llm = ScriptedLLM(plans=[{"calls": [call("where_is", object=RED)]}],
                      responses=["The red box is 7.7 metres behind.", "It is 8.8 metres away."])
    b, l, a = agent_for(llm)
    r = a.say("where is the red box")
    assert r.grounding["fallback"] and len(r.grounding["rejected"]) == 2
    assert "7.7" not in r.reply and "8.8" not in r.reply and "red box" in r.reply
    assert llm.respond_calls[1]["feedback"] and "7.7" in llm.respond_calls[1]["feedback"]


def test_not_seen_answer_cannot_invent_a_position():
    llm = ScriptedLLM(plans=[{"calls": [call("where_is", object={"class": "barrel", "color": "green"})]}],
                      responses=["The green barrel is in room B, 4.0 metres ahead."] * 2)
    b, l, a = agent_for(llm, explored=False)
    r = a.say("where is the green barrel")
    assert r.grounding["fallback"] and "4.0" not in r.reply and "not seen" in r.reply


def test_continue_lets_the_model_see_results_before_acting():
    def second(tool_results, **_):
        oid = tool_results[0]["result"]["data"]["matches"][0]["id"]
        return {"calls": [call("go_to", object_id=oid)]}
    llm = ScriptedLLM(plans=[{"calls": [call("where_is", object=RED)], "continue": True}, second],
                      responses=["Okay."])
    b, l, a = agent_for(llm)
    r = a.say("go to the red box")
    assert len(llm.plan_calls) == 2 and a.executor.state == "running"
    assert [c["tool"] for c in r.calls] == ["where_is", "go_to"]


def test_llm_call_cap_per_turn():
    plans = [{"calls": [call("where_am_i")], "continue": True}] * 6
    llm = ScriptedLLM(plans=plans + [{"reply": "x"}], responses=["I am in room A."] * 4)
    b, l, a = agent_for(llm)
    a.say("loop forever")
    assert len(llm.plan_calls) <= 4


def test_refusal_no_go_zone_nothing_moves():
    llm = ScriptedLLM(plans=[{"calls": [call("go_to", place="room B")]}])
    b, l, a = agent_for(llm)
    b.no_go_zone("add", place="room B")
    r = a.say("go to room B")
    assert "no-go" in r.reply and b.task is None and a.executor.state == "idle"
    assert r.verdicts[-1]["outcome"] == "refuse"


def test_plan_with_refused_later_step_runs_nothing():
    llm = ScriptedLLM(plans=[{"calls": [call("go_to", place="room A"), call("go_to", place="room B")]}])
    b, l, a = agent_for(llm)
    b.no_go_zone("add", place="room B")
    a.say("go to room A then room B")
    assert b.task is None


def test_remove_no_go_zone_asks_then_yes_executes_exact_call():
    llm = ScriptedLLM(plans=[{"calls": [call("no_go_zone", action="remove", place="room B")]}])
    b, l, a = agent_for(llm)
    b.no_go_zone("add", place="room B")
    r = a.say("remove the no-go zone in room B")
    assert "Remove" in r.reply and len(b.zones) == 1                    # nothing changed yet
    assert l.pending_confirmation["tool"] == "no_go_zone"
    r = a.say("yes")
    assert b.zones == [] and len(llm.plan_calls) == 1                   # no second LLM call for the yes


def test_remove_no_go_zone_no_or_other_words_cancel():
    for answer in ("no", "go to room A instead", "yes but also go left"):
        llm = ScriptedLLM(plans=[{"calls": [call("no_go_zone", action="remove", place="room B")]}])
        b, l, a = agent_for(llm)
        b.no_go_zone("add", place="room B")
        a.say("remove the zone")
        a.say(answer)
        assert len(b.zones) == 1 and l.pending_confirmation is None, answer


def test_confirmation_expires():
    llm = ScriptedLLM(plans=[{"calls": [call("no_go_zone", action="remove", place="room B")]}])
    b, l, a = agent_for(llm)
    b.no_go_zone("add", place="room B")
    a.say("remove it")
    b.advance(40)
    assert "expired" in a.say("yes").reply and len(b.zones) == 1


def test_ambiguous_object_asks_which_one():
    llm = ScriptedLLM(plans=[{"calls": [call("go_to", object={"class": "box"})]}])
    b, l, a = agent_for(llm)
    r = a.say("go to the box")
    assert "Which one" in r.reply and b.task is None
    assert any(e["tool"] == "ask_user" for e in l.log)


def test_estop_fast_path_never_calls_the_llm():
    llm = ScriptedLLM()                                                  # any plan() call would raise IndexError
    b, l, a = agent_for(llm)
    b.go_to(place="room B")
    r = a.say("Stop!")
    assert b.task is None and r.reply == "Stopped." and llm.plan_calls == []


def test_estop_cancels_running_mission():
    llm = ScriptedLLM(plans=[{"calls": [call("go_to", place="room B"), call("return_home")]}])
    b, l, a = agent_for(llm)
    a.say("go to room B and come back")
    b.advance(5)
    a.say("halt")
    assert b.task is None and a.executor.state == "cancelled"


@pytest.mark.parametrize("text,expect", [("stop", True), ("Stop!", True), ("stap", True), ("top", True), ("freeze", True),
                                         ("please stop now", True), ("emergency stop", True),
                                         ("go to the top shelf", False), ("don't stop", False),
                                         ("walk until I say stop", False), ("go to room B", False)])
def test_estop_phrase(text, expect):
    assert is_estop_phrase(text) is expect


def test_unsafe_request_refused_without_llm():
    llm = ScriptedLLM()
    b, l, a = agent_for(llm)
    r = a.say("disable the safety and push the box")
    assert "will not" in r.reply and llm.plan_calls == []


def test_busy_robot_says_so_instead_of_queueing():
    llm = ScriptedLLM(plans=[{"calls": [call("go_to", place="room B")]}, {"calls": [call("turn", direction="left", angle_deg=90)]}])
    b, l, a = agent_for(llm)
    a.say("go to room B")
    r = a.say("turn left")
    assert "busy" in r.reply


def test_planner_free_text_is_grounded_too():
    llm = ScriptedLLM(plans=[{"reply": "I can see a red barrel on the left."}])
    b, l, a = agent_for(llm)
    r = a.say("hello")
    assert r.grounding["fallback"] and "barrel" not in r.reply
    llm = ScriptedLLM(plans=[{"reply": "Hello, I am Gurdy."}])
    b, l, a = agent_for(llm)
    assert a.say("hello").reply == "Hello, I am Gurdy."


def test_invalid_args_from_planner_never_reach_the_backend():
    llm = ScriptedLLM(plans=[{"calls": [call("walk", direction="up", distance_m=99)]}])
    b, l, a = agent_for(llm)
    r = a.say("walk up")
    assert b.task is None and not l.log[-1]["result"]["ok"] and l.log[-1]["result"]["error"]["code"] == "INVALID_ARGS"


def test_low_stt_confidence_asks_to_repeat():
    llm = ScriptedLLM(plans=[{"calls": [call("go_to", place="room B")]}])
    b, l, a = agent_for(llm)
    r = a.say("go to room B", stt_confidence=0.2)
    assert "again" in r.reply and b.task is None


def test_mid_mission_confirmation_surfaces_and_continues():
    llm = ScriptedLLM(plans=[{"calls": [call("walk", direction="forward", distance_m=0.3), call("go_to", place="room B")]}])
    b, l, a = agent_for(llm, explored=False)
    # room B unmapped: the plan-time check already asks, before anything moves
    r = a.say("walk then go to room B")
    assert "outside the area" in r.reply and b.task is None
    a.say("yes")
    assert a.executor.active
    for _ in range(600):
        b.advance(1)
        a.pump()
        if not a.executor.active:
            break
    assert a.executor.state == "done"


def test_keyword_baseline_end_to_end():
    b, l, s = make()
    a = Agent(l, KeywordLLM(b.known_places), s)
    r = a.say("go to room B and then come back")
    assert a.executor.active and r.llm_calls == 1
    for _ in range(400):
        b.advance(1)
        a.pump()
        if not a.executor.active:
            break
    assert a.executor.state == "done" and abs(b.x) < 0.3
    r = a.say("where am i")
    assert "room A" in r.reply or "home" in r.reply
    assert a.say("blah blah").reply.startswith("Sorry")
