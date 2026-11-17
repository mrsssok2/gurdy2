from brain.agent import MissionExecutor, SafetyChecker
from brain.mock import MockBackend
from brain.tools import ToolLayer

from test_safety import make


def ex_for(b, l, s):
    return MissionExecutor(l, s)


def types(evs):
    return [e["type"] for e in evs]


def test_two_step_plan_runs_in_order_and_finishes():
    b, l, s = make()
    ex = ex_for(b, l, s)
    ev = ex.submit([{"tool": "go_to", "args": {"place": "room B"}}, {"tool": "return_home", "args": {}}])
    assert ex.state == "running" and [x.status for x in ex.steps] == ["running", "pending"]
    ev += ex.run(b.advance, 600)
    assert ex.state == "done" and [x.status for x in ex.steps] == ["done", "done"]
    assert abs(b.x) < 0.3 and abs(b.y) < 0.3                       # back home
    assert types(ev).count("mission_finished") == 1
    assert ex.steps[0].result["target"]["place"] == "room B"


def test_robot_answers_questions_while_mission_runs():
    b, l, s = make()
    ex = ex_for(b, l, s)
    ex.submit([{"tool": "go_to", "args": {"place": "room B"}}])
    b.advance(10)
    assert l.call("where_am_i").ok and l.call("status").data["mode"] == "moving"
    ex.pump()
    assert ex.state == "running"


def test_failed_step_cancels_rest_and_reports():
    b, l, s = make()
    ex = ex_for(b, l, s)
    ex.submit([{"tool": "go_to", "args": {"place": "room B"}}, {"tool": "walk", "args": {"direction": "left", "distance_m": 1}}])
    b.add_obstacle(__import__("brain.mock", fromlist=["Rect"]).Rect(2.3, 0.0, 2.7, 1.0))   # close the doorway
    ev = ex.run(b.advance, 600)
    assert ex.state == "failed"
    assert [x.status for x in ex.steps] == ["failed", "cancelled"]
    assert "mission_failed" in types(ev)
    failed = next(e for e in ev if e["type"] == "mission_failed")
    assert failed["data"]["step"] == "go_to"


def test_cancel_stops_motion():
    b, l, s = make()
    ex = ex_for(b, l, s)
    ex.submit([{"tool": "go_to", "args": {"place": "room B"}}, {"tool": "return_home", "args": {}}])
    b.advance(5)
    ev = ex.cancel()
    assert ex.state == "cancelled" and b.task is None
    assert [x.status for x in ex.steps] == ["cancelled", "cancelled"]
    assert "mission_cancelled" in types(ev)
    assert not ex.active


def test_pause_and_resume():
    b, l, s = make()
    ex = ex_for(b, l, s)
    ex.submit([{"tool": "walk", "args": {"direction": "forward", "distance_m": 1.0}}])
    ex.pause()
    assert ex.state == "paused" and b.task.paused
    x0 = b.x
    b.advance(20)
    assert b.x == x0
    ex.resume()
    ex.run(b.advance, 60)
    assert ex.state == "done"


def test_step_is_rechecked_before_it_starts():
    """Plan was fine, but a leg fails during step 1 -> step 2 (not return_home) is refused at start time."""
    b, l, s = make()
    ex = ex_for(b, l, s)
    ex.submit([{"tool": "walk", "args": {"direction": "forward", "distance_m": 0.5}}, {"tool": "go_to", "args": {"place": "room B"}}])
    b.fail_leg("rear_left")
    ev = ex.run(b.advance, 60)
    assert ex.state == "failed" and ex.steps[1].reason == "degraded_motion"
    assert ex.steps[1].status == "failed"


def test_confirm_step_pauses_executor_until_approved():
    b, l, s = make(explored=False)                                  # room B unmapped -> confirm at step time
    ex = ex_for(b, l, s)
    ex.submit([{"tool": "walk", "args": {"direction": "forward", "distance_m": 0.3}}, {"tool": "go_to", "args": {"place": "room B"}}])
    ev = ex.run(b.advance, 60)
    assert ex.state == "awaiting_confirmation" and ex.steps[1].status == "pending"
    assert "confirmation_needed" in types(ev)
    assert b.task is None                                           # nothing moves while waiting
    ex.approve_and_continue()
    ex.run(b.advance, 600)
    assert ex.state == "done"


def test_instant_steps_and_events_pass_through():
    b, l, s = make()
    ex = ex_for(b, l, s)
    ev = ex.submit([{"tool": "stop", "args": {}},
                    {"tool": "walk", "args": {"direction": "left", "distance_m": 0.5}}])
    ev += ex.run(b.advance, 60)
    assert ex.state == "done"
    b.fail_leg("front_right")
    assert "leg_failure" in types(ex.pump())                        # unrelated backend events are not swallowed


def test_submit_while_active_is_rejected():
    import pytest
    b, l, s = make()
    ex = ex_for(b, l, s)
    ex.submit([{"tool": "walk", "args": {"direction": "forward", "distance_m": 1.0}}])
    with pytest.raises(RuntimeError):
        ex.submit([{"tool": "turn", "args": {"direction": "left", "angle_deg": 90}}])
