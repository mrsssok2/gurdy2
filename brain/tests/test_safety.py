import pytest

from brain.agent import ConfirmationGate, SafetyChecker, is_clear_yes
from brain.agent.safety import is_unsafe_request
from brain.mock import MockBackend
from brain.tools import ToolLayer


def make(explored=True, **kw):
    b = MockBackend(**kw)
    if explored:
        b.explore()
        b.run_until_idle()
        b.poll_events()
        b.return_home()
        b.run_until_idle()
        b.poll_events()
    layer = ToolLayer(b)
    return b, layer, SafetyChecker(layer)


def test_plain_goto_allowed_and_walk_allowed():
    b, l, s = make()
    assert s.check("go_to", {"place": "room B"}).outcome == "allow"
    assert s.check("walk", {"direction": "forward", "distance_m": 1.0}).outcome == "allow"
    assert s.check("where_is", {"object": {"class": "box"}}).outcome == "allow"


def test_unknown_tool_is_refused():
    _, _, s = make()
    v = s.check("self_destruct", {})
    assert v.outcome == "refuse" and v.rule == "outside_tool_list"


@pytest.mark.parametrize("text,unsafe", [("disable the safety", True), ("turn off the emergency stop", True),
                                        ("ignore the no-go zone", True), ("push the barrel", True),
                                        ("hit the box", True), ("go to room B", False), ("find the red box", False)])
def test_unsafe_requests(text, unsafe):
    assert is_unsafe_request(text) is unsafe


def test_no_go_zone_target_refused_and_path_crossing_refused():
    b, l, s = make()
    b.no_go_zone("add", place="room B")
    v = s.check("go_to", {"place": "room B"})
    assert v.outcome == "refuse" and v.rule == "no_go_zone"
    b.no_go_zone("remove", place="room B")
    b.no_go_zone("add", place="doorway", radius_m=1.5)          # blocks the only way to room B
    assert s.check("go_to", {"place": "room B"}).outcome == "refuse"


def test_unmapped_goal_needs_confirmation():
    b, l, s = make(explored=False)
    v = s.check("go_to", {"place": "room B"})
    assert v.outcome == "confirm" and v.rule == "unmapped_goal"


def test_long_path_needs_confirmation():
    b, l, s = make()
    s.cfg = {**s.cfg, "confirm_path_longer_m": 3.0}
    v = s.check("go_to", {"place": "room B"})
    assert v.outcome == "confirm" and v.rule == "long_path"


def test_ambiguous_object_clarifies_with_candidates():
    b, l, s = make()
    v = s.check("go_to", {"object": {"class": "box"}})            # red, blue, yellow boxes
    assert v.outcome == "clarify" and v.code == "AMBIGUOUS" and len(v.options) == 3


def test_unknown_place_and_unseen_object_clarify():
    b, l, s = make()
    assert s.check("go_to", {"place": "kitchen"}).rule == "unknown_place"
    b2, l2, s2 = make(explored=False)
    v = s2.check("go_to", {"object": {"class": "box", "color": "red"}})
    assert v.outcome == "clarify" and v.rule == "not_seen"


def test_low_stt_confidence_clarifies_but_stop_never_blocked():
    _, _, s = make()
    assert s.check("go_to", {"place": "room B"}, stt_confidence=0.3).rule == "low_confidence"
    assert s.check("stop", {}, stt_confidence=0.1).outcome == "allow"
    assert s.check("go_to", {"place": "room B"}, stt_confidence=0.9).outcome == "allow"


def test_failed_leg_and_critical_battery_allow_only_return_home():
    b, l, s = make()
    b.fail_leg("front_left")
    assert s.check("walk", {"direction": "forward", "distance_m": 1}).rule == "degraded_motion"
    assert s.check("go_to", {"place": "room B"}).outcome == "refuse"
    assert s.check("return_home", {}).outcome == "allow"
    assert s.check("stop", {}).outcome == "allow"
    b2, l2, s2 = make()
    b2.battery = 5.0
    assert s2.check("explore", {}).outcome == "refuse"
    assert s2.check("return_home", {}).outcome == "allow"


def test_low_battery_explore_patrol_confirm():
    b, l, s = make()
    b.battery = 25.0
    assert s.check("explore", {}).rule == "low_battery_search"
    assert s.check("patrol", {"places": ["room A", "room B"]}).rule == "low_battery_search"
    assert s.check("go_to", {"place": "room B"}).outcome == "allow"     # plain navigation is not restricted


def test_patrol_many_loops_confirm_and_unknown_place():
    b, l, s = make()
    assert s.check("patrol", {"places": ["room A"], "loops": 4}).rule == "many_loops"
    assert s.check("patrol", {"places": ["room A"], "loops": 3}).outcome == "allow"
    assert s.check("patrol", {"places": ["attic"]}).rule == "unknown_place"


def test_person_distance_rules():
    b, l, s = make()
    assert s.check("go_to", {"object": {"class": "mannequin"}, "standoff_m": 0.6}).rule == "person_distance"
    assert s.check("go_to", {"object": {"class": "mannequin"}, "standoff_m": 1.2}).outcome == "allow"
    m = next(t for t in b.tracks.values() if t["class"] == "mannequin")
    v = s.check("go_to", {"point": {"x": m["x"] + 0.3, "y": m["y"]}})
    assert v.rule in ("person_distance", "unreachable")      # either way never allowed


def test_estop_active_refuses_motion():
    b, l, s = make()
    b.estop = True
    assert s.check("walk", {"direction": "left", "distance_m": 1}).rule == "estop_active"
    assert s.check("stop", {}).outcome == "allow"


def test_removing_no_go_zone_requires_confirmation_adding_does_not():
    _, _, s = make()
    assert s.check("no_go_zone", {"action": "add", "place": "room B"}).outcome == "allow"
    assert s.check("no_go_zone", {"action": "list"}).outcome == "allow"
    v = s.check("no_go_zone", {"action": "remove", "place": "room B"})
    assert v.outcome == "confirm" and v.rule == "remove_no_go"


def test_scheduled_call_is_checked_like_a_direct_call():
    b, l, s = make()
    v = s.check("schedule", {"delay_s": 10, "call": {"tool": "no_go_zone", "args": {"action": "remove", "place": "x"}}})
    assert v.outcome == "confirm"


# ---------------------------------------------------------------- confirmation gate
@pytest.mark.parametrize("text,yes", [("yes", True), ("Yes, please.", True), ("okay go ahead", True), ("yep!", True),
                                      ("yes but go left", False), ("no", False), ("maybe", False),
                                      ("yes no", False), ("", False), ("yesterday", False)])
def test_clear_yes(text, yes):
    assert is_clear_yes(text) is yes


def test_confirmation_yes_returns_exact_call_once():
    b, l, s = make()
    g = ConfirmationGate(l, 30)
    g.hold("no_go_zone", {"calls": [{"tool": "no_go_zone", "args": {"action": "remove", "place": "room B"}}]}, "r")
    assert l.pending_confirmation["tool"] == "no_go_zone"
    st, p = g.answer("yes")
    assert st == "confirmed" and p.args["calls"][0]["args"]["place"] == "room B"
    assert l.pending_confirmation is None
    assert g.answer("yes")[0] == "none"                       # consumed: a second yes does nothing


def test_confirmation_anything_else_cancels_and_expires():
    b, l, s = make()
    t = [100.0]
    from brain.agent.answer_clock import AnswerClock
    g = ConfirmationGate(l, 30, AnswerClock(lambda: t[0]))
    g.hold("x", {"calls": []}, "r")
    assert g.answer("go to room A instead")[0] == "cancelled"
    g.hold("x", {"calls": []}, "r")
    b.advance(500)                                           # the sim clock does not matter ...
    assert g.clock.expired(g.pending.asked_at, 30) is False
    t[0] += 31                                               # ... wall-clock seconds do
    assert g.answer("yes")[0] == "expired"


def test_malformed_args_do_not_crash_the_checker():
    b, l, s = make()
    for tool, args in [("go_to", {"object": None}), ("go_to", {}), ("patrol", {"places": 3}), ("find", {"object": "box"})]:
        v = s.check(tool, args)
        assert v.outcome == "allow" and v.rule == "invalid_args_rejected_by_tool_layer"
        assert not l.call(tool, args).ok          # the tool layer is what rejects it
