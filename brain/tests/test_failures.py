"""Mission failures: one message, plain words, arrival tolerance, the rest of the plan offered, the robot really stops, room letters keep their capital."""
import re

import pytest

from brain.agent import Agent, ScriptedLLM, templates
from brain.agent.llm import KeywordLLM
from brain.ros import RosBackend
from brain.ros.fake_port import FakePort
from brain.tools import ToolLayer

PLACES = """map_id: fake_lab
places:
  home: {x: 0.0, y: 0.0, kind: waypoint}
  room a: {rect: [-1.0, -1.0, 2.0, 2.0], kind: room}
  room b: {rect: [3.0, -1.0, 6.0, 2.0], kind: room}
"""
BOX = {"id": 1, "class": "box", "color": "red", "x": 4.0, "y": 0.5, "z": 0.2, "hits": 9, "score": 0.8, "sigma_m": 0.04, "first_seen": 1.0, "last_seen": 2.0}
RAW = re.compile(r"stopped:|finished:|go_to|_|\bblocked\b\.$")


def make(tmp_path, pose=(0.0, 0.0, 0.0), objs=(), llm=None):
    f = tmp_path / "places.yaml"
    f.write_text(PLACES)
    port = FakePort(pose)
    port.set_objects(list(objs))
    b = RosBackend(port, f)
    return b, port, Agent(ToolLayer(b), llm or KeywordLLM(b.known_places))


def pump_until(agent, port, secs=40.0, dt=0.5):
    msgs = []
    for _ in range(int(secs / dt)):
        port.advance(dt)
        msgs += agent.pump()
        if not agent.executor.active:
            break
    return msgs


def test_failed_go_to_gives_one_plain_message_and_offers_the_rest(tmp_path):
    b, p, a = make(tmp_path, objs=[BOX])
    p.blocked.append((4.5, 0.5, 4.0))                    # every goal in room B aborts; the robot is still at home (far away)
    a.say("Go to room B and find the red box")
    msgs = pump_until(a, p)
    assert len(msgs) == 1, msgs                           # task_failed + mission_failed + path_blocked alert = ONE message
    m = msgs[0]
    assert m == "I couldn't reach room B: my path is blocked. I haven't looked for the red box yet. Shall I search from here?"
    assert not RAW.search(m.replace("my path is blocked.", "")), m
    assert a.gate.pending is not None and a.gate.pending.tool == "resume_plan"
    r = a.say("yes")                                      # runs exactly the unfinished step from here
    assert [s.tool for s in a.executor.steps] == ["find"] and "red box" in r.reply.lower()


def test_late_path_blocked_alert_is_not_spoken_again(tmp_path):
    b, p, a = make(tmp_path)
    p.blocked.append((4.5, 0.5, 4.0))
    a.say("go to room B")
    msgs = pump_until(a, p)
    assert len(msgs) == 1 and "my path is blocked" in msgs[0]
    p.alert("path_blocked", "warning", "The path is blocked: navigation gave up on the goal.", goal="7")      # the monitor node's own alert, late
    p.advance(0.5)
    assert a.pump() == []                                  # still within the quiet window: no second "path blocked"


def test_nav2_giving_up_inside_the_room_counts_as_arrived_and_the_plan_continues(tmp_path):
    b, p, a = make(tmp_path, pose=(3.3, 1.7, 0.0), objs=[BOX])     # inside room B, 2.2 m from its centre
    p.blocked.append((4.5, 0.5, 4.0))
    a.say("Go to room B and find the red box")
    msgs = pump_until(a, p)
    assert msgs[0] == "I'm in room B, close to the goal."
    assert not any("blocked" in m for m in msgs) and not any("couldn't" in m for m in msgs)
    assert [s.status for s in a.executor.steps] == ["done", "done"]         # find ran too (the box is known)
    assert any("red box" in m.lower() for m in msgs)


def test_giving_up_within_one_metre_of_the_goal_counts_as_arrived(tmp_path):
    b, p, a = make(tmp_path, pose=(2.9, 0.5, 0.0))        # just outside the rect, 1.6 m from the centre: NOT close
    p.blocked.append((4.5, 0.5, 4.0))
    a.say("go to room B")
    assert "my path is blocked" in " ".join(pump_until(a, p))
    b, p, a = make(tmp_path, pose=(3.8, 1.9, 0.0))        # inside the rect
    p.blocked.append((4.5, 0.5, 4.0))
    a.say("go to room B")
    assert pump_until(a, p)[0] == "I'm in room B, close to the goal."


def test_failure_stops_the_robot_for_real(tmp_path):
    b, p, a = make(tmp_path)
    p.blocked.append((4.5, 0.5, 4.0))
    a.say("go to room B")
    h = max(p.goals)
    pump_until(a, p)
    assert h in p.cancelled_goals                         # the Nav2 goal (and its recovery behaviours) was cancelled
    assert p.zero_holds and p.zero_holds[-1][1] >= 1.0 and p.cmd == (0.0, 0.0, 0.0)
    n = len(p.zero_holds)
    a.say("go to room A")
    a.say("stop")
    assert len(p.zero_holds) > n                          # and a user stop holds zero as well


def test_room_goal_avoids_the_gap_between_objects(tmp_path):
    cyl = {"id": 2, "class": "barrel", "color": "green", "x": 4.5, "y": 1.0, "z": 0.2, "hits": 9, "score": 0.8, "sigma_m": 0.04, "first_seen": 1.0, "last_seen": 2.0}
    b, p, a = make(tmp_path, objs=[BOX, cyl])             # the centre (4.5, 0.5) is 0.5 m from the box and 0.5 m from the barrel
    goals = b._place_goals(b.places["room B"])
    assert goals[0] != (4.5, 0.5) and all(min(((t["x"] - g[0]) ** 2 + (t["y"] - g[1]) ** 2) ** 0.5 for t in b._tracks()) >= 0.9 for g in goals)
    assert b.go_to(place="room B").ok
    gx, gy = p.goals[max(p.goals)]["x"], p.goals[max(p.goals)]["y"]
    assert (gx, gy) in goals and 3.0 <= gx <= 6.0 and -1.0 <= gy <= 2.0
    b2, p2, a2 = make(tmp_path)                           # no objects: the centre is still the goal
    assert b2._place_goals(b2.places["room B"])[0] == (4.5, 0.5)


def test_room_letters_keep_their_capital(tmp_path):
    b, p, a = make(tmp_path)
    assert "room B" in b.known_places() and "room b" not in b.known_places()
    assert templates.ack("go_to", {"place": "room b"}, {"estimate": {"distance_m": 4.5}}).startswith("Going to room B")
    r = a.say("go to room b")
    assert "room B" in r.reply and "room b" not in r.reply


@pytest.mark.parametrize("tool,reason", [("go_to", "blocked"), ("go_to", "no_path"), ("go_to", "timeout"), ("walk", "blocked"), ("find", "timeout"),
                                         ("patrol", "blocked"), ("explore", "stuck"), ("inspect", "blocked"), ("turn", "blocked"), ("return_home", "no_path"),
                                         ("go_to", "weird_reason")])
def test_failure_text_is_plain_words(tool, reason):
    t = templates.failure_text(tool, reason, {"place": "room b"} if tool in ("go_to", "inspect") else {})
    assert t and not re.search(r"stopped:|finished:|_", t) and t[0].isupper() and t.endswith(".")
    assert templates.failure_text("go_to", "cancelled") is None and templates.failure_text("go_to", "estop") is None
    assert "room B" in templates.failure_text("go_to", "blocked", {"place": "room b"})


def test_mock_backend_events_are_plain_too():
    from brain.mock import MockBackend
    b = MockBackend()
    b.go_to(place="room B")
    b.advance(2.0)
    b.stop()
    texts = [e["text"] for e in b.events]
    assert not any(re.search(r"finished: |stopped: ", t) for t in texts), texts
    assert all(e.get("text") for e in b.events)


def test_announce_drops_duplicates_of_one_failure():
    from brain.mock import MockBackend
    a = Agent(ToolLayer(MockBackend()), ScriptedLLM(plans=[], replans=[{}]))
    ev = [{"type": "task_failed", "severity": "warning", "stamp": 5.0, "text": "I couldn't reach room B: my path is blocked.", "data": {"tool": "go_to", "stopped_reason": "blocked", "task_id": 1}},
          {"type": "path_blocked", "severity": "warning", "stamp": 5.0, "text": "The path is blocked.", "data": {"task_id": 1}},
          {"type": "mission_failed", "severity": "warning", "stamp": 5.0, "text": "I couldn't reach room B: my path is blocked.", "data": {"step": "go_to", "reason": "blocked"}}]
    out = a._announce(ev)
    assert out == ["I couldn't reach room B: my path is blocked."]
    assert a._announce([ev[1]]) == []                      # a later alert for the same failure: silent
