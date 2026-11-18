"""Persistent memory + alerts, in the mock world (no ROS, no GPU)."""
from brain.agent import Agent
from brain.agent.alerts import AlertManager
from brain.agent.llm import KeywordLLM
from brain.memory import MemoryBackend, MemoryStore
from brain.mock import MockBackend
from brain.tools import ToolLayer


class Clock:
    def __init__(self, t=1_700_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def make(tmp_path, clock, seed=0):
    store = MemoryStore(tmp_path / "gurdy.db", "mock_flat_lab", wall=clock)
    inner = MockBackend(seed=seed)
    b = MemoryBackend(inner, store)
    return inner, b, Agent(ToolLayer(b), KeywordLLM(b.known_places))


def run(agent, b, inner, text):
    r = agent.say(text)
    inner.run_until_idle()
    return r.reply, agent.pump()


def test_object_last_seen_survives_restart(tmp_path):
    clock = Clock()
    inner, b, a = make(tmp_path, clock)
    run(a, b, inner, "explore")
    live = b.recall("object").data["records"]
    assert live
    cls = live[0]["class"]
    b.store.close()
    clock.t += 7200                                     # two hours later, new process, fresh world (nothing seen yet)
    inner2, b2, a2 = make(tmp_path, clock)
    assert not inner2.tracks
    recs = b2.recall("object", {"class": cls}).data["records"]
    assert recs and recs[0]["from_memory"] and recs[0]["stale"] and 7100 < recs[0]["age_s"] < 7300
    assert recs[0]["place"] == live[0]["place"]


def test_places_persist_and_are_restored(tmp_path):
    clock = Clock()
    inner, b, _ = make(tmp_path, clock)
    inner.x, inner.y = 1.0, 2.0
    assert b.save_place("my spot").ok
    b.store.close()
    inner2, b2, _ = make(tmp_path, clock)
    assert "my spot" in b2.known_places()
    assert inner2.places["my spot"] == (1.0, 2.0)


def test_mission_history_today_and_restart(tmp_path):
    clock = Clock()
    inner, b, a = make(tmp_path, clock)
    run(a, b, inner, "go to room a")
    clock.t += 600
    run(a, b, inner, "go home")
    ms = b.recall("mission").data["records"]
    assert [m["tool"] for m in ms] == ["go_to", "return_home"] and ms[0]["target"] == "room A"
    assert all(m["state"] in ("reached", "done") for m in ms)
    b.store.close()
    clock.t += 86400 * 2                                 # two days later
    _, b2, _ = make(tmp_path, clock)
    assert len(b2.recall("mission").data["records"]) == 2
    assert b2.recall("mission", {"since_s": 86400}).data["records"] == []     # nothing "today"


def test_interrupted_mission_marked_after_crash(tmp_path):
    clock = Clock()
    inner, b, a = make(tmp_path, clock)
    a.say("go to room a")
    inner.advance(2)
    b.poll_events()
    b.store.close()                                      # process died mid-mission
    clock.t += 5
    _, b2, _ = make(tmp_path, clock)
    assert [m["state"] for m in b2.recall("mission").data["records"]] == ["interrupted"]


def test_memory_is_per_map(tmp_path):
    clock = Clock()
    inner, b, a = make(tmp_path, clock)
    run(a, b, inner, "explore")
    n = len(b.recall("object").data["records"])
    assert n
    store = MemoryStore(tmp_path / "gurdy.db", "other_map", wall=clock)
    assert store.objects() == [] and store.missions() == [] and store.places() == []


def test_where_did_you_last_see_reply_has_age(tmp_path):
    clock = Clock()
    inner, b, a = make(tmp_path, clock)
    run(a, b, inner, "explore")
    rec = b.recall("object").data["records"][0]
    b.store.close()
    clock.t += 1500
    inner2, b2, a2 = make(tmp_path, clock)
    r = ToolLayer(b2).call("recall", {"what": "object", "query": {"class": rec["class"]}})
    from brain.agent import templates
    txt = templates.summarize("recall", {}, r if isinstance(r, dict) else r.to_dict())
    assert rec["place"] in txt and "minutes ago" in txt


# ---------------------------------------------------------------- alerts
def ev(t, sev="warning", stamp=0.0, **d):
    return {"stamp": stamp, "type": t, "severity": sev, "source": "t", "text": "x", "data": d}


def test_alert_texts_come_from_event_fields():
    m = AlertManager()
    assert m.handle(ev("battery_low", battery_pct=23.6)) == "Battery is low, 24 percent (simulated)."
    assert m.handle(ev("leg_failure", "critical", leg="front_left")) == "The front left leg has failed."
    assert m.handle(ev("path_blocked", blocked_by="table", task_id=1)) == "My path is blocked by the table."
    assert m.handle(ev("stuck", seconds=5)) == "I am stuck. I have not been making progress for 5 seconds."
    assert m.handle(ev("object_new", "info", **{"class": "fire_extinguisher", "color": "red"})) == "I see a new red fire extinguisher."
    assert m.handle(ev("task_started")) is None


def test_alert_rate_limit_and_idle_rule():
    m = AlertManager()
    assert m.handle(ev("path_blocked", stamp=0, task_id=1))
    assert m.handle(ev("path_blocked", stamp=10, task_id=1)) is None          # same key inside 30 s
    assert m.handle(ev("path_blocked", stamp=10, task_id=2))                  # another task = another key
    assert m.handle(ev("path_blocked", stamp=31, task_id=1))                  # cooldown over
    assert m.handle(ev("object_new", "info", stamp=0, object_id=1, **{"class": "box"}), idle=False) is None
    assert m.handle(ev("object_new", "info", stamp=1, object_id=1, **{"class": "box"}), idle=True)
    assert m.handle(ev("object_new", "info", stamp=100, object_id=1, **{"class": "box"})) is None   # info cooldown 300 s
    assert m.handle(ev("leg_failure", "critical", leg="a"), idle=False)       # critical ignores the idle rule
    assert m.suppressed == 3


def test_agent_speaks_leg_failure_and_battery_once(tmp_path):
    clock = Clock()
    inner, b, a = make(tmp_path, clock)
    inner.fail_leg("front_left")
    msgs = a.pump()
    assert "The front left leg has failed." in msgs
    inner._emit("leg_failure", "critical", "again", leg="front_left")
    assert a.pump() == []                                                    # rate-limited, not repeated
    inner.legs["front_left"] = "ok"
    inner.battery = 25.05
    inner.walk("forward", 1.0)
    inner.run_until_idle()
    msgs = a.pump()
    assert any(m.startswith("Battery is low") for m in msgs)


# ---------------------------------------------------------------- MemoryBackend over RosBackend (FakePort, no ROS)
def _ros(tmp_path, db_name="m.db", objs=()):
    from brain.ros import RosBackend
    from brain.ros.fake_port import FakePort
    from brain.tests.test_ros_backend import PLACES
    f = tmp_path / f"places_{db_name}.yaml"            # a fresh places file per "run", as start_agent_chat.sh does
    f.write_text(PLACES)
    port = FakePort()
    port.set_objects(list(objs))
    inner = RosBackend(port, f)
    return port, inner, MemoryBackend(inner, MemoryStore(tmp_path / db_name, "fake_lab"))


def _pump(port, b, secs=60, dt=0.2):
    for _ in range(int(secs / dt)):
        port.advance(dt)
        b.tick()
        b.poll_events()
        if b.task is None:
            break


def test_ros_backend_implements_protocol_with_memory(tmp_path):
    from brain.tools import missing_methods
    _, _, b = _ros(tmp_path)
    assert missing_methods(b) == []


def test_ros_memory_survives_restart(tmp_path):
    from brain.tests.test_ros_backend import BOX
    port, inner, b = _ros(tmp_path, objs=[BOX])
    assert b.go_to(place="room b").ok
    _pump(port, b)
    port.x = 1.0
    assert b.save_place("desk").ok
    assert b.recall("object", {"class": "box"}).data["records"][0]["from_memory"]
    b.store.close()
    # new process: fresh places file, no tracks (the detector has not seen anything yet), same DB
    port2, inner2, b2 = _ros(tmp_path)
    assert inner2.places["desk"] == {"x": 1.0, "y": 0.5, "kind": "waypoint"}       # restored in RosBackend's own format
    assert "desk" in inner2.known_places()
    names = {p["name"] for p in b2.recall("place").data["records"]}
    assert {"desk", "room a", "room b", "home"} <= names                           # saved place does not hide the rooms
    rec = b2.recall("object", {"class": "box"}).data["records"]
    assert len(rec) == 1 and rec[0]["from_memory"] and rec[0]["color"] == "red"
    missions = b2.recall("mission").data["records"]
    assert [m["tool"] for m in missions] == ["go_to"] and missions[0]["state"] == "reached"
