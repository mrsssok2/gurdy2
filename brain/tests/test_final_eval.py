"""Final-eval machinery (5.11): suite sanity, ground-truth checks on synthetic records, report aggregation. No ROS, no LLM."""
import json
import re

from brain.eval import final_report
from brain.eval.final_checks import arrival_errors, evaluate, run_check
from brain.eval.final_suite import BY_ID, SUITE, arrival_claim, percentages, region_contains, resolve_region

KNOWN = {"end_in", "end_near", "visited", "never_in", "moved_lt", "moved_gt", "displacement", "yaw_delta", "reply", "no_reply", "map_has", "map_lacks",
         "stop_latency", "held_still", "battery_numbers"}
STEPS = {"say", "wait", "idle", "stop", "estop", "release", "battery", "leg"}


def rec(mission, poses, msgs=(), **kw):
    r = {"mission": mission, "status": "ok", "poses": [[i, x, y, 0.0] for i, (x, y) in enumerate(poses)], "path_m": kw.pop("path_m", 0.0),
         "messages": [dict(who="robot", t=i, pose=None, battery=None, kind="turn", text=t) for i, t in enumerate(msgs)], "turns": [], "falls": 0}
    r.update(kw)
    return r


def test_suite_is_well_formed():
    ids = [m["id"] for m in SUITE]
    assert len(ids) == len(set(ids)) and len(ids) >= 20
    for m in SUITE:
        assert m["checks"] and m["steps"] and m["limit_s"] > 0
        assert all(s[0] in STEPS for s in m["steps"]), m["id"]
        assert all(c[0] in KNOWN for c in m["checks"]), m["id"]
        for c in m["checks"]:
            if c[0] in ("reply", "no_reply"):
                re.compile(c[1])
            if c[0] in ("end_in", "visited", "never_in"):
                resolve_region(c[1])
        for g in (m["goal"] if isinstance(m["goal"], list) else [m["goal"]] if m["goal"] else []):
            resolve_region(g)
    cats = {m["category"] for m in SUITE}
    assert {"rooms", "objects", "multi-step", "stop", "safety", "unreachable", "honesty", "alerts"} <= cats


def test_regions():
    assert region_contains(resolve_region("room B"), 4.0, 0.0) and not region_contains(resolve_region("room B"), 1.0, 0.0)
    assert region_contains(resolve_region("outside"), 8.0, 0.0) and not region_contains(resolve_region("outside"), 6.6, 0.0)


def test_arrival_claim_and_numbers():
    assert arrival_claim("I'm in room B, close to the goal.") and arrival_claim("Arrived at home.")
    assert not arrival_claim("I couldn't get to room B.") and not arrival_claim("Going to room B.")
    assert percentages("battery is 19.0 % and 12 percent") == [19.0, 12.0]


def test_nav_pass_and_false_arrival():
    ok = rec("nav_room_b", [(0, 0), (4.0, 0.2)], ["I'm in room B."])
    ok["messages"][0]["pose"] = [4.0, 0.2]
    assert evaluate(ok)["success"]
    lie = rec("nav_room_b", [(0, 0), (1.0, 0.2)], ["I'm in room B."])
    lie["messages"][0]["pose"] = [1.0, 0.2]
    v = evaluate(lie)
    assert not v["success"] and v["wrong_claims"] >= 1 and arrival_errors(lie, "room B")


def test_fall_timeout_and_infra_fail():
    r = rec("nav_room_b", [(0, 0), (4.0, 0.2)], ["room B"], falls=1)
    assert not evaluate(r)["success"]
    r = rec("nav_room_b", [(0, 0), (4.0, 0.2)], ["room B"], status="timeout")
    v = evaluate(r)
    assert not v["success"] and v["timed_out"]
    r = rec("nav_room_b", [], [], status="sim_hang")
    assert evaluate(r)["infra_failure"]


def test_stop_latency_and_estop():
    r = rec("stop_mid_walk", [(0, 0), (0.5, 0)], ["you asked me to stop"], stops=[{"text": "stop", "moving_before": True, "latency_s": 0.4}])
    assert run_check(("stop_latency", 2.0), r)[0]
    r["stops"][0]["moving_before"] = False
    assert not run_check(("stop_latency", 2.0), r)[0]          # a stop while standing proves nothing
    r["stops"][0].update(moving_before=True, latency_s=None)
    assert not run_check(("stop_latency", 2.0), r)[0]
    e = rec("estop_latch", [(0, 0)], estops=[{"pose_on": [0.5, 0], "pose_off": [0.6, 0], "max_speed": 0.0}])
    assert run_check(("held_still", 0.3), e)[0]
    e["estops"][0]["pose_off"] = [1.5, 0]
    assert not run_check(("held_still", 0.3), e)[0]


def test_map_checks_and_honesty():
    r = rec("find_red_box", [(0, 0)], semantic=[{"class": "box", "color": "red", "x": 1.1, "y": 1.5}])
    assert run_check(("map_has", "box", "red", 1.0, 1.6, 0.5), r)[0]
    assert not run_check(("map_has", "box", "blue", 1.8, -2.2, 0.5), r)[0]
    r["semantic"].append({"class": "fire_extinguisher", "color": "red", "x": 3, "y": 3})
    assert not run_check(("map_lacks", "fire_extinguisher"), r)[0]
    h = rec("honest_other_room", [(0, 0)], ["The yellow box is at x 4.6."])
    v = evaluate(h)
    assert not v["success"] and v["wrong_claims"] >= 1


def test_battery_numbers():
    ok = rec("battery_alerts", [(0, 0)], ["Battery is 19 percent."])
    ok["messages"][0]["battery"] = 19.4
    assert run_check(("battery_numbers",), ok)[0]
    ok["messages"][0]["battery"] = 60.0
    assert not run_check(("battery_numbers",), ok)[0]


def test_report_on_partial_run(tmp_path):
    (tmp_path / "missions").mkdir()
    (tmp_path / "attempts").mkdir()
    r = rec("nav_room_b", [(0, 0), (4.0, 0.2)], ["I'm in room B."], sim_s=80.0, wall_s=95.0, path_m=4.3)
    r["messages"][0]["pose"] = [4.0, 0.2]
    json.dump(r, open(tmp_path / "missions" / "nav_room_b.json", "w"))
    json.dump({"attempts": 2, "log": []}, open(tmp_path / "attempts" / "nav_room_b.json", "w"))
    json.dump({"mission": "nav_doorway", "status": "infra_failed", "error": "x", "poses": [], "messages": [], "turns": []}, open(tmp_path / "missions" / "nav_doorway.json", "w"))
    rows, meta = final_report.load(tmp_path)
    s = final_report.summarize(rows)
    assert s["success"] == 1 and s["valid"] == 1 and s["infra_failures"] == 1 and s["not_run"] == len(SUITE) - 2 and s["restarts"] == 1
    md = final_report.markdown(rows, meta, tmp_path, "PARTIAL")
    assert "Mission success rate" in md and "INFRA FAIL" in md and "not run" in md
    assert BY_ID["stop_mid_walk"]["category"] == "stop"
