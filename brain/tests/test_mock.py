import math

import pytest

from brain.mock import MockBackend, Rect
from brain.tools import ToolLayer

RED_BOX = {"class": "box", "color": "red"}


def finish(b: MockBackend, limit=3600):
    b.run_until_idle(limit)
    ev = [e for e in b.poll_events() if e["type"] in ("task_finished", "task_failed")]
    assert ev, "task did not finish"
    return ev[-1]["data"]


def test_initial_state_nothing_seen_and_deterministic():
    b = MockBackend(seed=1)
    assert b.where_is(RED_BOX).code == "NOT_SEEN"
    assert b.where_am_i().data["place"] == "room A"
    b2 = MockBackend(seed=1)
    for x in (b, b2):
        x.explore()
        x.run_until_idle()
    assert b.tracks == b2.tracks


def test_virtual_clock_is_fast():
    import time
    b = MockBackend()
    t = time.time()
    b.go_to(place="room B")
    b.run_until_idle()
    assert b.t > 30 and time.time() - t < 3           # tens of virtual seconds, ~no wall time


def test_go_to_place_through_doorway():
    b = MockBackend()
    r = b.go_to(place="the room B")                    # article tolerated
    assert r.ok and r.data["state"] == "started" and r.data["estimate"]["distance_m"] > 4
    d = finish(b)
    assert d["reached"] and d["stopped_reason"] == "reached" and d["final_distance_m"] < 0.05
    assert b.where_am_i().data["place"] == "room B"
    assert 4.3 < d["travelled_m"] < 5.5                # path via the doorway, not through the wall


def test_unknown_place_lists_known_places():
    r = MockBackend().go_to(place="kitchen")
    assert r.code == "UNKNOWN_PLACE" and "room B" in r.error["known_places"]


def test_busy_and_stop():
    b = MockBackend()
    b.go_to(place="room B")
    assert b.walk("forward", 1).code == "BUSY"
    b.advance(5)
    r = b.stop()
    assert r.data["cancelled_task"]["tool"] == "go_to" and b.task is None
    assert b.poll_events()[-1]["data"]["stopped_reason"] == "cancelled"
    assert b.status().data["mode"] == "idle"


def test_pause_resume():
    b = MockBackend()
    b.go_to(place="room B")
    b.advance(3)
    b.mission_control("pause")
    x = b.x
    b.advance(10)
    assert b.x == x and b.status().data["mode"] == "paused"
    b.mission_control("resume")
    assert finish(b)["reached"]


def test_walk_blocked_by_wall():
    b = MockBackend()
    r = b.walk("backward", 3.0)                        # west wall is 2 m behind the start
    assert r.ok
    d = finish(b)
    assert d["stopped_reason"] == "blocked" and 1.3 < d["moved_m"] < 1.9
    assert any(e["type"] == "path_blocked" for e in b.events)


def test_turn_and_heading():
    b = MockBackend()
    b.turn("left", 90)
    d = finish(b)
    assert abs(d["turned_deg"] - 90) < 1 and abs(b.where_am_i().data["heading_deg"] - 90) < 1


def test_find_explores_then_finds_and_reports_bearing():
    b = MockBackend()
    r = b.find(RED_BOX)
    assert r.ok
    d = finish(b)
    assert d["found"] and d["source"] == "live" and d["object"]["class"] == "box" and d["object"]["color"] == "red"
    m = b.where_is(RED_BOX).data["matches"][0]
    assert abs(m["x"] - 1.0) < 0.2 and abs(m["y"] - 1.6) < 0.2          # within the observation noise
    assert m["bearing_word"] in ("ahead", "ahead-left", "left") and m["last_seen_age_s"] < 5


def test_find_from_memory_is_instant_and_approach_walks():
    b = MockBackend()
    b.explore()
    b.run_until_idle()
    b.poll_events()
    b.find(RED_BOX)
    d = finish(b)
    assert d["found"] and d["source"] == "memory" and d["searched_s"] == 0
    b.find({"class": "barrel", "color": "green"}, approach=True)
    d = finish(b)
    assert d["found"]
    assert 0.4 < math.hypot(b.x - 4.5, b.y - 1.6) < 0.9              # standoff 0.6 m, facing it
    assert b.where_is({"class": "barrel"}).data["matches"][0]["bearing_word"] == "ahead"


def test_find_missing_object_times_out_or_not_found():
    b = MockBackend()
    b.remove_object(1)
    b.find(RED_BOX)
    # the extinguisher is also red but class differs; there is no red box left in the world
    d = finish(b)
    assert not d["found"] and d["object"] is None


def test_find_timeout():
    b = MockBackend()
    b.find({"class": "mannequin"}, max_time_s=10)
    d = finish(b)
    assert d["stopped_reason"] == "timeout" and not d["found"]


def test_explore_maps_everything_and_reports_new_objects():
    b = MockBackend()
    est = b.explore().data["estimate"]
    assert est["duration_s"] > 60
    d = finish(b)
    assert d["mapped_pct"] == 100.0 and len(b.tracks) >= 5 and len(d["new_objects"]) == len(b.tracks)


def test_ambiguous_object_carries_candidates():
    b = MockBackend()
    b.explore()
    b.run_until_idle()
    r = b.go_to(object={"class": "box"})                # red, blue, yellow boxes
    assert r.code == "AMBIGUOUS" and len(r.error["candidates"]) == 3
    assert b.where_is({"class": "box"}).data["matches"][0]["distance_m"] <= b.where_is({"class": "box"}).data["matches"][-1]["distance_m"]


def test_go_to_object_by_colour_and_id():
    b = MockBackend()
    b.explore()
    b.run_until_idle()
    tid = b.where_is({"class": "box", "color": "yellow"}).data["matches"][0]["id"]
    assert b.go_to(object_id=tid, standoff_m=0.8).ok
    d = finish(b)
    assert d["reached"] and d["target"]["object"]["color"] == "yellow"
    assert b.go_to(object_id=99).code == "NOT_SEEN"


def test_no_go_zone_blocks_target_and_path():
    b = MockBackend()
    b.explore()
    b.run_until_idle()
    assert b.go_to(place="home").ok
    b.run_until_idle()
    assert b.where_am_i().data["home_distance_m"] < 0.05
    b.poll_events()
    assert b.no_go_zone("add", place="room B", radius_m=1.0).ok
    assert b.go_to(place="room B").code == "IN_NO_GO_ZONE"
    assert b.path_check(place="room B").data["crosses_no_go"]
    # block the doorway: a zone on the doorway makes room B unreachable
    b.no_go_zone("remove", place="room B")
    b.no_go_zone("add", place="doorway", radius_m=1.2)
    assert b.go_to(place="room B").code == "IN_NO_GO_ZONE"
    assert b.walk("forward", 0.5).ok
    assert b.no_go_zone("list").data["zones"][0]["name"] == "doorway"
    assert b.no_go_zone("remove", place="nowhere").code == "UNKNOWN_PLACE"


def test_path_check_blocked_by_obstacle_across_doorway():
    b = MockBackend()
    assert b.path_check(place="room B").data["reachable"]
    b.add_obstacle(Rect(2.3, -0.1, 2.7, 1.1))           # plug the doorway
    r = b.path_check(place="room B").data
    assert not r["reachable"] and r["blocked_by"] == "unknown obstacle"
    assert b.go_to(place="room B").code == "NO_PATH"


def test_path_check_leaves_mapped_area():
    b = MockBackend()
    assert b.path_check(place="room B").data["leaves_mapped_area"]
    b.explore()
    b.run_until_idle()
    assert not b.path_check(place="room B").data["leaves_mapped_area"]


def test_whats_nearby_and_relation():
    b = MockBackend()
    b.explore()
    b.run_until_idle()
    near = b.whats_nearby(radius_m=3.0).data
    assert all(o["distance_m"] <= 3.0 for o in near["objects"]) and near["place"]
    assert near["obstacles"], "walls or the table must show up as obstacle sectors"
    r = b.relation({"class": "box", "color": "red"}, {"class": "box", "color": "blue"})
    assert r.ok and r.data["distance_m"] == pytest.approx(math.hypot(0.8, 3.8), abs=0.3) and not r.data["same_place"] is None
    assert b.relation({"class": "box", "color": "red"}, {"class": "chair"}).code == "NOT_SEEN"


def test_status_battery_is_labelled_simulated_and_drains():
    b = MockBackend()
    s0 = b.status().data
    assert s0["battery_simulated"] and s0["battery_pct"] == 100 and set(s0["legs"].values()) == {"ok"}
    b.go_to(place="room B")
    b.run_until_idle()
    assert b.status().data["battery_pct"] < 100


def test_leg_failure_alert_and_refusal_of_motion():
    b = MockBackend()
    b.fail_leg("front_left")
    assert b.status().data["legs"]["front_left"] == "failed" and b.status().data["alerts_active"] == ["leg_failure"]
    assert b.walk("forward", 1).code == "REFUSED"
    assert b.return_home().ok is False or b.where_am_i().data["home_distance_m"] == 0   # already home -> trivial
    ev = b.poll_events()
    assert ev[-1]["type"] == "leg_failure" and ev[-1]["severity"] == "critical"


def test_why_and_report_and_recall():
    b = MockBackend()
    b.walk("backward", 3.0)
    b.run_until_idle()
    why = b.why("stop").data["events"]
    assert why and why[-1]["type"] in ("path_blocked", "task_failed")
    rep = b.report().data["summary"]
    assert rep["state"] == "blocked" and rep["distance_m"] > 1 and rep["alerts"]
    assert b.report(mission_id=99).code == "UNAVAILABLE"
    b.explore()
    b.run_until_idle()
    recs = b.recall("object", {"class": "box"}).data["records"]
    assert len(recs) == 3 and all(r["map_id"] and r["stale"] is False for r in recs)
    b.advance(2000)
    assert all(r["stale"] for r in b.recall("object").data["records"])
    assert b.recall("object", {"since_s": 100}).data["records"] == []
    assert b.recall("place").data["records"]


def test_inspect_confirms_moved_and_missing():
    b = MockBackend()
    b.explore()
    b.run_until_idle()
    b.poll_events()
    tid = b.where_is({"class": "box", "color": "red"}).data["matches"][0]["id"]
    b.move_object(1, 1.0, 2.3)
    b.inspect(object_id=tid)
    d = finish(b)
    assert d["confirmed"] and d["moved_m"] > 0.4
    b.remove_object(1)
    b.inspect(object_id=tid)
    d = finish(b)
    assert not d["confirmed"] and any(e["type"] == "object_missing" for e in b.events)


def test_patrol_reports_new_objects_per_place():
    b = MockBackend()
    b.patrol(["room A", "room B"], loops=1)
    d = finish(b)
    assert d["loops_done"] == 1 and [v["place"] for v in d["visits"]] == ["room A", "room B"]
    assert sum(len(v["new"]) for v in d["visits"]) == len(b.tracks) > 0


def test_schedule_fires_later():
    b = MockBackend()
    assert b.schedule(30, {"tool": "walk", "args": {"direction": "forward", "distance_m": 0.5}}).ok
    assert b.schedule(5, {"tool": "walk", "args": {"direction": "up"}}).code == "INVALID_ARGS"
    b.advance(29)
    assert b.task is None
    b.advance(2)
    assert b.task is not None and b.task.tool == "walk"


def test_save_place_then_go_back():
    b = MockBackend()
    b.go_to(place="room B")
    b.run_until_idle()
    b.save_place("charging corner")
    b.return_home()
    b.run_until_idle()
    assert b.go_to(place="Charging Corner").ok
    assert "charging corner" in b.known_places()


def test_describe_view_is_marked_unverified():
    b = MockBackend()
    b.turn("left", 60)
    b.run_until_idle()
    d = b.describe_view().data
    assert d["unverified"] and isinstance(d["detections"], list)


def test_focus_and_state_block():
    b = MockBackend()
    L = ToolLayer(b)
    assert "KNOWN PLACES home, room A, room B, doorway" in L.state_block()
    L.call("find", {"object": RED_BOX})
    b.run_until_idle()
    for e in b.poll_events():
        if e["type"] == "task_finished":
            L.note_task_finished(e)
    L.call("where_is", {"object": RED_BOX})
    blk = L.state_block()
    assert "object=#" in blk and "red box" in blk and "last_task=find(" in blk and "(simulated)" in blk
    L.call("go_to", {"place": "room B"})
    assert L.focus["place"] == "room B"
    assert [x["tool"] for x in L.log][:2] == ["find", "where_is"]
