"""Looking around: find / explore must turn a full circle in each room (the camera sees only a cone), and explore must not report "done" for an
already mapped area without going there. Different objects / areas than the Gazebo eval missions. Run in the brain image: pytest brain/tests/test_search_sweep.py"""
import math

import pytest

from brain.ros import RosBackend
from brain.ros.fake_port import FakePort

PLACES = """map_id: fake_lab
places:
  home: {x: 0.0, y: 0.0, kind: waypoint}
  room a: {rect: [-2.0, -3.0, 2.5, 3.0], kind: room}
  room b: {rect: [2.5, -3.0, 6.5, 3.0], kind: room}
  stairs: {x: 1.0, y: -5.0, kind: hazard}
"""
CONE = {"id": 7, "class": "traffic_cone", "color": "orange", "x": -1.5, "y": -2.2, "z": 0.2, "hits": 9, "score": 0.8, "sigma_m": 0.04}
BARREL = {"id": 8, "class": "barrel", "color": "blue", "x": 6.0, "y": 2.4, "z": 0.3, "hits": 9, "score": 0.8, "sigma_m": 0.04}


def make(tmp_path, pose=(0.0, 0.0, 0.0)):
    f = tmp_path / "places.yaml"
    f.write_text(PLACES)
    port = FakePort(pose)
    return RosBackend(port, f), port


def run(b, port, secs, dt=0.2):
    for _ in range(int(secs / dt)):
        port.advance(dt)
        b.tick()
        if b.task is None:
            break


def final(b):
    ev = [e for e in b.events if e["type"] in ("task_finished", "task_failed")]
    return ev[-1]["data"] if ev else None


def total_turn(port_cmds_yaw: list[float]) -> float:
    return sum(abs(((b - a) + math.pi) % (2 * math.pi) - math.pi) for a, b in zip(port_cmds_yaw, port_cmds_yaw[1:]))


def test_find_turns_to_look_behind_and_finds_a_cone_in_the_corner(tmp_path):
    b, p = make(tmp_path)
    p.hide_objects([CONE])                                    # far corner of room A, behind the robot's travel direction
    assert b.find({"class": "traffic_cone", "color": "orange"}).ok
    run(b, p, 400)
    d = final(b)
    assert d["found"] and d["source"] == "live" and d["object"]["id"] == 7


def test_without_the_scan_the_same_search_misses_it(tmp_path):
    """The behaviour before the fix, kept as a regression guard that the scenario really needs the turn."""
    b, p = make(tmp_path)
    b.SCAN_QUARTERS = 0
    p.hide_objects([CONE])
    b.find({"class": "traffic_cone"}, max_time_s=200)
    run(b, p, 260)
    d = final(b)
    assert d["stopped_reason"] == "timeout" and not d["found"]


def test_find_searches_the_second_room_too(tmp_path):
    b, p = make(tmp_path)
    p.hide_objects([BARREL])                                  # corner of room B
    assert b.find({"class": "barrel"}).ok
    run(b, p, 500)
    d = final(b)
    assert d["found"] and d["object"]["id"] == 8


def test_the_scan_stops_as_soon_as_the_object_is_seen_and_covers_a_full_circle_otherwise(tmp_path):
    b, p = make(tmp_path)
    yaws: list[float] = []
    b.find({"class": "chair"}, max_time_s=500)                # not in this world: the whole sweep runs
    for _ in range(int(500 / 0.2)):
        p.advance(0.2)
        b.tick()
        yaws.append(p.yaw)
        if b.task is None:
            break
    assert final(b)["stopped_reason"] == "timeout" and not final(b)["found"]
    assert total_turn(yaws) > 2 * math.radians(330)           # a (nearly) full circle in each of the two rooms (+ driving turns)


def test_explore_of_an_already_mapped_room_still_goes_there_and_looks_around(tmp_path):
    b, p = make(tmp_path)
    p.explore_finish_after = 0.0                              # the explorer has nothing left to explore and ends at once
    p.hide_objects([BARREL])
    assert b.explore(area="room B").ok
    run(b, p, 400)
    d = final(b)
    assert d["stopped_reason"] == "done"
    assert b.where_am_i().data["place"] == "room b" or b.where_am_i().data["place"] == "room B"
    assert [o["id"] for o in d["new_objects"]] == [8]         # and what it saw there is reported


def test_explore_that_really_explores_does_not_add_a_sweep(tmp_path):
    b, p = make(tmp_path)
    p.explore_finish_after = 40.0
    b.explore(area="room A")
    run(b, p, 200)
    assert final(b)["stopped_reason"] == "done"
    assert p.nav_log == []                                    # no extra navigation goals after a real exploration


def test_sweep_never_enters_a_hazard_or_a_no_go_zone(tmp_path):
    b, p = make(tmp_path)
    assert b.no_go_zone("add", place="room B").ok
    steps = b._sweep_steps()
    assert [s["name"].lower() for s in steps if s["kind"] == "nav"] == ["room a"]
    assert b._sweep_steps("stairs") == []


def test_wait_step_stands_still_for_its_time(tmp_path):
    b, p = make(tmp_path)
    p.hide_objects([])
    b.find({"class": "chair"}, max_time_s=30)
    run(b, p, 8)
    assert b.task is not None                                 # still scanning after 8 s: waits and turns take time
