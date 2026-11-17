"""Battery model, leg-health estimator and alert monitor (pure Python, synthetic signals)."""
import math

import pytest

from gurdy_core.ops_monitors import AlertMonitor, BatteryModel, LegHealth, Snapshot


def test_battery_drains_faster_when_moving():
    b = BatteryModel(100.0)
    b.step(0.0, 0.0)
    idle = b.step(100.0, 0.0)
    m = BatteryModel(100.0)
    m.step(0.0, 0.1)
    moving = m.step(100.0, 0.1)
    assert idle == pytest.approx(100 - 0.2) and moving == pytest.approx(100 - 3.0)
    assert BatteryModel(1.0).step(0.0, 0.1) == 1.0 and m.step(50.0, 0.1) == moving      # time going backwards: no change


def walk_signal(t, leg=None, mode=None):
    """Targets: sine on every joint; measured = target lagging a little; the broken leg follows `mode`."""
    q_ref = [0.5 * math.sin(2 * math.pi * 1.0 * t + k) for k in range(18)]
    q = [0.5 * math.sin(2 * math.pi * 1.0 * (t - 0.02) + k) for k in range(18)]
    if leg is not None:
        i = ["lf", "lm", "lr", "rf", "rm", "rr"].index(leg)
        for j in range(3 * i, 3 * i + 3):
            q[j] = 0.0 if mode == "limp" else 0.1          # limp: collapsed at 0; locked: frozen
    return q_ref, q


def run_health(h, secs, leg=None, mode=None, t0=0.0, hz=50):
    for k in range(int(secs * hz)):
        t = t0 + k / hz
        h.step(t, *walk_signal(t, leg, mode))


def test_healthy_walk_no_failure():
    h = LegHealth()
    run_health(h, 10)
    assert set(h.states.values()) == {"ok"} and not h.failed


@pytest.mark.parametrize("mode", ["limp", "locked"])
def test_failed_leg_detected_and_latched(mode):
    h = LegHealth()
    run_health(h, 3)
    run_health(h, 8, leg="rm", mode=mode, t0=3.0)
    assert h.states["middle_right"] == "failed" and sum(v == "failed" for v in h.states.values()) == 1
    assert 4.0 < h.failed["rm"] < 9.0               # after the EMA + hold time, not instantly
    run_health(h, 3, t0=11.0)
    assert h.states["middle_right"] == "failed"      # latched
    h.reset()
    assert set(h.states.values()) == {"ok"}


def test_standing_still_is_not_judged():
    h = LegHealth()
    for k in range(500):
        t = k / 50
        q = [0.1] * 18
        h.step(t, q, [0.0] * 6 + [0.1] * 12)          # constant targets, one leg off: nothing is moving
    assert not h.failed


def snap(t, **kw):
    return Snapshot(t=t, **kw)


def types(alerts):
    return [a["type"] for a in alerts]


def test_fall_latches_until_upright():
    m = AlertMonitor()
    assert m.step(snap(0.0, tilt_deg=70)) == []
    assert types(m.step(snap(0.6, tilt_deg=70))) == ["fall"]
    assert m.step(snap(5.0, tilt_deg=80)) == []
    m.step(snap(6.0, tilt_deg=5))
    m.step(snap(7.0, tilt_deg=70))
    assert types(m.step(snap(8.0, tilt_deg=70))) == ["fall"]


def test_stuck_after_timeout_only_when_commanded():
    m = AlertMonitor()
    for t in range(0, 8):
        assert m.step(snap(t, cmd_speed=0.1, speed=0.0)) == []
    assert types(m.step(snap(8.5, cmd_speed=0.1, speed=0.0))) == ["stuck"]
    assert m.step(snap(30, cmd_speed=0.1, speed=0.0)) == []              # once per episode
    m2 = AlertMonitor()
    assert all(m2.step(snap(t, cmd_speed=0.0, speed=0.0)) == [] for t in range(60))
    m3 = AlertMonitor()
    assert all(m3.step(snap(t, cmd_speed=0.1, speed=0.09)) == [] for t in range(60))


def test_path_blocked_per_aborted_goal():
    m = AlertMonitor()
    assert types(m.step(snap(1.0, nav_aborted_ids=("g1",)))) == ["path_blocked"]
    assert m.step(snap(2.0, nav_aborted_ids=("g1",))) == []
    assert types(m.step(snap(20.0, nav_aborted_ids=("g1", "g2")))) == ["path_blocked"]


def test_battery_low_then_critical_and_rearm():
    m = AlertMonitor()
    assert m.step(snap(0, battery_pct=50)) == []
    a = m.step(snap(1, battery_pct=19.0))
    assert types(a) == ["battery_low"] and "simulated" in a[0]["text"]
    assert m.step(snap(2, battery_pct=18.0)) == []
    assert types(m.step(snap(20, battery_pct=9.0))) == ["battery_critical"]
    assert m.step(snap(40, battery_pct=None)) == []                       # unknown: nothing
    m.step(snap(50, battery_pct=80))
    assert types(m.step(snap(100, battery_pct=15))) == ["battery_low"]    # re-armed after a recharge


def test_leg_failure_once_per_leg():
    m = AlertMonitor()
    legs = {"front_left": "ok", "rear_right": "failed"}
    a = m.step(snap(1, legs=legs))
    assert types(a) == ["leg_failure"] and a[0]["data"]["leg"] == "rear_right" and a[0]["severity"] == "critical"
    assert m.step(snap(2, legs=legs)) == []
    assert types(m.step(snap(3, legs={**legs, "front_left": "failed"}))) == ["leg_failure"]


def test_new_object_not_for_the_first_snapshot():
    m = AlertMonitor()
    o = lambda i, c="box", col="red": {"id": i, "class": c, "color": col}
    assert m.step(snap(0, objects=[o(1)])) == []
    a = m.step(snap(1, objects=[o(1), o(2, "traffic_cone", "orange")]))
    assert types(a) == ["object_new"] and a[0]["text"] == "New object: orange traffic cone." and a[0]["severity"] == "info"
    assert m.step(snap(2, objects=[o(1), o(2)])) == []
    assert m.step(snap(3, objects=None)) == []
