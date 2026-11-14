"""Tripod gait math tests (no simulator)."""
import itertools

import numpy as np
import pytest

from gurdy_core.gait import TripodGait, _rot, se2_step
from gurdy_core.robot import JOINT_TYPES, LEG_NAMES, load_robot


@pytest.fixture(scope="module")
def robot():
    return load_robot()


@pytest.fixture
def gait(robot):
    return TripodGait(robot)


def _world(pose, foot_xy):
    x, y, th = pose
    return np.array([x, y]) + _rot(th) @ foot_xy


def _integrate(pose, v, wz, dt):
    d, th = se2_step(np.asarray(v, float), wz, dt)
    x, y, a = pose
    return (*(np.array([x, y]) + _rot(a) @ d), a + th)


def test_zero_command_holds_stance(gait):
    for _ in range(200):
        q = gait.step((0, 0, 0))
    assert q == pytest.approx(np.tile(gait.q_stance, 6), abs=1e-12)
    assert not gait.active and gait.ik_failures == 0


def test_se2_step_matches_fine_integration():
    v, wz, dt = np.array([0.1, -0.05]), 0.6, 0.02
    d, th = se2_step(v, wz, dt)
    def f(p):                                   # RK4 reference integration of the twist
        return np.array([*(_rot(p[2]) @ v), wz])
    p, n = np.zeros(3), 200
    h = dt / n
    for _ in range(n):
        k1 = f(p); k2 = f(p + h / 2 * k1); k3 = f(p + h / 2 * k2); k4 = f(p + h * k3)
        p = p + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    assert d == pytest.approx(p[:2], abs=1e-12) and th == pytest.approx(p[2])


@pytest.mark.parametrize("cmd", [(0.1, 0, 0), (0, 0.08, 0), (0, 0, 0.5), (0.1, -0.05, 0.3)])
def test_stance_feet_stay_fixed_in_world(gait, cmd):
    """If the body moves exactly with the (shaped) command, stance feet do not slip."""
    pose = (0.0, 0.0, 0.0)
    prev = {leg: None for leg in LEG_NAMES}
    worst = 0.0
    for _ in range(int(3 * gait.cycle / gait.dt)):
        swinging_before = set(gait.swinging_legs())
        gait.step(cmd)
        c = gait.body_twist                       # twist the gait assumed this tick (incl. start-up)
        pose = _integrate(pose, c[:2], c[2], gait.dt)
        swinging_after = set(gait.swinging_legs())
        for leg in LEG_NAMES:
            w = _world(pose, gait.feet[leg][:2])
            if leg not in swinging_before and leg not in swinging_after and prev[leg] is not None:
                worst = max(worst, np.linalg.norm(w - prev[leg]))
            prev[leg] = w if leg not in swinging_after else None
    assert worst < 1e-9


def test_exactly_one_tripod_swings(gait):
    for _ in range(int(2 * gait.cycle / gait.dt)):
        gait.step((0.1, 0, 0))
        sw = gait.swinging_legs()
        assert sorted(sw) in [sorted(g) for g in gait.groups]
        for leg in LEG_NAMES:
            if leg not in sw:
                assert gait.feet[leg][2] == pytest.approx(gait.ground_z)
        assert max(gait.feet[l][2] for l in sw) <= gait.ground_z + gait.height + 1e-12


def test_command_limits_never_violate_joint_limits(robot):
    """Every combination of command extremes, 3 cycles each: no IK failure, joints in limits."""
    lim = robot.command_limits
    lo = np.array([robot.joint_limits[t][0] for t in JOINT_TYPES] * 6)
    hi = np.array([robot.joint_limits[t][1] for t in JOINT_TYPES] * 6)
    for sx, sy, sw in itertools.product((-1, 0, 1), repeat=3):
        g = TripodGait(robot)
        cmd = (sx * lim["vx"], sy * lim["vy"], sw * lim["wz"])
        for _ in range(int(3 * g.cycle / g.dt)):
            q = g.step(cmd)
            assert np.all(q >= lo - 1e-9) and np.all(q <= hi + 1e-9), cmd
            assert np.all(q[2::3] >= 0.0), "knee bent backwards"
        assert g.ik_failures == 0, cmd


def test_feet_symmetric_about_home_from_first_step(robot):
    """Start-up ramp: stance strokes stay within +-max_stride/2 of home (no full-stroke transient)."""
    g = TripodGait(robot)
    worst = 0.0
    for _ in range(int(2 * g.cycle / g.dt)):
        g.step((0.15, 0, 0))
        for leg in LEG_NAMES:
            if leg not in g.swinging_legs():
                worst = max(worst, np.linalg.norm(g.feet[leg][:2] - g.home[leg]))
    assert worst <= g.max_stride / 2 + 1e-9, worst


def test_random_command_changes_no_ik_failures(robot):
    rng = np.random.default_rng(5)
    lim = np.array([robot.command_limits[k] for k in ("vx", "vy", "wz")])
    g = TripodGait(robot)
    cmd = np.zeros(3)
    for k in range(3000):                         # 60 s of operator-like input
        if k % 25 == 0:
            cmd = rng.uniform(-1, 1, 3) * lim * (rng.random(3) < 0.7)
        g.step(cmd)
    assert g.ik_failures == 0


def test_stride_cap_scales_command(gait):
    c = gait.shape_command((10.0, 10.0, 10.0))
    strokes = [np.linalg.norm(gait._foot_velocity(c[:2], c[2], gait.home[l])) * gait.t_stance
               for l in LEG_NAMES]
    assert max(strokes) <= gait.max_stride + 1e-12
    small = gait.shape_command((0.02, 0, 0))
    assert small == pytest.approx([0.02, 0, 0])


def _max_joint_speed(robot, cmd_fn, ticks):
    g = TripodGait(robot)
    q_prev = g.step(cmd_fn(0))
    worst = 0.0
    for k in range(1, ticks):
        q = g.step(cmd_fn(k))
        worst = max(worst, np.max(np.abs(q - q_prev)) / g.dt)
        q_prev = q
    return worst, g


def test_joint_speeds_within_servo_limit(robot):
    """Constant extreme commands, and reversals every 13 ticks (mid-step): implied joint speed
    never exceeds the servo velocity limit (6 rad/s)."""
    vmax = robot.joint_common["velocity"]
    lim = robot.command_limits
    for cmd in [(lim["vx"], 0, 0), (0, lim["vy"], 0), (0, 0, lim["wz"]), (lim["vx"], lim["vy"], lim["wz"])]:
        w, g = _max_joint_speed(robot, lambda k: cmd, 200)
        assert w <= vmax, (cmd, w)
    flip = lambda k: (0.15, 0, 0) if (k // 13) % 2 == 0 else (-0.15, 0.1, -0.6)
    w, g = _max_joint_speed(robot, flip, 300)
    assert w <= vmax and g.ik_failures == 0, w


def test_command_is_rate_limited(gait):
    gait.step((0.15, 0, 0))
    assert gait.cmd[0] == pytest.approx(gait.max_acc[0] * gait.dt)


def test_stops_after_command_returns_to_zero(gait):
    for _ in range(37):
        gait.step((0.1, 0.05, 0.4))
    n = 0
    while gait.active and n < 200:
        gait.step((0, 0, 0))
        n += 1
    assert not gait.active
    assert n * gait.dt <= 2.0 * gait.cycle + 1e-9      # decelerate (<=0.3 s) + finish stepping
    assert gait.joint_targets() == pytest.approx(np.tile(gait.q_stance, 6), abs=1e-9)
