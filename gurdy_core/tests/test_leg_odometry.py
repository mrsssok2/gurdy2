"""Leg odometry math (no simulator): Jacobian, exact recovery from a synthetic rigid-body motion."""
import numpy as np
import pytest

from gurdy_core.gait import TripodGait
from gurdy_core.kinematics import leg_jacobian
from gurdy_core.leg_odometry import LegOdomConfig, LegOdometry, leg_foot_jacobian, quat_to_rotation
from gurdy_core.robot import LEG_NAMES, load_robot


@pytest.fixture(scope="module")
def robot():
    return load_robot()


def test_analytic_jacobian_matches_numeric(robot):
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform([-0.5, -0.5, 0.1], [0.5, 0.9, 1.8])
        assert leg_foot_jacobian(robot, q) == pytest.approx(leg_jacobian(robot, q), abs=1e-6)


def _synthetic(robot, v, wz):
    """All six feet planted: joint velocities that exactly cancel the body motion."""
    odo = LegOdometry(robot, LegOdomConfig(rolling_correction=False))
    q = np.tile(TripodGait(robot).q_stance, 6)
    p, _ = odo.feet(q, np.zeros(18))
    qd = np.zeros(18)
    for i, leg in enumerate(LEG_NAMES):
        pd = -(np.array([*v, 0.0]) + wz * np.array([-p[i, 1], p[i, 0], 0.0]))
        m = robot.mounts[leg]
        c, s = np.cos(m.yaw), np.sin(m.yaw)
        pl = np.array([c * pd[0] + s * pd[1], -s * pd[0] + c * pd[1], pd[2]])      # body -> leg frame
        qd[3 * i:3 * i + 3] = np.linalg.solve(leg_foot_jacobian(robot, q[3 * i:3 * i + 3]), pl)
    return odo, q, qd


@pytest.mark.parametrize("v,wz", [((0.1, 0.0), 0.0), ((0.0, -0.08), 0.0), ((0.05, 0.03), 0.3), ((0.0, 0.0), -0.4)])
def test_recovers_planted_feet_motion(robot, v, wz):
    odo, q, qd = _synthetic(robot, v, wz)
    est = odo.estimate(q, qd)
    assert est.valid and len(est.stance) == 6
    assert est.v == pytest.approx(v, abs=1e-9) and est.wz == pytest.approx(wz, abs=1e-9)
    g = odo.estimate(q, qd, gyro_z=wz)
    assert g.v == pytest.approx(v, abs=1e-9) and g.gyro_assisted and g.wz == pytest.approx(wz, abs=1e-9)


def test_swing_foot_is_rejected(robot):
    odo, q, qd = _synthetic(robot, (0.1, 0.0), 0.0)
    q = q.copy()
    qd[0:3] = [0.0, 3.0, -2.0]                  # lf swings with a very different velocity
    q[1] += 0.4                                 # ... and is lifted (higher than the others)
    est = odo.estimate(q, qd)
    assert "lf" not in est.stance and est.v[0] == pytest.approx(0.1, abs=0.01)


def test_sliding_foot_is_rejected_by_gate(robot):
    odo, q, qd = _synthetic(robot, (0.1, 0.0), 0.0)
    qd[3:6] += leg_foot_jacobian(robot, q[3:6]) @ np.zeros(3)
    pert = np.linalg.solve(leg_foot_jacobian(robot, q[3:6]), np.array([0.2, 0.0, 0.0]))
    qd[3:6] += pert                              # lm foot slides 0.2 m/s while "in stance"
    est = odo.estimate(q, qd)
    assert "lm" not in est.stance and est.v[0] == pytest.approx(0.1, abs=1e-3)


def test_too_few_stance_feet_is_invalid(robot):
    odo = LegOdometry(robot)
    q = np.tile(TripodGait(robot).q_stance, 6).reshape(6, 3)
    q[:5, 1] += 0.5                              # five feet up: only one at ground height
    assert not odo.estimate(q.ravel(), np.zeros(18)).valid


def test_tilt_compensated_stance_detection(robot):
    odo = LegOdometry(robot)
    pitch = 0.15
    c, s = np.cos(pitch), np.sin(pitch)
    Rwb = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    pw = np.array([[*odo.mount_xy[leg], -0.1] for leg in LEG_NAMES])    # all feet on the floor (world z equal)
    p = pw @ Rwb                                   # the same feet seen from the pitched body
    assert np.ptp(p[:, 2]) > 0.03                  # in the body frame heights differ by more than the tolerance
    assert odo.stance_candidates(p, Rwb).all()
    assert not odo.stance_candidates(p).all()


def test_quaternion_identity():
    assert quat_to_rotation(1, 0, 0, 0) == pytest.approx(np.eye(3))
    R = quat_to_rotation(np.cos(0.25), 0, 0, np.sin(0.25))     # yaw 0.5
    assert R[1, 0] == pytest.approx(np.sin(0.5))


def test_pose_integration(robot):
    odo = LegOdometry(robot)
    odo, q, qd = _synthetic(robot, (0.1, 0.0), 0.3)
    est = odo.estimate(q, qd)
    for _ in range(100):
        odo.update(est, 0.01)
    assert odo.yaw == pytest.approx(0.3, abs=1e-9)
