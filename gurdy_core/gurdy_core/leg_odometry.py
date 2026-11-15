"""Leg (kinematic) odometry: body velocity from joint states, no ROS imports.

Idea: a foot that stands on the ground does not move in the world, so the body moves opposite to
that foot's motion *relative to the body*. With foot position p_i (body frame, from FK) and its
velocity pdot_i = J(q) qdot (from joint velocities), a stance foot satisfies

    pdot_i = -(v + w x p_i)        ->   v = -(pdot_i + w x p_i)

v = (vx, vy) is the body velocity in the body frame, w the body angular velocity. Averaging over
all stance feet gives (vx, vy, wz); with >= 3 non-collinear stance feet wz is observable from the
legs alone (planar least squares), otherwise (or when a gyro is given) w comes from outside.

Stance detection needs no gait phase, so it works for the tripod gait *and* the RL policy: feet
whose height (gravity-aligned if an orientation is given) is within `stance_height_tol` of the
lowest foot are candidates; candidates whose implied velocity disagrees with the rest (they are
sliding or just lifting off) are rejected one at a time (`outlier_gate`).

Limits (accepted, measured in docs/results/phase3_leg_odometry.md): feet slip a little, so the
estimate drifts (scale error of a few %); heading from the legs alone drifts more than a gyro.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .kinematics import leg_fk
from .robot import LEG_NAMES, Robot


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def quat_to_rotation(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Body -> world rotation matrix from a unit quaternion (w, x, y, z)."""
    n = np.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def leg_foot_jacobian(robot: Robot, q) -> np.ndarray:
    """Analytic d(foot)/dq (3x3) in the leg frame; matches kinematics.leg_jacobian."""
    l1, l2, l3 = robot.lengths
    yaw, pitch, knee = q
    a = pitch - knee
    r = l1 + l2 * np.cos(pitch) + l3 * np.cos(a)
    dr_dp = -l2 * np.sin(pitch) - l3 * np.sin(a)
    dr_dk = l3 * np.sin(a)
    dz_dp = l2 * np.cos(pitch) + l3 * np.cos(a)
    dz_dk = -l3 * np.cos(a)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([[-r * sy, dr_dp * cy, dr_dk * cy],
                     [r * cy, dr_dp * sy, dr_dk * sy],
                     [0.0, dz_dp, dz_dk]])


@dataclass
class LegOdomConfig:
    stance_height_tol: float = 0.015   # m above the lowest foot that still counts as stance
    outlier_gate: float = 0.05         # m/s: a stance foot whose implied velocity is further away is dropped
    min_stance_feet: int = 2           # fewer -> no estimate (covariance is made huge)
    # Covariance model: sigma = sqrt((s0^2 + (k*speed)^2) / n_stance). Slip grows with speed.
    sigma_v0: float = 0.012            # m/s
    slip_v: float = 0.20               # fraction of speed
    sigma_w0: float = 0.03             # rad/s
    slip_w: float = 0.20               # fraction of |wz|
    big_var: float = 1e3
    rolling_correction: bool = True    # foot is a sphere: track the contact point, not the centre


@dataclass
class LegOdomEstimate:
    valid: bool
    v: np.ndarray = field(default_factory=lambda: np.zeros(2))    # body-frame (vx, vy)
    wz: float = 0.0
    cov: np.ndarray = field(default_factory=lambda: np.eye(3) * 1e3)   # (vx, vy, wz)
    stance: tuple[str, ...] = ()
    gyro_assisted: bool = False


class LegOdometry:
    """Stateless velocity estimate (`estimate`) + dead-reckoned pose (`update`)."""

    def __init__(self, robot: Robot, cfg: LegOdomConfig | None = None):
        self.robot = robot
        self.cfg = cfg or LegOdomConfig()
        self.mount_xy = {leg: np.array(robot.mounts[leg].xy) for leg in LEG_NAMES}
        self.mount_R = {leg: _rot_z(robot.mounts[leg].yaw) for leg in LEG_NAMES}
        self.reset()

    def reset(self) -> None:
        self.x = self.y = self.yaw = 0.0

    # ------------------------------------------------------------ kinematics
    def feet(self, q: np.ndarray, qd: np.ndarray, R_wb: np.ndarray | None = None):
        """Foot positions and velocities in the body frame, each (6, 3). q, qd: 18 values in
        robot.joint_names order (leg-major: hip_yaw, hip_pitch, knee per leg).

        With `rolling_correction` the returned point is the foot sphere's contact point with the
        ground (centre minus radius along world-up), and its velocity includes the tibia rotation:
        a rolling sphere has a moving centre but a stationary contact point."""
        q = np.asarray(q, dtype=float).reshape(6, 3)
        qd = np.asarray(qd, dtype=float).reshape(6, 3)
        up = np.array([0.0, 0.0, 1.0]) if R_wb is None else R_wb[2, :]      # world z in body coordinates
        p = np.zeros((6, 3))
        pd = np.zeros((6, 3))
        for i, leg in enumerate(LEG_NAMES):
            pl = leg_fk(self.robot, q[i])
            vl = leg_foot_jacobian(self.robot, q[i]) @ qd[i]
            R = self.mount_R[leg]
            p[i, :2] = self.mount_xy[leg] + R @ pl[:2]
            p[i, 2] = pl[2]
            pd[i, :2] = R @ vl[:2]
            pd[i, 2] = vl[2]
            if self.cfg.rolling_correction:
                yaw = q[i, 0]
                axis = np.array([-np.sin(yaw), np.cos(yaw), 0.0])         # hip-pitch/knee axis in the leg frame
                omega_leg = np.array([0.0, 0.0, qd[i, 0]]) + (qd[i, 2] - qd[i, 1]) * axis
                omega = np.array([*(R @ omega_leg[:2]), omega_leg[2]])
                rho = -self.robot.foot_radius * up
                p[i] += rho
                pd[i] += np.cross(omega, rho)
        return p, pd

    def stance_candidates(self, p: np.ndarray, R_wb: np.ndarray | None = None) -> np.ndarray:
        """Boolean mask (6,) of feet near the lowest one (gravity-aligned height if R_wb given)."""
        z = p[:, 2] if R_wb is None else (p @ R_wb.T)[:, 2]
        return z - z.min() < self.cfg.stance_height_tol

    # ------------------------------------------------------------ estimate
    def estimate(self, q, qd, R_wb: np.ndarray | None = None, gyro_z: float | None = None) -> LegOdomEstimate:
        """Body-frame velocity from joint positions/velocities.

        R_wb: body->world rotation (from the IMU quaternion), used to find the lowest feet.
        gyro_z: if given, vx, vy are computed with this angular rate (their error no longer depends
        on the noisy leg-derived rotation). `wz` in the result is always the leg-derived rate, so it
        stays an independent measurement for the EKF; the gyro goes to the EKF through the IMU.
        """
        cfg = self.cfg
        p, pd = self.feet(q, qd, R_wb)
        idx = list(np.flatnonzero(self.stance_candidates(p, R_wb)))
        if len(idx) < cfg.min_stance_feet:
            return LegOdomEstimate(valid=False)
        while True:
            v, w = self._solve(p[idx], pd[idx], gyro_z)
            res = np.array([np.linalg.norm(pd[i, :2] + v + w * np.array([-p[i, 1], p[i, 0]])) for i in idx])
            if res.max() <= cfg.outlier_gate or len(idx) <= cfg.min_stance_feet:
                break
            idx.pop(int(res.argmax()))
        v_leg, wz = self._solve(p[idx], pd[idx], None)       # planar least squares: leg-only rotation
        if gyro_z is None:
            v = v_leg
        n = len(idx)
        # two feet cannot separate rotation from translation reliably: wz nearly unobserved
        wz_scale = 10.0 if len(idx) < 3 else 1.0
        sv = np.sqrt((cfg.sigma_v0**2 + (cfg.slip_v * np.linalg.norm(v)) ** 2) / n)
        sw = np.sqrt((cfg.sigma_w0**2 + (cfg.slip_w * abs(wz)) ** 2) / n) * wz_scale
        cov = np.diag([sv**2, sv**2, sw**2])
        return LegOdomEstimate(True, v, wz, cov, tuple(LEG_NAMES[i] for i in idx), gyro_z is not None)

    @staticmethod
    def _solve(p: np.ndarray, pd: np.ndarray, gyro_z: float | None):
        if gyro_z is not None:
            perp = np.stack([-p[:, 1], p[:, 0]], axis=1)
            v = -(pd[:, :2] + gyro_z * perp).mean(axis=0)
            return v, float(gyro_z)
        n = len(p)
        A = np.zeros((2 * n, 3))
        b = np.zeros(2 * n)
        for k in range(n):
            A[2 * k] = [1.0, 0.0, -p[k, 1]]
            A[2 * k + 1] = [0.0, 1.0, p[k, 0]]
            b[2 * k:2 * k + 2] = -pd[k, :2]
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        return sol[:2], float(sol[2])

    # ------------------------------------------------------------ pose
    def update(self, est: LegOdomEstimate, dt: float, yaw_rate: float | None = None) -> None:
        """Dead-reckon the pose with the estimated body-frame velocity (held if invalid).
        yaw_rate overrides the leg-derived wz (e.g. the gyro)."""
        if not est.valid or dt <= 0:
            return
        wz = est.wz if yaw_rate is None else yaw_rate
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        self.x += (c * est.v[0] - s * est.v[1]) * dt
        self.y += (s * est.v[0] + c * est.v[1]) * dt
        self.yaw += wz * dt
