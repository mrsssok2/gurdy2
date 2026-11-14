"""Tripod gait: body velocity command (vx, vy, wz) -> 18 joint targets. Pure Python, no ROS.

Two tripods ({lf, rm, lr} and {rf, lm, rr}) alternate. All foot positions live in the BODY frame
(x forward, y left, z up, origin on the hip-axis plane).

* Stance feet move by the exact rigid-body (SE(2)) motion the command implies, so if the body
  really moves with the commanded twist they stay fixed on the ground (tested to 1e-9).
* Swing feet follow a cycloid profile towards a touchdown point half a stroke ahead of the nominal
  foot position. Each tick the foot covers its cycloid share of the REMAINING distance to the
  current target, so command changes mid-swing never make a foot jump.
* Start-up: in the first half cycle after starting, the stance tripod moves at half speed (it
  starts at home, not at +stroke/2), so every foot is symmetric about home from the first step;
  this doubles as a gentle acceleration ramp.
* Command is clamped to robot.yaml command_limits, scaled down uniformly if any foot's stance
  stroke would exceed gait.yaml max_stride, then rate-limited (max_lin_accel / max_ang_accel), so a
  sudden command reversal cannot drag a late-swing touchdown target (and the foot) across a tick.
* Zero command: keeps stepping until every foot is back at its nominal position, then stops
  (phase frozen, output = nominal stance).

Used by the ROS gait node (gurdy_gait) and by the MuJoCo runner — same code in both sims.
"""
from __future__ import annotations

import numpy as np

from .kinematics import IKError, leg_ik, nominal_foot
from .robot import LEG_NAMES, Robot, load_gait

_EPS_CMD = 1e-6
_HOME_TOL = 1e-4      # m, foot counts as "home" (stop condition)


def se2_step(v: np.ndarray, wz: float, dt: float) -> tuple[np.ndarray, float]:
    """Exact body displacement (in the body frame at the start) and rotation for a constant
    twist (vx, vy, wz) held for dt."""
    th = wz * dt
    if abs(th) < 1e-12:
        return v * dt, th
    s, c = np.sin(th), np.cos(th)
    V = np.array([[s, -(1 - c)], [1 - c, s]]) / wz      # SE(2) left Jacobian times 1/wz
    return V @ v, th


def _rot(th: float) -> np.ndarray:
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


class TripodGait:
    def __init__(self, robot: Robot, gait: dict | None = None, dt: float | None = None):
        self.robot = robot
        g = (gait or load_gait())["tripod"]
        self.groups = [list(grp) for grp in g["groups"]]
        self.cycle = float(g["cycle_time"])
        self.duty = float(g["duty_factor"])
        self.height = float(g["step_height"])
        self.max_stride = float(g["max_stride"])
        self.max_acc = np.array([g["max_lin_accel"], g["max_lin_accel"], g["max_ang_accel"]], dtype=float)
        if not 0.5 <= self.duty < 1.0:
            raise ValueError("tripod needs 0.5 <= duty_factor < 1")
        self.dt = dt if dt is not None else 1.0 / robot.control["policy_rate_hz"]
        self.t_stance = self.duty * self.cycle
        self.swing_frac = 1.0 - self.duty
        self.group_of = {leg: k for k, grp in enumerate(self.groups) for leg in grp}

        nf = nominal_foot(robot)                     # leg frame
        self.ground_z = float(nf[2])
        self.home = {}
        for leg in LEG_NAMES:
            m = robot.mounts[leg]
            self.home[leg] = np.array(m.xy) + _rot(m.yaw) @ nf[:2]
        self.q_stance = leg_ik(robot, nf)
        lim = robot.command_limits
        self.cmd_limits = np.array([lim["vx"], lim["vy"], lim["wz"]])
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.phase = 0.0
        self.active = False
        self.feet = {leg: np.array([*self.home[leg], self.ground_z]) for leg in LEG_NAMES}
        self.liftoff = {leg: self.feet[leg][:2].copy() for leg in LEG_NAMES}
        self.q = {leg: self.q_stance.copy() for leg in LEG_NAMES}
        self.ik_failures = 0
        self.cmd = np.zeros(3)
        self.body_twist = np.zeros(3)
        self.startup = False

    def swing_progress(self, leg: str, phase: float | None = None) -> float | None:
        """Swing progress in [0, 1) if the leg is swinging at `phase`, else None."""
        ph = self.phase if phase is None else phase
        start = 0.5 * self.group_of[leg]
        rel = (ph - start) % 1.0
        return rel / self.swing_frac if rel < self.swing_frac else None

    def swinging_legs(self) -> list[str]:
        if not self.active:
            return []
        return [leg for leg in LEG_NAMES if self.swing_progress(leg) is not None]

    # ------------------------------------------------------------------ command shaping
    def shape_command(self, cmd) -> np.ndarray:
        """Clamp to limits, then scale so the largest stance stroke <= max_stride."""
        c = np.clip(np.asarray(cmd, dtype=float), -self.cmd_limits, self.cmd_limits)
        v, wz = c[:2], c[2]
        stroke = max(np.linalg.norm(self._foot_velocity(v, wz, self.home[leg])) * self.t_stance
                     for leg in LEG_NAMES)
        if stroke > self.max_stride:
            c *= self.max_stride / stroke
        return c

    def _rate_limit(self, target: np.ndarray) -> np.ndarray:
        step = self.max_acc * self.dt
        return self.cmd + np.clip(target - self.cmd, -step, step)

    @staticmethod
    def _foot_velocity(v, wz, p_xy) -> np.ndarray:
        """Velocity of a ground point relative to the body (= minus body velocity at that point)."""
        return -(np.asarray(v) + wz * np.array([-p_xy[1], p_xy[0]]))

    # ------------------------------------------------------------------ main update
    def step(self, cmd) -> np.ndarray:
        """Advance one control period with body twist command (vx, vy, wz).
        Returns 18 joint targets in robot.joint_names order."""
        c = self._rate_limit(self.shape_command(cmd))
        self.cmd = c
        moving = np.any(np.abs(c) > _EPS_CMD)
        if moving and not self.active:
            self.active = True
            self.startup = True
        if not self.active:
            return self.joint_targets()

        v, wz = c[:2], c[2]
        old_phase = self.phase
        new_phase = (self.phase + self.dt / self.cycle) % 1.0
        motion = 0.5 if self.startup else 1.0          # stance speed factor (start-up ramp)
        disp, th = se2_step(motion * v, motion * wz, self.dt)
        self.body_twist = motion * c                   # twist the stance feet assume (no slip)
        R_inv = _rot(-th)

        for leg in LEG_NAMES:
            s_old = self.swing_progress(leg, old_phase)
            s_new = self.swing_progress(leg, new_phase)
            foot = self.feet[leg]
            if s_new is None:
                if s_old is not None:            # touchdown this tick: land on the target
                    foot[:2] = self._touchdown(leg, v, wz)
                else:                            # stance: move with the ground (exact SE(2))
                    foot[:2] = R_inv @ (foot[:2] - disp)
                foot[2] = self.ground_z
            else:
                lifting = s_old is None or s_new < s_old
                if lifting:
                    self.liftoff[leg] = foot[:2].copy()
                k_old = 0.0 if lifting else self._cycloid(s_old)
                k_new = self._cycloid(s_new)
                target = self._touchdown(leg, v, wz)
                foot[:2] += (target - foot[:2]) * (k_new - k_old) / (1.0 - k_old)
                foot[2] = self.ground_z + self.height * 0.5 * (1 - np.cos(2 * np.pi * s_new))
        self.phase = new_phase

        # Stop once the command is zero and every foot is home, at a group switch.
        crossed = any(self.swing_progress(leg, old_phase) is not None and
                      self.swing_progress(leg, new_phase) is None for leg in LEG_NAMES)
        if crossed:
            self.startup = False
        if not moving and crossed and all(
                np.linalg.norm(self.feet[l][:2] - self.home[l]) < _HOME_TOL for l in LEG_NAMES):
            self.active = False
            self.phase = 0.0 if new_phase < 0.25 or new_phase >= 0.75 else 0.5
            for leg in LEG_NAMES:
                self.feet[leg] = np.array([*self.home[leg], self.ground_z])
        return self.joint_targets()

    @staticmethod
    def _cycloid(s: float) -> float:
        return s - np.sin(2 * np.pi * s) / (2 * np.pi)          # 0 -> 1, zero slope at both ends

    def _touchdown(self, leg, v, wz) -> np.ndarray:
        """Touchdown point: nominal + half the stance stroke ahead (opposite to foot motion)."""
        return self.home[leg] - 0.5 * self._foot_velocity(v, wz, self.home[leg]) * self.t_stance

    # ------------------------------------------------------------------ IK
    def foot_in_leg_frame(self, leg: str) -> np.ndarray:
        m = self.robot.mounts[leg]
        xy = _rot(-m.yaw) @ (self.feet[leg][:2] - np.array(m.xy))
        return np.array([xy[0], xy[1], self.feet[leg][2]])

    def joint_targets(self) -> np.ndarray:
        out = []
        for leg in LEG_NAMES:
            try:
                self.q[leg] = leg_ik(self.robot, self.foot_in_leg_frame(leg))
            except IKError:
                self.ik_failures += 1            # keep the last valid angles for this leg
            out.append(self.q[leg])
        return np.concatenate(out)
