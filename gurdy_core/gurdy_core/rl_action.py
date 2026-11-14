"""Policy action -> joint targets, shared by the MuJoCo env and the ROS 2 policy node (no ROS).

    a_filt = alpha * clip(a, -1, 1) + (1 - alpha) * a_filt_prev      (EMA low-pass, alpha=1: off)
    q_ref  = clip(q_stance + scale * a_filt, q_lo, q_hi)

The filter is part of the controller: the policy observes a_filt_prev as its "previous action".
alpha 0.5 (plan C1, 2026-11-14): exploration noise of a fresh policy hurt tracking 3x less, while
the tripod gait executed through the filter scored the same (1.287 vs 1.285 reward/step).
"""
from __future__ import annotations

import numpy as np

from .kinematics import leg_ik, nominal_foot
from .robot import JOINT_TYPES, LEG_NAMES, Robot


class ActionMapper:
    def __init__(self, robot: Robot, action_scale: dict, alpha: float = 1.0):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        n = len(LEG_NAMES)
        self.alpha = float(alpha)
        self.q_stance = np.tile(leg_ik(robot, nominal_foot(robot)), n)
        self.scale = np.tile([action_scale[t] for t in JOINT_TYPES], n)
        self.q_lo = np.tile([robot.joint_limits[t][0] for t in JOINT_TYPES], n)
        self.q_hi = np.tile([robot.joint_limits[t][1] for t in JOINT_TYPES], n)
        self.a_filt = np.zeros(len(self.q_stance))

    def reset(self) -> None:
        self.a_filt[:] = 0.0

    def __call__(self, action) -> np.ndarray:
        """Filter the action and return joint targets (robot.joint_names order)."""
        a = np.minimum(np.maximum(np.asarray(action, dtype=float), -1.0), 1.0)
        if self.alpha < 1.0:
            a = self.alpha * a + (1.0 - self.alpha) * self.a_filt
        self.a_filt = a
        return np.minimum(np.maximum(self.q_stance + self.scale * a, self.q_lo), self.q_hi)
