"""Velocity-command conditioning shared by the MuJoCo env and the ROS 2 policy node (no ROS).

Explicit /cmd_vel deadband (step 2.8, dead-zone fix): a command whose planar speed is below
LINEAR_DEADBAND (m/s) has its linear part set to zero; |wz| below ANGULAR_DEADBAND (rad/s) is set
to zero. Training applies the same function to every sampled command, so the policy is never asked
for a speed it was taught to treat as "stand", and everything above the deadband must be tracked.
"""
from __future__ import annotations

import numpy as np

LINEAR_DEADBAND = 0.015    # m/s, on sqrt(vx^2 + vy^2)
ANGULAR_DEADBAND = 0.05    # rad/s, on |wz|


def apply_deadband(command, lin: float = LINEAR_DEADBAND, ang: float = ANGULAR_DEADBAND) -> np.ndarray:
    """(vx, vy, wz) -> (vx, vy, wz) with the deadband applied (returns a new float array)."""
    c = np.array(command, dtype=float)
    if np.hypot(c[0], c[1]) < lin:
        c[0] = c[1] = 0.0
    if abs(c[2]) < ang:
        c[2] = 0.0
    return c


# --- minimum speed (deployment, Nav2 -> policy) -------------------------------------------------------------
# The policy stands still for |v| below ~0.03 m/s / 0.10 rad/s from rest (phase2_terrain.md). Nav2 (MPPI) starts from
# standstill with at most ax_max * model_dt = 0.04 m/s, so a command below the minimum must be LIFTED to it, never zeroed
# (zeroing deadlocked the start, phase3_gazebo_results.md).
MIN_LINEAR_SPEED = 0.05    # m/s, on sqrt(vx^2 + vy^2)
MIN_ANGULAR_SPEED = 0.15   # rad/s, on |wz|
MIN_LINEAR_EPSILON = LINEAR_DEADBAND    # m/s: |(vx, vy)| at or below the policy deadband (0.015) is an exact zero
MIN_ANGULAR_EPSILON = ANGULAR_DEADBAND  # rad/s: |wz| at or below the policy deadband (0.05) is an exact zero


def lift_to_min_speed(command, lin_min: float = MIN_LINEAR_SPEED, ang_min: float = MIN_ANGULAR_SPEED,
                      lin_eps: float = MIN_LINEAR_EPSILON,
                      ang_eps: float = MIN_ANGULAR_EPSILON) -> np.ndarray:
    """(vx, vy, wz) -> command with small non-zero parts lifted to the minimum speed.

    Linear part: |(vx, vy)| <= lin_eps (m/s) -> exactly 0; < 1 -> the vector is scaled up to lin_min keeping its
    direction; otherwise unchanged. Angular part: the same on |wz| with ang_eps (rad/s), keeping the sign. The two parts are
    independent (a pure turn stays a pure turn; a tiny turn next to a real translation is lifted to ang_min)."""
    c = np.array(command, dtype=float)
    n = float(np.hypot(c[0], c[1]))
    if n <= lin_eps:
        c[0] = c[1] = 0.0
    elif n < lin_min:
        c[:2] *= lin_min / n
    w = abs(c[2])
    if w <= ang_eps:
        c[2] = 0.0
    elif w < ang_min:
        c[2] = np.sign(c[2]) * ang_min
    return c
