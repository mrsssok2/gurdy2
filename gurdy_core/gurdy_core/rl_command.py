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
