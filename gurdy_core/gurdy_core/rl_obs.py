"""Policy observation builder — shared by the MuJoCo training env and the ROS 2 policy node, so the
policy sees exactly the same numbers in both sims. Pure NumPy, no ROS, no simulator.

Only quantities a real robot can measure (no ground-truth position or linear velocity):

    frame (65) = [ gravity direction in body frame (3)      <- IMU orientation (yaw not used)
                   body angular velocity (3)                <- IMU gyro
                   joint positions - stance (18)            <- encoders
                   joint velocities * 0.1 (18)              <- encoders
                   previous action (18)                     <- policy's own output
                   velocity command (vx, vy, wz) (3)        <- /cmd_vel
                   gait clock (sin, cos) (2) ]              <- internal timer
    observation = last HISTORY frames, oldest first (195)

Joint order is robot.joint_names. Further normalisation (running mean/std) is learned in training and
folded into the exported policy.
"""
from __future__ import annotations

import numpy as np

HISTORY = 3
JOINT_VEL_SCALE = 0.1
FRAME_LAYOUT = (("gravity", 3), ("gyro", 3), ("joint_pos", 18), ("joint_vel", 18),
                ("prev_action", 18), ("command", 3), ("clock", 2))
FRAME_DIM = sum(n for _, n in FRAME_LAYOUT)          # 65
OBS_DIM = HISTORY * FRAME_DIM                         # 195


def frame_slices() -> dict[str, slice]:
    out, i = {}, 0
    for name, n in FRAME_LAYOUT:
        out[name] = slice(i, i + n)
        i += n
    return out


def gravity_in_body(quat_wxyz) -> np.ndarray:
    """Unit gravity direction expressed in the body frame, from the body orientation quaternion.
    Level robot -> (0, 0, -1). Independent of yaw."""
    w, x, y, z = quat_wxyz
    # third row of R (world->body transpose of body->world) times (0, 0, -1)
    return -np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])


_SLICES = None


def build_frame(gravity, gyro, joint_pos, joint_vel, q_stance, prev_action, command, phase) -> np.ndarray:
    global _SLICES
    if _SLICES is None:
        _SLICES = frame_slices()
    f = np.empty(FRAME_DIM)
    s = _SLICES
    f[s["gravity"]] = gravity
    f[s["gyro"]] = gyro
    f[s["joint_pos"]] = np.asarray(joint_pos) - q_stance
    f[s["joint_vel"]] = np.asarray(joint_vel) * JOINT_VEL_SCALE
    f[s["prev_action"]] = prev_action
    f[s["command"]] = command
    f[s["clock"]] = (np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase))
    return f


class ObsHistory:
    """Fixed-length frame history; reset() fills it with copies of the first frame."""

    def __init__(self, history: int = HISTORY):
        self.history = history
        self.buf = np.zeros((history, FRAME_DIM))

    def reset(self, frame: np.ndarray) -> np.ndarray:
        self.buf[:] = frame
        return self.buf.reshape(-1).copy()

    def push(self, frame: np.ndarray) -> np.ndarray:
        self.buf[:-1] = self.buf[1:]
        self.buf[-1] = frame
        return self.buf.reshape(-1).copy()
