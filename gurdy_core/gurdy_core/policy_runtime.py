"""Deployment-side RL policy controller — pure NumPy (+ optional onnxruntime), no ROS, no simulator.

Everything the ROS 2 node `gurdy_policy` does per 50 Hz tick lives here, so it can be unit-tested
against the MuJoCo training env (same sensor inputs -> same observation vector, same joint targets):

    sensors (IMU quaternion, gyro, joint pos/vel) + /cmd_vel
      -> deadband (gurdy_core.rl_command) + clip to robot command limits
      -> frame (gurdy_core.rl_obs.build_frame) -> 3-frame history -> 195-value observation
      -> policy (ONNX: VecNormalize baked in, deterministic mean action)
      -> ActionMapper (clip, EMA filter alpha, stance offsets, joint limits) -> 18 joint targets

Tick order mirrors GurdyEnv: observation at tick k carries gait-clock phase k*dt/cycle and the
filtered action of tick k-1 as "previous action"; the first observation fills the whole history.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .rl_action import ActionMapper
from .rl_command import apply_deadband
from .rl_obs import ObsHistory, build_frame, gravity_in_body
from .robot import Robot


def load_meta(onnx_path: str) -> dict:
    with open(os.path.splitext(onnx_path)[0] + ".json") as f:
        return json.load(f)


class OnnxPolicy:
    """obs (195) -> raw mean action (18). CPU onnxruntime, single thread (a 0.5 MB MLP)."""

    def __init__(self, onnx_path: str):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    def __call__(self, obs) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)[None]
        return self.sess.run(None, {self.input_name: x})[0][0].astype(float)


class PolicyController:
    def __init__(self, robot: Robot, act_fn, meta: dict):
        self.robot = robot
        self.act_fn = act_fn
        self.meta = meta
        self.mapper = ActionMapper(robot, meta["action_scale"], meta["action_filter_alpha"])
        self.q_stance = self.mapper.q_stance
        self.dt = 1.0 / float(meta["policy_rate_hz"])
        self.clock_period = float(meta["clock_period_s"])
        self.cmd_lim = np.array([robot.command_limits[k] for k in ("vx", "vy", "wz")])
        self.hist = ObsHistory()
        self.reset()

    def reset(self) -> None:
        """Back to the episode-start state the policy was trained from (rest, zero history)."""
        self.mapper.reset()
        self.steps = 0
        self.command = np.zeros(3)
        self.q_ref = self.q_stance.copy()
        self._started = False

    def condition_command(self, command) -> np.ndarray:
        """Deadband (shared with training), then clip to the robot's command limits."""
        return np.clip(apply_deadband(command), -self.cmd_lim, self.cmd_lim)

    def observe(self, quat_wxyz, gyro, q, qd, command) -> np.ndarray:
        """Build the 195-value observation for this tick (does not advance the tick)."""
        self.command = self.condition_command(command)
        frame = build_frame(gravity=gravity_in_body(quat_wxyz), gyro=gyro, joint_pos=q, joint_vel=qd,
                            q_stance=self.q_stance, prev_action=self.mapper.a_filt, command=self.command,
                            phase=(self.steps * self.dt / self.clock_period) % 1.0)
        if not self._started:
            self._started = True
            return self.hist.reset(frame)
        return self.hist.push(frame)

    def step(self, quat_wxyz, gyro, q, qd, command) -> np.ndarray:
        """One 50 Hz tick: sensors + command -> joint targets (robot.joint_names order)."""
        obs = self.observe(quat_wxyz, gyro, q, qd, command)
        self.q_ref = self.mapper(self.act_fn(obs))
        self.steps += 1
        return self.q_ref
