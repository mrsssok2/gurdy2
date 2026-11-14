"""Gurdy MuJoCo Gymnasium environment (Phase 2).

* Model: sim_mujoco.build_mjcf(robot.yaml) — the same model that passed the Phase 1 parity tests.
* Timing: policy at robot.yaml policy_rate_hz (50 Hz); each action is held for
  1 / (policy_rate * mujoco_timestep) = 10 physics steps; PD actuators run every 2 ms step (500 Hz).
* Action (option A): q_ref = q_stance + scale * clip(a, -1, 1), clipped to the joint limits;
  scale = (hip_yaw 0.3, hip_pitch 0.5, knee 0.5) rad.
* Observation: gurdy_core.rl_obs (proprioception only, 3-frame history, 195 values).
* Episode: 20 s; command resampled every 5 s (15 % zero commands); terminates on a fall
  (body below FALL_Z, tilt above FALL_TILT, or body box touching the ground).
* reset(options={"command": (vx, vy, wz)}) fixes the command (evaluation/tests); with
  config resample_command false it is kept for the whole episode.
* info: ground-truth diagnostics for logging/reward/eval ONLY (never part of the observation).

Config: rl/configs/*.yaml, section `env` (defaults below).
"""
from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np

from gurdy_core.experiments import FALL_TILT, FALL_Z
from gurdy_core.kinematics import leg_ik, nominal_foot
from gurdy_core.rl_obs import OBS_DIM, ObsHistory, build_frame, gravity_in_body
from gurdy_core.robot import JOINT_TYPES, LEG_NAMES, load_gait, load_robot
from sim_mujoco.mjcf import build_mjcf

DEFAULTS = {
    "episode_s": 20.0,
    "command_period_s": 5.0,
    "resample_command": True,
    "zero_command_prob": 0.15,
    "action_scale": {"hip_yaw": 0.3, "hip_pitch": 0.5, "knee": 0.5},
    "init_joint_noise": 0.05,
}


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


class GurdyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: dict | None = None, robot_yaml: str | None = None):
        super().__init__()
        self.cfg = {**DEFAULTS, **(config or {})}
        self.robot = load_robot(robot_yaml)
        r = self.robot
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(r))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.policy_dt = 1.0 / r.control["policy_rate_hz"]
        self.n_sub = int(round(self.policy_dt / m.opt.timestep))
        self.max_steps = int(round(self.cfg["episode_s"] / self.policy_dt))
        self.cmd_steps = int(round(self.cfg["command_period_s"] / self.policy_dt))
        self.clock_period = float(load_gait()["tripod"]["cycle_time"])

        self.q_stance = np.tile(leg_ik(r, nominal_foot(r)), len(LEG_NAMES))
        self.scale = np.tile([self.cfg["action_scale"][t] for t in JOINT_TYPES], len(LEG_NAMES))
        self.q_lo = np.tile([r.joint_limits[t][0] for t in JOINT_TYPES], len(LEG_NAMES))
        self.q_hi = np.tile([r.joint_limits[t][1] for t in JOINT_TYPES], len(LEG_NAMES))
        self.cmd_lim = np.array([r.command_limits[k] for k in ("vx", "vy", "wz")])

        self.qadr = np.array([m.joint(n).qposadr[0] for n in r.joint_names])
        self.vadr = np.array([m.joint(n).dofadr[0] for n in r.joint_names])
        self.gyro_adr = m.sensor("imu_gyro").adr[0]
        self.body_geom = m.geom("body").id
        self.foot_geoms = np.array([m.geom(f"{leg}_foot").id for leg in LEG_NAMES])
        self.floor_geom = m.geom("floor").id
        self.key = m.key("stance").id

        self.action_space = gym.spaces.Box(-1.0, 1.0, (len(r.joint_names),), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.hist = ObsHistory()

    # ------------------------------------------------------------------ helpers
    def _sample_command(self) -> np.ndarray:
        if self.np_random.random() < self.cfg["zero_command_prob"]:
            return np.zeros(3)
        return self.np_random.uniform(-1.0, 1.0, 3) * self.cmd_lim

    def _frame(self) -> np.ndarray:
        d = self.data
        return build_frame(
            gravity=gravity_in_body(d.qpos[3:7]),
            gyro=d.sensordata[self.gyro_adr:self.gyro_adr + 3],
            joint_pos=d.qpos[self.qadr], joint_vel=d.qvel[self.vadr], q_stance=self.q_stance,
            prev_action=self.prev_action, command=self.command,
            phase=(self.steps * self.policy_dt / self.clock_period) % 1.0)

    def _body_velocity(self):
        """Linear velocity in the body frame (free-joint qvel[0:3] is world frame) and yaw rate."""
        d = self.data
        R = d.xmat[1].reshape(3, 3)                    # body 1 = base_link
        return R.T @ d.qvel[0:3], d.qvel[5]            # qvel[3:6] is already body frame

    def foot_contacts(self) -> np.ndarray:
        """Boolean per leg (LEG_NAMES order): foot sphere touching the floor."""
        d = self.data
        out = np.zeros(len(LEG_NAMES), dtype=bool)
        if d.ncon:
            g = d.contact.geom[:d.ncon]
            touching = g[(g == self.floor_geom).any(axis=1)].ravel()
            out = np.isin(self.foot_geoms, touching)
        return out

    def _fallen(self) -> bool:
        d = self.data
        g = gravity_in_body(d.qpos[3:7])
        if d.qpos[2] < FALL_Z or np.arccos(np.clip(-g[2], -1, 1)) > FALL_TILT:
            return True
        if d.ncon:                                     # vectorised: Python loops over contacts were 30 % of step time
            return bool(np.any(d.contact.geom[:d.ncon] == self.body_geom))
        return False

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        m, d = self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, self.key)
        noise = self.np_random.uniform(-1, 1, len(self.qadr)) * self.cfg["init_joint_noise"]
        d.qpos[self.qadr] = np.clip(self.q_stance + noise, self.q_lo, self.q_hi)
        d.ctrl[:] = self.q_stance
        mujoco.mj_forward(m, d)
        self.steps = 0
        self.prev_action = np.zeros(self.action_space.shape[0])
        self.q_ref = self.q_stance.copy()
        fixed = (options or {}).get("command")
        self.fixed_command = fixed is not None
        self.command = np.asarray(fixed, dtype=float) if self.fixed_command else self._sample_command()
        obs = self.hist.reset(self._frame())
        return obs.astype(np.float32), {"command": self.command.copy()}

    def step(self, action):
        a = np.minimum(np.maximum(np.asarray(action, dtype=float), -1.0), 1.0)
        d = self.data
        self.q_ref = np.minimum(np.maximum(self.q_stance + self.scale * a, self.q_lo), self.q_hi)
        d.ctrl[:] = self.q_ref
        cmd = self.command.copy()                      # command this step was taken under
        mujoco.mj_step(self.model, d, nstep=self.n_sub)
        self.steps += 1

        v, wz = self._body_velocity()
        # Placeholder reward (step 2.1): tracking only. Full reward = step 2.3.
        e = np.sum((v[:2] - cmd[:2]) ** 2) / 0.1 ** 2 + (wz - cmd[2]) ** 2 / 0.3 ** 2
        reward = float(np.exp(-e))

        terminated = self._fallen()
        truncated = self.steps >= self.max_steps
        self.prev_action = a
        if (self.cfg["resample_command"] and not self.fixed_command
                and self.steps % self.cmd_steps == 0):
            self.command = self._sample_command()
        obs = self.hist.push(self._frame())
        info = {"command": cmd, "body_vel": v, "yaw_rate": float(wz), "body_z": float(d.qpos[2]),
                "foot_contact": self.foot_contacts()}
        return obs.astype(np.float32), reward, terminated, truncated, info
