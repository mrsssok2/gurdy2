"""Gurdy MuJoCo Gymnasium environment (Phase 2).

* Model: sim_mujoco.build_mjcf(robot.yaml) — the same model that passed the Phase 1 parity tests.
* Timing: policy at robot.yaml policy_rate_hz (50 Hz); each action is held for
  1 / (policy_rate * mujoco_timestep) = 10 physics steps; PD actuators run every 2 ms step (500 Hz).
* Action (option A): q_ref = q_stance + scale * clip(a, -1, 1), clipped to the joint limits;
  scale = (hip_yaw 0.3, hip_pitch 0.7, knee 0.6) rad (the tripod gait needs up to 0.47 / 0.56 /
  0.51 rad: about 20 % headroom so the policy can at least represent the baseline gait).
* Observation: gurdy_core.rl_obs (proprioception only, 3-frame history, 195 values).
* Episode: 20 s; command resampled every 5 s (15 % zero commands); terminates on a fall
  (body below FALL_Z, tilt above FALL_TILT, or body box touching the ground).
* reset(options={"command": (vx, vy, wz)}) fixes the command (evaluation/tests); with
  config resample_command false it is kept for the whole episode.
* info: ground-truth diagnostics for logging/reward/eval ONLY (never part of the actor observation).
* Step 2.8: terrain (envs/terrain.py: heightfield per env, per-env curriculum over rough / slopes /
  steps; flat episodes use the original plane), small commands (small_command_prob, 15-40 % of the
  limits) and the shared /cmd_vel deadband (gurdy_core.rl_command) on every sampled command.
  Falls are judged relative to the ground under the feet.
* Step 2.7: domain randomisation (envs/randomization.py; ranges scaled by attr dr_scale), action
  latency (previous targets held for k substeps), pushes, sensor noise on the ACTOR's inputs only;
  obs_mode "asym" returns {"policy": 195 actor values, "privileged": 17 clean critic-only values}.

Config: rl/configs/*.yaml, sections `env` and `reward` (defaults below and in envs/reward.py).
Reward: envs/reward.py; info["reward_terms"] has every weighted term (for TensorBoard).
"""
from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np

from gurdy_core.experiments import FALL_TILT, FALL_Z
from gurdy_core.kinematics import leg_ik, nominal_foot
from gurdy_core.rl_action import ActionMapper
from gurdy_core.rl_command import apply_deadband
from gurdy_core.rl_obs import OBS_DIM, ObsHistory, build_frame, gravity_in_body
from gurdy_core.robot import JOINT_TYPES, LEG_NAMES, load_gait, load_robot
from sim_mujoco.mjcf import build_mjcf

from . import reward as R
from . import terrain as T
from .randomization import PRIVILEGED_DIM, DomainRandomizer

DEFAULTS = {
    "episode_s": 20.0,
    "command_period_s": 5.0,
    "resample_command": True,
    "axis_aligned_prob": 0.0,       # 2.7b: share of commands that are pure +-vx / +-vy / +-wz (30-100 % of limit)
    "rest_start_prob": 0.0,         # 2.7b: share of command switches preceded by 1-2 s of zero command
    "zero_command_prob": 0.15,
    "small_command_prob": 0.0,      # 2.8: share of commands at 15-40 % of the limits (half axis-aligned)
    "small_command_range": [0.15, 0.40],
    "command_deadband": False,      # 2.8: gurdy_core.rl_command.apply_deadband on every sampled command
    "terrain": {},                  # 2.8: see envs/terrain.py DEFAULTS (enabled false = floor plane only)
    "terrain_fixed": None,          # eval: {"type": "rough", "value": 0.02} (no curriculum)
    "action_scale": {"hip_yaw": 0.3, "hip_pitch": 0.7, "knee": 0.6},
    "init_joint_noise": 0.05,
    "action_filter_alpha": 1.0,     # EMA on actions: a_f = alpha*a + (1-alpha)*a_f_prev (1.0 = off)
    "measure_energy": False,        # eval: integrate sum|tau*qdot| over every 2 ms substep (slower)
    "record_substeps": False,       # eval: info["substeps"] = (qdot, tau, qpos) at every 2 ms substep
    "obs_mode": "flat",             # "flat": Box(195) | "asym": Dict(policy 195, privileged 17)
    "domain_randomization": {},     # see envs/randomization.py DEFAULTS (enabled false = nominal)
    "dr_fixed": None,               # eval scenario: fixed DR values (e.g. {"friction": 0.5})
}


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


class GurdyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: dict | None = None, robot_yaml: str | None = None,
                 reward_config: dict | None = None):
        super().__init__()
        self.cfg = {**DEFAULTS, **(config or {})}
        rc = reward_config or {}
        self.rcfg = {**R.DEFAULTS, **rc, "weights": {**R.DEFAULTS["weights"], **rc.get("weights", {})}}
        self.kappa = float(self.rcfg["kappa"])        # annealed by the training callback via set_attr
        self.penalty_scale = float(self.rcfg["penalty_scale"])   # curriculum, set via set_attr
        self.eff_progress = 1.0          # efficiency-weight curriculum progress, set via set_attr
        self.dr_scale = 1.0              # domain-randomisation range scale, set via set_attr
        self.robot = load_robot(robot_yaml)
        r = self.robot
        self.tcfg = T.merged(self.cfg["terrain"])
        self.terrain_on = bool(self.tcfg["enabled"] or self.cfg["terrain_fixed"])
        hf = None
        if self.terrain_on:
            self.tgen = T.TerrainGenerator(self.tcfg)
            self.elev = float(self.tcfg["max_elevation"])
            hf = {"n": self.tgen.n, "half": self.tgen.half, "elevation": self.elev, "z_offset": -self.elev / 2}
        self.curriculum = None
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(r, hfield=hf))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.policy_dt = 1.0 / r.control["policy_rate_hz"]
        self.n_sub = int(round(self.policy_dt / m.opt.timestep))
        self.max_steps = int(round(self.cfg["episode_s"] / self.policy_dt))
        self.cmd_steps = int(round(self.cfg["command_period_s"] / self.policy_dt))
        self.clock_period = float(load_gait()["tripod"]["cycle_time"])

        self.mapper = ActionMapper(r, self.cfg["action_scale"], self.cfg["action_filter_alpha"])
        self.q_stance, self.scale = self.mapper.q_stance, self.mapper.scale
        self.q_lo, self.q_hi = self.mapper.q_lo, self.mapper.q_hi
        self.cmd_lim = np.array([r.command_limits[k] for k in ("vx", "vy", "wz")])

        self.qadr = np.array([m.joint(n).qposadr[0] for n in r.joint_names])
        self.vadr = np.array([m.joint(n).dofadr[0] for n in r.joint_names])
        self.gyro_adr = m.sensor("imu_gyro").adr[0]
        self.body_geom = m.geom("body").id
        self.foot_geoms = np.array([m.geom(f"{leg}_foot").id for leg in LEG_NAMES])
        self.floor_geom = m.geom("floor").id
        self.terrain_geom = m.geom("terrain").id if self.terrain_on else -1
        self.ground_geom = self.floor_geom             # the geom the robot stands on this episode
        self.heights = None                            # terrain height map of this episode (None = plane)
        self.terrain_type, self.terrain_value, self.terrain_level = "flat", 0.0, 0
        self.foot_sites = [m.site(f"{leg}_foot").id for leg in LEG_NAMES]
        self.air_timer = R.FootAirTimer(len(LEG_NAMES), self.policy_dt)
        self._vel6 = np.zeros(6)
        self.key = m.key("stance").id

        self.action_space = gym.spaces.Box(-1.0, 1.0, (len(r.joint_names),), np.float32)
        box = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        if self.cfg["obs_mode"] == "asym":
            self.observation_space = gym.spaces.Dict({
                "policy": box, "privileged": gym.spaces.Box(-np.inf, np.inf, (PRIVILEGED_DIM,), np.float32)})
        else:
            self.observation_space = box
        self.hist = ObsHistory()
        legs = [m.body(f"{leg}_{s}").id for leg in LEG_NAMES for s in ("coxa", "femur", "tibia")]
        self.dr = DomainRandomizer(m, self.cfg["domain_randomization"], self.foot_geoms, self.floor_geom,
                                   m.body("base_link").id, legs, self.vadr,
                                   nominal_joint_friction=float(r.joint_common["friction"]))

    # ------------------------------------------------------------------ helpers
    def _sample_command(self) -> np.ndarray:
        c = self._sample_raw_command()
        return apply_deadband(c) if self.cfg["command_deadband"] else c

    def _sample_raw_command(self) -> np.ndarray:
        rng = self.np_random
        if rng.random() < self.cfg["zero_command_prob"]:
            return np.zeros(3)
        if self.cfg["small_command_prob"] > 0 and rng.random() < self.cfg["small_command_prob"]:
            # 2.8 dead-zone fix: largest component at 15-40 % of its limit
            if rng.random() < 0.5:
                d = np.zeros(3)
                d[int(rng.integers(3))] = rng.choice([-1.0, 1.0])
            else:
                d = rng.uniform(-1.0, 1.0, 3)
                d /= max(np.max(np.abs(d)), 1e-9)
            return d * rng.uniform(*self.cfg["small_command_range"]) * self.cmd_lim
        if self.cfg["axis_aligned_prob"] > 0 and rng.random() < self.cfg["axis_aligned_prob"]:
            c = np.zeros(3)
            ax = int(rng.integers(3))
            c[ax] = rng.choice([-1.0, 1.0]) * rng.uniform(0.3, 1.0) * self.cmd_lim[ax]
            return c
        return rng.uniform(-1.0, 1.0, 3) * self.cmd_lim

    def _new_command(self) -> np.ndarray:
        """Sample a command; with rest_start_prob it is delayed by 1-2 s of zero command, so the
        robot must start it from rest (2.7b: trains out the 'don't start' trap)."""
        c = self._sample_command()
        self.pending_command = None
        p = self.cfg["rest_start_prob"]
        if p > 0 and np.any(c != 0) and self.np_random.random() < p:
            self.pending_command = c
            self.rest_until = self.steps + int(self.np_random.uniform(1.0, 2.0) / self.policy_dt)
            return np.zeros(3)
        return c

    def _frame(self) -> np.ndarray:
        """Actor observation frame; sensor noise (DR) applied here only — the critic stays clean."""
        d = self.data
        g = gravity_in_body(d.qpos[3:7])
        gyro = d.sensordata[self.gyro_adr:self.gyro_adr + 3].copy()
        qp, qv = d.qpos[self.qadr].copy(), d.qvel[self.vadr].copy()
        ns = self.dr.params["noise_scale"]
        if ns > 0:
            n, rng = self.dr.cfg["noise"], self.np_random
            g = g + rng.normal(0, n["gravity"] * ns, 3)
            gyro += rng.normal(0, n["gyro"] * ns, 3)
            qp += rng.normal(0, n["joint_pos"] * ns, len(qp))
            qv += rng.normal(0, n["joint_vel"] * ns, len(qv))
        return build_frame(gravity=g, gyro=gyro, joint_pos=qp, joint_vel=qv, q_stance=self.q_stance,
                           prev_action=self.prev_action, command=self.command,
                           phase=(self.steps * self.policy_dt / self.clock_period) % 1.0)

    def _privileged(self, v_body, contact) -> np.ndarray:
        return np.concatenate([v_body, self.data.qvel[3:6], contact.astype(float), self.dr.privileged()])

    def _obs(self, policy_obs, v_body, contact):
        if self.cfg["obs_mode"] != "asym":
            return policy_obs.astype(np.float32)
        return {"policy": policy_obs.astype(np.float32),
                "privileged": self._privileged(v_body, contact).astype(np.float32)}

    def _schedule_push(self):
        lo, hi = self.dr.cfg["push"]["interval_s"]
        self.next_push = self.steps + int(self.np_random.uniform(lo, hi) / self.policy_dt)

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
            touching = g[(g == self.ground_geom).any(axis=1)].ravel()
            out = np.isin(self.foot_geoms, touching)
        return out

    def foot_velocities_xy(self) -> np.ndarray:
        """World-frame horizontal velocity of each foot centre (6 x 2)."""
        out = np.empty((len(self.foot_sites), 2))
        for i, sid in enumerate(self.foot_sites):
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, sid, self._vel6, 0)
            out[i] = self._vel6[3:5]
        return out

    def _ground_ref(self) -> float:
        """Ground height under the robot: median terrain height at the 6 feet (0 on the plane)."""
        if self.heights is None:
            return 0.0
        xs = self.data.site_xpos[self.foot_sites]
        return float(np.median([self.tgen.height_at(self.heights, x, y) for x, y, _ in xs]))

    def _setup_terrain(self):
        """Choose this episode's terrain (curriculum or fixed), upload the heightfield, switch geoms."""
        m = self.model
        if not self.terrain_on:
            return
        fixed = self.cfg["terrain_fixed"]
        if fixed:
            self.terrain_type, self.terrain_value, self.terrain_level = fixed["type"], float(fixed["value"]), -1
        else:
            if self.curriculum is None:
                self.curriculum = T.TerrainCurriculum(self.tcfg, self.np_random)
            self.terrain_type = self.curriculum.sample_type(self.np_random)
            self.terrain_level = self.curriculum.level(self.terrain_type)
            self.terrain_value = T.difficulty_value(self.tcfg, self.terrain_type, self.terrain_level)
        on_terrain = self.terrain_type != "flat"
        m.geom_contype[self.floor_geom] = 0 if on_terrain else 1
        m.geom_contype[self.terrain_geom] = 1 if on_terrain else 0
        m.geom_friction[self.terrain_geom] = m.geom_friction[self.floor_geom]   # DR friction
        self.ground_geom = self.terrain_geom if on_terrain else self.floor_geom
        if on_terrain:
            h = self.tgen.generate(self.terrain_type, self.terrain_value, self.np_random)
            m.hfield_data[:] = np.clip((h + self.elev / 2) / self.elev, 0.0, 1.0).ravel().astype(np.float32)
            self.heights = h
        else:
            self.heights = None

    def _fallen(self) -> bool:
        d = self.data
        g = gravity_in_body(d.qpos[3:7])
        if d.qpos[2] - self._ground_ref() < FALL_Z or np.arccos(np.clip(-g[2], -1, 1)) > FALL_TILT:
            return True
        if d.ncon:                                     # vectorised: Python loops over contacts were 30 % of step time
            return bool(np.any(d.contact.geom[:d.ncon] == self.body_geom))
        return False

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        m, d = self.model, self.data
        self.dr.sample(self.np_random, self.dr_scale, self.cfg["dr_fixed"])
        self.dr.apply(d)
        self._setup_terrain()
        mujoco.mj_resetDataKeyframe(m, d, self.key)
        if self.heights is not None:                   # lift so that the highest foot just touches the ground
            mujoco.mj_kinematics(m, d)
            r_f = self.robot.foot_radius + self.tcfg["res"]
            d.qpos[2] += max(self.tgen.max_near(self.heights, x, y, r_f) for x, y, _ in d.site_xpos[self.foot_sites])
        noise = self.np_random.uniform(-1, 1, len(self.qadr)) * self.cfg["init_joint_noise"]
        d.qpos[self.qadr] = np.clip(self.q_stance + noise, self.q_lo, self.q_hi)
        d.ctrl[:] = self.q_stance
        mujoco.mj_forward(m, d)
        self.steps = 0
        self.prev_action = np.zeros(self.action_space.shape[0])
        self.prev_action2 = np.zeros(self.action_space.shape[0])
        self.mapper.reset()
        self._schedule_push()
        self.q_ref = self.q_stance.copy()
        self.qd_prev = d.qvel[self.vadr].copy()
        self.air_timer.reset()
        self.track_vel = np.zeros(3)                   # filtered (vx, vy, wz), robot starts at rest
        fixed = (options or {}).get("command")
        self.fixed_command = fixed is not None
        self.pending_command = None
        self.command = np.asarray(fixed, dtype=float) if self.fixed_command else self._new_command()
        self.cmd_age = 0                               # steps since the command last changed
        self.ep_err_sum, self.ep_err_n = 0.0, 0        # episode tracking error (terrain curriculum)
        obs = self.hist.reset(self._frame())
        v0, _ = self._body_velocity()
        return self._obs(obs, v0, self.foot_contacts()), {"command": self.command.copy(),
                                                           "dr": dict(self.dr.params)}

    def step(self, action):
        d = self.data
        self.q_ref = self.mapper(action)               # clip, EMA filter (gurdy_core), joint targets
        a = self.mapper.a_filt                         # filtered action = what the policy observes next
        cmd = self.command.copy()                      # command this step was taken under
        q_cmd = np.clip(self.q_ref + self.dr.params["target_offset"], self.q_lo, self.q_hi)   # DR: PD-side offset
        lat = int(self.dr.params["latency_substeps"])  # previous targets held for `lat` substeps
        energy = 0.0
        sub = []
        if self.cfg["measure_energy"] or self.cfg["record_substeps"]:
            dt = self.model.opt.timestep
            for k in range(self.n_sub):
                if k == lat:
                    d.ctrl[:] = q_cmd
                mujoco.mj_step(self.model, d)
                energy += float(np.sum(np.abs(d.actuator_force * d.qvel[self.vadr]))) * dt
                if self.cfg["record_substeps"]:
                    sub.append((d.qvel[self.vadr].copy(), d.actuator_force.copy(), d.qpos.copy()))
        else:
            if lat:
                mujoco.mj_step(self.model, d, nstep=lat)
            d.ctrl[:] = q_cmd
            mujoco.mj_step(self.model, d, nstep=self.n_sub - lat)
        self.steps += 1
        if self.dr.params["push"] and self.steps >= self.next_push:   # random push (velocity kick)
            ang = self.np_random.uniform(0, 2 * np.pi)
            lo, hi = self.dr.cfg["push"]["speed"]
            sp = self.np_random.uniform(lo, hi) * self.dr.params["push_scale"]
            d.qvel[0:2] += sp * np.array([np.cos(ang), np.sin(ang)])
            self._schedule_push()

        v, wz = self._body_velocity()
        w_body = d.qvel[3:6].copy()
        qd = d.qvel[self.vadr].copy()
        contact = self.foot_contacts()
        terminated = self._fallen()
        self.track_vel = R.ema_update(self.track_vel, (v[0], v[1], wz), self.policy_dt,
                                      self.rcfg["track_filter_s"])
        raw = R.terms(
            v_body=v, w_body=w_body, track_vel=self.track_vel, gravity_body=gravity_in_body(d.qpos[3:7]), command=cmd,
            kappa=self.kappa, cfg=self.rcfg, joint_vel=qd, joint_vel_prev=self.qd_prev,
            dt=self.policy_dt, torque=d.actuator_force, action=a, action_prev=self.prev_action,
            foot_contact=contact, foot_vel_xy=self.foot_velocities_xy(),
            air_bonus=self.air_timer.update(contact, self.rcfg["air_time_target"]),
            q=d.qpos[self.qadr], q_stance=self.q_stance, fallen=terminated, action_prev2=self.prev_action2)
        reward, terms, clipped = R.combine(
            R.weighted(raw, self.rcfg["weights"], self.rcfg["curriculum_start_fraction"], self.eff_progress),
            self.penalty_scale, self.rcfg["only_positive"])
        self.qd_prev = qd
        truncated = self.steps >= self.max_steps
        if np.any(np.abs(cmd) > 1e-6) and self.cmd_age >= int(1.0 / self.policy_dt):
            self.ep_err_sum += float(np.linalg.norm((self.track_vel - cmd) / self.cmd_lim)
                                     / np.linalg.norm(cmd / self.cmd_lim))
            self.ep_err_n += 1
        level_change = 0
        if (terminated or truncated) and self.curriculum is not None:
            err = self.ep_err_sum / self.ep_err_n if self.ep_err_n else None
            level_change = self.curriculum.update(self.terrain_type, terminated, err, self.np_random)
        self.prev_action2 = self.prev_action
        self.prev_action = a
        if self.pending_command is not None and self.steps >= self.rest_until:
            self.command, self.pending_command = self.pending_command, None
        if (self.cfg["resample_command"] and not self.fixed_command
                and self.steps % self.cmd_steps == 0):
            self.command = self._new_command()
        self.cmd_age = 0 if not np.array_equal(self.command, cmd) else self.cmd_age + 1
        obs = self._obs(self.hist.push(self._frame()), v, contact)
        info = {"terrain": self.terrain_type, "terrain_level": self.terrain_level,
                "terrain_value": self.terrain_value, "terrain_level_change": level_change, "command": cmd, "body_vel": v, "yaw_rate": float(wz), "body_z": float(d.qpos[2]),
                "foot_contact": contact, "track_vel": self.track_vel.copy(), "reward_terms": terms,
                "reward_clipped": clipped, "energy_J": energy, "substeps": sub}
        return obs, reward, terminated, truncated, info
