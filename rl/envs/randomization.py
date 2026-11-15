"""Domain randomisation for the Gurdy MuJoCo env (step 2.7).

Per episode (reset): ground/foot friction, body mass scale + off-centre payload, per-link mass scale,
motor strength (effort limit, kp), joint damping, joint Coulomb friction, action latency.
Per step (env): sensor noise on the actor's observation, random pushes.
Leg failure (Experiment B) is present but off by default (leg_failure_prob 0).

Ranges are widened from nominal by `scale` in [0, 1] (training curriculum `dr_scale`):
    lo' = nominal + scale * (lo - nominal),  hi' = nominal + scale * (hi - nominal).
Evaluation scenarios pass `fixed` values instead of sampling (everything else nominal).
"""
from __future__ import annotations

import mujoco
import numpy as np

DEFAULTS = {
    "enabled": False,
    "friction": [0.5, 1.25],             # nominal 1.0 (floor and feet set to the same value)
    "body_mass_scale": [0.85, 1.15],
    "payload_kg": [0.0, 0.3],            # extra mass on the body ...
    "payload_offset_m": 0.02,            # ... placed up to this far off-centre (x, y)
    "link_mass_scale": [0.9, 1.1],       # per leg link, independent
    "effort_scale": [0.85, 1.15],
    "kp_scale": [0.9, 1.1],
    "damping_scale": [0.8, 1.2],
    "joint_friction": [0.0, 0.04],       # N·m (nominal from robot.yaml, 0.02)
    "latency_substeps": [0, 5],          # 2 ms substeps (0-10 ms)
    "noise": {"gyro": 0.05, "gravity": 0.02, "joint_pos": 0.005, "joint_vel": 0.1},
    "push": {"interval_s": [5.0, 8.0], "speed": [0.1, 0.2]},
    "leg_failure_prob": 0.0,             # Experiment B (not used in 2.7)
}

PRIVILEGED_DIM = 17   # body lin vel 3, ang vel 3, foot contacts 6, friction, payload, mass, effort, latency


class DomainRandomizer:
    def __init__(self, model: mujoco.MjModel, cfg: dict | None, foot_geoms, floor_geom, base_body,
                 leg_bodies, vadr, nominal_friction=1.0, nominal_joint_friction=0.02):
        c = {**DEFAULTS, **(cfg or {})}
        c["noise"] = {**DEFAULTS["noise"], **(cfg or {}).get("noise", {})}
        c["push"] = {**DEFAULTS["push"], **(cfg or {}).get("push", {})}
        self.cfg = c
        self.m = model
        self.foot_geoms, self.floor_geom = np.asarray(foot_geoms), floor_geom
        self.base, self.legs, self.vadr = base_body, np.asarray(leg_bodies), np.asarray(vadr)
        self.nom_friction = nominal_friction
        self.nom_jfric = nominal_joint_friction
        # nominal copies
        self.geom_friction0 = model.geom_friction.copy()
        self.body_mass0 = model.body_mass.copy()
        self.body_inertia0 = model.body_inertia.copy()
        self.body_ipos0 = model.body_ipos.copy()
        self.forcerange0 = model.actuator_forcerange.copy()
        self.gainprm0 = model.actuator_gainprm.copy()
        self.biasprm0 = model.actuator_biasprm.copy()
        self.damping0 = model.dof_damping.copy()
        self.frictionloss0 = model.dof_frictionloss.copy()
        self.params = self.nominal()

    # ------------------------------------------------------------------ sampling
    def nominal(self) -> dict:
        return {"friction": self.nom_friction, "body_mass_scale": 1.0, "payload_kg": 0.0,
                "payload_offset": np.zeros(2), "link_mass_scale": np.ones(len(self.legs)),
                "effort_scale": 1.0, "kp_scale": 1.0, "damping_scale": 1.0,
                "joint_friction": self.nom_jfric, "latency_substeps": 0, "noise_scale": 0.0,
                "push": False, "push_scale": 0.0, "leg_failure": -1}

    def _range(self, key, nominal, scale, rng):
        lo, hi = self.cfg[key]
        return rng.uniform(nominal + scale * (lo - nominal), nominal + scale * (hi - nominal))

    def sample(self, rng, scale: float, fixed: dict | None = None) -> dict:
        p = self.nominal()
        if self.cfg["enabled"] and scale > 0:
            p["friction"] = self._range("friction", self.nom_friction, scale, rng)
            p["body_mass_scale"] = self._range("body_mass_scale", 1.0, scale, rng)
            p["payload_kg"] = self._range("payload_kg", 0.0, scale, rng)
            p["payload_offset"] = rng.uniform(-1, 1, 2) * self.cfg["payload_offset_m"] * scale
            lo, hi = self.cfg["link_mass_scale"]
            p["link_mass_scale"] = rng.uniform(1 + scale * (lo - 1), 1 + scale * (hi - 1), len(self.legs))
            p["effort_scale"] = self._range("effort_scale", 1.0, scale, rng)
            p["kp_scale"] = self._range("kp_scale", 1.0, scale, rng)
            p["damping_scale"] = self._range("damping_scale", 1.0, scale, rng)
            p["joint_friction"] = self._range("joint_friction", self.nom_jfric, scale, rng)
            lat_hi = int(round(scale * self.cfg["latency_substeps"][1]))
            p["latency_substeps"] = int(rng.integers(self.cfg["latency_substeps"][0], lat_hi + 1))
            p["noise_scale"] = scale
            p["push"] = True
            p["push_scale"] = scale
            if rng.random() < self.cfg["leg_failure_prob"]:
                p["leg_failure"] = int(rng.integers(0, 6))
        for k, v in (fixed or {}).items():
            p[k] = np.asarray(v, dtype=float) if k == "payload_offset" else v
        if (fixed or {}).get("push") and "push_scale" not in fixed:
            p["push_scale"] = 1.0
        self.params = p
        return p

    # ------------------------------------------------------------------ applying
    def apply(self, data: mujoco.MjData) -> None:
        m, p = self.m, self.params
        m.geom_friction[:] = self.geom_friction0
        for g in list(self.foot_geoms) + [self.floor_geom]:
            m.geom_friction[g, 0] = p["friction"]
        m.body_mass[:] = self.body_mass0
        m.body_inertia[:] = self.body_inertia0
        m.body_ipos[:] = self.body_ipos0
        mb = self.body_mass0[self.base] * p["body_mass_scale"]
        pay = p["payload_kg"]
        m.body_mass[self.base] = mb + pay
        m.body_inertia[self.base] = self.body_inertia0[self.base] * p["body_mass_scale"]
        if pay > 0:
            off = np.array([*p["payload_offset"], 0.0])
            m.body_ipos[self.base] = (mb * self.body_ipos0[self.base] + pay * off) / (mb + pay)
        for b, s in zip(self.legs, p["link_mass_scale"]):
            m.body_mass[b] = self.body_mass0[b] * s
            m.body_inertia[b] = self.body_inertia0[b] * s
        m.actuator_forcerange[:] = self.forcerange0 * p["effort_scale"]
        m.actuator_gainprm[:, 0] = self.gainprm0[:, 0] * p["kp_scale"]
        m.actuator_biasprm[:, 1] = self.biasprm0[:, 1] * p["kp_scale"]
        m.dof_damping[:] = self.damping0
        m.dof_damping[self.vadr] = self.damping0[self.vadr] * p["damping_scale"]
        m.dof_frictionloss[:] = self.frictionloss0
        m.dof_frictionloss[self.vadr] = p["joint_friction"]
        if p["leg_failure"] >= 0:                    # Experiment B: limp leg = no motor torque
            sl = slice(3 * p["leg_failure"], 3 * p["leg_failure"] + 3)
            m.actuator_forcerange[sl] = 0.0
        mujoco.mj_setConst(m, data)

    def privileged(self) -> np.ndarray:
        """Episode constants for the critic (appended to body velocity and contacts by the env)."""
        p = self.params
        return np.array([p["friction"], p["payload_kg"], p["body_mass_scale"], p["effort_scale"],
                         p["latency_substeps"] * 0.002])
