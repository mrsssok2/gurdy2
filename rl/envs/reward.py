"""Reward for Gurdy locomotion (step 2.3). Pure NumPy: every term is unit-testable without MuJoCo.

Tracking (dominant) — ONE joint term, so standing can't farm part of it:
    r_track = exp( -||v_xy - c_xy||^2 / s_v^2  -  (w_z - c_w)^2 / s_w^2 )
    s_v = max(kappa * ||c_xy||, sigma_v_floor),  s_w = max(kappa * |c_w|, sigma_w_floor)
The tracked velocity is the body velocity low-pass filtered (EMA, track_filter_s = 0.2 s, about a
quarter stride): the command is a mean velocity, and a tripod stride naturally surges and nearly
stops at every tripod switch (measured: forward 0.013-0.137 m/s around a 0.096 mean, std 0.033;
EMA std 0.008). Unfiltered, half-speed walking scored almost as well as full-speed walking.
Command-relative sigmas make "standing under a move command" score exp(-1/kappa^2) for EVERY
command size (0.21 at kappa 0.8, 0.018 at 0.5); floors keep zero / pure-turn commands well defined
(user decision 2026-11-14: 0.02 m/s, 0.05 rad/s). kappa is annealed 0.8 -> 0.5 during training.

All other terms are small shaping penalties/bonuses; weights live in rl/configs/*.yaml (`reward`).

Combining (fix after the first smoke run, where noise-driven penalties made every step ~-1.2 so
that falling early beat surviving):
  * PENALTY_TERMS are multiplied by penalty_scale (curriculum 0.1 -> 1.0 during training);
  * only_positive: r = max(0, sum of all terms except fall) + fall term — the fall penalty is a
    separate terminal term, so a fall always loses future reward AND costs |fall|.
"""
from __future__ import annotations

import numpy as np

DEFAULTS = {
    "kappa": 0.8,
    "sigma_v_floor": 0.02,     # m/s
    "sigma_w_floor": 0.05,     # rad/s
    "air_time_target": 0.2,    # s; touchdowns after longer swings are rewarded (only when moving)
    "track_filter_s": 0.2,     # s; EMA time constant of the tracked velocity (0 = instantaneous)
    "only_positive": True,     # clip the per-step sum (except the fall term) at >= 0
    "penalty_scale": 1.0,      # multiplier on PENALTY_TERMS (training curriculum sets it)
    "weights": {
        "track": 1.5,
        "alive": 0.05,
        "fall": -10.0,
        "tilt": -1.0,
        "vz": -1.0,
        "ang_xy": -0.05,
        "energy": -2e-3,
        "torque": -5e-4,
        "joint_acc": -2.5e-7,
        "action_rate": -0.01,
        "foot_slip": -1.0,
        "air_time": 0.5,
        "stand_still": -0.1,
    },
}

_ZERO = 1e-6
PENALTY_TERMS = ("tilt", "vz", "ang_xy", "energy", "torque", "joint_acc", "action_rate", "foot_slip")


def is_moving(command) -> bool:
    return bool(np.any(np.abs(command) > _ZERO))


def tracking(v_xy, w_z, command, kappa, sigma_v_floor, sigma_w_floor) -> float:
    c = np.asarray(command, dtype=float)
    s_v = max(kappa * float(np.linalg.norm(c[:2])), sigma_v_floor)
    s_w = max(kappa * abs(float(c[2])), sigma_w_floor)
    e = float(np.sum((np.asarray(v_xy) - c[:2]) ** 2)) / s_v ** 2 + (float(w_z) - c[2]) ** 2 / s_w ** 2
    return float(np.exp(-e))


def ema_update(filtered, value, dt, tau):
    """One step of an exponential moving average; tau <= 0 returns the raw value."""
    if tau <= 0:
        return np.asarray(value, dtype=float).copy()
    return filtered + (np.asarray(value) - filtered) * min(1.0, dt / tau)


class FootAirTimer:
    """Per-foot air time; on touchdown returns (air_time - target) for feet that just landed."""

    def __init__(self, n_feet: int, dt: float):
        self.dt = dt
        self.air = np.zeros(n_feet)
        self.prev_contact = np.ones(n_feet, dtype=bool)

    def reset(self):
        self.air[:] = 0.0
        self.prev_contact[:] = True

    def update(self, contact, target: float) -> float:
        contact = np.asarray(contact, dtype=bool)
        touchdown = contact & ~self.prev_contact
        bonus = float(np.sum((self.air[touchdown] + self.dt) - target))   # this step counted as air
        self.air[~contact] += self.dt
        self.air[contact] = 0.0
        self.prev_contact = contact.copy()
        return bonus


def terms(*, v_body, w_body, track_vel, gravity_body, command, kappa, cfg, joint_vel, joint_vel_prev, dt,
          torque, action, action_prev, foot_contact, foot_vel_xy, air_bonus, q, q_stance,
          fallen) -> dict:
    """Unweighted term values for one policy step."""
    moving = is_moving(command)
    return {
        "track": tracking(track_vel[:2], track_vel[2], command, kappa, cfg["sigma_v_floor"], cfg["sigma_w_floor"]),
        "alive": 1.0,
        "fall": 1.0 if fallen else 0.0,
        "tilt": float(gravity_body[0] ** 2 + gravity_body[1] ** 2),
        "vz": float(v_body[2] ** 2),
        "ang_xy": float(w_body[0] ** 2 + w_body[1] ** 2),
        "energy": float(np.sum(np.abs(torque * joint_vel))),
        "torque": float(np.sum(torque ** 2)),
        "joint_acc": float(np.sum(((joint_vel - joint_vel_prev) / dt) ** 2)),
        "action_rate": float(np.sum((action - action_prev) ** 2)),
        "foot_slip": float(np.sum(np.sum(np.asarray(foot_vel_xy) ** 2, axis=1)[np.asarray(foot_contact, bool)])),
        "air_time": float(air_bonus) if moving else 0.0,
        "stand_still": 0.0 if moving else float(np.sum(np.abs(q - q_stance))),
    }


def weighted(term_values: dict, weights: dict) -> dict:
    return {k: weights[k] * v for k, v in term_values.items()}


def combine(weighted_terms: dict, penalty_scale: float, only_positive: bool):
    """Apply the penalty curriculum and the non-negative clip.
    Returns (reward, applied_terms, clipped)."""
    applied = {k: (v * penalty_scale if k in PENALTY_TERMS else v) for k, v in weighted_terms.items()}
    body = sum(v for k, v in applied.items() if k != "fall")
    clipped = bool(only_positive and body < 0.0)
    reward = (max(0.0, body) if only_positive else body) + applied.get("fall", 0.0)
    return float(reward), applied, clipped
