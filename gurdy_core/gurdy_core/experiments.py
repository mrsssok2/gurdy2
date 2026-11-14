"""Simulator-independent test protocols + metrics (Gazebo and MuJoCo run the SAME code).

Step response (fixed base, legs hanging): from the nominal stance, step one joint of leg `lf`
by +amp, hold, step back. Amplitudes: 0.05 rad (linear, no torque saturation) and 0.3 rad
(torque-saturated). Metrics per step: rise time 10-90 %, overshoot %, steady error ("sag")
= q_ref - mean(q) over the last 0.3 s before the return.

Standing / chatter (free base on flat ground): hold the nominal stance, settle, then record.
Chatter metrics per joint: RMS joint velocity, peak-to-peak position, torque std; plus the
dominant velocity frequency. Sag: mean (q - q_ref) per joint type; body height from FK.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kinematics import leg_fk, leg_ik, nominal_foot
from .robot import JOINT_TYPES, LEG_NAMES, Robot

STEP_LEG = "lf"
STEP_AMPS = (0.05, 0.30)
STEP_SETTLE = 2.0      # s before the first step
STEP_HOLD = 1.0        # s at the stepped value
STEP_REST = 1.0        # s back at stance before the next step
STEADY_WINDOW = 0.3    # s at the end of a hold used for the steady error

STAND_SETTLE = 5.0     # s
STAND_RECORD = 60.0    # s

# Gazebo <-> MuJoCo parity tolerances (step: per step; stand: whole robot).
PARITY_TOL = {
    "rise_time_rel": 0.15,       # |gz - mj| / mj, or within rise_time_abs_s
    "rise_time_abs_s": 0.005,
    "overshoot_pp": 3.0,         # percentage points
    "steady_error_rad": 0.003,   # joint Coulomb-friction model difference (deadband 1 mrad)
    "body_height_m": 0.002,      # standing
}

# Chatter pass thresholds (per joint, over the recording window).
CHATTER_LIMITS = {"qd_rms": 0.01, "q_p2p": 0.005, "tau_std": 0.05}   # rad/s, rad, N·m


@dataclass(frozen=True)
class StepEvent:
    t: float               # s after the protocol start
    q_ref: np.ndarray      # 18 joint references (robot.joint_names order)


@dataclass(frozen=True)
class StepWindow:
    joint: str
    joint_index: int
    amp: float
    t_cmd: float
    t_return: float
    q_from: float
    q_to: float


def stance_vector(robot: Robot) -> np.ndarray:
    q = leg_ik(robot, nominal_foot(robot))
    return np.tile(q, len(LEG_NAMES))


def step_protocol(robot: Robot) -> tuple[list[StepEvent], list[StepWindow], float]:
    """Reference schedule, the analysis windows and the total duration."""
    q0 = stance_vector(robot)
    events = [StepEvent(0.0, q0.copy())]
    windows = []
    t = STEP_SETTLE
    for amp in STEP_AMPS:
        for jt in JOINT_TYPES:
            name = f"{STEP_LEG}_{jt}"
            i = robot.joint_names.index(name)
            q1 = q0.copy()
            q1[i] += amp
            events.append(StepEvent(t, q1))
            events.append(StepEvent(t + STEP_HOLD, q0.copy()))
            windows.append(StepWindow(name, i, amp, t, t + STEP_HOLD, q0[i], q1[i]))
            t += STEP_HOLD + STEP_REST
    return events, windows, t


def detect_onset(t, tau, t_nominal, search=0.1, frac=0.5, jump=None):
    """Time the PD torque jumps after a reference step (robust to command-delivery delay).
    Searches [t_nominal - 0.01, t_nominal + search]."""
    t = np.asarray(t)
    tau = np.asarray(tau)
    sel = np.where((t >= t_nominal - 0.01) & (t <= t_nominal + search))[0]
    if len(sel) < 2:
        return float(t_nominal)
    base = tau[sel[0]]
    thr = frac * (jump if jump is not None else np.max(np.abs(tau[sel] - base)))
    for k in sel:
        if abs(tau[k] - base) > thr:
            return float(t[k - 1]) if k > 0 else float(t[k])
    return float(t_nominal)


def step_metrics(t, q, qd, t0, t_end, q_from, q_to) -> dict:
    t = np.asarray(t)
    q = np.asarray(q)
    m = (t >= t0) & (t < t_end)
    tt, qq = t[m] - t0, q[m]
    q_ss = float(np.mean(q[m & (t >= t_end - STEADY_WINDOW)]))
    # Rise time / overshoot relative to the FINAL value (standard definition); the offset from
    # the target is reported separately as steady_error.
    x = (qq - q_from) / (q_ss - q_from)
    def first(level):
        idx = np.where(x >= level)[0]
        return float(tt[idx[0]]) if len(idx) else float("nan")
    t10, t90 = first(0.1), first(0.9)
    return {
        "rise_time_s": t90 - t10,
        "overshoot_pct": float(max(0.0, (np.max(x) - 1.0) * 100.0)),
        "steady_error_rad": float(q_to - q_ss),
        "peak_velocity": float(np.max(np.abs(np.asarray(qd)[m]))),
    }


def analyze_step(robot: Robot, t, q, qd, tau) -> list[dict]:
    """t: (N,), q/qd/tau: (N, 18) in robot.joint_names order. Returns one dict per step."""
    _, windows, _ = step_protocol(robot)
    t = np.asarray(t)
    q, qd, tau = np.asarray(q), np.asarray(qd), np.asarray(tau)
    out = []
    for w in windows:
        jump = min(robot.control["kp"] * w.amp, robot.joint_common["effort"])
        t0 = detect_onset(t, tau[:, w.joint_index], w.t_cmd, jump=jump)
        t1 = t0 + (w.t_return - w.t_cmd)
        r = step_metrics(t, q[:, w.joint_index], qd[:, w.joint_index], t0, t1, w.q_from, w.q_to)
        r.update({"joint": w.joint, "amp": w.amp, "onset_delay_s": t0 - w.t_cmd})
        out.append(r)
    return out


def body_height_from_fk(robot: Robot, q18) -> float:
    """Hip-axis height above flat ground, assuming all six feet touch it (body level)."""
    zs = []
    for k, leg in enumerate(LEG_NAMES):
        zs.append(leg_fk(robot, q18[3 * k:3 * k + 3])[2])
    return float(-np.mean(zs) + robot.foot_radius)


def analyze_stand(robot: Robot, t, q, qd, tau) -> dict:
    t = np.asarray(t)
    q, qd, tau = np.asarray(q), np.asarray(qd), np.asarray(tau)
    m = t >= t[0] + STAND_SETTLE
    q, qd, tau, tt = q[m], qd[m], tau[m], t[m]
    q_ref = stance_vector(robot)
    per = {
        "qd_rms": np.sqrt(np.mean(qd ** 2, axis=0)),
        "q_p2p": np.ptp(q, axis=0),
        "tau_std": np.std(tau, axis=0),
    }
    dt = float(np.median(np.diff(tt)))
    spec = np.abs(np.fft.rfft(qd - qd.mean(axis=0), axis=0)) ** 2
    freqs = np.fft.rfftfreq(len(tt), dt)
    tot = spec[1:].sum(axis=0)
    dom = freqs[1:][np.argmax(spec[1:], axis=0)]
    sag = (q - q_ref).mean(axis=0)
    by_type = {jt: float(np.mean(sag[[3 * k + j for k in range(len(LEG_NAMES))]]))
               for j, jt in enumerate(JOINT_TYPES)}
    worst = {k: float(v.max()) for k, v in per.items()}
    heights = [body_height_from_fk(robot, row) for row in q[:: max(1, len(q) // 200)]]
    return {
        "duration_s": float(tt[-1] - tt[0]),
        "samples": int(len(tt)),
        "sample_rate_hz": 1.0 / dt,
        "worst": worst,
        "worst_joint": {k: robot.joint_names[int(np.argmax(v))] for k, v in per.items()},
        "chatter_pass": all(worst[k] < CHATTER_LIMITS[k] for k in CHATTER_LIMITS),
        "chatter_limits": CHATTER_LIMITS,
        "dominant_qd_freq_hz_median": float(np.median(dom[tot > 0])) if np.any(tot > 0) else 0.0,
        "sag_rad_by_type": by_type,
        "body_height_fk_m": float(np.mean(heights)),
        "nominal_body_height_m": robot.stance["body_height"],
    }


def parity_check_step(gz_steps: list[dict], mj_steps: list[dict]) -> list[dict]:
    """Per-step pass/fail against PARITY_TOL."""
    tol = PARITY_TOL
    out = []
    for a, b in zip(gz_steps, mj_steps):
        dr = abs(a["rise_time_s"] - b["rise_time_s"])
        checks = {
            "rise_time": dr <= max(tol["rise_time_abs_s"], tol["rise_time_rel"] * b["rise_time_s"]),
            "overshoot": abs(a["overshoot_pct"] - b["overshoot_pct"]) <= tol["overshoot_pp"],
            "steady_error": abs(a["steady_error_rad"] - b["steady_error_rad"]) <= tol["steady_error_rad"],
        }
        out.append({"joint": a["joint"], "amp": a["amp"], **checks, "pass": all(checks.values())})
    return out


# ====================================================================== velocity tracking
# (t_start, duration, (vx, vy, wz)); 2 s stops between segments; then 20 s of random commands.
TRACK_SETTLE = 3.0
TRACK_SEGMENT = 12.0
TRACK_STOP = 2.0
TRACK_SKIP = 2.0            # s ignored at the start of each segment (acceleration, start-up)
TRACK_SEGMENTS = [
    ("forward", (0.10, 0.0, 0.0)),
    ("backward", (-0.10, 0.0, 0.0)),
    ("strafe_left", (0.0, 0.08, 0.0)),
    ("strafe_right", (0.0, -0.08, 0.0)),
    ("turn_left", (0.0, 0.0, 0.3)),
    ("turn_right", (0.0, 0.0, -0.3)),
    ("combined", (0.08, 0.04, 0.2)),
]
TRACK_RANDOM = 20.0
TRACK_TOL = 0.25            # AC: mean velocity within +-25 % of command
FALL_Z = 0.06               # m, body height below this = fall
FALL_TILT = 0.6             # rad (~34 deg) roll/pitch = fall


def tracking_protocol(seed: int = 7):
    """List of (t, cmd) changes, list of analysis windows (name, t0, t1, cmd), total duration."""
    rng = np.random.default_rng(seed)
    changes, windows = [(0.0, (0.0, 0.0, 0.0))], []
    t = TRACK_SETTLE
    for name, cmd in TRACK_SEGMENTS:
        changes.append((t, cmd))
        windows.append((name, t + TRACK_SKIP, t + TRACK_SEGMENT, cmd))
        t += TRACK_SEGMENT
        changes.append((t, (0.0, 0.0, 0.0)))
        t += TRACK_STOP
    lim = np.array([0.15, 0.10, 0.60])
    for k in range(int(TRACK_RANDOM / 2.0)):
        changes.append((t, tuple(float(x) for x in rng.uniform(-1, 1, 3) * lim)))
        t += 2.0
    changes.append((t, (0.0, 0.0, 0.0)))
    t += TRACK_STOP
    return changes, windows, t


def cmd_at(changes, t):
    cmd = changes[0][1]
    for tc, c in changes:
        if tc <= t + 1e-9:
            cmd = c
    return cmd


def analyze_tracking(t, x, y, yaw, z, roll, pitch) -> dict:
    """Body-frame mean velocities per segment from a ground-truth pose trace."""
    t, x, y = np.asarray(t), np.asarray(x), np.asarray(y)
    yaw_u = np.unwrap(np.asarray(yaw))
    _, windows, total = tracking_protocol()
    out = []
    for name, t0, t1, cmd in windows:
        m = np.where((t >= t0) & (t <= t1))[0]
        i0, i1 = m[0], m[-1]
        dt = t[i1] - t[i0]
        # body-frame velocity: rotate each world displacement by the heading at that sample
        dx, dy = np.diff(x[i0:i1 + 1]), np.diff(y[i0:i1 + 1])
        c, s = np.cos(yaw_u[i0:i1]), np.sin(yaw_u[i0:i1])
        vbx = np.sum(c * dx + s * dy) / dt
        vby = np.sum(-s * dx + c * dy) / dt
        wz = (yaw_u[i1] - yaw_u[i0]) / dt
        meas = np.array([vbx, vby, wz])
        cmd = np.array(cmd)
        k = int(np.argmax(np.abs(cmd)))                 # main axis of this segment
        err = (meas[k] - cmd[k]) / cmd[k]
        out.append({"segment": name, "cmd": cmd.tolist(), "measured": meas.tolist(),
                    "main_axis_error_pct": float(100 * err), "pass": bool(abs(err) <= TRACK_TOL)})
    fell = bool(np.any(np.asarray(z) < FALL_Z) or np.any(np.abs(roll) > FALL_TILT) or np.any(np.abs(pitch) > FALL_TILT))
    return {"segments": out, "fell": fell, "duration_s": float(t[-1] - t[0]),
            "min_body_z": float(np.min(z)), "max_tilt_rad": float(max(np.max(np.abs(roll)), np.max(np.abs(pitch)))),
            "all_pass": bool(all(o["pass"] for o in out) and not fell)}


def quat_to_rpy(qw, qx, qy, qz):
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return roll, pitch, yaw
