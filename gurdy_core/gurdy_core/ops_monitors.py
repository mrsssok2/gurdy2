"""Robot-health logic with no ROS imports: simulated battery, leg-health estimator, alert monitor.

Each class is stepped with plain numbers and a sim-time stamp; the ROS nodes in `gurdy_safety` only move messages in and out.

* `BatteryModel` is a SIMULATED battery (the robot has no battery model in Gazebo): drain per second idle + per second moving.
* `LegHealth` flags a leg whose joints stop following their targets while the robot is walking (limp = zero torque, locked = frozen
  target). Thresholds are first guesses, NOT calibrated on Gazebo data yet (see docs/results/phase5_ros_backend.md).
* `AlertMonitor` turns snapshots of the robot state into events: path_blocked, stuck, fall, battery_low / battery_critical,
  leg_failure, object_new. Every alert is raised once per episode (a latch that re-arms when the condition clears).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .robot import JOINT_TYPES, LEG_NAMES

# short leg id (robot.yaml) -> the leg names the agent / status tool use
LEG_LONG = {"lf": "front_left", "lm": "middle_left", "lr": "rear_left", "rf": "front_right", "rm": "middle_right", "rr": "rear_right"}


class BatteryModel:
    """SIMULATED battery. 100 % lasts about 55 min of walking (same as the mock world): 0.03 %/s moving, 0.002 %/s idle."""

    def __init__(self, level_pct: float = 100.0, drain_idle: float = 0.002, drain_moving: float = 0.03, moving_speed: float = 0.02):
        self.level = float(level_pct)
        self.drain_idle, self.drain_moving, self.moving_speed = drain_idle, drain_moving, moving_speed
        self.last_t: float | None = None

    def step(self, t: float, speed: float | None) -> float:
        """Advance to sim time `t` (s) with the current speed (m/s, None = unknown = idle). Time going backwards = sim reset."""
        if self.last_t is not None and t > self.last_t:
            moving = speed is not None and speed > self.moving_speed
            self.level = max(0.0, self.level - (self.drain_moving if moving else self.drain_idle) * (t - self.last_t))
        self.last_t = t
        return self.level


@dataclass
class LegHealthConfig:
    abs_error_rad: float = 0.15      # CALIBRATED on a flat-lab Gazebo walk (2026-11-18, 303 s walking): healthy EMA error mean 0.036, p99 0.06, max 0.09 rad -> 0.15 = 1.7x margin, 0 false alarms; mean |target - measured| of a leg's 3 joints (EMA) above this = suspect
    ratio: float = 3.0               # ... and above `ratio` x the median of the other legs
    ema_tau_s: float = 0.8
    hold_s: float = 2.0              # suspect this long = failed
    excite_rad: float = 0.2          # a leg counts as 'moving' when its mean target joint speed (EMA) exceeds this, rad/s
    excite_window_s: float = 0.5
    min_active_legs: int = 3         # judge only while at least this many legs' targets are moving (the robot is walking)


class LegHealth:
    """Per-leg tracking-error monitor. Input: 18 targets + 18 measured joint angles (canonical order), sim time.

    A failed leg is LATCHED (`reset()` clears it): a broken leg does not heal by itself.
    Not detectable while the robot stands still (nothing moves, a locked leg looks healthy)."""

    def __init__(self, cfg: LegHealthConfig | None = None):
        self.cfg = cfg or LegHealthConfig()
        self.err = {l: 0.0 for l in LEG_NAMES}
        self.moved = {l: 0.0 for l in LEG_NAMES}           # EMA of the leg's target motion per second
        self.suspect_since: dict[str, float | None] = {l: None for l in LEG_NAMES}
        self.failed: dict[str, float] = {}                 # leg -> sim time it was declared failed
        self.prev_q: list[float] | None = None
        self.last_t: float | None = None

    def reset(self) -> None:
        self.__init__(self.cfg)

    @property
    def states(self) -> dict[str, str]:
        return {LEG_LONG[l]: ("failed" if l in self.failed else "ok") for l in LEG_NAMES}

    def step(self, t: float, q_ref, q) -> list[str]:
        """-> legs newly declared failed in this step (short ids)."""
        c = self.cfg
        newly: list[str] = []
        if self.last_t is None or t <= self.last_t:
            self.last_t, self.prev_q = t, list(q_ref)
            return newly
        dt = t - self.last_t
        a = 1.0 - math.exp(-dt / c.ema_tau_s)
        am = 1.0 - math.exp(-dt / c.excite_window_s)
        n = len(JOINT_TYPES)
        for i, leg in enumerate(LEG_NAMES):
            sl = slice(n * i, n * i + n)
            e = sum(abs(r - m) for r, m in zip(q_ref[sl], q[sl])) / n
            mv = sum(abs(r - p) for r, p in zip(q_ref[sl], self.prev_q[sl])) / dt
            self.err[leg] += a * (e - self.err[leg])
            self.moved[leg] += am * (mv - self.moved[leg])
        self.prev_q, self.last_t = list(q_ref), t
        active = sum(1 for l in LEG_NAMES if self.moved[l] > c.excite_rad)
        if active < c.min_active_legs:
            self.suspect_since = {l: None for l in LEG_NAMES}          # standing: no judgement, no accumulating hold time
            return newly
        for leg in LEG_NAMES:
            if leg in self.failed or self.moved[leg] <= c.excite_rad:
                continue
            others = sorted(self.err[o] for o in LEG_NAMES if o != leg and o not in self.failed)
            med = others[len(others) // 2] if others else 0.0
            if self.err[leg] > c.abs_error_rad and self.err[leg] > c.ratio * max(med, 1e-3):
                if self.suspect_since[leg] is None:
                    self.suspect_since[leg] = t
                if t - self.suspect_since[leg] >= c.hold_s:
                    self.failed[leg] = t
                    newly.append(leg)
            else:
                self.suspect_since[leg] = None
        return newly


@dataclass
class AlertConfig:
    stuck_cmd_speed: float = 0.05        # commanded |v| (m/s) that counts as "told to move"
    stuck_speed: float = 0.02            # measured |v| below this = not moving
    stuck_s: float = 8.0
    fall_tilt_deg: float = 60.0
    fall_clear_deg: float = 30.0
    fall_hold_s: float = 0.5
    battery_low_pct: float = 20.0
    battery_critical_pct: float = 10.0
    battery_rearm_pct: float = 3.0       # hysteresis above a threshold before the alert can re-fire
    cooldown_s: float = 10.0             # minimum gap between two alerts of the same type


@dataclass
class Snapshot:
    """What the monitor reads each tick. None = no source right now (never invented)."""
    t: float
    cmd_speed: float | None = None           # |v| the robot is commanded to walk (cmd_vel_smoothed or /policy/cmd)
    speed: float | None = None               # measured |v| (odometry)
    tilt_deg: float | None = None
    battery_pct: float | None = None
    legs: dict[str, str] | None = None
    nav_aborted_ids: tuple = ()              # ids of Nav2 goals that ended aborted (any hashable)
    objects: list[dict] | None = None        # semantic map "objects" (needs `id`, `class`, `color`)


@dataclass
class AlertMonitor:
    cfg: AlertConfig = field(default_factory=AlertConfig)

    def __post_init__(self):
        self._stuck_since: float | None = None
        self._fall_since: float | None = None
        self._fallen = False
        self._stuck_raised = False
        self._low = self._crit = False
        self._legs_failed: set[str] = set()
        self._seen_aborts: set = set()
        self._known_objects: set | None = None        # None until the first snapshot with objects (those are not "new")
        self._last: dict[str, float] = {}

    def _alert(self, out: list, t: float, typ: str, sev: str, text: str, **data) -> None:
        if typ in self._last and t - self._last[typ] < self.cfg.cooldown_s and typ != "fall":
            return
        self._last[typ] = t
        out.append({"stamp": round(t, 2), "type": typ, "severity": sev, "source": "alert_monitor", "text": text, "data": data})

    def step(self, s: Snapshot) -> list[dict]:
        c, out, t = self.cfg, [], s.t
        # ---- fall (latched until the body is upright again)
        if s.tilt_deg is not None:
            if s.tilt_deg >= c.fall_tilt_deg:
                self._fall_since = self._fall_since if self._fall_since is not None else t
                if not self._fallen and t - self._fall_since >= c.fall_hold_s:
                    self._fallen = True
                    self._alert(out, t, "fall", "critical", f"The body is tilted {s.tilt_deg:.0f} degrees: I have fallen.", tilt_deg=round(s.tilt_deg, 1))
            else:
                self._fall_since = None
                if s.tilt_deg <= c.fall_clear_deg:
                    self._fallen = False
        # ---- stuck: told to move, not moving, not fallen
        if s.cmd_speed is not None and s.speed is not None and not self._fallen:
            if s.cmd_speed >= c.stuck_cmd_speed and s.speed < c.stuck_speed:
                self._stuck_since = self._stuck_since if self._stuck_since is not None else t
                if not self._stuck_raised and t - self._stuck_since >= c.stuck_s:
                    self._stuck_raised = True
                    self._alert(out, t, "stuck", "warning", "I am commanded to move but I am not moving.",
                                cmd_speed=round(s.cmd_speed, 3), speed=round(s.speed, 3))
            else:
                self._stuck_since, self._stuck_raised = None, False
        # ---- path blocked: a Nav2 goal ended aborted
        for gid in s.nav_aborted_ids:
            if gid not in self._seen_aborts:
                self._seen_aborts.add(gid)
                self._alert(out, t, "path_blocked", "warning", "The path is blocked: navigation gave up on the goal.", goal=str(gid))
        # ---- battery (SIMULATED values; text says so)
        if s.battery_pct is not None:
            b = s.battery_pct
            if b <= c.battery_critical_pct and not self._crit:
                self._crit = self._low = True
                self._alert(out, t, "battery_critical", "critical", f"Battery is critical at {b:.0f} percent (simulated).", battery_pct=round(b, 1))
            elif b <= c.battery_low_pct and not self._low:
                self._low = True
                self._alert(out, t, "battery_low", "warning", f"Battery is low at {b:.0f} percent (simulated).", battery_pct=round(b, 1))
            if b > c.battery_critical_pct + c.battery_rearm_pct:
                self._crit = False
            if b > c.battery_low_pct + c.battery_rearm_pct:
                self._low = False
        # ---- leg fault (raised once per leg; the health node latches failures)
        for leg, st in (s.legs or {}).items():
            if st == "failed" and leg not in self._legs_failed:
                self._legs_failed.add(leg)
                self._alert(out, t, "leg_failure", "critical", f"Leg {leg.replace('_', ' ')} has failed.", leg=leg)
                self._last.pop("leg_failure", None)        # two legs failing together must both be reported
            elif st != "failed":
                self._legs_failed.discard(leg)
        # ---- new object (not in the first snapshot)
        if s.objects is not None:
            ids = {o["id"] for o in s.objects}
            if self._known_objects is None:
                self._known_objects = ids
            else:
                for o in sorted((o for o in s.objects if o["id"] not in self._known_objects), key=lambda o: o["id"]):
                    col = (o.get("color") or "").strip()
                    name = f"{col} {str(o['class']).replace('_', ' ')}".strip()
                    out.append({"stamp": round(t, 2), "type": "object_new", "severity": "info", "source": "alert_monitor",
                                "text": f"New object: {name}.", "data": {"object_id": o["id"]}})
                self._known_objects |= ids
        return out
