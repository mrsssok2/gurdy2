"""Load robot.yaml (the single source of truth) and derive everything computed from it.

Pure Python (numpy + PyYAML), no ROS imports. Used by the xacro helpers, the MJCF generator,
IK and the gait. Inertias are always computed from shapes here — never typed by hand.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

LEG_NAMES = ("lf", "lm", "lr", "rf", "rm", "rr")
JOINT_TYPES = ("hip_yaw", "hip_pitch", "knee")
SEGMENTS = ("coxa", "femur", "tibia")
# Joint that drives each segment, and that joint's axis expressed in the segment's own frame
# (segments extend along local +x). hip_pitch "+ = up" -> axis -y; knee "+ = down" -> axis +y.
SEGMENT_JOINT = {"coxa": "hip_yaw", "femur": "hip_pitch", "tibia": "knee"}
JOINT_AXIS = {"hip_yaw": (0.0, 0.0, 1.0), "hip_pitch": (0.0, -1.0, 0.0), "knee": (0.0, 1.0, 0.0)}
_MOUNT_OF = {"f": "front", "m": "middle", "r": "rear"}


def default_robot_yaml() -> Path:
    """Find robot.yaml: $GURDY_ROBOT_YAML, then the repo checkout, then the installed ROS share."""
    env = os.environ.get("GURDY_ROBOT_YAML")
    if env:
        return Path(env)
    repo = Path(__file__).resolve().parents[2]
    p = repo / "ros2_ws/src/gurdy_description/config/robot.yaml"
    if p.exists():
        return p
    try:  # installed via colcon
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("gurdy_description")) / "config/robot.yaml"
    except Exception as e:  # pragma: no cover
        raise FileNotFoundError("robot.yaml not found; set GURDY_ROBOT_YAML") from e


def default_gait_yaml() -> Path:
    return Path(__file__).resolve().parent / "config" / "gait.yaml"


# ---------------------------------------------------------------- inertia from shapes
def box_inertia(mass: float, sx: float, sy: float, sz: float) -> np.ndarray:
    """Diagonal inertia (about COM) of a solid box with full side lengths sx, sy, sz."""
    return mass / 12.0 * np.array([sy**2 + sz**2, sx**2 + sz**2, sx**2 + sy**2])


def cylinder_x_inertia(mass: float, radius: float, length: float) -> np.ndarray:
    """Diagonal inertia (about COM) of a solid cylinder whose axis is local x."""
    i_perp = mass * (3 * radius**2 + length**2) / 12.0
    return np.array([0.5 * mass * radius**2, i_perp, i_perp])


@dataclass(frozen=True)
class Segment:
    name: str
    length: float
    shape: str
    mass: float
    size_yz: tuple[float, float] | None = None
    radius: float | None = None

    @property
    def com(self) -> np.ndarray:
        return np.array([self.length / 2.0, 0.0, 0.0])

    def shape_inertia(self) -> np.ndarray:
        if self.shape == "box":
            return box_inertia(self.mass, self.length, *self.size_yz)
        if self.shape == "cylinder":
            return cylinder_x_inertia(self.mass, self.radius, self.length)
        raise ValueError(f"unknown shape {self.shape!r} for segment {self.name}")


@dataclass(frozen=True)
class Mount:
    leg: str
    xy: tuple[float, float]
    yaw: float


class Robot:
    """Parsed robot.yaml with derived quantities."""

    def __init__(self, data: dict, source: Path | None = None):
        self.raw = data
        self.source = source
        self.name = data["name"]
        b = data["body"]
        self.body_size = tuple(b["size"])
        self.body_mass = float(b["mass"])
        legs = data["legs"]
        self.segments = {
            n: Segment(name=n, length=s["length"], shape=s["shape"], mass=s["mass"],
                       size_yz=tuple(s["size_yz"]) if "size_yz" in s else None,
                       radius=s.get("radius"))
            for n, s in legs["segments"].items()
        }
        self.foot_radius = float(legs["foot"]["radius"])
        self.foot_friction = float(legs["foot"]["friction"])
        self.mounts = {}
        for leg in LEG_NAMES:
            m = legs["mounts"][_MOUNT_OF[leg[1]]]
            sign = 1.0 if leg[0] == "l" else -1.0
            self.mounts[leg] = Mount(leg, (m["xy"][0], sign * m["xy"][1]), sign * m["yaw"])
        j = data["joints"]
        self.joint_common = dict(j["all"])
        self.joint_limits = {t: (float(j[t]["lower"]), float(j[t]["upper"])) for t in JOINT_TYPES}
        self.control = dict(data["control"])
        self.stance = dict(data["nominal_stance"])
        self.command_limits = dict(data["command_limits"])
        self.sensors = data["sensors"]
        self.sim = dict(data["sim"])
        self._validate()

    # ------------------------------------------------------------ convenience
    @property
    def lengths(self) -> tuple[float, float, float]:
        return tuple(self.segments[s].length for s in SEGMENTS)

    @property
    def joint_names(self) -> list[str]:
        """All 18 joints, in the canonical order used everywhere (controllers, RL obs/actions)."""
        return [f"{leg}_{t}" for leg in LEG_NAMES for t in JOINT_TYPES]

    @property
    def leg_mass(self) -> float:
        return sum(s.mass for s in self.segments.values())

    @property
    def total_mass(self) -> float:
        return self.body_mass + len(LEG_NAMES) * self.leg_mass

    def body_inertia(self) -> np.ndarray:
        return box_inertia(self.body_mass, *self.body_size)

    def link_inertia(self, segment: str) -> np.ndarray:
        """Diagonal COM inertia of a leg segment (same value in URDF and MJCF).

        No armature is folded in: Gazebo cannot model joint armature, so it is 0 in both sims
        (see robot.yaml); servo back-EMF is modelled as joint damping instead.
        """
        return self.segments[segment].shape_inertia()

    # ------------------------------------------------------------ checks
    def _validate(self) -> None:
        lo, hi = self.joint_limits["knee"]
        if lo < 0.0:
            raise ValueError(f"knee lower limit must be >= 0 (one-sided, no backward bend); got {lo}")
        for t, (lo, hi) in self.joint_limits.items():
            if not lo < hi:
                raise ValueError(f"{t}: lower {lo} must be < upper {hi}")
        for s in self.segments.values():
            if s.mass <= 0 or s.length <= 0:
                raise ValueError(f"segment {s.name} needs positive mass and length")
        if float(self.joint_common.get("armature", 0.0)) != 0.0:
            raise ValueError("joints.all.armature must be 0: Gazebo cannot model it (sim parity)")
        pd_dt = 1.0 / self.control["pd_rate_hz"]
        if not np.isclose(self.sim["mujoco_timestep"], pd_dt):
            raise ValueError("sim.mujoco_timestep must equal 1/control.pd_rate_hz (same PD rate in both sims)")
        ratio = pd_dt / self.sim["gazebo_step"]
        if not np.isclose(ratio, round(ratio)):
            raise ValueError("1/pd_rate_hz must be a whole multiple of sim.gazebo_step")


@lru_cache(maxsize=None)
def _load_cached(path: str) -> Robot:
    with open(path) as f:
        return Robot(yaml.safe_load(f), Path(path))


def load_robot(path: str | os.PathLike | None = None) -> Robot:
    return _load_cached(str(Path(path or default_robot_yaml()).resolve()))


def load_gait(path: str | os.PathLike | None = None) -> dict:
    with open(path or default_gait_yaml()) as f:
        return yaml.safe_load(f)
