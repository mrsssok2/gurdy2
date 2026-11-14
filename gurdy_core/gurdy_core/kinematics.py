"""Single-leg forward/inverse kinematics and static joint torques.

Leg frame: origin on the hip-yaw axis at body mid-plane height, x pointing outward along the
mount yaw, z up. Joint vector q = (hip_yaw, hip_pitch, knee) with the robot.yaml conventions:
hip_pitch + = femur up, knee + = tibia bends down (knee >= 0 always; never bends backwards).
"""
from __future__ import annotations

import numpy as np

from .robot import LEG_NAMES, Robot


class IKError(ValueError):
    """Target unreachable or outside joint limits."""


def leg_fk(robot: Robot, q) -> np.ndarray:
    """Foot-centre position in the leg frame."""
    l1, l2, l3 = robot.lengths
    yaw, pitch, knee = q
    r = l1 + l2 * np.cos(pitch) + l3 * np.cos(pitch - knee)
    z = l2 * np.sin(pitch) + l3 * np.sin(pitch - knee)
    return np.array([r * np.cos(yaw), r * np.sin(yaw), z])


def leg_ik(robot: Robot, foot, check_limits: bool = True) -> np.ndarray:
    """Joint angles placing the foot centre at `foot` (leg frame).

    Always returns the knee-up branch, i.e. knee in [0, pi]: the backward-bending solution
    is never produced.
    """
    l1, l2, l3 = robot.lengths
    x, y, z = foot
    yaw = np.arctan2(y, x)
    r = np.hypot(x, y) - l1
    d2 = r * r + z * z
    d = np.sqrt(d2)
    if d > l2 + l3 or d < abs(l2 - l3):
        raise IKError(f"foot {foot} out of reach (distance {d:.4f} m)")
    cos_inner = np.clip((l2 * l2 + l3 * l3 - d2) / (2 * l2 * l3), -1.0, 1.0)
    knee = np.pi - np.arccos(cos_inner)                    # >= 0 by construction
    cos_a = np.clip((l2 * l2 + d2 - l3 * l3) / (2 * l2 * d), -1.0, 1.0)
    pitch = np.arctan2(z, r) + np.arccos(cos_a)
    q = np.array([yaw, pitch, knee])
    if check_limits:
        for val, name in zip(q, ("hip_yaw", "hip_pitch", "knee")):
            lo, hi = robot.joint_limits[name]
            if not lo - 1e-9 <= val <= hi + 1e-9:
                raise IKError(f"{name}={val:.3f} outside limits [{lo}, {hi}] for foot {foot}")
    return q


def nominal_foot(robot: Robot) -> np.ndarray:
    """Nominal stance foot centre in the leg frame (foot sphere touching flat ground)."""
    s = robot.stance
    return np.array([s["foot_reach"], 0.0, -(s["body_height"] - robot.foot_radius)])


def nominal_joint_angles(robot: Robot) -> dict[str, float]:
    """All 18 joints in the nominal stance, keyed by joint name."""
    q = leg_ik(robot, nominal_foot(robot))
    return {f"{leg}_{t}": float(v) for leg in LEG_NAMES
            for t, v in zip(("hip_yaw", "hip_pitch", "knee"), q)}


def leg_jacobian(robot: Robot, q, eps: float = 1e-7) -> np.ndarray:
    """d(foot)/dq, 3x3 (central differences; cheap and exact enough for statics)."""
    q = np.asarray(q, dtype=float)
    J = np.zeros((3, 3))
    for i in range(3):
        dq = np.zeros(3)
        dq[i] = eps
        J[:, i] = (leg_fk(robot, q + dq) - leg_fk(robot, q - dq)) / (2 * eps)
    return J


def segment_coms(robot: Robot, q) -> list[tuple[float, np.ndarray]]:
    """(mass, COM position in leg frame) of coxa, femur, tibia."""
    l1, l2, l3 = robot.lengths
    yaw, pitch, knee = q
    u = np.array([np.cos(yaw), np.sin(yaw)])

    def pt(r, z):
        return np.array([r * u[0], r * u[1], z])

    knee_r, knee_z = l1 + l2 * np.cos(pitch), l2 * np.sin(pitch)
    segs = robot.segments
    return [
        (segs["coxa"].mass, pt(l1 / 2, 0.0)),
        (segs["femur"].mass, pt(l1 + l2 / 2 * np.cos(pitch), l2 / 2 * np.sin(pitch))),
        (segs["tibia"].mass, pt(knee_r + l3 / 2 * np.cos(pitch - knee),
                                knee_z + l3 / 2 * np.sin(pitch - knee))),
    ]


def static_leg_torques(robot: Robot, q, foot_force) -> np.ndarray:
    """Joint torques the motors must supply to hold pose q against a ground reaction force on
    the foot (leg frame, N) plus the leg's own weight. Returned with motor sign convention."""
    g = robot.sim["gravity"]
    tau_ground = leg_jacobian(robot, q).T @ np.asarray(foot_force, dtype=float)
    # Gravity on each segment: generalized force = J_com^T * (0, 0, -m g); numerical J_com.
    tau_grav = np.zeros(3)
    eps = 1e-7
    for i in range(3):
        dq = np.zeros(3)
        dq[i] = eps
        plus = segment_coms(robot, np.asarray(q) + dq)
        minus = segment_coms(robot, np.asarray(q) - dq)
        for (m, cp), (_, cm) in zip(plus, minus):
            tau_grav[i] += -m * g * (cp[2] - cm[2]) / (2 * eps)
    # Motors must cancel external generalized forces.
    return -(tau_ground + tau_grav)


def tripod_static_torques(robot: Robot) -> np.ndarray:
    """Per-joint holding torque in the nominal stance with only 3 feet on the ground."""
    weight = robot.total_mass * robot.sim["gravity"]
    return static_leg_torques(robot, leg_ik(robot, nominal_foot(robot)), (0.0, 0.0, weight / 3.0))
