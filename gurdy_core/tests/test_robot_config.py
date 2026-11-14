"""Tests for robot.yaml loading, derived mass/inertia, joint limits, stance IK and torque headroom."""
import copy

import numpy as np
import pytest

from gurdy_core.kinematics import (IKError, joint_effective_inertias, leg_fk, leg_ik, nominal_foot,
                                   nominal_joint_angles, pd_stability_ratios, tripod_static_torques)
from gurdy_core.robot import (LEG_NAMES, SEGMENTS, Robot, box_inertia, cylinder_x_inertia,
                              load_gait, load_robot)


@pytest.fixture(scope="module")
def robot():
    return load_robot()


# ------------------------------------------------------------------ mass / structure
def test_total_mass(robot):
    expected = 1.40 + 6 * (0.10 + 0.12 + 0.06)
    assert robot.total_mass == pytest.approx(expected)
    assert 2.5 < robot.total_mass < 3.5, "agreed design target is ~3 kg"


def test_eighteen_unique_joints(robot):
    names = robot.joint_names
    assert len(names) == 18 and len(set(names)) == 18
    assert names[:3] == ["lf_hip_yaw", "lf_hip_pitch", "lf_knee"]


def test_right_side_is_mirrored(robot):
    for side in "fmr":
        left, right = robot.mounts["l" + side], robot.mounts["r" + side]
        assert right.xy[0] == left.xy[0] and right.xy[1] == -left.xy[1]
        assert right.yaw == -left.yaw


# ------------------------------------------------------------------ inertia from geometry
def test_box_and_cylinder_formulas():
    # Cube: all three equal m*a^2/6.
    assert box_inertia(6.0, 1, 1, 1) == pytest.approx([1, 1, 1])
    # Thin rod along x: I_perp -> m L^2 / 12, I_axial -> 0.
    rod = cylinder_x_inertia(12.0, 1e-6, 1.0)
    assert rod == pytest.approx([0, 1, 1], abs=1e-9)


def test_inertias_match_mujoco(robot):
    """Independent check: MuJoCo computes inertia from the same geoms + mass."""
    mujoco = pytest.importorskip("mujoco")
    bx, by, bz = (s / 2 for s in robot.body_size)
    parts = [f"<body name='body'><geom type='box' size='{bx} {by} {bz}' mass='{robot.body_mass}'/></body>"]
    for name in SEGMENTS:
        s = robot.segments[name]
        if s.shape == "box":
            geom = f"type='box' pos='{s.length/2} 0 0' size='{s.length/2} {s.size_yz[0]/2} {s.size_yz[1]/2}'"
        else:
            geom = f"type='cylinder' fromto='0 0 0 {s.length} 0 0' size='{s.radius}'"
        parts.append(f"<body name='{name}'><geom {geom} mass='{s.mass}'/></body>")
    m = mujoco.MjModel.from_xml_string(f"<mujoco><worldbody>{''.join(parts)}</worldbody></mujoco>")
    got = m.body("body").inertia
    assert np.sort(got) == pytest.approx(np.sort(robot.body_inertia()), rel=1e-6)
    for name in SEGMENTS:
        got = m.body(name).inertia
        assert np.sort(got) == pytest.approx(np.sort(robot.segments[name].shape_inertia()), rel=1e-6), name


def test_link_inertias_are_physical_rigid_bodies(robot):
    """Triangle inequality A + B >= C for every link (MuJoCo rejects violations)."""
    for name in ["body"] + list(SEGMENTS):
        I = robot.body_inertia() if name == "body" else robot.link_inertia(name)
        for k in range(3):
            assert I[k] <= I[(k + 1) % 3] + I[(k + 2) % 3] + 1e-15, name


def test_armature_is_zero_in_both_sims(robot):
    assert robot.joint_common["armature"] == 0.0
    bad = copy.deepcopy(robot.raw)
    bad["joints"]["all"]["armature"] = 0.005
    with pytest.raises(ValueError, match="armature"):
        Robot(bad)


def test_damping_gives_servo_torque_speed_line(robot):
    """effort / damping = top speed under full torque; must not exceed the joint velocity limit,
    otherwise DART's hard velocity limit (Gazebo) and MuJoCo (no limit) diverge."""
    jc = robot.joint_common
    top_speed = jc["effort"] / jc["damping"]
    assert top_speed <= jc["velocity"] * 1.0001
    assert top_speed >= 0.95 * jc["velocity"], "damping should not make the servo much slower than spec"


def test_effective_inertia_matches_simple_cases(robot):
    """Knee inertia = tibia about its end (pose independent); yaw with leg straight out is the
    largest lever arm."""
    tib = robot.segments["tibia"]
    i_knee = robot.link_inertia("tibia")[1] + tib.mass * (tib.length / 2) ** 2
    for q in ([0, 0, 0], [0.3, 0.5, 1.2], [-0.5, -0.8, 2.3]):
        assert joint_effective_inertias(robot, q)[2] == pytest.approx(i_knee)
    assert joint_effective_inertias(robot, [0, 0, 0])[0] > joint_effective_inertias(robot, [0, 0, 2.0])[0]


def test_explicit_pd_is_stable_for_every_joint(robot):
    """Sampled PD at pd_rate_hz, worst-case pose per joint: kd*dt/I and omega*dt (hard limit 2
    for both) must stay well below 2 (< 0.5 => 4x margin)."""
    r = pd_stability_ratios(robot)
    assert np.all(r["kd_dt_over_I"] < 0.5), r
    assert np.all(r["omega_dt"] < 1.0), r
    # Old gains (kd 0.3, no armature) would have been unstable on the knee: documents the decision.
    dt = 1.0 / robot.control["pd_rate_hz"]
    assert 0.30 * dt / r["I_min"][2] > 2.0


# ------------------------------------------------------------------ knee is one-sided
def test_knee_limit_is_one_sided(robot):
    lo, hi = robot.joint_limits["knee"]
    assert lo == 0.0 and hi > 0.0


def test_negative_knee_limit_rejected(robot):
    bad = copy.deepcopy(robot.raw)
    bad["joints"]["knee"]["lower"] = -0.1
    with pytest.raises(ValueError, match="one-sided"):
        Robot(bad)


def test_positive_knee_bends_tibia_down(robot):
    # Femur horizontal: knee 0 -> foot at hip height; knee > 0 -> foot below.
    assert leg_fk(robot, [0, 0, 0])[2] == pytest.approx(0.0)
    assert leg_fk(robot, [0, 0, 0.5])[2] < 0.0


def test_ik_never_returns_backward_knee(robot):
    rng = np.random.default_rng(0)
    lims = robot.joint_limits
    n_ok = 0
    for _ in range(2000):
        q = np.array([rng.uniform(*lims[t]) for t in ("hip_yaw", "hip_pitch", "knee")])
        foot = leg_fk(robot, q)
        sol = leg_ik(robot, foot, check_limits=False)
        assert sol[2] >= 0.0
        assert leg_fk(robot, sol) == pytest.approx(foot, abs=1e-9)
        n_ok += 1
    assert n_ok == 2000


def test_ik_rejects_unreachable(robot):
    with pytest.raises(IKError):
        leg_ik(robot, [1.0, 0.0, 0.0])


# ------------------------------------------------------------------ nominal stance
def test_stance_ik(robot):
    target = nominal_foot(robot)
    q = leg_ik(robot, target)                 # raises if outside limits
    assert leg_fk(robot, q) == pytest.approx(target, abs=1e-12)
    # Foot sphere touches the ground: foot centre is foot_radius above it.
    assert target[2] == pytest.approx(-(robot.stance["body_height"] - robot.foot_radius))
    # Body bottom clears the ground.
    assert robot.stance["body_height"] - robot.body_size[2] / 2 > 0.05
    angles = nominal_joint_angles(robot)
    assert len(angles) == 18 and all(angles[f"{l}_knee"] > 0 for l in LEG_NAMES)


def test_tripod_torque_headroom(robot):
    """Holding torque on 3 legs must leave >= 2x headroom to the effort limit."""
    tau = np.abs(tripod_static_torques(robot))
    effort = robot.joint_common["effort"]
    assert tau[0] < 1e-6, "vertical load should not load hip_yaw"
    assert effort / tau.max() >= 2.0, f"headroom {effort / tau.max():.2f}x, torques {tau}"


# ------------------------------------------------------------------ config consistency
def test_pd_rate_consistency_enforced(robot):
    bad = copy.deepcopy(robot.raw)
    bad["sim"]["mujoco_timestep"] = 0.001
    with pytest.raises(ValueError, match="pd_rate_hz"):
        Robot(bad)


def test_gait_groups_form_alternating_tripods():
    groups = load_gait()["tripod"]["groups"]
    assert sorted(groups[0] + groups[1]) == sorted(LEG_NAMES)
    for g in groups:
        sides = [leg[0] for leg in g]
        assert sorted(sides) in (["l", "l", "r"], ["l", "r", "r"])
        assert {leg[1] for leg in g} == {"f", "m", "r"}
