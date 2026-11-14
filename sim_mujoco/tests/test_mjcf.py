"""MJCF <-> robot.yaml/gurdy_core parity (the URDF has the same checks in gurdy_description)."""
import mujoco
import numpy as np
import pytest

from gurdy_core.kinematics import joint_effective_inertias, leg_fk
from gurdy_core.robot import JOINT_TYPES, LEG_NAMES, load_robot
from sim_mujoco.mjcf import build_mjcf


@pytest.fixture(scope="module")
def robot():
    return load_robot()


@pytest.fixture(scope="module", params=[False, True], ids=["free", "fixed"])
def model(request, robot):
    return mujoco.MjModel.from_xml_string(build_mjcf(robot, fixed_base=request.param))


def test_total_mass(model, robot):
    assert model.body_mass.sum() == pytest.approx(robot.total_mass, rel=1e-9)


def test_joints_limits_dynamics(model, robot):
    jc = robot.joint_common
    for name in robot.joint_names:
        j = model.joint(name)
        t = name.split("_", 1)[1]
        assert tuple(j.range) == pytest.approx(robot.joint_limits[t])
        assert model.dof_damping[j.dofadr[0]] == pytest.approx(jc["damping"])
        assert model.dof_frictionloss[j.dofadr[0]] == pytest.approx(jc["friction"])
        assert model.dof_armature[j.dofadr[0]] == 0.0


def test_actuators_are_the_pd_law(model, robot):
    c = robot.control
    assert model.nu == 18
    for i, name in enumerate(robot.joint_names):
        assert model.actuator(i).name == name
        assert model.actuator_gainprm[i, 0] == pytest.approx(c["kp"])
        assert model.actuator_biasprm[i, :3] == pytest.approx([0, -c["kp"], -c["kd"]])
        assert model.actuator_forcerange[i] == pytest.approx([-robot.joint_common["effort"], robot.joint_common["effort"]])
    assert model.opt.timestep == pytest.approx(1.0 / c["pd_rate_hz"])


def test_fk_and_inertia_match_gurdy_core(robot):
    m = mujoco.MjModel.from_xml_string(build_mjcf(robot, fixed_base=True, base_z=0.0))
    d = mujoco.MjData(m)
    rng = np.random.default_rng(3)
    M = np.zeros((m.nv, m.nv))
    for _ in range(20):
        qs = {leg: np.array([rng.uniform(*robot.joint_limits[t]) for t in JOINT_TYPES]) for leg in LEG_NAMES}
        for leg, q in qs.items():
            for t, v in zip(JOINT_TYPES, q):
                d.qpos[m.joint(f"{leg}_{t}").qposadr[0]] = v
        mujoco.mj_forward(m, d)
        mujoco.mj_fullM(m, M, d.qM)
        for leg, q in qs.items():
            mt = robot.mounts[leg]
            c, s = np.cos(mt.yaw), np.sin(mt.yaw)
            expected = np.array([mt.xy[0], mt.xy[1], 0]) + np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ leg_fk(robot, q)
            assert d.site(f"{leg}_foot").xpos == pytest.approx(expected, abs=1e-9)
            dofs = [m.joint(f"{leg}_{t}").dofadr[0] for t in JOINT_TYPES]
            assert np.diag(M)[dofs] == pytest.approx(joint_effective_inertias(robot, q), rel=1e-6)


def test_no_self_collision(robot):
    m = mujoco.MjModel.from_xml_string(build_mjcf(robot))
    floor = m.geom("floor").id
    for g in range(m.ngeom):
        if g == floor:
            continue
        for h in range(m.ngeom):
            if h in (g, floor):
                continue
            ok = (m.geom_contype[g] & m.geom_conaffinity[h]) or (m.geom_contype[h] & m.geom_conaffinity[g])
            assert not ok, (m.geom(g).name, m.geom(h).name)
