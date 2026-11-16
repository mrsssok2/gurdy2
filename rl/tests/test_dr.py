"""Step 2.7: domain randomisation, latency, noise, pushes, privileged obs, asymmetric policy, transfer."""
import mujoco
import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from envs import reward as R
from envs.gurdy_env import GurdyEnv, load_config
from envs.randomization import DEFAULTS as DR, PRIVILEGED_DIM
from gurdy_core.rl_obs import FRAME_DIM, frame_slices
from policies import AsymmetricActorCriticPolicy, transfer_from_flat

BASE = load_config("configs/flat.yaml")


def env_with(dr=None, fixed=None, obs_mode="flat", **kw):
    cfg = {**BASE["env"], "obs_mode": obs_mode, "domain_randomization": dr or {}, "dr_fixed": fixed, **kw}
    return GurdyEnv(cfg, reward_config=BASE["reward"])


def test_disabled_is_nominal_and_no_accumulation():
    e = env_with({"enabled": True})
    m = e.model
    nominal = (m.body_mass.copy(), m.geom_friction.copy(), m.actuator_forcerange.copy(), m.dof_damping.copy())
    e.dr_scale = 1.0
    for s in range(5):
        e.reset(seed=s)
    e.dr_scale = 0.0
    e.reset(seed=9)                                   # scale 0 -> back to nominal exactly
    for a, b in zip(nominal, (m.body_mass, m.geom_friction, m.actuator_forcerange, m.dof_damping)):
        assert np.allclose(a, b)


def test_sampled_params_within_ranges_and_applied():
    e = env_with({"enabled": True})
    e.dr_scale = 1.0
    m = e.model
    base_mass0 = e.dr.body_mass0[m.body("base_link").id]
    for s in range(30):
        _, info = e.reset(seed=s)
        p = info["dr"]
        assert DR["friction"][0] <= p["friction"] <= DR["friction"][1]
        assert DR["payload_kg"][0] <= p["payload_kg"] <= DR["payload_kg"][1]
        assert 0 <= p["latency_substeps"] <= DR["latency_substeps"][1]
        assert m.geom_friction[e.foot_geoms[0], 0] == pytest.approx(p["friction"])
        assert m.geom_friction[e.floor_geom, 0] == pytest.approx(p["friction"])
        assert m.body_mass[m.body("base_link").id] == pytest.approx(base_mass0 * p["body_mass_scale"] + p["payload_kg"])
        assert m.actuator_forcerange[0, 1] == pytest.approx(e.robot.joint_common["effort"] * p["effort_scale"])
        kp = e.robot.control["kp"] * p["kp_scale"]
        assert m.actuator_gainprm[0, 0] == pytest.approx(kp) and m.actuator_biasprm[0, 1] == pytest.approx(-kp)
        assert m.dof_frictionloss[e.vadr[0]] == pytest.approx(p["joint_friction"])


def test_latency_holds_previous_targets():
    def first_step_qvel(lat):
        e = env_with(fixed={"latency_substeps": lat})
        e.reset(seed=0)
        e.step(np.full(18, 0.8))
        return e.data.qvel[e.vadr].copy()
    q0, q5 = first_step_qvel(0), first_step_qvel(5)
    assert not np.allclose(q0, q5)
    # latency 0 must equal the plain env (regression)
    e = env_with()
    e.reset(seed=0)
    e.step(np.full(18, 0.8))
    assert np.allclose(e.data.qvel[e.vadr], q0)


def test_noise_only_in_actor_observation():
    e = env_with(fixed={"noise_scale": 1.0}, obs_mode="asym")
    obs, _ = e.reset(seed=1)
    for _ in range(20):
        obs, _, _, _, info = e.step(np.zeros(18))
    clean = e.dr.params["noise_scale"]
    e.dr.params["noise_scale"] = 0.0
    clean_frame = e._frame()
    e.dr.params["noise_scale"] = clean
    S = frame_slices()
    newest = obs["policy"][-FRAME_DIM:]
    assert not np.allclose(newest[S["joint_pos"]], clean_frame[S["joint_pos"]], atol=1e-4)
    priv = obs["privileged"]
    assert priv.shape == (PRIVILEGED_DIM,)
    assert priv[:3] == pytest.approx(info["body_vel"], abs=1e-5)           # clean true velocity
    assert np.array_equal(priv[6:12] > 0.5, info["foot_contact"])


def test_push_changes_velocity():
    e = env_with(fixed={"push": True})
    e.reset(seed=2)
    e.next_push = 10
    v = []
    for _ in range(12):
        e.step(np.zeros(18))
        v.append(np.linalg.norm(e.data.qvel[0:2]))
    assert v[9] > v[8] + 0.05                      # kick of 0.1-0.2 m/s at step 10


def test_leg_failure_limp_zeroes_that_legs_torque_from_the_failure_step():
    e = env_with(fixed={"leg_failure": 1, "failure_type": "limp", "failure_step": 5}, obs_mode="asym",
                 failure_privileged=True)
    obs, _ = e.reset(seed=0, options={"command": (0.1, 0, 0)})
    assert np.all(e.model.actuator_forcerange[3:6] != 0)
    for k in range(10):
        obs, *_, info = e.step(np.zeros(18))
        if k < 5:
            assert not info["leg_failed"] and obs["privileged"][17:].sum() == 0
    assert info["leg_failed"]
    assert np.all(e.model.actuator_forcerange[3:6] == 0) and np.all(e.model.actuator_forcerange[0:3] != 0)
    assert np.allclose(np.abs(e.data.actuator_force[3:6]), 0)
    f = obs["privileged"][17:]
    assert f.shape == (8,) and f[1] == 1 and f[6] == 1 and f.sum() == 2
    e.reset(seed=1)                                   # next episode starts healthy again (forcerange restored)
    assert np.all(e.model.actuator_forcerange[3:6] != 0)


def test_leg_failure_locked_holds_the_failure_angles_and_actor_obs_unchanged():
    e = env_with(fixed={"leg_failure": 2, "failure_type": "locked", "failure_step": 20})
    e.reset(seed=0, options={"command": (0.1, 0, 0)})
    for _ in range(19):
        e.step(np.zeros(18))
    q0 = e.data.qpos[e.qadr][6:9].copy()
    for _ in range(40):
        obs, *_ = e.step(np.random.uniform(-1, 1, 18))
    assert obs.shape == (195,)                        # actor sees no failure flag
    assert np.allclose(e.q_lock, e.data.qpos[e.qadr][6:9], atol=0.2) and e.q_lock is not None
    assert np.allclose(e.data.ctrl[6:9], e.q_lock)
    assert np.all(np.abs(e.data.qpos[e.qadr][6:9] - q0) < 0.2)


def test_failure_sampling_probability_types_and_time():
    e = env_with({"enabled": True, "leg_failure_prob": 0.5, "failure_time_s": [2.0, 6.0]})
    ps = []
    for s in range(200):
        e.reset(seed=s)
        ps.append(e.dr.params)
    failed = [p for p in ps if p["leg_failure"] >= 0]
    assert 70 < len(failed) < 130
    assert {p["failure_type"] for p in failed} == {"limp", "locked"} and {p["leg_failure"] for p in failed} == set(range(6))
    assert all(100 <= p["failure_step"] <= 300 for p in failed)


def test_asym_env_check_and_spaces():
    e = env_with({"enabled": True}, obs_mode="asym")
    check_env(e, skip_render_check=True)
    assert e.observation_space["privileged"].shape == (PRIVILEGED_DIM,)


def test_reward_second_difference_and_curriculum():
    t = R.terms(v_body=np.zeros(3), w_body=np.zeros(3), track_vel=np.zeros(3), gravity_body=np.array([0, 0, -1.0]),
                command=np.zeros(3), kappa=0.5, cfg=R.DEFAULTS, joint_vel=np.zeros(18), joint_vel_prev=np.zeros(18),
                dt=0.02, torque=np.zeros(18), action=np.full(18, 0.3), action_prev=np.full(18, 0.1),
                foot_contact=np.ones(6, bool), foot_vel_xy=np.zeros((6, 2)), air_bonus=0.0, q=np.zeros(18),
                q_stance=np.zeros(18), fallen=False, action_prev2=np.full(18, 0.2))
    assert t["action_2nd"] == pytest.approx(18 * 0.3 ** 2)            # 0.3 - 0.2 + 0.2 = 0.3
    w = {k: 0.0 for k in t} | {"energy": -0.02, "action_2nd": -2.0}
    vals = {k: 0.0 for k in t} | {"energy": 10.0, "action_2nd": 1.0}
    start = {"energy": 0.1, "action_2nd": 0.0}
    a0 = R.weighted(vals, w, start, 0.0)
    a1 = R.weighted(vals, w, start, 1.0)
    ah = R.weighted(vals, w, start, 0.5)
    assert a0["energy"] == pytest.approx(-0.02) and a1["energy"] == pytest.approx(-0.2)
    assert a0["action_2nd"] == 0.0 and a1["action_2nd"] == pytest.approx(-2.0)
    assert ah["energy"] == pytest.approx(-0.11)


def _asym_model():
    venv = DummyVecEnv([lambda: env_with({"enabled": True}, obs_mode="asym")])
    return PPO(AsymmetricActorCriticPolicy, venv, policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
               device="cpu", seed=0)


def test_actor_ignores_privileged_critic_uses_it():
    pol = _asym_model().policy
    obs = {"policy": torch.randn(4, 195), "privileged": torch.randn(4, PRIVILEGED_DIM)}
    obs2 = {"policy": obs["policy"], "privileged": torch.randn(4, PRIVILEGED_DIM)}
    with torch.no_grad():
        assert torch.allclose(pol.get_distribution(obs).distribution.mean, pol.get_distribution(obs2).distribution.mean)
        assert not torch.allclose(pol.predict_values(obs), pol.predict_values(obs2))


def test_transfer_from_flat_preserves_actor_and_critic():
    flat = PPO("MlpPolicy", DummyVecEnv([lambda: env_with()]),
               policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])), device="cpu", seed=1)
    asym = _asym_model()
    transfer_from_flat(asym.policy, flat.policy)
    x = torch.randn(5, 195)
    obs = {"policy": x, "privileged": torch.randn(5, PRIVILEGED_DIM)}
    with torch.no_grad():
        assert torch.allclose(asym.policy.get_distribution(obs).distribution.mean,
                              flat.policy.get_distribution(x).distribution.mean, atol=1e-6)
        assert torch.allclose(asym.policy.predict_values(obs), flat.policy.predict_values(x), atol=1e-6)


# ---------------------------------------------------------------- step 2.9b: sim-to-sim gap randomisation
GAP_DR = {"enabled": True, "latency_substeps": [0, 10], "target_offset": {"hip_yaw": 0.02, "hip_pitch": 0.06, "knee": 0.08},
          "contact_timeconst": [0.01, 0.04], "contact_dampratio": [0.8, 1.5]}


def test_gap_dr_off_by_default_is_nominal():
    e = env_with({"enabled": True})
    e.dr_scale = 1.0
    _, info = e.reset(seed=1)
    assert np.all(info["dr"]["target_offset"] == 0.0)
    assert np.allclose(e.model.geom_solref[e.foot_geoms], e.dr.solref0[e.foot_geoms])


def test_gap_dr_sampled_in_range_and_applied_to_targets():
    e = env_with(GAP_DR)
    e.dr_scale = 1.0
    seen_lat, tcs = set(), []
    for s in range(60):
        _, info = e.reset(seed=s)
        p = info["dr"]
        off = p["target_offset"].reshape(6, 3)
        assert np.all(np.abs(off[:, 0]) <= 0.02) and np.all(np.abs(off[:, 1]) <= 0.06) and np.all(np.abs(off[:, 2]) <= 0.08)
        assert 0.01 <= p["contact_timeconst"] <= 0.04 and 0.8 <= p["contact_dampratio"] <= 1.5
        assert np.allclose(e.model.geom_solref[e.foot_geoms], [[p["contact_timeconst"], p["contact_dampratio"]]] * 6)
        seen_lat.add(p["latency_substeps"])
        tcs.append(p["contact_timeconst"])
    assert max(seen_lat) == 10 and min(tcs) < 0.015 and max(tcs) > 0.035
    e.dr_scale = 1.0
    e.reset(seed=3, options={"command": [0.1, 0, 0]})
    e.step(np.zeros(18, np.float32))
    exp = np.clip(e.q_ref + e.dr.params["target_offset"], e.q_lo, e.q_hi)
    assert np.allclose(e.data.ctrl, exp)                  # zero latency or held: after the step the offset target is applied
    assert np.all(e.data.ctrl >= e.q_lo - 1e-12) and np.all(e.data.ctrl <= e.q_hi + 1e-12)


def test_gap_dr_fixed_scenario_and_scale_zero():
    off = np.full(18, 0.03)
    e = env_with(GAP_DR, fixed={"target_offset": off, "contact_timeconst": 0.04, "latency_substeps": 10})
    _, info = e.reset(seed=0)
    assert np.allclose(info["dr"]["target_offset"], off) and info["dr"]["latency_substeps"] == 10
    e2 = env_with(GAP_DR)
    e2.dr_scale = 0.0
    _, info = e2.reset(seed=0)
    assert np.all(info["dr"]["target_offset"] == 0) and np.allclose(e2.model.geom_solref[e2.foot_geoms], e2.dr.solref0[e2.foot_geoms])


def test_extend_privileged_keeps_actor_and_critic_values():
    from policies import extend_privileged
    mk = lambda **kw: PPO(AsymmetricActorCriticPolicy, DummyVecEnv([lambda: env_with({"enabled": True}, obs_mode="asym", **kw)]),
                          policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])), device="cpu", seed=0)
    old, new = mk(), mk(failure_privileged=True)
    extend_privileged(new.policy, old.policy)
    pol = torch.randn(4, 195)
    po, pn = torch.randn(4, PRIVILEGED_DIM), torch.randn(4, 8)
    o_old = {"policy": pol, "privileged": po}
    o_new = {"policy": pol, "privileged": torch.cat([po, pn], 1)}
    with torch.no_grad():
        assert torch.allclose(old.policy.get_distribution(o_old).distribution.mean, new.policy.get_distribution(o_new).distribution.mean)
        assert torch.allclose(old.policy.predict_values(o_old), new.policy.predict_values(o_new), atol=1e-6)
