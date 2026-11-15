"""Step 2.9: ONNX export parity and deployment controller == training env (no Gazebo, no ROS)."""
import json
import os

import numpy as np
import pytest

from envs.gurdy_env import GurdyEnv
from gurdy_core.policy_runtime import OnnxPolicy, PolicyController
from gurdy_core.rl_obs import frame_slices
from gurdy_core.robot import load_robot

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MODEL = os.path.join(ROOT, "models", "dr_energy_ft_6M")
needs_model = pytest.mark.skipif(not os.path.exists(MODEL + ".onnx"), reason="models/ not exported")


@pytest.fixture(scope="module")
def meta():
    with open(MODEL + ".json") as f:
        return json.load(f)


@needs_model
def test_onnx_matches_recorded_torch_actions():
    fx = np.load(MODEL + "_parity.npz")
    out = np.array([OnnxPolicy(MODEL + ".onnx")(o) for o in fx["obs"]])
    assert out.shape == fx["action"].shape == (len(fx["obs"]), 18)
    assert np.max(np.abs(out - fx["action"])) < 1e-5


@needs_model
def test_onnx_matches_sb3_predict_on_live_checkpoint(meta):
    """Full original path (VecNormalize in float64 + SB3 deterministic predict) vs the ONNX graph."""
    ckpt = os.path.join(ROOT, meta["checkpoint"])
    if not os.path.exists(ckpt):
        pytest.skip("checkpoint not available")
    import pickle

    from stable_baselines3 import PPO

    from train import vecnorm_for
    model = PPO.load(ckpt, device="cpu")
    with open(vecnorm_for(ckpt), "rb") as f:
        vn = pickle.load(f)
    obs = np.load(MODEL + "_parity.npz")["obs"][:100]
    onnx = OnnxPolicy(MODEL + ".onnx")
    worst = 0.0
    for o in obs:
        n = vn.normalize_obs({"policy": o[None].astype(np.float64),
                              "privileged": np.zeros((1, 17))})["policy"].astype(np.float32)
        feats = {"policy": __import__("torch").as_tensor(n), "privileged": __import__("torch").zeros(1, 17)}
        ref = model.policy.get_distribution(feats).distribution.mean.detach().numpy()[0]
        worst = max(worst, float(np.max(np.abs(onnx(o) - ref))))
    assert worst < 1e-5, worst


def _env(meta):
    return GurdyEnv({"obs_mode": "flat", "action_scale": meta["action_scale"],
                     "action_filter_alpha": meta["action_filter_alpha"], "resample_command": False,
                     "domain_randomization": {"enabled": False}})


def _sensors(env):
    d = env.data
    return d.qpos[3:7].copy(), d.sensordata[env.gyro_adr:env.gyro_adr + 3].copy(), \
        d.qpos[env.qadr].copy(), d.qvel[env.vadr].copy()


@needs_model
def test_controller_observation_and_targets_equal_env(meta):
    env = _env(meta)
    rng = np.random.default_rng(0)
    actions = [rng.uniform(-1.5, 1.5, 18) for _ in range(60)]      # beyond +-1: both must clip
    seq = iter(actions)
    ctl = PolicyController(load_robot(), lambda obs: next(seq), meta)
    assert ctl.dt == pytest.approx(env.policy_dt) and ctl.clock_period == pytest.approx(env.clock_period)
    cmds = {0: (0.08, 0.0, 0.0), 20: (0.0, -0.05, 0.2), 40: (0.01, 0.0, 0.02)}   # last: inside deadband
    def raw_at(k):                                       # /cmd_vel held at tick k
        return np.array(cmds[max(c for c in cmds if c <= k)])

    obs, _ = env.reset(seed=3, options={"command": ctl.condition_command(raw_at(0))})
    for k in range(60):
        q_ref = ctl.step(*_sensors(env), raw_at(k))
        # obs the controller built == obs the env returned for this tick
        assert np.max(np.abs(ctl.hist.buf.reshape(-1) - obs)) < 1e-6, k
        # env resamples its command at the end of the step, before building the next obs; the
        # training env holds the deadbanded command
        env.command = ctl.condition_command(raw_at(k + 1))
        obs, *_ = env.step(actions[k])
        assert np.max(np.abs(q_ref - env.q_ref)) < 1e-12, k
    assert np.all(ctl.command == 0)                       # (0.01, 0, 0.02) -> deadband -> stand


@needs_model
def test_command_conditioning(meta):
    ctl = PolicyController(load_robot(), lambda o: np.zeros(18), meta)
    assert np.all(ctl.condition_command((0.01, 0.01, 0.04)) == 0)
    assert ctl.condition_command((0.1, 0.0, 0.0))[0] == pytest.approx(0.1)
    big = ctl.condition_command((5.0, -5.0, 9.0))
    assert np.all(np.abs(big) <= ctl.cmd_lim + 1e-12)


@needs_model
def test_reset_restores_episode_start(meta):
    env = _env(meta)
    ctl = PolicyController(load_robot(), lambda o: np.full(18, 0.3), meta)
    env.reset(seed=1)
    for _ in range(5):
        ctl.step(*_sensors(env), (0.1, 0, 0))
    ctl.reset()
    ctl.observe(*_sensors(env), (0, 0, 0))
    f = ctl.hist.buf
    S = frame_slices()
    assert np.allclose(f, f[0]) and np.all(f[0][S["prev_action"]] == 0) and f[0][S["clock"]] == pytest.approx([0, 1])
