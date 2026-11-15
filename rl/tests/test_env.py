"""Step 2.2 environment tests."""
import inspect

import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from envs.gurdy_env import GurdyEnv, load_config
from gurdy_core.rl_obs import FRAME_DIM, HISTORY, OBS_DIM, build_frame, frame_slices
from gurdy_core.robot import LEG_NAMES

S = frame_slices()


@pytest.fixture
def env():
    return GurdyEnv(load_config("configs/flat.yaml")["env"])


def newest(obs):
    return obs[-FRAME_DIM:]


def test_check_env(env):
    check_env(env, skip_render_check=True)


def test_observation_layout_at_reset(env):
    obs, info = env.reset(seed=1)
    assert obs.shape == (OBS_DIM,) == (HISTORY * FRAME_DIM,) == (195,)
    frames = obs.reshape(HISTORY, FRAME_DIM)
    assert np.allclose(frames, frames[0])                       # history filled with first frame
    f = newest(obs)
    assert f[S["gravity"]] == pytest.approx([0, 0, -1], abs=1e-6)
    assert np.all(np.abs(f[S["joint_pos"]]) <= env.cfg["init_joint_noise"] + 1e-6)
    assert np.all(f[S["prev_action"]] == 0)
    assert f[S["command"]] == pytest.approx(info["command"])
    assert f[S["clock"]] == pytest.approx([0, 1])


def test_history_shifts(env):
    env.reset(seed=2)
    o1, *_ = env.step(np.full(18, 0.2))
    o2, *_ = env.step(np.full(18, -0.2))
    f1, f2 = o2.reshape(HISTORY, FRAME_DIM)[-2:]
    assert np.allclose(f1, newest(o1))                          # previous newest moved back one slot
    assert f2[S["prev_action"]] == pytest.approx(np.full(18, -0.2))


def test_observation_has_no_privileged_inputs(env):
    """World position, heading and world-frame linear velocity must not change the observation:
    the policy only sees what the robot's own sensors measure."""
    env.reset(seed=3)
    for _ in range(30):
        env.step(env.np_random.uniform(-0.3, 0.3, 18))
    d = env.data
    mujoco.mj_forward(env.model, d)     # refresh sensors (mj_step evaluates them before integrating)
    base = env._frame()
    qpos, qvel = d.qpos.copy(), d.qvel.copy()
    # translate + rotate about world z + add world-frame linear velocity
    d.qpos[0:2] += [3.0, -2.0]
    yaw = np.array([np.cos(0.6), 0, 0, np.sin(0.6)])
    q = np.empty(4)
    mujoco.mju_mulQuat(q, yaw, qpos[3:7])
    d.qpos[3:7] = q
    d.qvel[0:3] += [0.5, -0.3, 0.0]
    mujoco.mj_forward(env.model, d)
    assert env._frame() == pytest.approx(base, abs=1e-9)
    d.qpos[:], d.qvel[:] = qpos, qvel
    # and the obs builder only accepts sensor-type inputs
    params = set(inspect.signature(build_frame).parameters)
    assert params == {"gravity", "gyro", "joint_pos", "joint_vel", "q_stance", "prev_action", "command", "phase"}


def test_targets_always_within_joint_limits(env):
    env.reset(seed=4)
    rng = np.random.default_rng(0)
    for k in range(200):
        a = rng.uniform(-10, 10, 18) if k % 2 else np.sign(rng.uniform(-1, 1, 18)) * 5
        _, _, term, trunc, _ = env.step(a)
        assert np.all(env.data.ctrl >= env.q_lo - 1e-12) and np.all(env.data.ctrl <= env.q_hi + 1e-12)
        assert np.all(env.data.ctrl[2::3] >= 0.0), "knee target bent backwards"
        if term or trunc:
            env.reset()


def test_deterministic_seeding():
    def rollout(seed):
        e = GurdyEnv(load_config("configs/flat.yaml")["env"])
        obs, _ = e.reset(seed=seed)
        rng = np.random.default_rng(7)
        out = [obs]
        for _ in range(300):                                    # crosses a command resample
            o, r, *_ = e.step(rng.uniform(-0.5, 0.5, 18))
            out.append(np.append(o, r))
        return np.concatenate(out)
    a, b, c = rollout(11), rollout(11), rollout(12)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_command_sampling(env):
    env.reset(seed=5)
    cmds = np.array([env._sample_command() for _ in range(4000)])
    assert np.all(np.abs(cmds) <= env.cmd_lim + 1e-12)
    zero = np.all(cmds == 0, axis=1).mean()
    assert 0.12 < zero < 0.18                                   # zero_command_prob 0.15


def test_command_resampled_every_period_and_fixed_option(env):
    env.reset(seed=6)
    first = env.command.copy()
    for _ in range(env.cmd_steps - 1):
        env.step(np.zeros(18))
    assert np.array_equal(env.command, first)
    env.step(np.zeros(18))
    assert not np.array_equal(env.command, first)               # resampled after cmd_steps
    env.reset(seed=6, options={"command": (0.1, 0.0, 0.0)})
    for _ in range(2 * env.cmd_steps):
        _, _, _, _, info = env.step(np.zeros(18))
    assert env.command == pytest.approx([0.1, 0, 0])


def test_truncation_at_episode_end():
    e = GurdyEnv({"episode_s": 1.0})
    e.reset(seed=0)
    for k in range(1, 51):
        _, _, term, trunc, _ = e.step(np.zeros(18))
        assert not term
        assert trunc == (k == 50)


def test_fall_detection(env):
    env.reset(seed=8)
    assert not env._fallen()
    d = env.data
    tilt = np.array([np.cos(0.4), np.sin(0.4), 0, 0])           # 0.8 rad roll > FALL_TILT
    d.qpos[3:7] = tilt
    mujoco.mj_forward(env.model, d)
    assert env._fallen()
    env.reset(seed=8)
    d.qpos[2] = 0.05                                            # body below FALL_Z
    mujoco.mj_forward(env.model, d)
    assert env._fallen()


def test_standing_still_all_feet_down(env):
    env.reset(seed=9, options={"command": (0, 0, 0)})
    for _ in range(50):
        _, _, term, _, info = env.step(np.zeros(18))
    assert not term
    assert info["foot_contact"].all() and len(info["foot_contact"]) == len(LEG_NAMES)
    assert abs(info["body_z"] - env.robot.stance["body_height"]) < 0.01


def test_axis_aligned_and_rest_start_commands():
    e = GurdyEnv({**load_config("configs/flat.yaml")["env"], "axis_aligned_prob": 0.25, "rest_start_prob": 0.25})
    e.reset(seed=3)
    cmds = np.array([e._sample_command() for _ in range(8000)])
    moving = cmds[np.any(cmds != 0, axis=1)]
    pure = np.sum(moving != 0, axis=1) == 1
    assert 0.2 < pure.mean() < 0.32                      # ~25 % (+ rare chance single-axis from uniform)
    assert np.all(np.abs(cmds) <= e.cmd_lim + 1e-12)
    # rest starts: a zero command that is later replaced by the pending command without resampling
    rests, starts = 0, 0
    for ep in range(40):
        e.reset(seed=100 + ep)
        for _ in range(e.max_steps):
            before = e.command.copy()
            _, _, term, trunc, _ = e.step(np.zeros(18))
            if not np.any(before) and np.any(e.command) and e.steps % e.cmd_steps != 0:
                starts += 1                              # pending command released after a rest
            if term or trunc:
                break
        rests += 1
    assert starts > 20


def test_new_options_off_keep_old_sampling():
    cfg = load_config("configs/flat.yaml")["env"]
    a, b = GurdyEnv(cfg), GurdyEnv({**cfg, "axis_aligned_prob": 0.0, "rest_start_prob": 0.0})
    a.reset(seed=4)
    b.reset(seed=4)
    assert np.array_equal([a._sample_command() for _ in range(50)], [b._sample_command() for _ in range(50)])
