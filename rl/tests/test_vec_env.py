"""BatchedSubprocVecEnv must behave exactly like SB3's DummyVecEnv + Monitor (same seeds/actions)."""
import numpy as np
import pytest
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.gurdy_env import GurdyEnv
from vec_env import BatchedSubprocVecEnv


def _env():
    return GurdyEnv({"episode_s": 0.5})           # short episodes: exercise auto-reset


def test_matches_dummy_vec_env():
    ref = make_vec_env(_env, n_envs=6, seed=3, vec_env_cls=DummyVecEnv)
    bat = BatchedSubprocVecEnv(_env, n_workers=3, envs_per_worker=2)
    bat.seed(3)
    try:
        o_ref, o_bat = ref.reset(), bat.reset()
        assert np.allclose(o_ref, o_bat)
        rng = np.random.default_rng(0)
        episodes = 0
        for _ in range(60):                       # > 2 episodes of 25 steps each
            a = rng.uniform(-1, 1, (6, 18)).astype(np.float32)
            o1, r1, d1, i1 = ref.step(a)
            o2, r2, d2, i2 = bat.step(a)
            assert np.allclose(o1, o2) and np.allclose(r1, r2) and np.array_equal(d1, d2)
            for x, y in zip(i1, i2):
                assert x.get("TimeLimit.truncated") == y.get("TimeLimit.truncated")
                if "episode" in x:
                    episodes += 1
                    assert x["episode"]["r"] == pytest.approx(y["episode"]["r"])
                    assert x["episode"]["l"] == y["episode"]["l"]
                    assert np.allclose(x["terminal_observation"], y["terminal_observation"])
        assert episodes >= 12
        assert bat.get_attr("max_steps") == [25] * 6
    finally:
        bat.close()
        ref.close()


def _asym_env():
    return GurdyEnv({"episode_s": 0.5, "obs_mode": "asym", "domain_randomization": {"enabled": True}})


def test_dict_observations_match_dummy_vec_env():
    ref = make_vec_env(_asym_env, n_envs=4, seed=5, vec_env_cls=DummyVecEnv)
    bat = BatchedSubprocVecEnv(_asym_env, n_workers=2, envs_per_worker=2)
    bat.seed(5)
    try:
        o1, o2 = ref.reset(), bat.reset()
        assert set(o1) == set(o2) == {"policy", "privileged"}
        for k in o1:
            assert np.allclose(o1[k], o2[k])
        rng = np.random.default_rng(1)
        for _ in range(40):
            a = rng.uniform(-1, 1, (4, 18)).astype(np.float32)
            (p1, r1, d1, _), (p2, r2, d2, i2) = ref.step(a), bat.step(a)
            for k in p1:
                assert np.allclose(p1[k], p2[k])
            assert np.allclose(r1, r2) and np.array_equal(d1, d2)
            for info in i2:
                if "terminal_observation" in info:
                    assert set(info["terminal_observation"]) == {"policy", "privileged"}
    finally:
        bat.close()
        ref.close()
