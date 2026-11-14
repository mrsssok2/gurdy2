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
