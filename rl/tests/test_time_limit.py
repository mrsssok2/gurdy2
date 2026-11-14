"""Time-limit truncations must bootstrap from the value function; falls (terminations) must not.
Checked end-to-end: BatchedSubprocVecEnv -> SB3 PPO.collect_rollouts -> rollout buffer."""
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from envs.gurdy_env import GurdyEnv
from vec_env import BatchedSubprocVecEnv


class FallingEnv(GurdyEnv):
    """Falls (terminates) on its 4th step."""

    def _fallen(self):
        return self.steps == 4


def short_env():
    return GurdyEnv({"episode_s": 0.2})              # 10 steps, then time-limit truncation


def falling_env():
    return FallingEnv({"episode_s": 0.2})


class Record(BaseCallback):
    """Records raw rewards and V(terminal_obs) with the SAME weights SB3 bootstraps with
    (the policy is updated after the rollout, so values must be taken here)."""

    def _on_rollout_start(self):
        self.raw, self.infos, self.dones, self.vals = [], [], [], []

    def _on_step(self):
        self.raw.append(np.array(self.locals["rewards"], copy=True))   # before SB3's bootstrap
        infos, dones = self.locals["infos"], np.array(self.locals["dones"])
        vals = {}
        for i, info in enumerate(infos):
            if dones[i] and info.get("terminal_observation") is not None:
                obs = torch.as_tensor(info["terminal_observation"][None], dtype=torch.float32)
                with torch.no_grad():
                    vals[i] = self.model.policy.predict_values(obs).item()
        self.infos.append(infos)
        self.dones.append(dones)
        self.vals.append(vals)
        return True


def _collect(env_fn):
    venv = BatchedSubprocVecEnv(env_fn, n_workers=2, envs_per_worker=2)
    venv.seed(0)
    model = PPO("MlpPolicy", venv, n_steps=24, batch_size=48, gamma=0.9, device="cpu", seed=0)
    cb = Record()
    model.learn(total_timesteps=1, callback=cb)          # one rollout of 24 steps x 4 envs
    venv.close()
    return model, cb


def test_truncation_bootstraps_with_value():
    model, cb = _collect(short_env)
    buf = model.rollout_buffer.rewards                   # (n_steps, n_envs), after bootstrap
    checked = 0
    for t, (raw, infos, dones) in enumerate(zip(cb.raw, cb.infos, cb.dones)):
        for i, info in enumerate(infos):
            if dones[i]:
                assert info["TimeLimit.truncated"] is True
                v = cb.vals[t][i]
                assert abs(buf[t, i] - (raw[i] + model.gamma * v)) < 1e-5
                checked += 1
            else:
                assert abs(buf[t, i] - raw[i]) < 1e-6
    assert checked >= 8                                  # 4 envs x 2 truncations in 24 steps


def test_falls_are_not_bootstrapped():
    model, cb = _collect(falling_env)
    buf = model.rollout_buffer.rewards
    falls = 0
    for t, (raw, infos, dones) in enumerate(zip(cb.raw, cb.infos, cb.dones)):
        for i, info in enumerate(infos):
            if dones[i]:
                assert info["TimeLimit.truncated"] is False
                falls += 1
            assert abs(buf[t, i] - raw[i]) < 1e-6        # never bootstrapped
    assert falls >= 16                                   # every 4 steps, 4 envs, 24 steps
