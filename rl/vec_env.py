"""BatchedSubprocVecEnv: SB3-compatible VecEnv running `envs_per_worker` environments in each of
`n_workers` processes. One pipe message per worker per step (not per env), and workers don't import
torch (see vec_worker.py). Provides the same info keys SB3's Monitor + SubprocVecEnv would:
"episode", "terminal_observation", "TimeLimit.truncated".
"""
from __future__ import annotations

import multiprocessing as mp

import cloudpickle
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from vec_worker import worker


class BatchedSubprocVecEnv(VecEnv):
    def __init__(self, env_fn, n_workers: int, envs_per_worker: int, start_method: str = "forkserver"):
        self.n_workers, self.k = n_workers, envs_per_worker
        ctx = mp.get_context(start_method)
        self.remotes, self.procs = [], []
        for _ in range(n_workers):
            remote, work_remote = ctx.Pipe()
            fns = cloudpickle.dumps([env_fn] * envs_per_worker)
            p = ctx.Process(target=worker, args=(work_remote, remote, fns), daemon=True)
            p.start()
            work_remote.close()
            self.remotes.append(remote)
            self.procs.append(p)
        self.remotes[0].send(("get_attr", "observation_space"))
        obs_space = self.remotes[0].recv()[0]
        self.remotes[0].send(("get_attr", "action_space"))
        act_space = self.remotes[0].recv()[0]
        super().__init__(n_workers * envs_per_worker, obs_space, act_space)
        self._actions = None
        self.closed = False

    def _split(self, arr):
        return [arr[i * self.k:(i + 1) * self.k] for i in range(self.n_workers)]

    def step_async(self, actions):
        for r, a in zip(self.remotes, self._split(np.asarray(actions))):
            r.send(("step", a))
        self.waiting = True

    def step_wait(self):
        res = [r.recv() for r in self.remotes]
        self.waiting = False
        obs = np.concatenate([o for o, _, _, _ in res])
        rews = np.concatenate([x for _, x, _, _ in res]).astype(np.float32)
        dones = np.concatenate([d for _, _, d, _ in res])
        infos = [i for _, _, _, inf in res for i in inf]
        return obs, rews, dones, infos

    def reset(self):
        seeds = self._seeds if self._seeds else [None] * self.num_envs
        for r, s in zip(self.remotes, self._split(np.array(seeds, dtype=object))):
            r.send(("reset", (list(s), self._options[0] if self._options else None)))
        res = [r.recv() for r in self.remotes]
        self.reset_infos = [i for _, inf in res for i in inf]
        self._reset_seeds()
        self._reset_options()
        return np.concatenate([o for o, _ in res])

    def close(self):
        if self.closed:
            return
        if getattr(self, "waiting", False):
            for r in self.remotes:
                r.recv()
        for r in self.remotes:
            r.send(("close", None))
        for p in self.procs:
            p.join(timeout=5)
        self.closed = True

    def _flat(self, cmd, data, indices):
        for r in self.remotes:
            r.send((cmd, data))
        vals = [v for r in self.remotes for v in r.recv()]
        idx = self._get_indices(indices)
        return [vals[i] for i in idx]

    def get_attr(self, attr_name, indices=None):
        return self._flat("get_attr", attr_name, indices)

    def set_attr(self, attr_name, value, indices=None):
        for r in self.remotes:
            r.send(("set_attr", (attr_name, value)))
        for r in self.remotes:
            r.recv()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return self._flat("env_method", (method_name, method_args, method_kwargs), indices)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._get_indices(indices))
