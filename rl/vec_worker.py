"""Worker process for BatchedSubprocVecEnv. Deliberately imports NO stable_baselines3 / torch,
so each worker stays small (SB3's own SubprocVecEnv worker pulls in torch: ~340 MiB per worker)."""
import time

import cloudpickle
import numpy as np


def stack(obs_list):
    """Stack per-env observations; Dict observations are stacked per key."""
    if isinstance(obs_list[0], dict):
        return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}
    return np.stack(obs_list)


def worker(remote, parent_remote, env_fns_bytes):
    parent_remote.close()
    envs = [fn() for fn in cloudpickle.loads(env_fns_bytes)]
    n = len(envs)
    ep_ret, ep_len, t0 = np.zeros(n), np.zeros(n, dtype=int), time.time()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, rews, dones, infos = [], np.zeros(n), np.zeros(n, dtype=bool), []
                for i, (env, a) in enumerate(zip(envs, data)):
                    o, r, term, trunc, info = env.step(a)
                    ep_ret[i] += r
                    ep_len[i] += 1
                    done = term or trunc
                    info = dict(info)
                    info["TimeLimit.truncated"] = bool(trunc and not term)
                    if done:
                        info["terminal_observation"] = o
                        # same keys and rounding as SB3's Monitor wrapper
                        info["episode"] = {"r": round(float(ep_ret[i]), 6), "l": int(ep_len[i]),
                                           "t": round(time.time() - t0, 6)}
                        ep_ret[i], ep_len[i] = 0.0, 0
                        o, _ = env.reset()
                    obs.append(o)
                    rews[i], dones[i] = r, done
                    infos.append(info)
                remote.send((stack(obs), rews, dones, infos))
            elif cmd == "reset":
                seeds, options = data
                out = [env.reset(seed=s, options=options) for env, s in zip(envs, seeds)]
                ep_ret[:], ep_len[:] = 0.0, 0
                remote.send((stack([o for o, _ in out]), [i for _, i in out]))
            elif cmd == "get_attr":
                remote.send([getattr(env.unwrapped, data) for env in envs])
            elif cmd == "set_attr":
                name, value = data
                for env in envs:
                    setattr(env.unwrapped, name, value)
                remote.send(None)
            elif cmd == "env_method":
                name, args, kwargs = data
                remote.send([getattr(env.unwrapped, name)(*args, **kwargs) for env in envs])
            elif cmd == "close":
                for env in envs:
                    env.close()
                remote.close()
                break
    except KeyboardInterrupt:
        pass
