"""Step 2.1 benchmark: env throughput vs number of parallel workers, RAM, and PPO update cost.

    docker compose run --rm rl python3 rl/bench.py [--workers 1 4 6 8 10] [--steps 2000]

Prints a markdown table and writes JSON to --out. All numbers are measured, nothing estimated.
"""
import argparse
import json
import os
import time

import mujoco
import numpy as np
import psutil


def tree_rss_mib() -> float:
    p = psutil.Process()
    return sum(c.memory_info().rss for c in [p] + p.children(recursive=True)) / 2**20


def host_benchmark() -> float:
    """Single-thread MuJoCo stepping RTF (same as scripts/measure_rtf.py) to show host clock state."""
    from sim_mujoco.mjcf import build_mjcf
    m = mujoco.MjModel.from_xml_string(build_mjcf())
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    t = time.perf_counter()
    for _ in range(20000):
        mujoco.mj_step(m, d)
    return 20000 * m.opt.timestep / (time.perf_counter() - t)


def bench_single(steps: int) -> dict:
    from envs.gurdy_env import GurdyEnv
    env = GurdyEnv()
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    acts = rng.uniform(-0.3, 0.3, (steps, 18))
    t = time.perf_counter()
    resets = 0
    for k in range(steps):
        _, _, term, trunc, _ = env.step(acts[k])
        if term or trunc:
            env.reset()
            resets += 1
    dt = time.perf_counter() - t
    # physics only, same number of mj_step calls
    m, d = env.model, env.data
    t = time.perf_counter()
    for _ in range(steps):
        mujoco.mj_step(m, d, nstep=env.n_sub)
    phys = time.perf_counter() - t
    return {"steps_per_s": steps / dt, "physics_only_steps_per_s": steps / phys,
            "python_overhead_pct": 100 * (dt - phys) / dt, "resets": resets}


def bench_vec(n: int, vec_steps: int) -> dict:
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from envs.gurdy_env import GurdyEnv
    venv = make_vec_env(GurdyEnv, n_envs=n, seed=0, vec_env_cls=SubprocVecEnv)
    venv.reset()
    rng = np.random.default_rng(1)
    for _ in range(50):                                   # warm-up
        venv.step(rng.uniform(-0.3, 0.3, (n, 18)))
    peak = tree_rss_mib()
    t = time.perf_counter()
    for k in range(vec_steps):
        venv.step(rng.uniform(-0.3, 0.3, (n, 18)))
        if k % 200 == 0:
            peak = max(peak, tree_rss_mib())
    dt = time.perf_counter() - t
    peak = max(peak, tree_rss_mib())
    venv.close()
    return {"workers": n, "env_steps_per_s": n * vec_steps / dt, "ram_tree_mib": peak}


def _env_fn():
    from envs.gurdy_env import GurdyEnv
    return GurdyEnv()


def bench_batched(workers: int, per_worker: int, vec_steps: int) -> dict:
    from vec_env import BatchedSubprocVecEnv
    venv = BatchedSubprocVecEnv(_env_fn, workers, per_worker)
    venv.seed(0)
    venv.reset()
    n = venv.num_envs
    rng = np.random.default_rng(1)
    for _ in range(20):
        venv.step(rng.uniform(-0.3, 0.3, (n, 18)))
    peak = tree_rss_mib()
    t = time.perf_counter()
    for k in range(vec_steps):
        venv.step(rng.uniform(-0.3, 0.3, (n, 18)))
        if k % 100 == 0:
            peak = max(peak, tree_rss_mib())
    dt = time.perf_counter() - t
    venv.close()
    return {"workers": workers, "envs_per_worker": per_worker, "num_envs": n,
            "env_steps_per_s": n * vec_steps / dt, "ram_tree_mib": peak}


def bench_ppo_batched(workers: int, per_worker: int, n_steps: int, torch_threads: int) -> dict:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize
    from vec_env import BatchedSubprocVecEnv
    torch.set_num_threads(torch_threads)
    venv = VecNormalize(BatchedSubprocVecEnv(_env_fn, workers, per_worker))
    n = venv.num_envs
    model = PPO("MlpPolicy", venv, n_steps=n_steps, batch_size=4096, n_epochs=5, learning_rate=3e-4,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
                policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                                   log_std_init=float(np.log(0.35))),
                device="cpu", verbose=0)
    t = time.perf_counter()
    model.learn(total_timesteps=2 * n * n_steps)
    total = time.perf_counter() - t
    t = time.perf_counter()
    model.train()
    train = time.perf_counter() - t
    peak = tree_rss_mib()
    venv.close()
    per_iter = total / 2
    return {"workers": workers, "envs_per_worker": per_worker, "num_envs": n, "n_steps": n_steps,
            "torch_threads": torch_threads, "samples_per_iter": n * n_steps, "iter_s": per_iter,
            "train_s": train, "collect_s": per_iter - train,
            "overall_samples_per_s": n * n_steps / per_iter, "ram_tree_mib": peak}


def bench_ppo(n: int, torch_threads: int) -> dict:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from envs.gurdy_env import GurdyEnv
    torch.set_num_threads(torch_threads)
    venv = VecNormalize(make_vec_env(GurdyEnv, n_envs=n, seed=0, vec_env_cls=SubprocVecEnv))
    model = PPO("MlpPolicy", venv, n_steps=1024, batch_size=2048, n_epochs=5, learning_rate=3e-4,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, target_kl=None,
                policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                                   log_std_init=float(np.log(0.35))),
                device="cpu", verbose=0)
    t = time.perf_counter()
    model.learn(total_timesteps=2 * n * 1024)              # 2 iterations: collect + train each
    total = time.perf_counter() - t
    t = time.perf_counter()
    model.train()                                          # one update on the filled buffer
    train = time.perf_counter() - t
    peak = tree_rss_mib()
    venv.close()
    per_iter = total / 2
    return {"workers": n, "torch_threads": torch_threads, "samples_per_iter": n * 1024,
            "iter_s": per_iter, "train_s": train, "collect_s": per_iter - train,
            "overall_samples_per_s": n * 1024 / per_iter, "ram_tree_mib": peak}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--steps", type=int, default=2000, help="vectorised steps per worker count")
    ap.add_argument("--ppo-workers", type=int, default=8)
    ap.add_argument("--out", default="docs/results/phase2_bench.json")
    a = ap.parse_args()

    res = {"cpu": os.cpu_count(), "host_benchmark_before": host_benchmark()}
    res["single"] = bench_single(5000)
    res["vec"] = [bench_vec(n, a.steps) for n in a.workers]
    res["batched"] = [bench_batched(w, k, max(200, a.steps // k)) for w, k in
                      [(4, 4), (6, 4), (8, 2), (8, 4), (8, 8), (10, 4)]]
    best = max(res["batched"], key=lambda r: r["env_steps_per_s"])
    res["ppo_batched"] = [bench_ppo_batched(best["workers"], best["envs_per_worker"],
                                            n_steps=max(64, 16384 // best["num_envs"]), torch_threads=t)
                          for t in (2, 4)]
    res["host_benchmark_after"] = host_benchmark()

    s = res["single"]
    print(f"host benchmark (MuJoCo stepping RTF): before {res['host_benchmark_before']:.0f}, "
          f"after {res['host_benchmark_after']:.0f}")
    print(f"single env: {s['steps_per_s']:.0f} steps/s (physics only {s['physics_only_steps_per_s']:.0f}, "
          f"Python overhead {s['python_overhead_pct']:.0f} %)\n")
    print("| workers | env steps/s | RAM (process tree) |\n|---|---|---|")
    for v in res["vec"]:
        print(f"| {v['workers']} | {v['env_steps_per_s']:.0f} | {v['ram_tree_mib']:.0f} MiB |")
    print("\nBatchedSubprocVecEnv:\n| workers x envs | env steps/s | RAM (process tree) |\n|---|---|---|")
    for v in res["batched"]:
        print(f"| {v['workers']} x {v['envs_per_worker']} = {v['num_envs']} | {v['env_steps_per_s']:.0f} | {v['ram_tree_mib']:.0f} MiB |")
    for p in res["ppo_batched"]:
        print(f"\nPPO ({p['workers']}x{p['envs_per_worker']} envs, n_steps {p['n_steps']}, torch threads {p['torch_threads']}, "
              f"{p['samples_per_iter']} samples/iter): iteration {p['iter_s']:.1f} s = collect {p['collect_s']:.1f} s "
              f"+ update {p['train_s']:.1f} s -> {p['overall_samples_per_s']:.0f} samples/s overall; RAM {p['ram_tree_mib']:.0f} MiB")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
