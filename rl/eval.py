"""Step 2.5: deterministic evaluation of policies vs the tripod gait (MuJoCo, flat ground).

    python3 rl/eval.py --run flat_bc_v1 --checkpoints 10000000 30000000 latest [--video]

Two protocols, identical for every controller:
 1. Phase 1 tracking protocol (gurdy_core.experiments.tracking_protocol, 123 s, commands at 20 Hz,
    analysed with the SAME analyze_tracking code as Phase 1): per-segment main-axis error, falls,
    mechanical energy (sum |tau*qdot| integrated every 2 ms substep) and cost of transport
    CoT = E / (m g d), d = horizontal path length.
 2. 100 episodes of 20 s with the training command distribution (fixed seeds 0..99): fall rate and
    per-command-window tracking error: mean body velocity over seconds 1-5 of each 5 s window vs
    the command, error = ||(v - c)/lim|| / ||c/lim|| (lim = command limits, mixes m/s and rad/s).
Policies act deterministically (mean action); observations normalised with the checkpoint's
VecNormalize statistics. Results: rl/runs/<run>/eval/*.json and a markdown table on stdout.
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


class Controller:
    """'tripod' or a policy checkpoint; .act(obs, env) -> action."""

    def __init__(self, spec):
        self.spec = spec
        if spec != "tripod":
            import torch
            from stable_baselines3 import PPO
            from train import vecnorm_for
            torch.set_num_threads(1)
            self.model = PPO.load(spec, device="cpu")
            with open(vecnorm_for(spec), "rb") as f:
                self.vn = pickle.load(f)

    def reset(self, env):
        if self.spec == "tripod":
            from gurdy_core.gait import TripodGait
            self.gait = TripodGait(env.robot)

    def act(self, obs, env):
        if self.spec == "tripod":
            return (self.gait.step(env.command) - env.q_stance) / env.scale
        o = self.vn.normalize_obs(obs[None].astype(np.float64)).astype(np.float32)
        return self.model.policy.predict(o, deterministic=True)[0][0]


def _env(cfg, episode_s, fixed):
    from envs.gurdy_env import GurdyEnv
    env_cfg = {**cfg["env"], "episode_s": episode_s, "measure_energy": True}
    if fixed:
        env_cfg["resample_command"] = False
    env = GurdyEnv(env_cfg, reward_config=cfg["reward"])
    env.penalty_scale = 1.0
    env.kappa = cfg["training"]["kappa_anneal"]["end"]
    return env


def run_protocol(args):
    """Phase 1 tracking protocol for one controller. Returns metrics + per-step trace."""
    spec, cfg, record = args
    from gurdy_core.experiments import analyze_tracking, cmd_at, quat_to_rpy, tracking_protocol
    changes, _, total = tracking_protocol()
    env = _env(cfg, total + 1.0, fixed=True)
    ctl = Controller(spec)
    obs, _ = env.reset(seed=0, options={"command": (0.0, 0.0, 0.0)})
    ctl.reset(env)
    rows, energy, path, prev_xy, frames = [], 0.0, 0.0, env.data.qpos[:2].copy(), []
    t = 0.0
    while t < total - 1e-9:
        env.command = np.array(cmd_at(changes, t), dtype=float)
        obs, _, term, _, info = env.step(ctl.act(obs, env))
        t = env.steps * env.policy_dt
        energy += info["energy_J"]
        xy = env.data.qpos[:2].copy()
        path += float(np.linalg.norm(xy - prev_xy))
        prev_xy = xy
        rows.append((t, *env.data.qpos[:7]))
        if record:
            frames.append(env.data.qpos.copy())
        if term:
            break
    a = np.array(rows)
    roll, pitch, yaw = quat_to_rpy(a[:, 4], a[:, 5], a[:, 6], a[:, 7])
    res = analyze_tracking(a[:, 0], a[:, 1], a[:, 2], yaw, a[:, 3], roll, pitch)
    m, g = env.robot.total_mass, env.robot.sim["gravity"]
    res.update({"controller": spec, "energy_J": energy, "path_m": path,
                "cost_of_transport": energy / (m * g * path) if path > 0 else float("nan"),
                "mean_power_W": energy / max(t, 1e-9)})
    return res, (np.array(frames) if record else None)


def run_episodes(args):
    """Training-distribution episodes (seeds) for one controller."""
    spec, cfg, seeds = args
    env = _env(cfg, cfg["env"]["episode_s"], fixed=False)
    ctl = Controller(spec)
    lim = env.cmd_lim
    out = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        ctl.reset(env)
        vels, cmds, energy, done, term = [], [], 0.0, False, False
        while not done:
            cmd = env.command.copy()
            obs, _, term, trunc, info = env.step(ctl.act(obs, env))
            vels.append([*info["body_vel"][:2], info["yaw_rate"]])
            cmds.append(cmd)
            energy += info["energy_J"]
            done = term or trunc
        vels, cmds = np.array(vels), np.array(cmds)
        errs = []
        w = env.cmd_steps
        for k in range(0, len(cmds) - w + 1, w):
            c = cmds[k]
            if not np.any(np.abs(c) > 1e-6) or not np.allclose(cmds[k:k + w], c):
                continue
            v = vels[k + int(1.0 / env.policy_dt):k + w].mean(axis=0)
            errs.append(float(np.linalg.norm((v - c) / lim) / np.linalg.norm(c / lim)))
        out.append({"seed": seed, "fell": bool(term), "steps": len(cmds), "window_errors": errs,
                    "energy_J": energy})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoints", nargs="+", default=["latest"])
    ap.add_argument("--config", default=None, help="defaults to the run's config.yaml")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--no-tripod", action="store_true")
    a = ap.parse_args()

    import yaml
    run = os.path.join(HERE, "runs", a.run)
    cfg = yaml.safe_load(open(a.config or os.path.join(run, "config.yaml")))
    out_dir = os.path.join(run, "eval")
    os.makedirs(out_dir, exist_ok=True)
    specs = [] if a.no_tripod else ["tripod"]
    for c in a.checkpoints:
        specs.append(os.path.join(run, "checkpoints", "ppo_latest.zip" if c == "latest" else f"ppo_{c}_steps.zip"))

    ctx = mp.get_context("forkserver")
    results = {}
    t0 = time.time()
    with ctx.Pool(a.workers) as pool:
        prot = pool.map(run_protocol, [(s, cfg, False) for s in specs])
        chunks = np.array_split(np.arange(a.episodes), a.workers)
        jobs = [(s, cfg, [int(x) for x in ch]) for s in specs for ch in chunks if len(ch)]
        eps = pool.map(run_episodes, jobs)
    for s, (p, _) in zip(specs, prot):
        ep = [e for (spec, _, _), lst in zip(jobs, eps) if spec == s for e in lst]
        errs = np.concatenate([e["window_errors"] for e in ep])
        name = "tripod" if s == "tripod" else os.path.basename(s).replace(".zip", "")
        results[name] = {
            "protocol": p,
            "episodes": {"n": len(ep), "fall_rate": float(np.mean([e["fell"] for e in ep])),
                         "window_error_mean": float(errs.mean()), "window_error_median": float(np.median(errs)),
                         "window_error_p90": float(np.percentile(errs, 90)),
                         "windows_below_20pct": float(np.mean(errs < 0.2)), "n_windows": int(len(errs))},
        }
    with open(os.path.join(out_dir, "eval.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"evaluated {len(specs)} controllers in {time.time() - t0:.0f} s\n")
    seg_names = [s["segment"] for s in next(iter(results.values()))["protocol"]["segments"]]
    print("| controller | " + " | ".join(seg_names) + " | falls (protocol) | energy J | CoT | 100-ep fall rate | window err mean / median / p90 | windows < 20 % |")
    print("|---" * (len(seg_names) + 7) + "|")
    for name, r in results.items():
        p, e = r["protocol"], r["episodes"]
        segs = " | ".join(f"{s['main_axis_error_pct']:+.1f} %" for s in p["segments"])
        print(f"| {name} | {segs} | {int(p['fell'])} | {p['energy_J']:.0f} | {p['cost_of_transport']:.2f} | "
              f"{100 * e['fall_rate']:.0f} % | {100 * e['window_error_mean']:.1f} / {100 * e['window_error_median']:.1f} / "
              f"{100 * e['window_error_p90']:.1f} % | {100 * e['windows_below_20pct']:.0f} % |")


if __name__ == "__main__":
    main()
