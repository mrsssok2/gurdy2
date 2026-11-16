"""Experiment B evaluation: leg failure (limp / locked), per failed leg x type, MuJoCo, flat floor, DR off.

    python3 rl/eval_failure.py --policies terrain_dr2:latest leg_failure:latest [--episodes 10] [--workers 10]

Controllers: tripod (open loop, no adaptation) + every given policy. Protocol: standard command distribution
(fixed per seed, same for every controller), 20 s episodes, the failure happens at the end of the first command window
(step = cmd_steps, 5 s). Metrics are computed on the windows AFTER the failure only (first second of each window skipped):
fall rate (whole episode), tracking error (episode mean of window errors, same definition as rl/eval.py), cost of transport
(energy / (m g path) after the failure). A no-failure row gives the regression check. Output: rl/runs/<last policy>/eval/eval_failure.json + markdown.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from eval import ROBOT_MG, Controller, _env, resolve        # noqa: E402

TYPES = ("limp", "locked")


def run_failure_episodes(args):
    spec, cfg, seeds, leg, ftype = args
    ctl = Controller(spec)
    env = _env(cfg, cfg["env"]["episode_s"], fixed=False, obs_mode=ctl.obs_mode)
    w = env.cmd_steps
    if leg is not None:
        env.cfg["dr_fixed"] = {"leg_failure": leg, "failure_type": ftype, "failure_step": w}
    lim = env.cmd_lim
    out = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        ctl.reset(env)
        vels, cmds, en, term, done = [], [], [], False, False
        pos = [env.data.qpos[:2].copy()]
        while not done:
            cmds.append(env.command.copy())
            obs, _, term, trunc, info = env.step(ctl.act(obs, env))
            vels.append([*info["body_vel"][:2], info["yaw_rate"]])
            en.append(info["energy_J"])
            pos.append(env.data.qpos[:2].copy())
            done = term or trunc
        vels, cmds, en, pos = map(np.array, (vels, cmds, en, pos))
        first = w if leg is not None else 0                  # post-failure windows only (all windows if no failure)
        errs = []
        for k in range(first, len(cmds) - w + 1, w):
            c = cmds[k]
            if not np.any(np.abs(c) > 1e-6) or not np.allclose(cmds[k:k + w], c):
                continue
            v = vels[k + int(1.0 / env.policy_dt):k + w].mean(axis=0)
            errs.append(float(np.linalg.norm((v - c) / lim) / np.linalg.norm(c / lim)))
        path = float(np.linalg.norm(np.diff(pos[first:], axis=0), axis=1).sum())
        out.append({"seed": seed, "fell": bool(term), "steps": len(cmds), "fall_step": len(cmds) if term else None,
                    "errs": errs, "energy_J": float(en[first:].sum()), "path_m": path})
    return out


def summarize(eps):
    per = [np.mean(e["errs"]) for e in eps if e["errs"]]
    path = sum(e["path_m"] for e in eps)
    return {"n": len(eps), "fall_rate": float(np.mean([e["fell"] for e in eps])),
            "error_mean": float(np.mean(per)) if per else float("nan"),
            "error_se": float(np.std(per, ddof=1) / np.sqrt(len(per))) if len(per) > 1 else float("nan"),
            "cot": float(sum(e["energy_J"] for e in eps) / (ROBOT_MG * max(path, 1e-9)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--episodes", type=int, default=10, help="episodes per (leg, type) cell")
    ap.add_argument("--nofail-episodes", type=int, default=50)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out")
    a = ap.parse_args()
    pols = [resolve(p) for p in a.policies]
    ctrls = [("tripod", "tripod", pols[0][2])] + pols
    cells = [(None, None)] + [(leg, t) for t in TYPES for leg in range(6)]
    jobs, keys = [], []
    t0 = time.time()
    for name, path, cfg in ctrls:
        for leg, t in cells:
            n = a.nofail_episodes if leg is None else a.episodes
            seeds = list(range(1000, 1000 + n))              # same seeds for every controller and cell
            jobs.append((path, cfg, seeds, leg, t))
            keys.append((name, leg, t))
    with mp.get_context("forkserver").Pool(a.workers) as pool:
        res = pool.map(run_failure_episodes, jobs, chunksize=1)
    table = {}
    for (name, leg, t), eps in zip(keys, res):
        table.setdefault(name, {})["none" if leg is None else f"{t}:{leg}"] = summarize(eps)
    out = a.out or os.path.join(HERE, "runs", a.policies[-1].split(":")[0], "eval", "eval_failure.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(table, open(out, "w"), indent=2)
    names = list(table)
    print(f"failure eval: {len(names)} controllers x {len(cells)} cells in {time.time() - t0:.0f} s -> {out}")
    print("cell = falls % / tracking error % (post-failure windows) / CoT\n")
    print("| failure | " + " | ".join(names) + " |\n|---" + "|---" * len(names) + "|")
    keys_order = ["none"] + [f"{t}:{leg}" for t in TYPES for leg in range(6)]
    for k in keys_order:
        print(f"| {k} | " + " | ".join(
            f"{100 * table[n][k]['fall_rate']:.0f} / {100 * table[n][k]['error_mean']:.0f} / {table[n][k]['cot']:.1f}"
            for n in names) + " |")
    for t in TYPES:                                          # pooled over the 6 legs
        print(f"| **{t} (all legs)** | " + " | ".join(
            "{:.0f} / {:.0f} / {:.1f}".format(
                100 * np.mean([table[n][f"{t}:{l}"]["fall_rate"] for l in range(6)]),
                100 * np.mean([table[n][f"{t}:{l}"]["error_mean"] for l in range(6)]),
                np.mean([table[n][f"{t}:{l}"]["cot"] for l in range(6)])) for n in names) + " |")


if __name__ == "__main__":
    main()
