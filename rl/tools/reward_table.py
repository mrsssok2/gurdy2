"""Expected per-step reward of reference policies under the training command distribution.

    docker compose run --rm rl python3 rl/tools/reward_table.py [--episodes 4]

Policies: random = the initial PPO policy (mean 0, std = exp(log_std_init)), stand (a = 0),
tripod (Phase 1 gait following the env command). Penalty scale 0.1 (training start) and 1.0.
"""
import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.gurdy_env import GurdyEnv          # noqa: E402
from envs.reward import PENALTY_TERMS        # noqa: E402
from gurdy_core.gait import TripodGait       # noqa: E402


def run(cfg, policy, pscale, episodes, std):
    rows = []
    for ep in range(episodes):
        env = GurdyEnv(cfg["env"], reward_config=cfg["reward"])
        env.penalty_scale = pscale
        env.reset(seed=100 + ep)
        gait = TripodGait(env.robot)
        rng = np.random.default_rng(ep)
        ret, done, k = 0.0, False, 0
        acc = {"r": 0.0, "body": 0.0, "clip": 0, "track": 0.0, "pen": 0.0, "n": 0}
        while not done:
            if policy == "random":
                a = rng.normal(0.0, std, 18)
            elif policy == "stand":
                a = np.zeros(18)
            else:
                a = (gait.step(env.command) - env.q_stance) / env.scale
            _, r, term, trunc, info = env.step(a)
            t = info["reward_terms"]
            acc["r"] += r
            acc["body"] += sum(v for kk, v in t.items() if kk != "fall")
            acc["clip"] += int(info["reward_clipped"])
            acc["track"] += t["track"]
            acc["pen"] += sum(t[p] for p in PENALTY_TERMS)
            acc["n"] += 1
            done = term or trunc
        acc["fell"] = int(term)
        acc["ret"] = acc["r"]
        rows.append(acc)
    n = sum(r["n"] for r in rows)
    return {"reward": sum(r["r"] for r in rows) / n, "unclipped": sum(r["body"] for r in rows) / n,
            "clipped": sum(r["clip"] for r in rows) / n, "track": sum(r["track"] for r in rows) / n,
            "penalties": sum(r["pen"] for r in rows) / n, "falls": sum(r["fell"] for r in rows),
            "ep_return": float(np.mean([r["ret"] for r in rows])), "episodes": len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "flat.yaml"))
    ap.add_argument("--episodes", type=int, default=4)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    std = float(np.exp(cfg["ppo"]["log_std_init"]))
    print(f"per-step means over {a.episodes} x {cfg['env']['episode_s']:.0f} s episodes, training command "
          f"distribution, kappa {cfg['reward']['kappa']}, random policy std {std:.2f}\n")
    print("| policy | penalty scale | reward/step (clipped) | unclipped sum | clipped steps | track | penalties | falls | episode return |")
    print("|---|---|---|---|---|---|---|---|---|")
    for pol in ("random", "stand", "tripod"):
        for ps in (0.1, 1.0):
            r = run(cfg, pol, ps, a.episodes, std)
            print(f"| {pol} | {ps} | {r['reward']:.3f} | {r['unclipped']:.3f} | {100*r['clipped']:.0f} % | "
                  f"{r['track']:.3f} | {r['penalties']:.3f} | {r['falls']}/{r['episodes']} | {r['ep_return']:.0f} |")


if __name__ == "__main__":
    main()
