"""Is standing still the reward-optimal answer to a tiny command? Per-step reward (all terms, final
energy/smoothness weights of the given run's config) for: stand (a = 0), tripod walking the command,
and the policy, at small single-axis commands; mean over seconds 4-10 of a 10 s episode from rest.
    python3 rl/tools/deadzone_reward.py --policy dr_energy_ft:10000000
"""
import argparse
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, HERE)
from eval import Controller, resolve        # noqa: E402
from envs.gurdy_env import GurdyEnv         # noqa: E402

CMDS = [(0.03, 0, 0), (0.045, 0, 0), (0.06, 0, 0), (0, 0.03, 0), (0, 0.05, 0), (0, 0, 0.10), (0, 0, 0.15), (0, 0, 0.20)]


def job(args):
    spec, cfg, cmd, who = args
    env = GurdyEnv({**cfg["env"], "episode_s": 11.0, "resample_command": False, "obs_mode":
                    Controller(spec).obs_mode if who == "policy" else cfg["env"]["obs_mode"],
                    "domain_randomization": {"enabled": False}, "axis_aligned_prob": 0.0, "rest_start_prob": 0.0},
                   reward_config=cfg["reward"])
    env.eff_progress, env.penalty_scale, env.kappa = 1.0, 1.0, cfg["reward"]["kappa"]
    ctl = Controller(spec if who == "policy" else "tripod")
    obs, _ = env.reset(seed=0, options={"command": cmd})
    ctl.reset(env)
    rs, terms, speed = [], {}, []
    ax = int(np.argmax(np.abs(cmd)))
    while env.steps * env.policy_dt < 10.0 - 1e-9:
        a = np.zeros(18) if who == "stand" else ctl.act(obs, env)
        obs, r, _, _, info = env.step(a)
        if env.steps * env.policy_dt > 4.0:
            rs.append(r)
            speed.append([*info["body_vel"][:2], info["yaw_rate"]][ax])
            for k, v in info["reward_terms"].items():
                terms[k] = terms.get(k, 0.0) + v
    n = len(rs)
    return cmd, who, float(np.mean(rs)), {k: v / n for k, v in terms.items()}, float(np.mean(speed) / cmd[ax])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    a = ap.parse_args()
    _, path, cfg = resolve(a.policy)
    jobs = [(path, cfg, c, w) for c in CMDS for w in ("stand", "tripod", "policy")]
    with mp.get_context("forkserver").Pool(10) as pool:
        out = pool.map(job, jobs)
    print(f"reward config: {a.policy} (kappa {cfg['reward']['kappa']}, floors {cfg['reward'].get('sigma_v_floor', 0.02)} m/s / "
          f"{cfg['reward'].get('sigma_w_floor', 0.05)} rad/s, final energy/smoothness weights)\n")
    print("| command | stand: reward/step (track) | tripod: reward/step (track, energy+smooth) | policy: reward/step (speed ratio) |")
    print("|---|---|---|---|")
    pen = ("energy", "torque", "action_rate", "action_2nd", "joint_acc", "foot_slip", "vz", "ang_xy", "tilt")
    for c in CMDS:
        row = {w: (r, t, s) for cc, w, r, t, s in out if cc == c for _ in [0]}
        st, tr, po = row["stand"], row["tripod"], row["policy"]
        print(f"| {c} | {st[0]:.3f} ({st[1]['track']:.3f}) | {tr[0]:.3f} ({tr[1]['track']:.3f}, {sum(tr[1][k] for k in pen):+.3f}) | "
              f"{po[0]:.3f} ({po[2]:.2f}) |")


if __name__ == "__main__":
    main()
