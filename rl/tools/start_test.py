"""Start-from-rest test: does the controller start moving for every pure command?

Stand (zero command) for 2 s + delay d (d = 0.0 ... 0.7 s, so the start lands at different gait-clock
phases), then apply a pure command for 8 s. Failure = main-axis mean velocity over the last 5 s
< 20 % of the command (deterministic policy, nominal physics).
    python3 rl/tools/start_test.py --controllers tripod flat_bc_v1:latest dr_energy_v1:latest
"""
import argparse
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, HERE)
from eval import Controller, _env, resolve   # noqa: E402

CMDS = {"+vx": (0.10, 0, 0), "-vx": (-0.10, 0, 0), "+vy": (0, 0.08, 0), "-vy": (0, -0.08, 0),
        "+wz": (0, 0, 0.3), "-wz": (0, 0, -0.3)}
DELAYS = np.arange(0.0, 0.71, 0.1)


def job(args):
    spec, cfg, name = args
    cmd = np.array(CMDS[name], float)
    axis = int(np.argmax(np.abs(cmd)))
    ctl = Controller(spec)
    res = []
    for d in DELAYS:
        env = _env(cfg, 12.0, fixed=True, obs_mode=ctl.obs_mode)
        obs, _ = env.reset(seed=0, options={"command": (0, 0, 0)})
        ctl.reset(env)
        vel = []
        while env.steps * env.policy_dt < 2.0 + d + 8.0 - 1e-9:
            t = env.steps * env.policy_dt
            env.command = cmd if t >= 2.0 + d - 1e-9 else np.zeros(3)
            obs, _, term, _, info = env.step(ctl.act(obs, env))
            if t >= 2.0 + d + 3.0:
                vel.append([*info["body_vel"][:2], info["yaw_rate"]][axis])
            if term:
                break
        ratio = float(np.mean(vel) / cmd[axis]) if vel else 0.0
        res.append(ratio)
    return spec, name, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controllers", nargs="+", required=True)
    a = ap.parse_args()
    specs, cfg = [], None
    for c in a.controllers:
        if c == "tripod":
            specs.append(("tripod", "tripod"))
        else:
            name, path, cfg = resolve(c)
            specs.append((name, path))
    jobs = [(path, cfg, k) for _, path in specs for k in CMDS]
    with mp.get_context("forkserver").Pool(10) as pool:
        out = pool.map(job, jobs)
    print(f"start-from-rest: {len(DELAYS)} start phases per command; cell = failed starts "
          f"(speed < 20 % of command) / mean speed ratio\n")
    print("| controller | " + " | ".join(CMDS) + " | total failed |\n|---" + "|---" * (len(CMDS) + 1) + "|")
    for label, path in specs:
        cells, tot = [], 0
        for k in CMDS:
            r = next(x for s, n, x in out if s == path and n == k)
            f = sum(v < 0.2 for v in r)
            tot += f
            cells.append(f"{f}/{len(r)} ({np.mean(r):.2f})")
        print(f"| {label} | " + " | ".join(cells) + f" | {tot}/{len(CMDS) * len(DELAYS)} |")


if __name__ == "__main__":
    main()
