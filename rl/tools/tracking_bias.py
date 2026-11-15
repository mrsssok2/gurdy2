"""Overshoot diagnosis: SIGNED tracking error per axis and commanded speed, measured three ways.
Signed error = (measured - command) / command: positive = faster than commanded (overshoot),
negative = slower (undershoot), -100 % = did not move.

For each controller and each single-axis command (4 speeds x 2 signs per axis), walk 12 s from rest
(deterministic, nominal physics) and average over the last 6 s:
  * body   : body-frame velocity from the env (what rl/eval.py's episode windows use), plain mean
  * reward : the same signal through the reward's 0.2 s EMA (what the reward compares to the command)
  * heading: world-pose displacement rotated into the yaw-only heading frame + world yaw rate
             (what the Phase 1 tracking protocol / analyze_tracking uses)
Then a least-squares fit measured = slope * command + offset per axis (scale vs offset error).

    python3 rl/tools/tracking_bias.py --controllers tripod dr_energy_v1:40000000 dr_energy_ft:10000000
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, HERE)
from eval import Controller, _env, resolve                      # noqa: E402
from gurdy_core.experiments import quat_to_rpy                  # noqa: E402

SPEEDS = {0: [0.03, 0.06, 0.10, 0.15], 1: [0.03, 0.05, 0.08, 0.10], 2: [0.10, 0.20, 0.30, 0.60]}
AXES = ["vx", "vy", "wz"]
T_TOTAL, T_AVG = 12.0, 6.0


def job(args):
    spec, cfg, axis, speed, pre = (args + (0.0,))[:5]   # pre: walk this many s at full speed first
    cmd = np.zeros(3)
    cmd[axis] = speed
    ctl = Controller(spec)
    env = _env(cfg, T_TOTAL + 1, fixed=True, obs_mode=ctl.obs_mode)
    big = np.zeros(3)
    big[axis] = np.sign(speed) * SPEEDS[axis][2]
    obs, _ = env.reset(seed=0, options={"command": big if pre else cmd})
    ctl.reset(env)
    body, ema, pose = [], [], []
    while env.steps * env.policy_dt < T_TOTAL - 1e-9:
        env.command = big if env.steps * env.policy_dt < pre else cmd
        obs, _, term, _, info = env.step(ctl.act(obs, env))
        if env.steps * env.policy_dt > T_TOTAL - T_AVG:
            body.append([*info["body_vel"][:2], info["yaw_rate"]])
            ema.append(info["track_vel"])
            pose.append(env.data.qpos[:7].copy())
        if term:
            break
    body, ema, pose = np.array(body), np.array(ema), np.array(pose)
    _, _, yaw = quat_to_rpy(pose[:, 3], pose[:, 4], pose[:, 5], pose[:, 6])
    yaw = np.unwrap(yaw)
    dxy = np.diff(pose[:, :2], axis=0)
    c, s = np.cos(yaw[:-1]), np.sin(yaw[:-1])
    dt = env.policy_dt * len(dxy)
    heading = np.array([np.sum(c * dxy[:, 0] + s * dxy[:, 1]) / dt, np.sum(-s * dxy[:, 0] + c * dxy[:, 1]) / dt,
                        (yaw[-1] - yaw[0]) / dt])
    return {"spec": spec, "axis": axis, "cmd": float(speed), "pre": pre, "fell": bool(term),
            "body": body.mean(0).tolist(), "reward": ema.mean(0).tolist(), "heading": heading.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controllers", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "docs", "results", "phase2_tracking_bias.json"))
    a = ap.parse_args()
    specs, cfg = [], None
    for c in a.controllers:
        if c != "tripod":
            cfg = resolve(c)[2]
    for c in a.controllers:
        specs.append(("tripod", "tripod") if c == "tripod" else (c, resolve(c)[1]))
    jobs = [(path, cfg, ax, sg * v) for _, path in specs for ax in range(3) for v in SPEEDS[ax] for sg in (1, -1)]
    # dead-zone check while already walking: 4 s at the 3rd speed, then the small command
    jobs += [(path, cfg, ax, sg * v, 4.0) for _, path in specs for ax in range(3) for v in SPEEDS[ax][:2] for sg in (1, -1)]
    with mp.get_context("forkserver").Pool(10) as pool:
        res = pool.map(job, jobs)
    walking = [r for r in res if r["pre"]]
    res = [r for r in res if not r["pre"]]
    json.dump({"from_rest": res, "after_walking": walking}, open(a.out, "w"), indent=2)

    print("Signed main-axis error (measured - command) / command (+ = faster than commanded), last 6 s of 12 s from rest.\n"
          "Three measurements: body = body-frame mean (eval episodes), reward = 0.2 s EMA (reward),"
          " heading = heading frame from world pose (Phase 1 protocol).\n")
    for label, path in specs:
        rows = [r for r in res if r["spec"] == path]
        print(f"### {label}\n")
        print("| axis | command | body | reward (EMA) | heading | off-axis (body) |\n|---|---|---|---|---|---|")
        for ax in range(3):
            for r in sorted([r for r in rows if r["axis"] == ax], key=lambda r: r["cmd"]):
                e = {k: 100 * (r[k][ax] - r["cmd"]) / r["cmd"] for k in ("body", "reward", "heading")}
                off = [f"{AXES[j]} {r['body'][j]:+.3f}" for j in range(3) if j != ax]
                print(f"| {AXES[ax]} | {r['cmd']:+.2f} | {e['body']:+.1f} % | {e['reward']:+.1f} % | "
                      f"{e['heading']:+.1f} % | {', '.join(off)}{' FELL' if r['fell'] else ''} |")
        print("\nFit measured = slope x command + offset (body frame):\n")
        print("| axis | slope (1 = no scale error) | offset | max |reward - body| | max |heading - body| |\n|---|---|---|---|---|")
        for ax in range(3):
            rr = [r for r in rows if r["axis"] == ax]
            x = np.array([r["cmd"] for r in rr])
            y = np.array([r["body"][ax] for r in rr])
            slope, off = np.polyfit(x, y, 1)
            d1 = max(abs(r["reward"][ax] - r["body"][ax]) for r in rr)
            d2 = max(abs(r["heading"][ax] - r["body"][ax]) for r in rr)
            unit = "rad/s" if ax == 2 else "m/s"
            print(f"| {AXES[ax]} | {slope:.3f} | {off:+.4f} {unit} | {d1:.4f} | {d2:.4f} |")
        print("\nSmall commands after 4 s of walking at the 3rd speed (speed ratio measured/command, body frame):\n")
        cells = []
        for ax in range(3):
            for r in sorted([r for r in walking if r["spec"] == path and r["axis"] == ax], key=lambda r: r["cmd"]):
                cells.append(f"{AXES[ax]} {r['cmd']:+.2f}: {r['body'][ax] / r['cmd']:.2f}")
        print(" · ".join(cells) + "\n")


if __name__ == "__main__":
    main()
