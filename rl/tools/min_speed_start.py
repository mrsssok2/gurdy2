"""Start check at the Nav2 operating point: from rest, does the policy start at exactly the lifted minimum speed?
6 commands (vx ±lin, vy ±lin, wz ±ang) x 8 start timings (rl/eval.py START_DELAYS); a start fails when the mean speed
3-11 s after the command is < 20 % of it (the gate used in every eval). Several levels can be given to find the smallest passing one.
    python3 rl/tools/min_speed_start.py --controllers terrain_dr2:15000000 --levels 0.05,0.15 0.06,0.18
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import eval as ev  # noqa: E402


def _run(job):
    path, cfg, key, vec = job      # forkserver workers do not see START_CMDS changes made in the parent
    ev.START_CMDS[key] = (vec, (0, 0, 0))
    return ev.run_start((path, cfg, key))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controllers", nargs="+", required=True)
    ap.add_argument("--levels", nargs="+", default=["0.05,0.15"], help="lin,ang pairs")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    levels = [tuple(float(v) for v in s.split(",")) for s in a.levels]
    cfg, specs = None, []
    for c in a.controllers:
        _, path, cfg = ev.resolve(c)[:3]
        specs.append((c, path))
    jobs, names = [], {}
    for lin, ang in levels:
        for tag, vec in {"vx+": (lin, 0, 0), "vx-": (-lin, 0, 0), "vy+": (0, lin, 0), "vy-": (0, -lin, 0),
                         "wz+": (0, 0, ang), "wz-": (0, 0, -ang)}.items():
            key = f"{lin}/{ang}/{tag}"
            for _, path in specs:
                jobs.append((path, cfg, key, vec))
    with mp.get_context("forkserver").Pool(10) as pool:
        out = pool.map(_run, jobs)
    res = {}
    for label, path in specs:
        for lin, ang in levels:
            row, fails = [], 0
            for tag in ("vx+", "vx-", "vy+", "vy-", "wz+", "wz-"):
                r = [x for s, n, x in out if s == path and n == f"{lin}/{ang}/{tag}"][0]
                f = int(sum(v < 0.2 for v in r))
                fails += f
                row.append((tag, f, float(np.mean(r)), float(np.min(r))))
            res[f"{label}@{lin}/{ang}"] = {"failed": fails, "n": 48, "cells": row}
            print(f"\n{label}  lin {lin} m/s  ang {ang} rad/s: failed starts {fails}/48 {'PASS' if fails == 0 else 'FAIL'}")
            print("| cmd | failed/8 | mean speed ratio | worst timing ratio |\n|---|---|---|---|")
            for tag, f, m, mn in row:
                print(f"| {tag} | {f}/8 | {m:.2f} | {mn:.2f} |")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
