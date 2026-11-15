"""Where does the ~10 % command-window tracking error come from? Splits rl/eval.py's 100 nominal
training-distribution episodes by the smallest non-zero command component (relative to its limit).
    python3 rl/tools/window_error_breakdown.py --controllers tripod dr_energy_ft:10000000
"""
import argparse
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, HERE)
from eval import resolve, run_episodes   # noqa: E402

LIM = np.array([0.15, 0.10, 0.60])
BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 1.01)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controllers", nargs="+", required=True)
    a = ap.parse_args()
    cfg = next(resolve(c)[2] for c in a.controllers if c != "tripod")
    specs = [("tripod", "tripod") if c == "tripod" else (c, resolve(c)[1]) for c in a.controllers]
    jobs = [(path, cfg, [int(x) for x in ch], "nominal") for _, path in specs
            for ch in np.array_split(np.arange(100), 5)]
    with mp.get_context("forkserver").Pool(10) as pool:
        res = pool.map(run_episodes, jobs)
    print("| controller | all windows | smallest non-zero component < 20 % of limit | 20-40 % | >= 40 % |\n|---|---|---|---|---|")
    for i, (label, _) in enumerate(specs):
        eps = [e for r in res[5 * i:5 * i + 5] for e in r]
        errs = np.concatenate([e["window_errors"] for e in eps])
        cmds = np.array([c for e in eps for c in e["window_commands"]])
        rel = np.abs(cmds) / LIM
        smallest = np.array([r[r > 1e-6].min() for r in rel])
        cells = [f"{100 * errs.mean():.1f} % (n={len(errs)})"]
        for lo, hi in BINS:
            m = (smallest >= lo) & (smallest < hi)
            cells.append(f"{100 * errs[m].mean():.1f} % (n={m.sum()})")
        print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
