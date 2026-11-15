"""Start-from-rest test (the 2.7b gate) as a standalone table; the implementation lives in rl/eval.py
(run_start), which also runs it in every evaluation.
    python3 rl/tools/start_test.py --controllers tripod flat_bc_v1:latest dr_energy_v1:40000000
"""
import argparse
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, HERE)
from eval import START_CMDS, START_DELAYS, resolve, run_start, start_gate_jobs, start_gate_summary  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controllers", nargs="+", required=True)
    a = ap.parse_args()
    specs, cfg = [], None
    for c in a.controllers:
        if c != "tripod":
            cfg = resolve(c)[2]
    for c in a.controllers:
        specs.append(("tripod", "tripod") if c == "tripod" else resolve(c)[:2])
    with mp.get_context("forkserver").Pool(10) as pool:
        out = pool.map(run_start, start_gate_jobs([(path, cfg) for _, path in specs]))
    print(f"start-from-rest: {len(START_DELAYS)} start phases per command; cell = failed starts "
          f"(speed < 20 % of command) / mean speed ratio\n")
    print("| controller | " + " | ".join(START_CMDS) + " | total failed | gate |\n|---" + "|---" * (len(START_CMDS) + 2) + "|")
    for label, path in specs:
        g = start_gate_summary(out, path)
        cells = [f"{g['per_command_failed'][k]}/{len(START_DELAYS)} ({g['per_command_speed_ratio'][k]:.2f})" for k in START_CMDS]
        print(f"| {label} | " + " | ".join(cells) + f" | {g['failed_starts']}/{g['n_starts']} | {'PASS' if g['pass'] else 'FAIL'} |")


if __name__ == "__main__":
    main()
