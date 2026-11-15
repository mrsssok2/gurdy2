"""Print the key training curves of a run as a table (from its TensorBoard event files).

    docker compose run --rm rl python3 rl/tools/summarize_run.py smoke [--points 6]
"""
import argparse
import glob
import os

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

KEYS = [
    ("rollout/ep_rew_mean", "return"), ("rollout/ep_len_mean", "ep_len"), ("gurdy/fall_rate", "fall"),
    ("gurdy/track_rel_error", "trk_err"), ("reward_terms/track", "r_track"),
    ("reward_terms/foot_slip", "r_slip"), ("reward_terms/joint_acc", "r_jacc"), ("reward_terms/energy", "r_energy"),
    ("reward_terms/ang_xy", "r_angxy"), ("reward_terms/action_rate", "r_arate"), ("reward_terms/vz", "r_vz"),
    ("reward_terms/torque", "r_torque"), ("reward_terms/action_2nd", "r_a2nd"), ("reward_terms/air_time", "r_air"), ("reward_terms/stand_still", "r_still"),
    ("reward_terms/tilt", "r_tilt"), ("reward_terms/fall", "r_fall"),
    ("gurdy/action_std", "eff_std"), ("train/std", "std"), ("train/entropy_loss", "entropy"), ("train/approx_kl", "kl"),
    ("train/clip_fraction", "clipfrac"), ("train/explained_variance", "expl_var"),
    ("train/value_loss", "value_loss"), ("train/policy_gradient_loss", "pg_loss"),
    ("gurdy/penalty_scale", "pen_scale"), ("gurdy/reward_clipped_frac", "clipped"),
    ("gurdy/eff_progress", "eff_prog"), ("gurdy/dr_scale", "dr_scale"),
    ("time/fps", "fps"), ("gurdy/ram_mib", "ram_MiB"), ("gurdy/kappa", "kappa"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--points", type=int, default=6)
    a = ap.parse_args()
    root = os.path.join(os.path.dirname(__file__), "..", "runs", a.run, "tb")
    files = sorted(glob.glob(os.path.join(root, "**", "events.out.tfevents.*"), recursive=True))
    series = {}
    for f in files:
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags()["scalars"]:
            for e in ea.Scalars(tag):
                series.setdefault(tag, {})[e.step] = e.value
    steps = sorted(series.get("rollout/ep_rew_mean", {}))
    if not steps:
        print("no data yet")
        return
    idx = np.unique(np.linspace(0, len(steps) - 1, a.points).round().astype(int))
    cols = [steps[i] for i in idx]
    print(f"run {a.run}: {len(steps)} logged iterations, last step {steps[-1]:,}\n")
    print("| metric | " + " | ".join(f"{s/1e6:.2f}M" for s in cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for tag, name in KEYS:
        if tag not in series:
            continue
        vals = []
        for s in cols:
            d = series[tag]
            near = min(d, key=lambda k: abs(k - s))   # nearest logged step
            v = d[near]
            vals.append(f"{v:.3g}" if abs(v) < 1e4 else f"{v:.0f}")
        print(f"| {name} | " + " | ".join(vals) + " |")


if __name__ == "__main__":
    main()
