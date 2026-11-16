#!/usr/bin/env python3
"""Experiment B: leg-failure table + chart from the eval JSON (no new runs).  python3 scripts/exp_b_report.py
Input: docs/results/exp_B_leg_failure.json (copy of rl/runs/leg_failure/eval/eval_failure.json)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = json.load(open("docs/results/exp_B_leg_failure.json"))
SRC = [("tripod (open loop)", "tripod", "#2a78d6"), ("terrain_dr2 15M (no adaptation)", "terrain_dr2:latest", "#eb6834"),
       ("leg_failure 30M (adapted)", "leg_failure:latest", "#1baf7a")]
TYPES = ("limp", "locked")


def pooled(k, t):
    """Mean over the 6 legs; SE from the 6 per-leg SEs (independent cells)."""
    c = [D[k][f"{t}:{l}"] for l in range(6)]
    return (np.mean([x["error_mean"] for x in c]), np.sqrt(sum(x["error_se"] ** 2 for x in c)) / 6,
            np.mean([x["fall_rate"] for x in c]), np.mean([x["cot"] for x in c]))


names = list(D)
md = ["| failure | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
fmt = lambda f, e, se, c: f"{100 * f:.0f} % / {100 * e:.0f} ± {100 * se:.0f} % / {c:.1f}"
for k in ["none"] + [f"{t}:{l}" for t in TYPES for l in range(6)]:
    md.append(f"| {k} | " + " | ".join(fmt(D[n][k]["fall_rate"], D[n][k]["error_mean"], D[n][k]["error_se"], D[n][k]["cot"])
                                       for n in names) + " |")
for t in TYPES:
    row = []
    for n in names:
        e, se, f, c = pooled(n, t)
        row.append("**" + fmt(f, e, se, c) + "**")
    md.append(f"| **{t}, all legs** | " + " | ".join(row) + " |")
TABLE = "\n".join(md) + "\n"

fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
w = 0.26
for ax, t in zip(axs, TYPES):
    for i, (lab, k, col) in enumerate(SRC):
        e = [100 * D[k][f"{t}:{l}"]["error_mean"] for l in range(6)] + [100 * pooled(k, t)[0]]
        se = [100 * D[k][f"{t}:{l}"]["error_se"] for l in range(6)] + [100 * pooled(k, t)[1]]
        x = np.r_[np.arange(6), 6.4] + (i - 1) * w
        ax.bar(x, e, w - 0.03, color=col, label=lab, yerr=se, error_kw=dict(lw=1, ecolor="#52514e", capsize=2))
        ax.text(x[-1], e[-1] + se[-1] + 3, f"{e[-1]:.0f}", ha="center", fontsize=8, color="#0b0b0b")
    ax.axhline(40, color="#52514e", lw=1, ls="--")
    ax.set_xticks(list(range(6)) + [6.4]); ax.set_xticklabels([f"leg {l}" for l in range(6)] + ["all legs"])
    ax.set_title(f"{t} leg", loc="left", fontsize=11); ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axs[0].set_ylabel("tracking error after the failure (%)")
axs[0].set_ylim(0, 118)
axs[1].legend(frameon=False, fontsize=8.5, loc="upper right")
fig.suptitle("Experiment B: tracking error after one leg fails (20 episodes per bar, mean ± SE; lower is better; dashed = 40 % criterion)", fontsize=10, x=0.01, ha="left")
fig.tight_layout(); fig.savefig("docs/results/exp_B_leg_failure.png", dpi=130); plt.close(fig)
print(TABLE)
