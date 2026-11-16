#!/usr/bin/env python3
"""Experiment A: terrain tables + charts from existing eval JSON (no new runs).  python3 scripts/exp_a_report.py"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = json.load(open("docs/results/phase2_terrain_eval.json"))
B = json.load(open("rl/runs/terrain_dr2/eval/eval_terrain.json"))
SRC = [("tripod", "tripod", A), ("ft-6M (flat DR)", "dr_energy_ft:6000000", A),
       ("terrain_v1 60M", "terrain_v1:latest", A), ("terrain_dr2 15M", "terrain_dr2:latest", B)]
COL = ["#7f7f7f", "#0072B2", "#E69F00", "#009E73"]
T = {"rough": ("peak-to-peak cm", 100), "slopes": ("slope deg", 1), "steps": ("step height cm", 100)}
cells = {t: sorted(A["tripod"][t], key=float) for t in T}
md = ["| terrain | level | " + " | ".join(n for n, *_ in SRC) + " |", "|---|---|" + "---|" * len(SRC)]
for t in T:
    for c in cells[t]:
        row = []
        for _, k, d in SRC:
            r = d[k][t][c]
            row.append(f"{100*r['success_rate']:.0f} % / {100*r['fall_rate']:.0f} % / {100*r['window_error_mean']:.0f} % / {r['cost_of_transport']:.1f}")
        md.append(f"| {t} | {float(c)*T[t][1]:g} | " + " | ".join(row) + " |")
open("docs/results/exp_A_table.md", "w").write("\n".join(md) + "\n")

for metric, ylab, fn, scale in [("success_rate", "success rate (%)", "exp_A_success.png", 100),
                                ("cost_of_transport", "cost of transport (-)", "exp_A_cot.png", 1)]:
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, t in zip(axs, T):
        xs = [float(c) * T[t][1] for c in cells[t]]
        for (n, k, d), col in zip(SRC, COL):
            ax.plot(xs, [d[k][t][c][metric] * scale for c in cells[t]], "-o", c=col, lw=2, ms=6, label=n,
                    mfc=col, mec="white")
        ax.set_title(t); ax.set_xlabel(T[t][0]); ax.grid(alpha=.25)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    axs[0].set_ylabel(ylab); axs[0].legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig("docs/results/" + fn, dpi=130); plt.close(fig)
print(open("docs/results/exp_A_table.md").read())
