#!/usr/bin/env python3
"""Time-lapse (GIF + MP4), coverage-vs-time plot and final map PNG from an exploration run (rl image: numpy, matplotlib, imageio, ffmpeg).
    python3 scripts/render_explore.py RUN_DIR OUT_PREFIX        e.g. docs/results/data/phase3/explore_gt docs/results/phase3_explore_gt
Reads RUN_DIR/snaps/*.npz + timeseries.csv + explore_log.json; writes OUT_PREFIX_timelapse.gif/.mp4, OUT_PREFIX_coverage.png.
Frames: unknown = grey, free = white, occupied = black, true world outline red, ground-truth trail blue, goals as orange crosses.
"""
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_world_flat_lab as g

run, prefix = Path(sys.argv[1]), sys.argv[2]
snaps = sorted((run / "snaps").glob("*.npz"))
ts = np.genfromtxt(run / "timeseries.csv", delimiter=",", names=True)
log = json.loads((run / "explore_log.json").read_text()) if (run / "explore_log.json").exists() else None
X0, X1, Y0, Y1 = g.X_MIN - 0.5, g.X_MAX + 0.5, g.Y_MIN - 0.5, g.Y_MAX + 0.5


def outline(ax):
    for _, (cx, cy), (sx, sy) in g.walls():
        ax.add_patch(plt.Rectangle((cx - sx / 2, cy - sy / 2), sx, sy, fill=False, ec="tab:red", lw=0.6, alpha=0.7))
    for _, k, (cx, cy), s, _ in g.OBJECTS:
        ax.add_patch(plt.Rectangle((cx - s[0] / 2, cy - s[1] / 2), s[0], s[1], fill=False, ec="tab:red", lw=0.6, alpha=0.7) if k == "box"
                     else plt.Circle((cx, cy), s[0], fill=False, ec="tab:red", lw=0.6, alpha=0.7))


def render(npz, path):
    d = np.load(npz)
    grid, res, org, t = d["grid"], float(d["res"]), d["origin"], float(d["t"])
    img = np.full(grid.shape, 0.6)
    img[(grid >= 0) & (grid < 50)] = 1.0
    img[grid >= 50] = 0.0
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=90)
    h, w = grid.shape
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, origin="lower", extent=(org[0], org[0] + w * res, org[1], org[1] + h * res))
    outline(ax)
    m = ts["t_s"] <= t
    ax.plot(ts["x"][m], ts["y"][m], c="tab:blue", lw=0.9)
    ax.plot(*d["pose"], "o", c="tab:blue", ms=5)
    if log:
        for gl in log["goals"]:
            if gl["t0"] <= t and not gl["final"]:
                ax.plot(*gl["xy"], "x", c="tab:orange", ms=4, mew=1)
    row = ts[np.searchsorted(ts["t_s"], t, side="right") - 1]
    ax.set_title(f"t = {t:5.0f} s   mapped {row['interior_fraction'] * 100:4.1f} %   path {float(d['path']):5.1f} m", fontsize=10)
    ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1); ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


frames_dir = run / "frames"
frames_dir.mkdir(exist_ok=True)
paths = []
for i, s in enumerate(snaps):
    p = frames_dir / f"{i:04d}.png"
    render(s, p)
    paths.append(p)
imgs = [imageio.imread(p) for p in paths]
hold = [imgs[-1]] * 8
imageio.mimsave(prefix + "_timelapse.gif", imgs + hold, duration=0.25, loop=0)
imageio.mimsave(prefix + "_timelapse.mp4", imgs + hold, fps=5, macro_block_size=2)
render(snaps[-1], prefix + "_map.png")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].plot(ts["t_s"], ts["interior_fraction"] * 100, c="tab:blue"); ax[0].set_ylabel("lab interior mapped (%)")
ax[1].plot(ts["t_s"], ts["path_m"], c="tab:green"); ax[1].set_ylabel("ground-truth path (m)")
for a in ax:
    a.set_xlabel("sim time (s)"); a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(prefix + "_coverage.png", dpi=110)
print(f"{len(snaps)} frames -> {prefix}_timelapse.gif/.mp4, {prefix}_map.png, {prefix}_coverage.png")
