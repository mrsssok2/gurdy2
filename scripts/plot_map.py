#!/usr/bin/env python3
"""PNG of a saved map with the flat_lab world outline and (optional) ground-truth path.
    python3 scripts/plot_map.py maps/NAME.yaml OUT.png [route.json]     (rl image: numpy, matplotlib, pyyaml)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_world_flat_lab as g
from map_quality import cells_xy, load_map

occ, res, org = load_map(Path(sys.argv[1]))
pts = cells_xy(occ, res, org)
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.scatter(pts[:, 0], pts[:, 1], s=2, c="k", label="map (occupied)")
for _, (cx, cy), (sx, sy) in g.walls():
    ax.add_patch(plt.Rectangle((cx - sx / 2, cy - sy / 2), sx, sy, fill=False, ec="tab:red", lw=0.8))
for _, k, (cx, cy), s, _ in g.OBJECTS:
    if k == "box":
        ax.add_patch(plt.Rectangle((cx - s[0] / 2, cy - s[1] / 2), s[0], s[1], fill=False, ec="tab:red", lw=0.8))
    else:
        ax.add_patch(plt.Circle((cx, cy), s[0], fill=False, ec="tab:red", lw=0.8))
if len(sys.argv) > 3:
    p = np.array(json.load(open(sys.argv[3]))["path"])
    ax.plot(p[:, 1], p[:, 2], c="tab:blue", lw=0.8, label="ground-truth path")
ax.plot([], [], c="tab:red", label="true world outline")
ax.set_aspect("equal"); ax.legend(loc="lower right", fontsize=7); ax.set_title(Path(sys.argv[1]).stem)
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=110)
