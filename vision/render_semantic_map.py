#!/usr/bin/env python3
"""Plot the SLAM occupancy map with ground-truth objects (hollow) and detected semantic objects (filled), PNG.

    python3 vision/render_semantic_map.py maps/NAME.yaml semantic.json out.png [--gt GT.json]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("map_yaml"); ap.add_argument("semantic"); ap.add_argument("out")
ap.add_argument("--gt", default=str(Path(__file__).resolve().parents[1] / "docs/results/data/phase4/semantic_lab_gt.json"))
a = ap.parse_args()
my = yaml.safe_load(open(a.map_yaml))
img = np.array(Image.open(Path(a.map_yaml).parent / my["image"]).convert("L"))
res, (ox, oy, _) = my["resolution"], my["origin"]
H, W = img.shape
ext = [ox, ox + W * res, oy, oy + H * res]
found, gt = json.load(open(a.semantic))["objects"], json.load(open(a.gt))["objects"]
classes = sorted({o["class"] for o in gt} | {o["class"] for o in found})
cmap = {c: plt.cm.tab10(i) for i, c in enumerate(classes)}
fig, ax = plt.subplots(figsize=(9, 6))
ax.imshow(img, cmap="gray", extent=ext, origin="upper")
for g in gt:
    ax.scatter(g["x"], g["y"], s=260, facecolors="none", edgecolors=cmap[g["class"]], linewidths=2)
for f in found:
    ax.scatter(f["x"], f["y"], s=60, color=cmap[f["class"]], marker="o", edgecolors="k")
    ax.annotate(f["class"].replace("_", " "), (f["x"], f["y"]), xytext=(4, 6), textcoords="offset points", fontsize=7, color="yellow")
for c in classes:
    ax.scatter([], [], color=cmap[c], label=c)
ax.scatter([], [], s=200, facecolors="none", edgecolors="w", label="ground truth (ring)")
ax.legend(fontsize=7, loc="upper left", framealpha=0.6)
ax.set_title(f"Semantic map: {len(found)} detected tracks (filled) vs {len(gt)} ground-truth objects (rings)")
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
fig.tight_layout(); fig.savefig(a.out, dpi=130)
print("wrote", a.out)
