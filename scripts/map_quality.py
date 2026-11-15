#!/usr/bin/env python3
"""Score saved occupancy maps against the flat_lab world (walls + boxes) or against each other.

    python3 scripts/map_quality.py maps/flat_lab_gt.yaml maps/flat_lab_ekf.yaml     # both vs the world
    python3 scripts/map_quality.py maps/flat_lab_ekf.yaml --ref maps/flat_lab_gt.yaml   # vs another map

Needs numpy + scipy + pyyaml (not in the sim image: run it in the rl image or on the host with pip deps).
Assumes the map frame = world frame, which holds when the robot spawns at the world origin and
slam_toolbox starts there (the default flat_lab runs).

Metrics (metres):
  precision_mean / p95   distance from each occupied map cell to the nearest true obstacle cell
  recall_10cm            fraction of true wall+object cells with a mapped cell within 10 cm
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
RES = 0.05


def read_pgm(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    m = re.match(rb"P5\s+(?:#[^\n]*\n\s*)*(\d+)\s+(\d+)\s+(\d+)\s", raw)
    w, h, _ = (int(g) for g in m.groups())
    return np.frombuffer(raw[m.end():], dtype=np.uint8, count=w * h).reshape(h, w)


def load_map(yaml_path: Path):
    meta = yaml.safe_load(yaml_path.read_text())
    img = read_pgm(yaml_path.parent / meta["image"])
    occ = (255 - img) / 255.0 >= float(meta.get("occupied_thresh", 0.65))
    return occ[::-1], float(meta["resolution"]), np.array(meta["origin"][:2])     # row 0 = lowest y


def cells_xy(occ, res, origin) -> np.ndarray:
    iy, ix = np.nonzero(occ)
    return np.stack([origin[0] + (ix + 0.5) * res, origin[1] + (iy + 0.5) * res], axis=1)


def world_cells() -> np.ndarray:
    import gen_world_flat_lab as g
    pts = []
    boxes = [(c, s) for _, c, s in g.walls()]
    boxes += [(c, (s[0], s[1])) for _, k, c, s, _ in g.OBJECTS if k == "box"]
    for (cx, cy), (sx, sy) in boxes:
        xs = np.arange(cx - sx / 2, cx + sx / 2 + 1e-9, RES / 2)
        ys = np.arange(cy - sy / 2, cy + sy / 2 + 1e-9, RES / 2)
        pts.append(np.array([(x, y) for x in xs for y in ys]))
    for _, k, (cx, cy), (r, _h), _ in [o for o in g.OBJECTS if o[1] == "cylinder"]:
        a = np.linspace(0, 2 * np.pi, 60)
        pts.append(np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1))
    return np.concatenate(pts)


def nearest(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree
    return cKDTree(dst).query(src)[0]


def score(map_yaml: Path, ref_pts: np.ndarray) -> dict:
    occ, res, origin = load_map(map_yaml)
    pts = cells_xy(occ, res, origin)
    d_map = nearest(pts, ref_pts)           # mapped -> true
    d_ref = nearest(ref_pts, pts)           # true -> mapped
    return {"occupied_cells": int(len(pts)), "precision_mean_m": float(d_map.mean()),
            "precision_p95_m": float(np.percentile(d_map, 95)), "recall_10cm": float((d_ref < 0.10).mean()),
            "recall_mean_m": float(d_ref.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("maps", nargs="+", type=Path)
    ap.add_argument("--ref", type=Path, help="reference map yaml (default: the flat_lab world geometry)")
    a = ap.parse_args()
    if a.ref:
        occ, res, origin = load_map(a.ref)
        ref = cells_xy(occ, res, origin)
    else:
        ref = world_cells()
    for m in a.maps:
        s = score(m, ref)
        print(f"{m.name:28s} cells {s['occupied_cells']:5d}  precision mean {s['precision_mean_m']:.3f} m "
              f"p95 {s['precision_p95_m']:.3f} m   recall<10cm {100 * s['recall_10cm']:.1f} %  "
              f"(mean {s['recall_mean_m']:.3f} m)")


if __name__ == "__main__":
    main()
