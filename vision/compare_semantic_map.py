#!/usr/bin/env python3
"""Compare a saved semantic map (detector_node save_path / one /semantic_map/objects message) with the world's
ground truth (docs/results/data/phase4/semantic_lab_gt.json).

    python3 vision/compare_semantic_map.py MAP.json [--gt GT.json] [--max-dist 0.6]

Greedy nearest match per class. Reports found / missed / false objects, per-object xy error, duplicates (more
confirmed tracks than ground-truth objects of that class). Map and GT must be in the same frame (world == map when
the robot spawns at the origin and SLAM starts there; with `target_frame:=odom` also true for ground-truth odometry).
"""
import argparse
import json
import math
from pathlib import Path

DEFAULT_GT = Path(__file__).resolve().parents[1] / "docs/results/data/phase4/semantic_lab_gt.json"


def compare(found: list, gt: list, max_dist: float = 0.6) -> dict:
    used, rows = set(), []
    for g in gt:
        best, bd = None, max_dist
        for i, f in enumerate(found):
            if i in used or f["class"] != g["class"]:
                continue
            d = math.hypot(f["x"] - g["x"], f["y"] - g["y"])
            if d < bd:
                best, bd = i, d
        if best is not None:
            used.add(best)
        rows.append({"name": g["name"], "class": g["class"], "found": best is not None, "err_m": bd if best is not None else None,
                     "color_ok": None if best is None or not g.get("color") else found[best].get("color") == g["color"]})
    n_found = sum(r["found"] for r in rows)
    errs = sorted(r["err_m"] for r in rows if r["found"])
    return {"rows": rows, "recall": n_found / len(gt) if gt else float("nan"),
            "false_or_duplicate": len(found) - len(used), "median_err_m": errs[len(errs) // 2] if errs else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("map")
    ap.add_argument("--gt", default=str(DEFAULT_GT))
    ap.add_argument("--max-dist", type=float, default=0.6)
    a = ap.parse_args()
    found = json.load(open(a.map))["objects"]
    res = compare(found, json.load(open(a.gt))["objects"], a.max_dist)
    for r in res["rows"]:
        print(f"{r['name']:15s} {r['class']:18s} " + (f"found, error {r['err_m']:.3f} m" if r["found"] else "MISSED")
              + ("" if r["color_ok"] is None else f", colour {'ok' if r['color_ok'] else 'WRONG'}"))
    print(f"recall {res['recall']:.2f}, false/duplicate tracks {res['false_or_duplicate']}, median error {res['median_err_m']}")


if __name__ == "__main__":
    main()
