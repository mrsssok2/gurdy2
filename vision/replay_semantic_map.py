#!/usr/bin/env python3
"""Replay a detector_node sightings log (parameter log_path) through SemanticMap with different confirmation rules and
compare each result with the world's ground truth. Same sightings, so the settings are compared like for like.

    python3 vision/replay_semantic_map.py docs/results/data/phase4/NAME/sightings.jsonl
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gurdy_core"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_semantic_map import DEFAULT_GT, compare  # noqa: E402
from gurdy_core.vision.semantic_map import SemanticMap  # noqa: E402

SETTINGS = [  # name, skip truncated, truncated weight, SemanticMap kwargs
    ("old: truncated skipped, 3 hits", True, 1.0, dict(min_hits=3)),
    ("truncated mapped (w 0.25), 3 hits only", False, 0.25, dict(min_hits=3)),
    ("truncated mapped (w 0.25), 3 hits + span 1 s + score 0.5 + scatter 0.5 (first try)", False, 0.25,
     dict(min_hits=3, min_span=1.0, min_mean_score=0.5, max_scatter=0.5)),
    ("5 hits + span 2 s + score 0.5 + scatter 0.5 (old node default)", False, 0.25, dict(min_hits=5, min_span=2.0, min_mean_score=0.5, max_scatter=0.5)),
    ("default + dominance 4x within 0.35 m", False, 0.25, dict(min_hits=5, min_span=2.0, min_mean_score=0.5, max_scatter=0.5, dominance_ratio=4.0)),
    ("8 hits + score 0.6 + dominance 4x (node default now)", False, 0.25, dict(min_hits=8, min_span=2.0, min_mean_score=0.6, max_scatter=0.5, dominance_ratio=4.0)),
    ("as default, but 10 hits + span 3 s + score 0.7", False, 0.25, dict(min_hits=10, min_span=3.0, min_mean_score=0.7, max_scatter=0.5)),
]


def run(log, skip_trunc, wt, kw):
    m = SemanticMap(ttl=10.0, **kw)
    for s in log:
        if skip_trunc and s["truncated"]:
            continue
        m.update(s["cls"], s["p"], s["score"], s["t"], range_m=s["range"], color=s["color"], weight_scale=wt if s["truncated"] else 1.0)
    return [t.to_dict() | {"class": t.cls} for t in m.confirmed()]


def main():
    log = [json.loads(ln) for ln in open(sys.argv[1])]
    gt = json.load(open(DEFAULT_GT))["objects"]
    print(f"{len(log)} sightings ({sum(s['truncated'] for s in log)} truncated)\n")
    print("| setting | found | missing | false tracks | median err [m] | low-hit (<10) false |\n|---|---|---|---|---|---|")
    for name, skip, wt, kw in SETTINGS:
        found = run(log, skip, wt, kw)
        r = compare(found, gt, 0.6)
        miss = [x["name"] for x in r["rows"] if not x["found"]]
        # false tracks = tracks that are not the match of any GT object
        used = set()
        for g in gt:
            best, bd = None, 0.6
            for i, f in enumerate(found):
                if i in used or f["class"] != g["class"]:
                    continue
                d = ((f["x"] - g["x"]) ** 2 + (f["y"] - g["y"]) ** 2) ** 0.5
                if d < bd:
                    best, bd = i, d
            if best is not None:
                used.add(best)
        fp = [f for i, f in enumerate(found) if i not in used]
        print(f"| {name} | {sum(x['found'] for x in r['rows'])}/{len(gt)} | {', '.join(miss) or '-'} | {len(fp)} | "
              f"{r['median_err_m']:.3f} | {sum(f['hits'] < 10 for f in fp)} |")
        if kw.get("min_span") == 2.0 and not skip:
            print("   false tracks:", [(f["class"], round(f["x"], 1), round(f["y"], 1), f["hits"]) for f in fp])


if __name__ == "__main__":
    main()
