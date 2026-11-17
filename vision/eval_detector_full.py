#!/usr/bin/env python3
"""Full detector comparison on a split: mAP50, mAP50-95, per-class P/R, confusion matrix, edge-truncated boxes, mannequins.

    docker compose run --rm vision python3 vision/eval_detector_full.py --models models/detector/det_v1_n.onnx \
        models/detector/det_v1_s.onnx --data datasets/synth_v1 --split test --out docs/results/phase4_detector_full.md

Speed and 3D error come from vision/eval_detector.py (same split). Metrics use gurdy_core.vision.metrics (all-point AP).
A GT box is "edge-truncated" if it touches the image border (within 2 px): the object is cut by the frame.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gurdy_core"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_detector import yolo_gt  # noqa: E402
from gurdy_core.vision.classes import CLASSES  # noqa: E402
from gurdy_core.vision.detector import OnnxDetector, iou_xyxy  # noqa: E402
from gurdy_core.vision.metrics import evaluate  # noqa: E402

THRS = [round(0.5 + 0.05 * i, 2) for i in range(10)]


def touches_edge(box, w, h, m=2.0):
    return box[0] <= m or box[1] <= m or box[2] >= w - m or box[3] >= h - m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    root = Path(a.data)
    files = sorted((root / "images" / a.split).glob("*.png"))
    imgs = [np.array(Image.open(f).convert("RGB")) for f in files]
    H, W = imgs[0].shape[:2]
    gts = [yolo_gt(root / "labels" / a.split / f"{f.stem}.txt", W, H) for f in files]
    n_gt = sum(len(g) for g in gts)
    out = [f"# Detector comparison on {a.data}/{a.split}: {len(files)} images, {n_gt} boxes", "",
           f"Operating point: score >= {a.conf}, IoU 0.5 for P/R and confusion. AP all-point interpolation; "
           f"mAP50-95 = mean over IoU 0.50:0.05:0.95.", ""]
    summary = []
    nC = len(CLASSES)
    for mp in a.models:
        name = Path(mp).stem
        det = OnnxDetector(mp, "CPUExecutionProvider", conf=0.05)
        per_image = [{"pred": [(d.cls, d.score, d.box) for d in det(im)], "gt": g} for im, g in zip(imgs, gts)]
        r50 = evaluate(per_image, nC, 0.5, a.conf)
        maps = [evaluate(per_image, nC, t, a.conf)["mAP"] for t in THRS]
        summary.append((name, r50["mAP"], float(np.mean(maps))))
        out += [f"## {name}", "", f"mAP50 **{r50['mAP']:.3f}**, mAP50-95 **{np.mean(maps):.3f}**", "",
                "| class | GT | AP50 | precision | recall |", "|---|---|---|---|---|"]
        for c, n in enumerate(CLASSES):
            out.append(f"| {n} | {r50[c]['n_gt']} | {r50[c]['ap']:.3f} | {r50[c]['precision']:.3f} | {r50[c]['recall']:.3f} |")
        cm = np.zeros((nC + 1, nC + 1), int)  # [gt, pred]; index nC = background / missed
        rec = []
        for p, g in zip(per_image, gts):
            preds = [(k, s, b) for (k, s, b) in p["pred"] if s >= a.conf]
            used = set()
            for gc, gb in g:
                best, bj = 0.0, -1
                for j, (pc, ps, pb) in enumerate(preds):
                    if j in used:
                        continue
                    iou = float(iou_xyxy(np.array([gb]), np.array([pb]))[0, 0])
                    if iou > best:
                        best, bj = iou, j
                if bj >= 0 and best >= 0.5:
                    used.add(bj)
                    cm[gc, preds[bj][0]] += 1
                else:
                    cm[gc, nC] += 1
            for j, (pc, ps, pb) in enumerate(preds):
                if j not in used:
                    cm[nC, pc] += 1
            for gc, gb in g:
                same = [float(iou_xyxy(np.array([gb]), np.array([pb]))[0, 0]) for (pc, ps, pb) in preds if pc == gc]
                bi = max(same) if same else 0.0
                rec.append((gc, touches_edge(gb, W, H), bi >= 0.5, bi, (gb[3] - gb[1]) / max(gb[2] - gb[0], 1)))
        names = list(CLASSES) + ["bg/missed"]
        out += ["", "Confusion matrix (rows = ground truth, columns = prediction; last column = missed, last row = false positive):", "",
                "| GT \\ pred | " + " | ".join(n[:8] for n in names) + " |", "|---|" + "---|" * len(names)]
        for i, n in enumerate(names):
            out.append(f"| {n} | " + " | ".join(str(v) for v in cm[i]) + " |")
        pairs = [(CLASSES[i], CLASSES[j], int(cm[i, j])) for i in range(nC) for j in range(nC) if i != j and cm[i, j]]
        out += ["", "Class confusions (GT -> predicted as): " + (", ".join(f"{x}->{y} {n}" for x, y, n in sorted(pairs, key=lambda t: -t[2])) or "none"), ""]
        rec = np.array(rec, float)
        out += ["Edge-truncated vs fully visible GT boxes (recall = same-class box with IoU >= 0.5 at the operating point):", "",
                "| subset | GT | recall | mean best IoU |", "|---|---|---|---|"]
        for label, m in [("all", np.ones(len(rec), bool)), ("edge-truncated", rec[:, 1] == 1), ("fully visible", rec[:, 1] == 0),
                         ("mannequin, all", rec[:, 0] == 5), ("mannequin, edge-truncated", (rec[:, 0] == 5) & (rec[:, 1] == 1)),
                         ("mannequin, fully visible", (rec[:, 0] == 5) & (rec[:, 1] == 0)),
                         ("mannequin, box h/w < 1.5 (legs/torso only)", (rec[:, 0] == 5) & (rec[:, 4] < 1.5)),
                         ("mannequin, box h/w >= 1.5", (rec[:, 0] == 5) & (rec[:, 4] >= 1.5))]:
            if m.sum():
                out.append(f"| {label} | {int(m.sum())} | {rec[m, 2].mean():.3f} | {rec[m, 3].mean():.3f} |")
        out.append("")
    out += ["## Summary", "", "| model | mAP50 | mAP50-95 |", "|---|---|---|"] + [f"| {n} | {m:.3f} | {m2:.3f} |" for n, m, m2 in summary]
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
