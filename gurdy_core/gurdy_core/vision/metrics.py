"""Detection metrics: per-class precision/recall at a fixed score threshold and AP@IoU (all-point interpolation)."""
from __future__ import annotations

import numpy as np

from .detector import iou_xyxy


def match(preds: list, gts: list, iou_thr: float = 0.5):
    """Greedy match by score within one image & class. preds: [(score, box)], gts: [box].
    Returns list of (score, is_tp) for the preds and the number of unmatched gts."""
    preds = sorted(preds, key=lambda p: -p[0])
    used, out = set(), []
    ious = iou_xyxy(np.array([p[1] for p in preds]), np.array(gts)) if preds and gts else None
    for k, (s, _) in enumerate(preds):
        tp = False
        if ious is not None:
            for j in np.argsort(-ious[k]):
                if ious[k, j] < iou_thr:
                    break
                if int(j) not in used:
                    used.add(int(j))
                    tp = True
                    break
        out.append((s, tp))
    return out, len(gts) - len(used)


def average_precision(scores: np.ndarray, tp: np.ndarray, n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    o = np.argsort(-np.asarray(scores))
    tp = np.asarray(tp, float)[o]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / n_gt, ctp / np.maximum(ctp + cfp, 1e-9)
    mrec, mpre = np.concatenate([[0], rec, [1]]), np.concatenate([[1], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())


def evaluate(per_image: list, n_classes: int, iou_thr: float = 0.5, score_thr: float = 0.35) -> dict:
    """per_image: [{'pred': [(cls, score, box)], 'gt': [(cls, box)]}]. Returns per-class AP/precision/recall."""
    res = {}
    for c in range(n_classes):
        sc, tp, ngt, ntp_thr, npred_thr = [], [], 0, 0, 0
        for im in per_image:
            p = [(s, b) for (k, s, b) in im["pred"] if k == c]
            g = [b for (k, b) in im["gt"] if k == c]
            ngt += len(g)
            m, _ = match(p, g, iou_thr)
            for s, t in m:
                sc.append(s)
                tp.append(t)
                if s >= score_thr:
                    npred_thr += 1
                    ntp_thr += int(t)
        res[c] = {"ap": average_precision(np.array(sc), np.array(tp), ngt), "n_gt": ngt,
                  "precision": ntp_thr / npred_thr if npred_thr else float("nan"),
                  "recall": ntp_thr / ngt if ngt else float("nan")}
    aps = [r["ap"] for r in res.values() if not np.isnan(r["ap"])]
    res["mAP"] = float(np.mean(aps)) if aps else float("nan")
    return res
