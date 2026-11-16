# Phase 4 — dataset check `datasets/synth_v1` (2026-11-16)

Render: 180 scenes, 0 failed, 16 retries, 30.5 min. Tools: `vision/dataset_report.py` (statistics + grid, numbers in
`phase4_dataset_stats.json`) and `vision/check_labels.py` (boxes vs independent ground-truth centres). Both run in the vision image.

| | train | val | test |
|---|---|---|---|
| images | 1499 | 323 | 314 |
| images with no label | 30 | 5 | 4 |
| box / barrel / fire_extinguisher / traffic_cone / toolbox / mannequin / chair | 298 / 362 / 353 / 354 / 299 / 373 / 352 | 74 / 45 / 78 / 89 / 93 / 48 / 103 | 55 / 94 / 61 / 90 / 46 / 74 / 67 |

- **Format:** 3408 boxes, 0 invalid lines (class id, coordinates in [0,1], size > 0), 0 duplicates, every image has a label file and the reverse.
- **Empty images:** 39 of 2136 (1.8 %) — useful as background examples, kept.
- **Box size (320x240 image):** width p5/p50/p95 = 14 / 56 / 190 px, height 25 / 86 / 197 px. Smallest side 6 px; 40 boxes < 8 px, 246 (7 %) < 16 px on
  one side. 131 boxes cover > 50 % of the image. 1 box per image at the median, at most 5.
- **Truncation:** 1895 boxes (56 %) touch the image border. The camera is 0.29 m high and often close, so tall objects are cut.
- **Boxes vs ground truth:** 2476 of 3408 boxes (73 %) contain a ground-truth centre of the same class (trial: 85 %). The other 932 are mostly truncated objects
  whose centre is outside the image (the checker only knows centres in view) or objects partly hidden behind another one: 785 touch the border, and mannequins
  alone account for 366 (366 of 495 mannequin boxes — from 0.29 m the camera mostly sees legs, the centre is above the frame). The frames inspected
  (`val/s0154_v10`, `val/s0076_v02`, `train/s0161_v04`) confirm this. Depth at the box centre minus GT depth: median −0.10 m (front surface vs centre).
- **Grid:** `phase4_dataset_grid.png`, 16 random labelled images (seed 0) including box and mannequin; boxes sit on the objects, distractors are unlabelled.

**Verdict: labels are correct, no re-render.** Things to watch in the evaluation, not fixed now:
- thin border slivers (a few px of an object) are labelled as full instances — label noise for the detector;
- the mannequin class is learned mostly from legs/lower body; a full-body mannequin far away is rare in the data;
- class counts per split are uneven in val (barrel 45, mannequin 48 vs chair 103), so per-class val AP is noisy.
