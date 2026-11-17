# Phase 4 — detector evaluation and semantic_lab test (2026-11-17)

Models: `det_v1_n` (yolo11n, 10.5 MB ONNX) and `det_v1_s` (yolo11s, 37.8 MB), both 300 epochs on `datasets/synth_v1`
(1499 / 323 / 314 images train / val / test), exported at 256x320. `det_v1_s` was resumed from `last.pt` after a power loss.
All numbers below are on the **held-out test split** (314 images, 487 boxes). Full tables and confusion matrices:
`phase4_detector_full.md` (`vision/eval_detector_full.py`), speed + 3D error: `phase4_detector_det_v1_{n,s}.md` (`vision/eval_detector.py`).

## Accuracy (score >= 0.35, IoU 0.5)
| model | mAP50 | mAP50-95 | weakest class (AP50) |
|---|---|---|---|
| det_v1_n | 0.900 | 0.834 | box 0.675 |
| det_v1_s | 0.916 | 0.855 | box 0.773 |

(Ultralytics' own val-split numbers for s: 0.943 / 0.890 — test is lower, as expected.) Per class, `fire_extinguisher` is ~1.0, barrel / toolbox /
mannequin 0.94-0.97, traffic_cone 0.92, chair 0.86, **box 0.68 / 0.77**: plain coloured boxes are missed (12 / 11 of 55) and sometimes
labelled toolbox (3 / 2). Class confusions are otherwise rare (<= 1 each). Median 3D centre error from depth: 0.066 m (n) / 0.065 m (s), p90 0.23 m.

## Edge-truncated boxes and mannequins
288 of 487 test boxes touch the image border; **all 74 mannequin boxes do** (the camera is 0.29 m high, so mannequins are mostly legs).
Recall on truncated boxes is *not* lower than on fully visible ones (n 0.924 vs 0.884; s 0.934 vs 0.894) — the "fully visible" set contains
the small and occluded objects. Mannequin recall 0.932 for both models; the legs-only ones (h/w < 1.5, 19 boxes) 0.947 (n) / 0.895 (s), too few to tell apart.

## Speed (320x240 frames, inference only incl. pre/post-processing, 12-thread host CPU, RTX 4050)
| model | ONNX CUDA | ONNX CPU |
|---|---|---|
| det_v1_n | 197 FPS | 93 FPS |
| det_v1_s | 177 FPS | 35 FPS |

The vision image's `onnxruntime-gpu` only finds CUDA when `LD_LIBRARY_PATH` contains the pip-installed libs
(`/opt/conda/lib/python3.11/site-packages/nvidia/{cudnn,cublas,cuda_runtime,curand,cufft,cuda_nvrtc,cusparse,cusolver,nvjitlink}/lib`);
without it the CUDA provider silently falls back to CPU (`libcublasLt.so.12` missing). Not yet fixed in the Dockerfile.

## Selection: det_v1_s
+1.6 / +2.1 points mAP, clearly better on `box`, equal 3D error. The detector node runs at 5 Hz, so the 35 FPS CPU speed of s is 7x headroom; n would only
matter if CPU load becomes the limit (it is 2.7x faster). `detector_node` default is now `det_v1_s.onnx`.

## Gazebo semantic_lab test (explore with SLAM + Nav2, policy gait, detector on CPU in the sim container)
`scripts/phase4_semantic_run.sh NAME det.onnx` — frontier exploration of the lab from an empty map, detector 5 Hz, objects de-duplicated in the `map` frame.
Compared with the world's ground truth (`vision/compare_semantic_map.py`, match radius 0.6 m, same-class nearest). One run per model, so no variance estimate.

| | det_v1_s | det_v1_n |
|---|---|---|
| GT objects found (of 11) | 9 | 9 |
| xy error, median / range | 0.035 m / 0.016-0.059 | 0.043 m / 0.009-0.095 |
| missed | mannequin, chair (see below) | mannequin, barrel_b (yellow box labelled barrel) |
| extra tracks | 5 | 6 |
| colour correct (3 coloured boxes) | 3/3 | 3/3 |
| interior mapped / sim time / RTF | 97.0 % / 418 s / 0.66 | 97.6 % / 472 s / 0.54 |

RTF is **not comparable** between the two: in the n run my annotated-frame saver (a second CPU detector) ran for the whole run, in the s run only for the second half.
For both, the robot returned to the start and exploration ended with no frontiers.

What went wrong (all are findings, not fixed yet):
- **Mannequin never mapped (both).** The detector does see it (legs, boxes in the saved frames), but `detector_node` skips map updates for boxes touching the image edge
  (centre bias) and a mannequin always touches it from 0.29 m. A fix needs a bottom-truncation-aware position estimate.
- **Unlabelled distractors become objects.** `green_cylinder` (4.5, 1.6) -> `box` (32 / 61 hits) and `table` (0.9, -1.4) -> `box` (18 / 9 hits): none of the distractors are in the training data (no hard negatives). Needs distractor classes/images in the dataset.
- s: a `chair` track on the green cylinder (7 hits, 0.9 m from the real chair, which was therefore missed); n found the chair (0.07 m) but also this false chair (4 hits).
- Low-hit ghosts (3-8 hits) near the red box (`toolbox`) and the table (`traffic_cone`); a `min_hits` of ~10 would remove them but also true tracks (the s run's barrel_a had 11, a cone 18).

Files: semantic map PNG `phase4_semantic_map.png` (s run; rings = ground truth, dots = tracks), detection frames `phase4_detections/` (montage + 8 frames),
raw results `data/phase4/sem_{s,n}/` (semantic.json, compare.txt, monitor.json, logs), SLAM maps `maps/sem_{s,n}.*`.
