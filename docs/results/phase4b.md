# Phase 4b — detector fixes (2026-11-17)

## 1. CUDA by default in the vision image
`docker/vision/Dockerfile` now sets `LD_LIBRARY_PATH` to the pip-installed `nvidia/*/lib` directories. Verified after `docker compose build vision`:
`InferenceSession(det_v1_s.onnx, [CUDA, CPU]).get_providers() -> ['CUDAExecutionProvider', 'CPUExecutionProvider']`, and `eval_detector.py`
(100 test images, no env override) gives CUDA 181.6 FPS vs CPU 35.7 FPS.

## 2. Edge-truncated detections are mapped
`detector_node` used to skip every box that touches the image border. From the 0.29 m camera that is most boxes (1057 of 1556 sightings in the run below)
and every mannequin. Now:
- depth comes from the visible part of the box (`box_depth`, unchanged);
- the horizontal centre is corrected when the box is cut at the left/right border (`geometry.visible_centre_u`: one apparent radius from the real edge; top/bottom
  cuts do not move it; cut on both sides = wider than the view -> skipped);
- the sighting enters the map with position weight 0.25 (`truncated_weight`), so a full view always dominates.

## 3. Track confirmation
`SemanticMap` confirms a track when ALL hold: >= `min_hits` sightings inside the association gate, spanning >= `min_span` s, mean detector score >= `min_mean_score`,
RMS scatter around the track <= `max_scatter` x gate; once confirmed it stays. Node defaults: 5 hits, 2 s, 0.5, 0.5. Unit tests: `gurdy_core/tests/test_vision.py`.

Gazebo run `sem_s2` (semantic_lab, frontier exploration, det_v1_s on CPU, 480 sim s, 97.4 % interior mapped), same sightings replayed with different rules
(`vision/replay_semantic_map.py`, result `docs/results/data/phase4/sem_s2/replay.md`):

| setting | found | false tracks | of those < 10 hits |
|---|---|---|---|
| old: truncated skipped, 3 hits | 7/11 (cone, mannequin, chair, barrel_b missing) | 5 | 2 |
| truncated mapped, 3 hits | 11/11 | 9 | 4 |
| + span 1 s, score 0.5, scatter 0.5 | 11/11 | 9 | 4 |
| 5 hits + span 2 s (node default) | 11/11 | 6 | 1 |
| 10 hits + span 3 s + score 0.7 | 11/11 | 4 | 0 |

With the node defaults the mannequin is found at 0.056 m (189 sightings), all 11 objects within 0.015-0.107 m (median 0.033 m), colours of the 3 coloured boxes correct.
The 6 false tracks are detector errors with many sightings, so no confirmation rule can remove them: table -> `box` (123 hits), green cylinder -> `box` (48), a `toolbox` on the red box (46),
a `barrel` on the yellow box (27), a `chair` on the green cylinder (13) and a `traffic_cone` next to the table (7). The stricter row was found on this same single run, so it is a hint, not a validated setting.
The first 3-hit rules alone did nothing because every false track here has many consistent sightings (the time-span / score rules only prune the ghosts).

## 4. Dataset v2 and retraining
Finding first: **v1 already contained random distractors** (0-8 boxes / cylinders / spheres per scene, random colours). The confusion therefore came from the box-shaped ones being
indistinguishable from the `box` class (plain coloured cubes) and from distractors rarely being the centre of a view. v2 (`vision/scene.py` `profile="v2"`, `objects.distractor_parts`):
- 2-10 unlabelled distractors per scene from 8 families (table with legs, solid bench slab like flat_lab's table, stool/cylinder, pillar, ball, plank, tall crate, stacked primitives); none is a 25-50 cm cube;
- 25 % of the views aimed at distractors; mannequin class weight 3, one mannequin forced in half the scenes, 40 % of object views aimed at a mannequin.
- Caveat: the benches / cylinders resemble the two distractors of `flat_lab`, so the semantic_lab test is no longer fully independent of the training distribution.

`datasets/synth_v2`: 240 scenes (168/36/36 train/val/test), 2250 images, 0 failed scenes (26 retries, 39.4 min): train 1566 / val 342 / test 342 images; boxes: mannequin 1563, box 314, barrel 348,
fire_extinguisher 284, traffic_cone 270, toolbox 313, chair 274 (v1 had 495 mannequin boxes). Label check (`docs/results/phase4b_label_check/`, 6 sample frames looked at: boxes sit on the labelled objects, distractors stay unlabelled; 63 of 2250 frames have no box): only 53 % of boxes contain a GT centre because mannequin
boxes are legs only (same effect as in v1: 27 %).

Training: `det_v2_s` = yolo11s, 300 epochs, patience 50, same hyper-parameters as `det_v1_s`, on `datasets/synth_v12.yaml` (v1 + v2 train/val; test splits stay separate so `det_v1_s` and `det_v2_s` are compared on
`synth_v1/test` and `synth_v2/test`). Started 06:02 UTC; not evaluated yet.

Monitor commands:

```bash
docker ps                                                     # gurdy2-train-det2 must be listed
tail -c 600 vision/runs/train_det_v2_s.log | tr '\r' '\n' | tail -3     # current epoch + speed
column -s, -t < vision/runs/det_v2_s/results.csv | cut -c1-110 | tail -5   # epoch, losses, P, R, mAP50, mAP50-95
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
docker rm -f gurdy2-train-det2                                # stop it (resume: train.py --model vision/runs/det_v2_s/weights/last.pt ... or `yolo train resume`)
# when finished: python3 vision/train.py --data-yaml datasets/synth_v12.yaml --name det_v2_s --model yolo11s.pt --epochs 300 --export-only   (in the vision image)
```
