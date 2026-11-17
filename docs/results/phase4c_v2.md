# Phase 4c — det_v2_s vs det_v1_s (2026-11-17)

det_v2_s = yolo11s on v1+v2 data (`datasets/synth_v12.yaml`), resumed after a power loss (epoch ~159), early stop ~220. ONNX `models/detector/det_v2_s.onnx`.
Test splits, score >= 0.35, IoU 0.5. Full tables: `phase4c_detector_full_synth_v{1,2}.md`. Note: semantic_lab is NOT independent of v2 (v2 distractor families resemble its distractors).

| | det_v1_s | det_v2_s |
|---|---|---|
| mAP50 / mAP50-95, v1 test | 0.916 / 0.855 | 0.922 / 0.865 |
| mAP50 / mAP50-95, v2 test | 0.824 / 0.757 | 0.871 / 0.806 |
| semantic_lab (Gazebo, one run each): found | 11/11 | 11/11 |
| median xy error | 0.033 m | 0.031 m |
| false / duplicate tracks (compare script) | 9 | 7 |
| mean RTF (v2 run) / interior mapped / sim time | – | 0.60 / 97.9 % / 710 s |

v1 run = `sem_s2` (the "6 false tracks" in phase4b.md is the count after replaying confirmation rules; the compare script counts 9 on the same file).
v2 false tracks: the table and green cylinder are STILL labelled `box` (125 / 133 hits), a 10-hit `toolbox` on the red box, three weak/ghost tracks (chair 15, box 6, chair 37 beside the cylinder).
So v2 helps on harder data (+4.7 pts mAP50 on v2 test) but did not remove the distractor -> `box` errors in Gazebo. One run each, no variance.
**Default detector: det_v2_s** (node, launch, run script updated). Data: `docs/results/data/phase4/sem_v2`.
