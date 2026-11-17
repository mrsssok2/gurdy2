# Phase 4 detector evaluation — det_v1_n.onnx on datasets/synth_v1/test (314 images)

| class | GT | AP50 | precision@0.35 | recall@0.35 |
|---|---|---|---|---|
| box | 55 | 0.675 | 0.727 | 0.727 |
| barrel | 94 | 0.967 | 1.000 | 0.947 |
| fire_extinguisher | 61 | 0.998 | 0.968 | 1.000 |
| traffic_cone | 90 | 0.921 | 0.954 | 0.922 |
| toolbox | 46 | 0.935 | 0.933 | 0.913 |
| mannequin | 74 | 0.942 | 0.945 | 0.932 |
| chair | 67 | 0.861 | 0.966 | 0.836 |
| **mAP50** | | **0.900** | | |

3D centre error (camera frame, 315 true positives): median 0.066 m, mean 0.137 m, p90 0.226 m; median range 2.32 m

| provider | active | FPS (inference only) |
|---|---|---|
| cuda | CUDAExecutionProvider | 197.4 |
| cpu | CPUExecutionProvider | 92.7 |

