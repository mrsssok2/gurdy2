# Phase 4 detector evaluation — det_v1_s.onnx on datasets/synth_v1/test (314 images)

| class | GT | AP50 | precision@0.35 | recall@0.35 |
|---|---|---|---|---|
| box | 55 | 0.773 | 0.808 | 0.764 |
| barrel | 94 | 0.957 | 1.000 | 0.957 |
| fire_extinguisher | 61 | 1.000 | 0.984 | 1.000 |
| traffic_cone | 90 | 0.921 | 0.943 | 0.911 |
| toolbox | 46 | 0.954 | 0.956 | 0.935 |
| mannequin | 74 | 0.941 | 0.945 | 0.932 |
| chair | 67 | 0.864 | 0.921 | 0.866 |
| **mAP50** | | **0.916** | | |

3D centre error (camera frame, 316 true positives): median 0.065 m, mean 0.132 m, p90 0.224 m; median range 2.31 m

| provider | active | FPS (inference only) |
|---|---|---|
| cuda | CUDAExecutionProvider | 176.8 |
| cpu | CPUExecutionProvider | 34.9 |

