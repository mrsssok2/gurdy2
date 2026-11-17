# Detector comparison on datasets/synth_v1/test: 314 images, 487 boxes

Operating point: score >= 0.35, IoU 0.5 for P/R and confusion. AP all-point interpolation; mAP50-95 = mean over IoU 0.50:0.05:0.95.

## det_v1_s

mAP50 **0.916**, mAP50-95 **0.855**

| class | GT | AP50 | precision | recall |
|---|---|---|---|---|
| box | 55 | 0.773 | 0.808 | 0.764 |
| barrel | 94 | 0.957 | 1.000 | 0.957 |
| fire_extinguisher | 61 | 1.000 | 0.984 | 1.000 |
| traffic_cone | 90 | 0.921 | 0.943 | 0.911 |
| toolbox | 46 | 0.954 | 0.956 | 0.935 |
| mannequin | 74 | 0.941 | 0.945 | 0.932 |
| chair | 67 | 0.864 | 0.921 | 0.866 |

Confusion matrix (rows = ground truth, columns = prediction; last column = missed, last row = false positive):

| GT \ pred | box | barrel | fire_ext | traffic_ | toolbox | mannequi | chair | bg/misse |
|---|---|---|---|---|---|---|---|---|
| box | 42 | 0 | 0 | 0 | 2 | 0 | 0 | 11 |
| barrel | 0 | 90 | 0 | 0 | 0 | 0 | 0 | 4 |
| fire_extinguisher | 0 | 0 | 60 | 0 | 0 | 0 | 0 | 1 |
| traffic_cone | 0 | 0 | 0 | 81 | 0 | 1 | 0 | 8 |
| toolbox | 0 | 0 | 0 | 0 | 43 | 0 | 0 | 3 |
| mannequin | 0 | 0 | 0 | 1 | 0 | 68 | 0 | 5 |
| chair | 0 | 0 | 1 | 0 | 0 | 0 | 58 | 8 |
| bg/missed | 10 | 0 | 1 | 5 | 0 | 4 | 5 | 0 |

Class confusions (GT -> predicted as): box->toolbox 2, traffic_cone->mannequin 1, mannequin->traffic_cone 1, chair->fire_extinguisher 1

Edge-truncated vs fully visible GT boxes (recall = same-class box with IoU >= 0.5 at the operating point):

| subset | GT | recall | mean best IoU |
|---|---|---|---|
| all | 487 | 0.918 | 0.876 |
| edge-truncated | 288 | 0.934 | 0.898 |
| fully visible | 199 | 0.894 | 0.844 |
| mannequin, all | 74 | 0.932 | 0.883 |
| mannequin, edge-truncated | 74 | 0.932 | 0.883 |
| mannequin, box h/w < 1.5 (legs/torso only) | 19 | 0.895 | 0.868 |
| mannequin, box h/w >= 1.5 | 55 | 0.945 | 0.888 |

## det_v2_s

mAP50 **0.922**, mAP50-95 **0.865**

| class | GT | AP50 | precision | recall |
|---|---|---|---|---|
| box | 55 | 0.725 | 0.745 | 0.745 |
| barrel | 94 | 0.979 | 1.000 | 0.957 |
| fire_extinguisher | 61 | 1.000 | 0.984 | 1.000 |
| traffic_cone | 90 | 0.954 | 0.966 | 0.933 |
| toolbox | 46 | 0.975 | 0.915 | 0.935 |
| mannequin | 74 | 0.928 | 0.908 | 0.932 |
| chair | 67 | 0.891 | 0.967 | 0.866 |

Confusion matrix (rows = ground truth, columns = prediction; last column = missed, last row = false positive):

| GT \ pred | box | barrel | fire_ext | traffic_ | toolbox | mannequi | chair | bg/misse |
|---|---|---|---|---|---|---|---|---|
| box | 41 | 0 | 0 | 0 | 2 | 0 | 0 | 12 |
| barrel | 0 | 90 | 0 | 0 | 0 | 0 | 0 | 4 |
| fire_extinguisher | 0 | 0 | 60 | 0 | 0 | 0 | 0 | 1 |
| traffic_cone | 0 | 0 | 0 | 84 | 0 | 0 | 0 | 6 |
| toolbox | 0 | 0 | 0 | 0 | 43 | 0 | 0 | 3 |
| mannequin | 0 | 0 | 1 | 0 | 0 | 68 | 0 | 5 |
| chair | 0 | 0 | 1 | 0 | 0 | 0 | 58 | 8 |
| bg/missed | 14 | 0 | 0 | 3 | 2 | 8 | 2 | 0 |

Class confusions (GT -> predicted as): box->toolbox 2, mannequin->fire_extinguisher 1, chair->fire_extinguisher 1

Edge-truncated vs fully visible GT boxes (recall = same-class box with IoU >= 0.5 at the operating point):

| subset | GT | recall | mean best IoU |
|---|---|---|---|
| all | 487 | 0.920 | 0.883 |
| edge-truncated | 288 | 0.938 | 0.910 |
| fully visible | 199 | 0.894 | 0.843 |
| mannequin, all | 74 | 0.932 | 0.905 |
| mannequin, edge-truncated | 74 | 0.932 | 0.905 |
| mannequin, box h/w < 1.5 (legs/torso only) | 19 | 0.895 | 0.876 |
| mannequin, box h/w >= 1.5 | 55 | 0.945 | 0.915 |

## Summary

| model | mAP50 | mAP50-95 |
|---|---|---|
| det_v1_s | 0.916 | 0.855 |
| det_v2_s | 0.922 | 0.865 |
