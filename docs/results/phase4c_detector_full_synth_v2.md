# Detector comparison on datasets/synth_v2/test: 342 images, 479 boxes

Operating point: score >= 0.35, IoU 0.5 for P/R and confusion. AP all-point interpolation; mAP50-95 = mean over IoU 0.50:0.05:0.95.

## det_v1_s

mAP50 **0.824**, mAP50-95 **0.757**

| class | GT | AP50 | precision | recall |
|---|---|---|---|---|
| box | 31 | 0.430 | 0.404 | 0.677 |
| barrel | 63 | 0.914 | 0.853 | 0.921 |
| fire_extinguisher | 48 | 0.974 | 0.904 | 0.979 |
| traffic_cone | 31 | 0.967 | 0.938 | 0.968 |
| toolbox | 37 | 0.913 | 0.917 | 0.892 |
| mannequin | 217 | 0.904 | 0.871 | 0.899 |
| chair | 52 | 0.664 | 0.500 | 0.865 |

Confusion matrix (rows = ground truth, columns = prediction; last column = missed, last row = false positive):

| GT \ pred | box | barrel | fire_ext | traffic_ | toolbox | mannequi | chair | bg/misse |
|---|---|---|---|---|---|---|---|---|
| box | 21 | 0 | 0 | 0 | 0 | 0 | 0 | 10 |
| barrel | 0 | 58 | 0 | 0 | 0 | 0 | 1 | 4 |
| fire_extinguisher | 0 | 0 | 47 | 0 | 0 | 0 | 0 | 1 |
| traffic_cone | 0 | 0 | 0 | 30 | 0 | 0 | 0 | 1 |
| toolbox | 0 | 0 | 0 | 0 | 33 | 0 | 0 | 4 |
| mannequin | 0 | 0 | 1 | 0 | 0 | 195 | 0 | 21 |
| chair | 0 | 1 | 0 | 0 | 0 | 0 | 44 | 7 |
| bg/missed | 31 | 9 | 4 | 2 | 3 | 29 | 45 | 0 |

Class confusions (GT -> predicted as): barrel->chair 1, mannequin->fire_extinguisher 1, chair->barrel 1

Edge-truncated vs fully visible GT boxes (recall = same-class box with IoU >= 0.5 at the operating point):

| subset | GT | recall | mean best IoU |
|---|---|---|---|
| all | 479 | 0.898 | 0.849 |
| edge-truncated | 356 | 0.921 | 0.877 |
| fully visible | 123 | 0.829 | 0.770 |
| mannequin, all | 217 | 0.903 | 0.864 |
| mannequin, edge-truncated | 212 | 0.910 | 0.871 |
| mannequin, fully visible | 5 | 0.600 | 0.552 |
| mannequin, box h/w < 1.5 (legs/torso only) | 55 | 0.927 | 0.907 |
| mannequin, box h/w >= 1.5 | 162 | 0.895 | 0.849 |

## det_v2_s

mAP50 **0.871**, mAP50-95 **0.806**

| class | GT | AP50 | precision | recall |
|---|---|---|---|---|
| box | 31 | 0.544 | 0.465 | 0.645 |
| barrel | 63 | 0.933 | 0.935 | 0.921 |
| fire_extinguisher | 48 | 0.999 | 0.960 | 1.000 |
| traffic_cone | 31 | 0.935 | 0.935 | 0.935 |
| toolbox | 37 | 0.919 | 1.000 | 0.919 |
| mannequin | 217 | 0.937 | 0.923 | 0.935 |
| chair | 52 | 0.832 | 0.911 | 0.788 |

Confusion matrix (rows = ground truth, columns = prediction; last column = missed, last row = false positive):

| GT \ pred | box | barrel | fire_ext | traffic_ | toolbox | mannequi | chair | bg/misse |
|---|---|---|---|---|---|---|---|---|
| box | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 11 |
| barrel | 0 | 58 | 0 | 0 | 0 | 0 | 1 | 4 |
| fire_extinguisher | 0 | 0 | 48 | 0 | 0 | 0 | 0 | 0 |
| traffic_cone | 0 | 0 | 0 | 29 | 0 | 0 | 0 | 2 |
| toolbox | 0 | 0 | 0 | 0 | 34 | 0 | 0 | 3 |
| mannequin | 0 | 0 | 0 | 0 | 0 | 203 | 0 | 14 |
| chair | 0 | 2 | 0 | 0 | 0 | 0 | 40 | 10 |
| bg/missed | 23 | 2 | 2 | 2 | 0 | 17 | 4 | 0 |

Class confusions (GT -> predicted as): chair->barrel 2, barrel->chair 1

Edge-truncated vs fully visible GT boxes (recall = same-class box with IoU >= 0.5 at the operating point):

| subset | GT | recall | mean best IoU |
|---|---|---|---|
| all | 479 | 0.906 | 0.868 |
| edge-truncated | 356 | 0.921 | 0.892 |
| fully visible | 123 | 0.862 | 0.797 |
| mannequin, all | 217 | 0.940 | 0.913 |
| mannequin, edge-truncated | 212 | 0.948 | 0.922 |
| mannequin, fully visible | 5 | 0.600 | 0.557 |
| mannequin, box h/w < 1.5 (legs/torso only) | 55 | 0.945 | 0.930 |
| mannequin, box h/w >= 1.5 | 162 | 0.938 | 0.908 |

## Summary

| model | mAP50 | mAP50-95 |
|---|---|---|
| det_v1_s | 0.824 | 0.757 |
| det_v2_s | 0.871 | 0.806 |
