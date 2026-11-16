# Phase 4 — object set, licenses, detector choice

## Object classes (7, inspection-robot set)
All models are **self-made procedural models** (SDF primitives + one generated OBJ lathe mesh for the cone) built in
`vision/objects.py` — no third-party assets, so they are licensed **MIT with this repo**. Gazebo Fuel was not used (per-model
licenses to track, network dependency); if Fuel models are added later, record name + license here first.

| id | class | built from | variants | Hinglish aliases (`resolve_class`) |
|---|---|---|---|---|
| 0 | box | box (+ tape strip) | cardboard or plain saturated colour, 25–50 cm | dabba, carton, peti |
| 1 | barrel | cylinder + 2 bands | 5 colours | drum, pipa |
| 2 | fire_extinguisher | cylinder, neck, handle, hose, label band | – | extinguisher, aag |
| 3 | traffic_cone | lathe mesh (OBJ) + white band + base plate | – | cone |
| 4 | toolbox | box, handle, latch | 4 colours | auzaar |
| 5 | mannequin | torso, head, neck, legs, arms | skin + 2 clothes colours | putla, insaan, aadmi, person |
| 6 | chair | seat, back, 4 legs | 4 colours | kursi |

`box` deliberately includes plain red/blue/yellow boxes: the coloured boxes of `flat_lab` are class `box`, and the colour is a
separate attribute (HSV rule on the crop, `gurdy_core/vision/colors.py`) so "laal dabba" = class box + colour red.
Gazebo label id = class id + 1 (0 = background). Distractors (unlabelled boxes/cylinders/spheres in random colours) occlude and
confuse but have no label.

## Detector library license
- **Ultralytics YOLO (v8/11): AGPL-3.0** (or paid Enterprise license). Using it unmodified for training in a private/academic
  project is fine; if the trained detector were shipped inside a closed-source product or offered as a network service, AGPL
  obligations would apply. Mitigation built in: Ultralytics lives **only in `docker/vision`** (training/export); the repo (MIT),
  the ROS node and the evaluation tool use only the exported **ONNX file** through `onnxruntime` + our own decoder
  (`gurdy_core/vision/detector.py`). The pretrained COCO weights carry the same AGPL terms.
- **Permissive alternatives** if that ever matters: **YOLOX (Apache-2.0)**, **RT-DETR via PaddleDetection / Hugging Face
  Transformers (Apache-2.0)**, **torchvision SSDlite320-MobileNetV3 / Faster R-CNN (BSD-3)**. Switching needs a trainer script and a
  matching `decode_*` in `gurdy_core/vision/detector.py` (the ROS node and the metrics do not change).

## Camera
The dataset camera uses the intrinsics of `robot.yaml` (320×240, hfov 1.047, range 0.1–5 m, pitch-down 0.17 rad ± 0.07),
height randomised 0.10–0.30 m around the robot's real camera height, roll ± 0.03 rad.
