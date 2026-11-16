# Phase 4 — run commands (after `leg_failure` training has finished)

Status 2026-11-16: vision image built + GPU-verified; **step 1 trial done** (5 scenes, all camera assumptions OK, labels checked: `docs/results/phase4_trial/README.md`); full render running in container `gurdy2-render` (`tail -f datasets/synth_v1/render.log`; resume with the plain `render` command, then `finalize`). Steps 3–5 not run.

Original status: **code and unit tests only** — nothing below has been run yet (no Gazebo, no rendering, no GPU training while
`leg_failure` trains). One heavy workload at a time (8 GB RAM / 6 GB VRAM): check `docker ps` first.

GPU check already done (light): `nvidia-smi` in WSL and `docker run --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L`
both show the RTX 4050 (6141 MiB).

```bash
# 0. one-time: build the vision image (~8 GB pull: PyTorch 2.5.1 + CUDA 12.1 runtime + Ultralytics) and check the GPU inside it
docker compose build vision
docker compose run --rm vision python3 -c "import torch, onnxruntime as o; print(torch.cuda.is_available(), o.get_available_providers())"

# 1. plan scenes (fast; host or any image with numpy+PIL+yaml) and render with Gazebo headless (sim image, resumable)
docker compose run --rm sim python3 vision/generate_dataset.py plan --out datasets/synth_v1 --scenes 200 --views 12 --seed 0
docker compose run --rm sim python3 vision/generate_dataset.py render --out datasets/synth_v1 --start 0 --count 5     # try 5 scenes first, LOOK at the images/labels
docker compose run --rm sim python3 vision/generate_dataset.py render --out datasets/synth_v1                          # the rest
docker compose run --rm sim python3 vision/generate_dataset.py finalize --out datasets/synth_v1                        # dataset.yaml + stats.json

# 2. sanity-check labels before training (boxes drawn on samples + comparison with ground truth); repeat on the full set, look for box/mannequin frames
docker compose run --rm sim python3 vision/check_labels.py --data datasets/synth_v1 --out docs/results/phase4_trial --samples 5

# 3. train + export ONNX (GPU) -> models/detector/det_v1.onnx
docker compose run --rm vision python3 vision/train.py --data datasets/synth_v1 --name det_v1 --epochs 60

# 4. mAP on the held-out test scenes + FPS on GPU and CPU + 3D error -> docs/results/phase4_detector.md
docker compose run --rm vision python3 vision/eval_detector.py --model models/detector/det_v1.onnx --data datasets/synth_v1 \
    --split test --providers cuda cpu --out docs/results/phase4_detector.md

# 5. ROS: test in the lab (builds the new packages first)
docker compose run --rm sim bash -c 'cd ros2_ws && colcon build --symlink-install --packages-select gurdy_perception gurdy_bringup gurdy_gazebo && source install/setup.bash && cd .. && \
    ros2 launch gurdy_bringup sim.launch.py gui:=false world:=semantic_lab nav:=slam navigation:=false perception:=true controller:=policy'
#    second terminal: drive around (teleop or scripts/nav_goal.sh / frontier explorer), then
#    ros2 topic echo /semantic_map/objects --once   (or detector_node parameter save_path:=/tmp/sem.json)
python3 vision/compare_semantic_map.py /tmp/sem.json          # recall, xy error, duplicates, colour vs semantic_lab_gt.json
```
Expected-but-unverified points to check in step 1 (they are the riskiest assumptions):
- `/world/synth/set_pose` moves the *static* `cam` model; if not, make the camera model non-static or use one world per view.
- boundingbox_camera publishes on `/cam/bbox` (the `Capture` class subscribes to that name; `gz topic -l` shows the real one).
- RGB, depth and bbox frames share the same sim stamp (the capture waits for a stamp present in all three).
- the labelled boxes cover only the *visible* part of an occluded object (`visible_2d`).
