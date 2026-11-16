#!/usr/bin/env python3
"""Synthetic detection dataset from Gazebo Harmonic (boundingbox_camera = automatic YOLO labels).

    # 1. plan: sample scenes, write worlds + textures + meshes (fast, no simulator needed)
    python3 vision/generate_dataset.py plan   --out datasets/synth_v1 --scenes 200 --views 12 --seed 0
    # 2. render: run each world headless in Gazebo, move the camera through its views, save frames (ROS image, `sim` service)
    python3 vision/generate_dataset.py render --out datasets/synth_v1 [--start 0 --count 50]   # resumable
    # 3. dataset.yaml (+ per-split counts) for the trainer
    python3 vision/generate_dataset.py finalize --out datasets/synth_v1

Layout:  images/{train,val,test}/s0007_v03.png  labels/<split>/*.txt (YOLO)  depth/<split>/*.png (uint16 mm)
         meta/<split>/*.json (camera pose, intrinsics, ground-truth object centres in the camera frame)
Split is BY SCENE (scene_idx % 20: 14 train / 3 val / 3 test), so no object layout is shared between splits.
Randomised per scene: room size, floor/wall colour + texture, sun + lamps + ambient + shadows, 1-6 labelled objects
(7 classes, size/colour variants), 0-8 distractors, camera height 0.10-0.30 m, pitch, roll; per frame: exposure,
Gaussian noise, motion blur (vision/augment.py). Depth is the clean Gazebo depth (same as the robot's sensor).
Label ids in Gazebo = class id + 1 (0 = unlabelled background). The render step needs gz-transport python
(`gz.transport13`, in the ROS image); it cannot run on the host Python.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gurdy_core"))
from gurdy_core.vision.classes import CLASSES, GZ_LABEL_OFFSET  # noqa: E402
from vision import scene as S  # noqa: E402
from vision.augment import augment_rgb  # noqa: E402
from vision.textures import write_textures  # noqa: E402

SPLIT_NAMES = [n for n, _ in S.SPLITS]


def cmd_plan(a):
    out = Path(a.out)
    (out / "worlds").mkdir(parents=True, exist_ok=True)
    (out / "assets" / "meshes").mkdir(parents=True, exist_ok=True)
    write_textures(str(out / "assets" / "textures"), S.N_TEXTURES, seed=a.seed)
    cam = {}
    if a.robot_yaml:
        import yaml
        d = yaml.safe_load(open(a.robot_yaml))["sensors"]["depth_camera"]
        cam = {k: d[k] for k in ("width", "height", "hfov", "range", "pitch_down")}
    for i in range(a.scenes):
        sc = S.sample_scene(a.seed * 100003 + i, a.views, cam)
        sc["index"], sc["split"] = i, S.split_for_scene(i)
        (out / "worlds" / f"scene_{i:04d}.json").write_text(json.dumps(sc))
        mesh_dir = out / "assets" / "meshes" / f"s{i:04d}"
        mesh_dir.mkdir(exist_ok=True)
        (out / "worlds" / f"scene_{i:04d}.sdf").write_text(
            S.world_sdf(sc, str(mesh_dir.resolve()), str((out / "assets" / "textures").resolve())))
    print(f"planned {a.scenes} scenes -> {out}/worlds  (train/val/test scenes: "
          f"{[sum(S.split_for_scene(i) == s for i in range(a.scenes)) for s in SPLIT_NAMES]})")


def rpy_to_quat(r, p, y):
    """Gazebo roll-pitch-yaw (extrinsic xyz) -> quaternion (x, y, z, w)."""
    cr, sr, cp, sp, cy, sy = np.cos(r / 2), np.sin(r / 2), np.cos(p / 2), np.sin(p / 2), np.cos(y / 2), np.sin(y / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


class Capture:
    """Subscribes to the camera topics via gz-transport; keeps the latest frames keyed by sim stamp (ns)."""

    def __init__(self, world: str):
        from gz.msgs10.annotated_axis_aligned_2d_box_v_pb2 import AnnotatedAxisAligned2DBox_V
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.image_pb2 import Image as GzImage
        from gz.msgs10.pose_pb2 import Pose
        from gz.transport13 import Node
        self.Pose, self.Boolean, self.world = Pose, Boolean, world
        self.node = Node()
        self.lock = threading.Lock()
        self.rgb, self.depth, self.bbox = {}, {}, {}
        self.node.subscribe(GzImage, "/cam/image", lambda m: self._put(self.rgb, m))
        self.node.subscribe(GzImage, "/cam/depth_image", lambda m: self._put(self.depth, m))
        self.node.subscribe(AnnotatedAxisAligned2DBox_V, "/cam/bbox", lambda m: self._put(self.bbox, m))
        self.latest = 0

    def _put(self, store, m):
        ns = m.header.stamp.sec * 10**9 + m.header.stamp.nsec
        with self.lock:
            store[ns] = m
            self.latest = max(self.latest, ns)
            for k in sorted(store)[:-6]:
                del store[k]

    def set_pose(self, pos, rpy) -> bool:
        qx, qy, qz, qw = rpy_to_quat(*rpy)
        req = self.Pose()
        req.name = "cam"
        req.position.x, req.position.y, req.position.z = pos
        req.orientation.x, req.orientation.y, req.orientation.z, req.orientation.w = qx, qy, qz, qw
        ok, rep = self.node.request(f"/world/{self.world}/set_pose", req, self.Pose, self.Boolean, 3000)
        return bool(ok and rep.data)

    def frame_after(self, t_ns: int, timeout: float = 15.0):
        """First (rgb, depth, bbox) triple with the same stamp > t_ns."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                common = sorted(k for k in self.rgb if k > t_ns and k in self.depth and k in self.bbox)
                if common:
                    k = common[0]
                    return self.rgb[k], self.depth[k], self.bbox[k]
            time.sleep(0.02)
        return None


def _img(m, channels):
    return np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, channels) if channels == 3 else \
        np.frombuffer(m.data, np.float32).reshape(m.height, m.width)


def render_scene(sc: dict, out: Path, a) -> int:
    idx, split = sc["index"], sc["split"]
    world = out / "worlds" / f"scene_{idx:04d}.sdf"
    env = {**os.environ, "GZ_PARTITION": f"synth{os.getpid()}"}
    os.environ["GZ_PARTITION"] = env["GZ_PARTITION"]            # for our own gz-transport node
    srv = subprocess.Popen(["gz", "sim", "-s", "-r", "--headless-rendering", "-v", "1", str(world)], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    n_ok, rng = 0, np.random.default_rng(sc["seed"] + 7)
    try:
        cap = Capture(S.WORLD_NAME)
        t0 = time.time()
        while cap.latest == 0 and time.time() - t0 < 60:       # wait for the first frames (server start-up)
            time.sleep(0.2)
        if cap.latest == 0:
            raise RuntimeError("no camera frames from Gazebo (headless rendering / GPU path?)")
        for vi, view in enumerate(sc["views"]):
            if not cap.set_pose(view["pos"], view["rpy"]):
                continue
            with cap.lock:
                t_set = cap.latest
            fr = cap.frame_after(t_set + int(0.35 * 1e9))       # skip frames rendered before/while the pose changed
            if fr is None:
                continue
            rgb_m, depth_m, bb = fr
            rgb, depth = _img(rgb_m, 3), _img(depth_m, 0)
            H, W = rgb.shape[:2]
            boxes = [(b.label - GZ_LABEL_OFFSET, b.box.min_corner.x, b.box.min_corner.y, b.box.max_corner.x, b.box.max_corner.y)
                     for b in bb.annotated_box if 0 <= b.label - GZ_LABEL_OFFSET < len(CLASSES)]
            lines = S.yolo_lines(boxes, W, H)
            if not lines and rng.random() > a.negative_keep:
                continue
            name = f"s{idx:04d}_v{vi:02d}"
            for sub in ("images", "labels", "depth", "meta"):
                (out / sub / split).mkdir(parents=True, exist_ok=True)
            Image.fromarray(augment_rgb(np.array(rgb), rng)).save(out / "images" / split / f"{name}.png")
            (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            d_mm = np.where(np.isfinite(depth), np.clip(depth * 1000, 0, 65535), 0).astype(np.uint16)
            Image.fromarray(d_mm).save(out / "depth" / split / f"{name}.png")
            (out / "meta" / split / f"{name}.json").write_text(json.dumps({
                "scene": idx, "view": vi, "cam_pos": view["pos"], "cam_rpy": view["rpy"], "camera": sc["camera"],
                "gt": S.gt_for_view(sc, view), "stamp_ns": rgb_m.header.stamp.sec * 10**9 + rgb_m.header.stamp.nsec}))
            n_ok += 1
    finally:
        srv.terminate()
        try:
            srv.wait(10)
        except subprocess.TimeoutExpired:
            srv.kill()
    return n_ok


def cmd_render(a):
    out = Path(a.out)
    scenes = sorted((out / "worlds").glob("scene_*.json"))[a.start:(a.start + a.count) if a.count else None]
    done_file = out / "rendered.txt"
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    for p in scenes:
        if p.stem in done:
            continue
        sc = json.loads(p.read_text())
        t0 = time.time()
        try:
            n = render_scene(sc, out, a)
        except Exception as e:                                  # keep going: one bad scene must not stop a long run
            print(f"{p.stem}: FAILED {e}")
            continue
        with open(done_file, "a") as fh:
            fh.write(p.stem + "\n")
        print(f"{p.stem} [{sc['split']}]: {n}/{len(sc['views'])} frames in {time.time() - t0:.0f} s", flush=True)


def cmd_finalize(a):
    out = Path(a.out).resolve()
    counts = {s: len(list((out / "images" / s).glob("*.png"))) for s in SPLIT_NAMES if (out / "images" / s).exists()}
    inst = np.zeros(len(CLASSES), int)
    for f in (out / "labels").glob("*/*.txt"):
        for ln in f.read_text().split("\n"):
            if ln.strip():
                inst[int(ln.split()[0])] += 1
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    (out / "dataset.yaml").write_text(f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n")
    (out / "stats.json").write_text(json.dumps({"images": counts, "instances": dict(zip(CLASSES, map(int, inst)))}, indent=1))
    print("images", counts, "instances", dict(zip(CLASSES, map(int, inst))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--out", required=True); p.add_argument("--scenes", type=int, default=200)
    p.add_argument("--views", type=int, default=12); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--robot-yaml", default="ros2_ws/src/gurdy_description/config/robot.yaml", help="camera intrinsics source")
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("render"); p.add_argument("--out", required=True); p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=0); p.add_argument("--negative-keep", type=float, default=0.1,
                                                                    help="probability to keep a frame with no labelled object")
    p.set_defaults(fn=cmd_render)
    p = sub.add_parser("finalize"); p.add_argument("--out", required=True); p.set_defaults(fn=cmd_finalize)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
