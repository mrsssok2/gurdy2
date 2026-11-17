#!/usr/bin/env python3
"""Save the latest /semantic_map/objects message (transient local) to a JSON file, and optionally annotated camera frames.

    python3 vision/save_semantic_map.py OUT.json                      # one message, then exit
    python3 vision/save_semantic_map.py --frames DIR MODEL.onnx       # run until SIGTERM: save annotated frames (own CPU detector, 1 Hz)
"""
import argparse
import os
import signal
import sys
from pathlib import Path

import numpy as np
import rclpy
from PIL import Image as PImage, ImageDraw
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gurdy_core"))
from gurdy_core.vision.detector import OnnxDetector  # noqa: E402


class Saver(Node):
    def __init__(self, a):
        super().__init__("save_semantic_map")
        self.a, self.last, self.n, self.t_last = a, None, 0, -1e9
        if a.frames:
            os.makedirs(a.frames, exist_ok=True)
            self.det = OnnxDetector(a.model or a.out, "CPUExecutionProvider", conf=0.35)
            self.create_subscription(Image, "camera/image", self.on_img, qos_profile_sensor_data)
        else:
            self.create_subscription(String, "semantic_map/objects", self.on_map, QoSProfile(
                depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE))

    def on_map(self, m):
        Path(self.a.out).write_text(m.data)
        print("saved", self.a.out)
        self.done = True

    def on_img(self, m):
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        if t - self.t_last < 1.0:
            return
        self.t_last = t
        rgb = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step // 3, 3)[:, :m.width]
        dets = self.det(np.ascontiguousarray(rgb))
        if not dets:
            return
        im = PImage.fromarray(rgb.copy())
        dr = ImageDraw.Draw(im)
        for d in dets:
            dr.rectangle(d.box, outline=(255, 255, 0), width=2)
            dr.text((d.box[0] + 2, d.box[1] + 2), f"{d.name} {d.score:.2f}", fill=(255, 255, 0))
        im.save(f"{self.a.frames}/t{t:07.1f}_{len(dets)}det.png")
        self.n += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?")
    ap.add_argument("model", nargs="?")
    ap.add_argument("--frames", default="")
    a = ap.parse_args()
    rclpy.init()
    n = Saver(a)
    n.done = False
    if a.frames:
        signal.signal(signal.SIGTERM, lambda *_: setattr(n, "done", True))
    t0 = n.get_clock().now()
    while rclpy.ok() and not n.done:
        rclpy.spin_once(n, timeout_sec=0.5)
        if not a.frames and (n.get_clock().now() - t0).nanoseconds > 20e9:
            print("no semantic map message in 20 s"); sys.exit(1)
    print("frames saved:", n.n) if a.frames else None


if __name__ == "__main__":
    main()
