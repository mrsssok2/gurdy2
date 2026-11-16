#!/usr/bin/env python3
"""Does the 2D lidar see the robot itself? Records /scan while standing (cmd 0) and walking (cmd vx) in an open world and,
for every return closer than --near metres, computes the 3-D distance of the hit point (on the lidar plane) to the robot's
leg chains (coxa->femur->tibia->foot, from TF) and to the body box. Run with the sim up (lidar:=true), nothing else running.
    python3 scripts/scan_selfhit.py OUT.json [--stand 20] [--walk 40] [--vx 0.1] [--near 0.5]
"""
import argparse
import json
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener

LEGS = ["lf", "lm", "lr", "rf", "rm", "rr"]
BODY_HALF = np.array([0.15, 0.09, 0.03])  # robot.yaml body.size / 2


def seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


class N(Node):
    def __init__(self):
        super().__init__("scan_selfhit")
        self.buf = Buffer()
        self.tl = TransformListener(self.buf, self)
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.scans = []
        for src in ("scan_raw", "scan"):   # raw lidar and the self-filtered scan, recorded side by side
            self.create_subscription(LaserScan, "/" + src, lambda m, src=src: self.on_scan(m, src), qos_profile_sensor_data)
        self.phase = "none"
        self.rows = []

    def pos(self, frame, ref="lidar_link"):
        t = self.buf.lookup_transform(ref, frame, rclpy.time.Time()).transform.translation
        return np.array([t.x, t.y, t.z])

    def on_scan(self, m, src):
        try:
            chains = {l: [self.pos(f"{l}_{s}") for s in ("coxa", "femur", "tibia", "foot")] for l in LEGS}
            body = self.pos("body_link")
            # body orientation in lidar frame -> rotate hit into body frame
            q = self.buf.lookup_transform("body_link", "lidar_link", rclpy.time.Time()).transform
        except Exception:
            return
        qx, qy, qz, qw = q.rotation.x, q.rotation.y, q.rotation.z, q.rotation.w
        R = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                      [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                      [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
        tb = np.array([q.translation.x, q.translation.y, q.translation.z])
        r = np.array(m.ranges)
        ang = m.angle_min + np.arange(len(r)) * m.angle_increment
        ok = np.isfinite(r) & (r >= m.range_min) & (r <= m.range_max)
        near = ok & (r < self.near)
        hits = []
        for rr, aa in zip(r[near], ang[near]):
            p = np.array([rr * math.cos(aa), rr * math.sin(aa), 0.0])
            dl = {(l, i): seg_dist(p, c[i], c[i + 1]) for l, c in chains.items() for i in range(3)}
            (leg, seg), dmin = min(dl.items(), key=lambda kv: kv[1])
            pb = R @ p + tb  # hit in body frame
            dbox = float(np.linalg.norm(np.maximum(np.abs(pb) - BODY_HALF, 0)))
            hits.append((float(rr), float(aa), leg, seg, dmin, dbox))
        self.rows.append({"src": src, "phase": self.phase, "min_range": float(r[ok].min()) if ok.any() else None,
                          "n_valid": int(ok.sum()), "n_near": int(near.sum()), "hits": hits,
                          "n_below_range_min": int((np.isfinite(r) & (r < m.range_min)).sum()),
                          "z_lidar_to_body": float(tb[2]) if False else float(-q.translation.z)})

    def run(self, phase, secs, vx):
        self.phase = phase
        tw = Twist()
        tw.linear.x = vx
        t0 = time.time()
        while time.time() - t0 < secs:
            self.pub.publish(tw)
            rclpy.spin_once(self, timeout_sec=0.05)


ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--stand", type=float, default=20)
ap.add_argument("--walk", type=float, default=40)
ap.add_argument("--vx", type=float, default=0.1)
ap.add_argument("--near", type=float, default=0.5)
a = ap.parse_args()
rclpy.init()
n = N()
n.near = a.near
t0 = time.time()
while len({r['src'] for r in n.rows}) < 2 and time.time() - t0 < 60:
    rclpy.spin_once(n, timeout_sec=0.1)
n.rows.clear()
n.run("stand", a.stand, 0.0)
n.run("walk", a.walk, a.vx)
n.run("stand_after", 5, 0.0)
json.dump(n.rows, open(a.out, "w"))
for src, ph in [(s_, p_) for s_ in ("scan_raw", "scan") for p_ in ("stand", "walk")]:
    rows = [r for r in n.rows if r["phase"] == ph and r["src"] == src]
    print(f"[{src}]", end=" ")
    mins = np.array([r["min_range"] for r in rows if r["min_range"] is not None])
    hits = [h for r in rows for h in r["hits"]]
    print(f"{ph}: scans={len(rows)} scans_with_return<{a.near}={sum(1 for r in rows if r['hits'])} "
          f"min_range min/median={mins.min() if len(mins) else None}/{np.median(mins) if len(mins) else None} "
          f"hits={len(hits)}")
    if hits:
        h = np.array([(x[0], x[4], x[5]) for x in hits])
        print(f"   hit range {h[:,0].min():.3f}-{h[:,0].max():.3f}; dist to nearest leg segment median {np.median(h[:,1]):.3f} max {h[:,1].max():.3f}; "
              f"dist to body box median {np.median(h[:,2]):.3f}")
