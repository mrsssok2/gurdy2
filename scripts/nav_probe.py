#!/usr/bin/env python3
"""Diagnosis: print per-10 s windows of the Nav2 velocity chain + ground-truth motion while goals run (run next to nav_eval.py).
cmd_vel_nav (MPPI) -> cmd_vel_smoothed (velocity_smoother) -> cmd_vel (collision_monitor) -> policy/cmd (after deadband)."""
import math
import sys
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rosgraph_msgs.msg import Clock

TOP = ["/cmd_vel_nav", "/cmd_vel_smoothed", "/cmd_vel", "/policy/cmd"]


class P(Node):
    def __init__(self):
        super().__init__("nav_probe")
        self.now, self.buf, self.pose = None, {t: [] for t in TOP}, None
        for t in TOP:
            self.create_subscription(Twist, t, lambda m, t=t: self.buf[t].append((m.linear.x, m.linear.y, m.angular.z)), 50)
        self.create_subscription(Odometry, "/odom/ground_truth", lambda m: setattr(self, "pose", (m.pose.pose.position.x, m.pose.pose.position.y)), 20)
        self.create_subscription(Clock, "/clock", lambda m: setattr(self, "now", m.clock.sec + m.clock.nanosec * 1e-9), 10)


rclpy.init()
n = P()
dur = float(sys.argv[1]) if len(sys.argv) > 1 else 150
while n.now is None or n.pose is None:
    rclpy.spin_once(n, timeout_sec=0.1)
t0, w0, p0 = n.now, n.now, n.pose
while n.now - t0 < dur:
    rclpy.spin_once(n, timeout_sec=0.1)
    if n.now - w0 >= 10:
        row = []
        for t in TOP:
            a = np.array(n.buf[t]) if n.buf[t] else np.zeros((0, 3))
            row.append(f"{t.split('/')[-1]}:n={len(a)} mean=({a[:,0].mean():+.3f},{a[:,1].mean():+.3f},{a[:,2].mean():+.3f})" if len(a) else f"{t.split('/')[-1]}:n=0")
            n.buf[t].clear()
        print(f"t={n.now-t0:5.0f} pos=({n.pose[0]:+.2f},{n.pose[1]:+.2f}) moved={math.hypot(n.pose[0]-p0[0], n.pose[1]-p0[1]):.2f} | " + " | ".join(row), flush=True)
        w0, p0 = n.now, n.pose
