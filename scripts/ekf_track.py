#!/usr/bin/env python3
"""Diagnosis: log GT / legs / EKF pose (x, y, yaw) every 2 s of sim time to OUT.json while drive_route.py runs."""
import json
import math
import sys
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rosgraph_msgs.msg import Clock

yaw = lambda q: math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
rclpy.init()
n = Node("ekf_track")
S, now, rows = {}, [None], []
for k, t in (("gt", "/odom/ground_truth"), ("legs", "/odom/legs"), ("ekf", "/odometry/filtered")):
    n.create_subscription(Odometry, t, lambda m, k=k: S.__setitem__(k, (m.pose.pose.position.x, m.pose.pose.position.y, yaw(m.pose.pose.orientation))), 50)
n.create_subscription(Clock, "/clock", lambda m: now.__setitem__(0, m.clock.sec + m.clock.nanosec * 1e-9), 10)
last = -10
try:
    while rclpy.ok():
        rclpy.spin_once(n, timeout_sec=0.1)
        if now[0] is not None and len(S) == 3 and now[0] - last >= 2.0:
            last = now[0]
            rows.append([round(now[0], 1)] + [round(v, 3) for k in ("gt", "legs", "ekf") for v in S[k]])
            json.dump(rows, open(sys.argv[1], "w"))
except KeyboardInterrupt:
    pass
