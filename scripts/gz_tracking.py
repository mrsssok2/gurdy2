#!/usr/bin/env python3
"""Velocity-tracking test in Gazebo: publish the shared /cmd_vel protocol (gurdy_core.experiments),
record /odom/ground_truth, print JSON. Needs: ros2 launch gurdy_bringup sim.launch.py gui:=false
"""
import json
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rosgraph_msgs.msg import Clock

from gurdy_core.experiments import analyze_tracking, cmd_at, quat_to_rpy, tracking_protocol


class Tracker(Node):
    def __init__(self):
        super().__init__("gz_tracking")
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.rows, self.sim_now = [], None
        self.create_subscription(Odometry, "/odom/ground_truth", self.on_odom, 100)
        self.create_subscription(Clock, "/clock", lambda m: setattr(self, "sim_now", m.clock.sec + m.clock.nanosec * 1e-9), 10)

    def on_odom(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.rows.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, p.x, p.y, p.z, q.w, q.x, q.y, q.z))


def main():
    rclpy.init()
    n = Tracker()
    changes, _, total = tracking_protocol()
    while n.sim_now is None or not n.rows or n.count_subscribers("/cmd_vel") == 0:
        rclpy.spin_once(n, timeout_sec=0.1)
    t0, w0, last_pub = n.sim_now, time.time(), -1.0
    n.rows.clear()
    while n.sim_now - t0 < total:
        ts = n.sim_now - t0
        if ts - last_pub >= 0.05:                      # 20 Hz like a teleop/Nav2 source
            c = cmd_at(changes, ts)
            msg = Twist()
            msg.linear.x, msg.linear.y, msg.angular.z = c
            n.pub.publish(msg)
            last_pub = ts
        rclpy.spin_once(n, timeout_sec=0.005)
    rtf = (n.sim_now - t0) / (time.time() - w0)
    a = np.array(n.rows)
    a = a[a[:, 0] >= t0]
    roll, pitch, yaw = quat_to_rpy(a[:, 4], a[:, 5], a[:, 6], a[:, 7])
    res = analyze_tracking(a[:, 0] - t0, a[:, 1], a[:, 2], yaw, a[:, 3], roll, pitch)
    res.update({"sim": "gazebo", "rtf": rtf})
    print(json.dumps(res, indent=2))
    rclpy.shutdown()


if __name__ == "__main__":
    main()
