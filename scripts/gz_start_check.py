#!/usr/bin/env python3
"""Gazebo start-from-rest check: for each of the 6 directions (vx±, vy±, wz±) publish the command on /cmd_vel after >= 3 s of zero
command and report the displacement in the start body frame over the last 6 s of a 10 s window (speed ratio = measured / commanded).
    python3 scripts/gz_start_check.py OUT.json LIN ANG        (policy running, ground-truth odom bridged)"""
import json
import math
import sys

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rosgraph_msgs.msg import Clock


def yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class N(Node):
    def __init__(self):
        super().__init__("gz_start_check")
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pose, self.now = None, None
        self.create_subscription(Clock, "/clock", lambda m: setattr(self, "now", m.clock.sec + m.clock.nanosec * 1e-9), 10)
        self.create_subscription(Odometry, "/odom/ground_truth", lambda m: setattr(
            self, "pose", (m.pose.pose.position.x, m.pose.pose.position.y, yaw(m.pose.pose.orientation))), 20)

    def hold(self, cmd, secs):
        t0 = self.now
        while self.now - t0 < secs:
            tw = Twist()
            tw.linear.x, tw.linear.y, tw.angular.z = (float(v) for v in cmd)
            self.pub.publish(tw)
            rclpy.spin_once(self, timeout_sec=0.05)


out, lin, ang = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
rclpy.init()
n = N()
while n.pose is None or n.now is None:
    rclpy.spin_once(n, timeout_sec=0.1)
res = {}
for tag, cmd in {"vx+": (lin, 0, 0), "vx-": (-lin, 0, 0), "vy+": (0, lin, 0), "vy-": (0, -lin, 0), "wz+": (0, 0, ang), "wz-": (0, 0, -ang)}.items():
    n.hold((0, 0, 0), 4.0)
    n.hold(cmd, 4.0)                      # first 4 s: start-up
    p0 = n.pose
    n.hold(cmd, 6.0)
    p1, dt = n.pose, 6.0
    dx, dy, dth = p1[0] - p0[0], p1[1] - p0[1], math.atan2(math.sin(p1[2] - p0[2]), math.cos(p1[2] - p0[2]))
    c, s = math.cos(p0[2]), math.sin(p0[2])
    bx, by = c * dx + s * dy, -s * dx + c * dy
    res[tag] = {"cmd": cmd, "dx_body": bx, "dy_body": by, "dyaw": dth, "window_s": dt}
    print(tag, {k: round(v, 3) if isinstance(v, float) else v for k, v in res[tag].items()}, flush=True)
n.hold((0, 0, 0), 1.0)
json.dump(res, open(out, "w"), indent=1)
