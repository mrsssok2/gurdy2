#!/usr/bin/env python3
"""Diagnosis: drive vx 0.1 for 20 s, then wz 0.3 for 15 s, then vy 0.08 for 15 s; print GT vs /odom/legs vs /odometry/filtered
displacement (body frame at segment start) and yaw change per segment. Needs sim with odom_source:=ekf."""
import math
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rosgraph_msgs.msg import Clock


def yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


rclpy.init()
n = Node("ekf_probe")
S = {"gt": None, "legs": None, "ekf": None}
now = [None]
for k, t in (("gt", "/odom/ground_truth"), ("legs", "/odom/legs"), ("ekf", "/odometry/filtered")):
    n.create_subscription(Odometry, t, lambda m, k=k: S.__setitem__(k, (m.pose.pose.position.x, m.pose.pose.position.y, yaw(m.pose.pose.orientation),
                                                                          m.twist.twist.linear.x, m.twist.twist.linear.y)), 50)
n.create_subscription(Clock, "/clock", lambda m: now.__setitem__(0, m.clock.sec + m.clock.nanosec * 1e-9), 10)
pub = n.create_publisher(Twist, "/cmd_vel", 10)
while now[0] is None or any(v is None for v in S.values()):
    rclpy.spin_once(n, timeout_sec=0.1)


def body(p0, p):                                    # displacement of p in the frame of p0
    dx, dy = p[0] - p0[0], p[1] - p0[1]
    c, s = math.cos(p0[2]), math.sin(p0[2])
    return c * dx + s * dy, -s * dx + c * dy


def seg(name, cmd, dur):
    p0 = {k: v for k, v in S.items()}
    t0 = now[0]
    while now[0] - t0 < dur:
        m = Twist(); m.linear.x, m.linear.y, m.angular.z = (float(c) for c in cmd)
        pub.publish(m)
        rclpy.spin_once(n, timeout_sec=0.05)
    pub.publish(Twist())
    t1 = now[0]
    while now[0] - t1 < 2.0:
        rclpy.spin_once(n, timeout_sec=0.05)
    out = [f"{name:8s} cmd={cmd}"]
    for k in ("gt", "legs", "ekf"):
        d = body(p0[k], S[k])
        out.append(f"{k}: d=({d[0]:+.3f},{d[1]:+.3f}) dyaw={math.degrees((S[k][2]-p0[k][2]+math.pi)%(2*math.pi)-math.pi*0):+.1f}")
    print(" | ".join(out), flush=True)


while now[0] < 5:
    rclpy.spin_once(n, timeout_sec=0.1)
seg("forward", (0.1, 0, 0), 20)
seg("turn", (0, 0, 0.3), 15)
seg("strafe", (0, 0.08, 0), 15)
seg("forward2", (0.1, 0, 0), 20)
