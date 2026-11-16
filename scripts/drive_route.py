#!/usr/bin/env python3
"""Scripted mapping drive for flat_lab: closed-loop waypoint follower on /odom/ground_truth -> /cmd_vel
(turn to face the waypoint, then walk; commands respect the policy minimum speeds 0.05 m/s / 0.15 rad/s).
Route: room A west loop, doorway (x 2.5, y 0.5), room B loop, back through the door, room A south, start.
    python3 scripts/drive_route.py OUT.json [--speed 0.12] [--max-s 1500]
Prints/saves: sim time, path length, per-waypoint times, stalls, final error to the start, GT path (2 Hz).
"""
import argparse
import json
import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rosgraph_msgs.msg import Clock

ROUTE = [(-1.4, 0.0), (-1.4, 2.2), (1.8, 2.2), (1.8, 0.5), (3.3, 0.5), (5.5, 0.5), (5.5, 2.3), (3.3, 2.3),
         (3.3, -2.3), (5.8, -2.3), (5.8, -0.5), (3.3, 0.5), (1.8, 0.5), (1.8, -0.7), (-1.4, -0.7), (-1.4, -2.4),
         (0.0, -2.4), (0.0, 0.0)]


class Driver(Node):
    def __init__(self):
        super().__init__("route_driver")
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(Odometry, "/odom/ground_truth", self.on_odom, 20)
        self.create_subscription(Clock, "/clock", lambda m: setattr(self, "now", m.clock.sec + m.clock.nanosec * 1e-9), 10)
        self.pose, self.now = None, None

    def on_odom(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.pose = (p.x, p.y, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--speed", type=float, default=0.12)
    ap.add_argument("--max-s", type=float, default=1500.0)
    a = ap.parse_args()
    rclpy.init()
    n = Driver()
    while n.pose is None or n.now is None or n.count_subscribers("/cmd_vel") == 0:
        rclpy.spin_once(n, timeout_sec=0.1)
    t0, last_pub, k = n.now, -1.0, 0
    path, length, prev = [], 0.0, n.pose
    stalls, reached, last_prog = [], [], (n.now, n.pose)
    while k < len(ROUTE) and n.now - t0 < a.max_s:
        rclpy.spin_once(n, timeout_sec=0.01)
        if n.now - last_pub < 0.05:
            continue
        last_pub = n.now
        x, y, yaw = n.pose
        length += math.hypot(x - prev[0], y - prev[1]); prev = n.pose
        if int(n.now - t0) % 1 == 0 and (not path or n.now - t0 - path[-1][0] >= 0.5):
            path.append((round(n.now - t0, 1), round(x, 3), round(y, 3), round(yaw, 3)))
        wx, wy = ROUTE[k]
        d = math.hypot(wx - x, wy - y)
        if d < 0.15:
            reached.append((k, round(n.now - t0, 1)))
            k += 1
            last_prog = (n.now, n.pose)
            continue
        if n.now - last_prog[0] > 25.0:                 # no 0.15 m of progress in 25 s: stall, skip waypoint
            if math.hypot(x - last_prog[1][0], y - last_prog[1][1]) < 0.15:
                stalls.append((k, round(n.now - t0, 1), round(x, 2), round(y, 2)))
                k += 1
            last_prog = (n.now, n.pose)
            continue
        if math.hypot(x - last_prog[1][0], y - last_prog[1][1]) >= 0.15:
            last_prog = (n.now, n.pose)
        h = wrap(math.atan2(wy - y, wx - x) - yaw)
        c = Twist()
        if abs(h) > 0.35:                                # turn in place first
            c.angular.z = math.copysign(min(0.3, max(0.15, 0.8 * abs(h))), h)
        else:
            c.linear.x = min(a.speed, max(0.05, 0.6 * d))
            if abs(h) > 0.06:
                c.angular.z = math.copysign(min(0.3, max(0.15, 1.2 * abs(h))), h)
        n.pub.publish(c)
    n.pub.publish(Twist())
    x, y, _ = n.pose
    res = {"sim_s": round(n.now - t0, 1), "path_m": round(length, 2), "waypoints": len(ROUTE), "reached": reached,
           "stalls": stalls, "final_xy": (round(x, 3), round(y, 3)), "final_err_to_start_m": round(math.hypot(x, y), 3),
           "complete": k >= len(ROUTE), "path": path}
    json.dump(res, open(a.out, "w"))
    print({k_: v for k_, v in res.items() if k_ != "path"})
    rclpy.shutdown()


if __name__ == "__main__":
    main()
