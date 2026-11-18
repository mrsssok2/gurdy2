#!/usr/bin/env python3
"""Record /joint_pd_controller/commands + /joint_states (+ /cmd_vel_smoothed) to an npz while the robot moves (leg-health calibration, 5.6).
  python3 scripts/leg_health_record.py OUT.npz SECONDS_SIM
"""
import sys
import numpy as np
import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist
from gurdy_core.robot import load_robot


class Rec(Node):
    def __init__(self):
        super().__init__("leg_health_record")
        self.idx = {n: i for i, n in enumerate(load_robot().joint_names)}
        self.ref = None; self.cmd = [0.0, 0.0, 0.0]; self.rows = []
        self.create_subscription(Float64MultiArray, "/joint_pd_controller/commands", lambda m: setattr(self, "ref", np.array(m.data)), 10)
        self.create_subscription(Twist, "/cmd_vel_smoothed", lambda m: setattr(self, "cmd", [m.linear.x, m.linear.y, m.angular.z]), 10)
        self.create_subscription(JointState, "/joint_states", self.js, 10)

    def js(self, m):
        if self.ref is None or len(self.ref) != 18:
            return
        q = np.zeros(18)
        for n, p in zip(m.name, m.position):
            if n in self.idx:
                q[self.idx[n]] = p
        self.rows.append([m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, *self.ref, *q, *self.cmd])


def main():
    out, secs = sys.argv[1], float(sys.argv[2])
    rclpy.init(); n = Rec(); t0 = None
    while rclpy.ok():
        rclpy.spin_once(n, timeout_sec=0.2)
        if n.rows:
            t0 = t0 or n.rows[0][0]
            if n.rows[-1][0] - t0 > secs:
                break
    np.savez_compressed(out, data=np.array(n.rows), joints=np.array(load_robot().joint_names))
    print("saved", out, len(n.rows))

main()
