#!/usr/bin/env python3
"""Step 2.9 diagnosis: record the policy's inputs/outputs in Gazebo while the shared tracking protocol runs
(run gz_tracking.py's protocol itself so the commands are identical). Saves an .npz:
  js_t, js_q, js_qd (/joint_states), imu_t, imu_quat(wxyz), imu_gyro, cmd_t, cmd_q (/joint_pd_controller/commands),
  od_t, od_xyz (ground truth). Times are sim stamps (s).
Usage (sim running with controller:=policy):  python3 scripts/policy_diag_gz.py OUT.npz
"""
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray

from gurdy_core.experiments import cmd_at, tracking_protocol
from gurdy_core.robot import load_robot

st = lambda h: h.stamp.sec + h.stamp.nanosec * 1e-9


class Rec(Node):
    def __init__(self):
        super().__init__("policy_diag")
        self.names = load_robot().joint_names
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.js, self.imu, self.cmd, self.od, self.sim_now = [], [], [], [], None
        self.rx_js, self.rx_imu = [], []
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(JointState, "/joint_states", self.on_js, 200)
        self.create_subscription(Imu, "/imu", self.on_imu, 500)
        self.create_subscription(Float64MultiArray, "/joint_pd_controller/commands", self.on_cmd, 200)
        self.create_subscription(Odometry, "/odom/ground_truth", self.on_od, 200)
        self.create_subscription(Clock, "/clock", lambda m: setattr(self, "sim_now", m.clock.sec + m.clock.nanosec * 1e-9), 10)

    def on_js(self, m):
        q, qd = np.zeros(18), np.zeros(18)
        for n, p, v in zip(m.name, m.position, m.velocity):
            q[self.idx[n]], qd[self.idx[n]] = p, v
        self.js.append((st(m.header), *q, *qd))
        self.rx_js.append(self.sim_now or 0.0)

    def on_imu(self, m):
        o, w = m.orientation, m.angular_velocity
        self.imu.append((st(m.header), o.w, o.x, o.y, o.z, w.x, w.y, w.z))
        self.rx_imu.append(self.sim_now or 0.0)

    def on_cmd(self, m):
        self.cmd.append((self.sim_now or 0.0, *m.data))

    def on_od(self, m):
        p = m.pose.pose.position
        self.od.append((st(m.header), p.x, p.y, p.z))


def main():
    rclpy.init()
    n = Rec()
    changes, _, total = tracking_protocol()
    while n.sim_now is None or not n.od or n.count_subscribers("/cmd_vel") == 0:
        rclpy.spin_once(n, timeout_sec=0.1)
    t0, last = n.sim_now, -1.0
    while n.sim_now - t0 < total:
        ts = n.sim_now - t0
        if ts - last >= 0.05:
            m = Twist()
            m.linear.x, m.linear.y, m.angular.z = cmd_at(changes, ts)
            n.pub.publish(m)
            last = ts
        rclpy.spin_once(n, timeout_sec=0.002)
    js, imu, cmd, od = (np.array(x) for x in (n.js, n.imu, n.cmd, n.od))
    np.savez(sys.argv[1], t0=t0, js_t=js[:, 0] - t0, js_q=js[:, 1:19], js_qd=js[:, 19:37],
             imu_t=imu[:, 0] - t0, imu_quat=imu[:, 1:5], imu_gyro=imu[:, 5:8],
             cmd_t=cmd[:, 0] - t0, cmd_q=cmd[:, 1:], rx_js=np.array(n.rx_js) - t0, rx_imu=np.array(n.rx_imu) - t0, od_t=od[:, 0] - t0, od_xyz=od[:, 1:])
    rclpy.shutdown()


if __name__ == "__main__":
    main()
