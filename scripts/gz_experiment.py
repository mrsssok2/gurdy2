#!/usr/bin/env python3
"""Run the shared step-response / standing protocol against a running Gazebo sim.

Needs: ros2 launch gurdy_gazebo sim.launch.py gui:=false fixed_base:=true   (step)
       ros2 launch gurdy_gazebo sim.launch.py gui:=false                    (stand)
Then:  python3 scripts/gz_experiment.py step|stand > result.json

References go to /joint_pd_controller/commands on the protocol's sim-time schedule (/clock);
/joint_states (500 Hz from joint_state_broadcaster) is recorded and analysed with the same
gurdy_core.experiments code as MuJoCo. Also reports Gazebo's real-time factor.
"""
import json
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from gurdy_core.experiments import (STAND_RECORD, STAND_SETTLE, analyze_stand, analyze_step,
                                    stance_vector, step_protocol)
from gurdy_core.robot import load_robot


class Runner(Node):
    def __init__(self, robot):
        super().__init__("gz_experiment")
        self.robot = robot
        self.names = robot.joint_names
        self.sim_now = None
        self.rows = []
        qos = QoSProfile(depth=5000, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(JointState, "/joint_states", self.on_js, qos)
        self.create_subscription(Clock, "/clock", self.on_clock, 10)
        self.pub = self.create_publisher(Float64MultiArray, "/joint_pd_controller/commands", 10)
        self.idx = None

    def on_clock(self, msg):
        self.sim_now = msg.clock.sec + msg.clock.nanosec * 1e-9

    def on_js(self, msg):
        if self.idx is None:
            pos = {n: i for i, n in enumerate(msg.name)}
            self.idx = np.array([pos[n] for n in self.names])
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.rows and t <= self.rows[-1][0]:
            self.rows.pop()          # duplicate stamp: keep the newest sample
        self.rows.append((t, np.asarray(msg.position)[self.idx], np.asarray(msg.velocity)[self.idx],
                          np.asarray(msg.effort)[self.idx]))

    def send(self, q):
        self.pub.publish(Float64MultiArray(data=[float(v) for v in q]))

    def run(self, schedule, duration):
        while self.sim_now is None or self.pub.get_subscription_count() == 0 or not self.rows:
            rclpy.spin_once(self, timeout_sec=0.1)
        t0 = self.sim_now
        wall0 = time.perf_counter()
        self.rows.clear()
        ev = list(schedule)
        while self.sim_now - t0 < duration:
            while ev and self.sim_now - t0 >= ev[0][0]:
                self.send(ev.pop(0)[1])
            rclpy.spin_once(self, timeout_sec=0.001)
        rtf = (self.sim_now - t0) / (time.perf_counter() - wall0)
        rows = [r for r in self.rows if r[0] >= t0]
        t = np.array([r[0] for r in rows]) - t0
        return t, np.array([r[1] for r in rows]), np.array([r[2] for r in rows]), np.array([r[3] for r in rows]), rtf


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "step"
    robot = load_robot()
    rclpy.init()
    node = Runner(robot)
    if mode == "step":
        events, _, total = step_protocol(robot)
        t, q, qd, tau, rtf = node.run([(e.t, e.q_ref) for e in events], total)
        out = {"sim": "gazebo", "rtf": rtf, "samples": len(t), "steps": analyze_step(robot, t, q, qd, tau)}
    else:
        t, q, qd, tau, rtf = node.run([(0.0, stance_vector(robot))], STAND_SETTLE + STAND_RECORD)
        out = analyze_stand(robot, t, q, qd, tau)
        out.update({"sim": "gazebo", "rtf": rtf})
    print(json.dumps(out, indent=2))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
