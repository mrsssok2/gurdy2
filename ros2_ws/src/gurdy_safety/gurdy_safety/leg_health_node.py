"""Leg health from joint tracking error: /joint_pd_controller/commands (targets) vs /joint_states -> /gurdy/leg_health.

Publishes a latched String JSON {"stamp", "legs": {"front_left": "ok"|"failed", ...}} at 1 Hz and on every change.
A failed leg stays failed until  ros2 service call /leg_health/reset std_srvs/srv/Trigger.
Thresholds (gurdy_core.ops_monitors.LegHealthConfig) are NOT calibrated on Gazebo data yet. A leg failure cannot be seen while the
robot stands still. Test aid: parameter `fake_failed_leg` (e.g. "rm") forces that leg to failed without any physics.
"""
import json

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger

from gurdy_core.ops_monitors import LEG_LONG, LegHealth, LegHealthConfig
from gurdy_core.robot import LEG_NAMES, load_robot


class LegHealthNode(Node):
    def __init__(self):
        super().__init__("leg_health_node")
        P = self.declare_parameter
        cfg = LegHealthConfig()
        P("robot_yaml", "")
        P("abs_error_rad", cfg.abs_error_rad)
        P("ratio", cfg.ratio)
        P("hold_s", cfg.hold_s)
        P("fake_failed_leg", "")
        g = lambda k: self.get_parameter(k).value
        cfg.abs_error_rad, cfg.ratio, cfg.hold_s = float(g("abs_error_rad")), float(g("ratio")), float(g("hold_s"))
        self.h = LegHealth(cfg)
        self.index = {n: i for i, n in enumerate(load_robot(g("robot_yaml") or None).joint_names)}
        self.q_ref = None
        self.q = None
        self.last_pub = None
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.pub = self.create_publisher(String, "/gurdy/leg_health", latched)
        self.create_subscription(Float64MultiArray, "/joint_pd_controller/commands", self._cmd, 10)
        self.create_subscription(JointState, "/joint_states", self._js, 10)
        self.create_service(Trigger, "/leg_health/reset", self._reset)
        self.create_timer(1.0, self._publish)

    def _cmd(self, m):
        self.q_ref = np.array(m.data)

    def _js(self, m):
        q = np.zeros(18)
        for n, p in zip(m.name, m.position):
            if n in self.index:
                q[self.index[n]] = p
        self.q = q
        if self.q_ref is None or len(self.q_ref) != 18:
            return
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        for leg in self.h.step(t, self.q_ref.tolist(), q.tolist()):
            self.get_logger().warn(f"leg {LEG_LONG[leg]} declared FAILED (tracking error {self.h.err[leg]:.2f} rad)")
        self._publish(only_on_change=True)

    def _reset(self, _req, resp):
        self.h.reset()
        resp.success, resp.message = True, "leg health reset"
        self._publish()
        return resp

    def _states(self):
        s = self.h.states
        fake = self.get_parameter("fake_failed_leg").value
        if fake in LEG_LONG:
            s[LEG_LONG[fake]] = "failed"
        return s

    def _publish(self, only_on_change=False):
        s = self._states()
        if only_on_change and s == self.last_pub:
            return
        self.last_pub = s
        self.pub.publish(String(data=json.dumps({"legs": s, "legs_order": [LEG_LONG[l] for l in LEG_NAMES]})))


def main():
    rclpy.init()
    n = LegHealthNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
