"""Alert monitor -> /gurdy/alerts (String JSON {stamp, type, severity, source, text, data}).

Types: path_blocked (Nav2 goal aborted), stuck, fall, battery_low / battery_critical, leg_failure, object_new.
Inputs: /clock, /odometry/filtered, /cmd_vel_smoothed (commanded speed), /imu, /battery, /gurdy/leg_health,
/navigate_to_pose/_action/status, semantic_map/objects. Logic: gurdy_core.ops_monitors.AlertMonitor (unit-tested).
"""
import json
import math

import rclpy
from action_msgs.msg import GoalStatus, GoalStatusArray
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import BatteryState, Imu
from std_msgs.msg import String

from gurdy_core.ops_monitors import AlertConfig, AlertMonitor, Snapshot


class AlertMonitorNode(Node):
    def __init__(self):
        super().__init__("alert_monitor_node")
        self.declare_parameter("odom_topic", "/odometry/filtered")
        self.declare_parameter("cmd_topic", "cmd_vel_smoothed")
        self.mon = AlertMonitor(AlertConfig())
        self.t = None
        self.speed = self.cmd_speed = self.tilt = self.battery = self.legs = self.objects = None
        self.cmd_time = None
        self.aborted: set = set()
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.pub = self.create_publisher(String, "/gurdy/alerts", 50)
        self.create_subscription(Clock, "/clock", lambda m: setattr(self, "t", m.clock.sec + m.clock.nanosec * 1e-9), 10)
        self.create_subscription(Odometry, self.get_parameter("odom_topic").value, self._odom, 10)
        self.create_subscription(Twist, self.get_parameter("cmd_topic").value, self._cmd, 10)
        self.create_subscription(Imu, "/imu", self._imu, qos_profile_sensor_data)
        self.create_subscription(BatteryState, "/battery", lambda m: setattr(self, "battery", 100.0 * m.percentage), 10)
        self.create_subscription(String, "/gurdy/leg_health", self._legs, latched)
        self.create_subscription(String, "semantic_map/objects", self._objects, latched)
        self.create_subscription(GoalStatusArray, "/navigate_to_pose/_action/status", self._nav_status, 10)
        self.create_timer(0.5, self.tick)

    def _odom(self, m):
        v = m.twist.twist.linear
        self.speed = math.hypot(v.x, v.y)

    def _cmd(self, m):
        self.cmd_speed, self.cmd_time = math.hypot(m.linear.x, m.linear.y) + abs(m.angular.z) * 0.1, self.t   # turning counts as "told to move"

    def _imu(self, m):
        q = m.orientation
        gz = 1 - 2 * (q.x * q.x + q.y * q.y)
        self.tilt = math.degrees(math.acos(max(-1.0, min(1.0, gz))))

    def _legs(self, m):
        try:
            self.legs = json.loads(m.data).get("legs")
        except ValueError:
            pass

    def _objects(self, m):
        try:
            self.objects = json.loads(m.data).get("objects")
        except ValueError:
            pass

    def _nav_status(self, m):
        for s in m.status_list:
            if s.status == GoalStatus.STATUS_ABORTED:
                self.aborted.add(bytes(s.goal_info.goal_id.uuid).hex())

    def tick(self):
        if self.t is None:
            return
        cmd = self.cmd_speed if (self.cmd_time is not None and self.t - self.cmd_time < 1.0) else 0.0
        snap = Snapshot(t=self.t, cmd_speed=cmd if self.cmd_speed is not None else None, speed=self.speed, tilt_deg=self.tilt,
                        battery_pct=self.battery, legs=self.legs, nav_aborted_ids=tuple(self.aborted), objects=self.objects)
        for a in self.mon.step(snap):
            self.get_logger().info(f"alert {a['type']}: {a['text']}")
            self.pub.publish(String(data=json.dumps(a)))


def main():
    rclpy.init()
    n = AlertMonitorNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
