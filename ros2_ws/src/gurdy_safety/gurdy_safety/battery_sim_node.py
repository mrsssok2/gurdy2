"""SIMULATED battery: /battery (sensor_msgs/BatteryState, percentage 0..1, location "SIMULATED"), 1 Hz.

Gazebo has no battery model. Drain: gurdy_core.ops_monitors.BatteryModel (0.03 %/s while moving, 0.002 %/s idle, time from /clock).
Set the level at run time to test low-battery behaviour:  ros2 param set /battery_sim_node level 15.0
"""
import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import BatteryState

from gurdy_core.ops_monitors import BatteryModel


class BatterySim(Node):
    def __init__(self):
        super().__init__("battery_sim_node")
        self.declare_parameter("level", 100.0)
        self.declare_parameter("drain_moving", 0.03)
        self.declare_parameter("drain_idle", 0.002)
        self.declare_parameter("odom_topic", "/odometry/filtered")
        self.model = BatteryModel(self.get_parameter("level").value, self.get_parameter("drain_idle").value,
                                  self.get_parameter("drain_moving").value)
        self.speed = None
        self.t = None
        self.create_subscription(Clock, "/clock", self._on_clock, 10)
        self.create_subscription(Odometry, self.get_parameter("odom_topic").value, self._odom, 10)
        self.pub = self.create_publisher(BatteryState, "/battery", 10)
        self.add_on_set_parameters_callback(self._params)
        self.create_timer(1.0, self.tick)           # wall timer; the model integrates sim time from /clock
        self.get_logger().warn("SIMULATED battery: the values on /battery are not measurements")

    def _params(self, params):
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name == "level":
                self.model.level = max(0.0, min(100.0, float(p.value)))
        return SetParametersResult(successful=True)

    def _on_clock(self, m):
        self.t = m.clock.sec + m.clock.nanosec * 1e-9

    def _odom(self, m):
        v = m.twist.twist.linear
        self.speed = math.hypot(v.x, v.y)

    def tick(self):
        if self.t is None:
            return
        lvl = self.model.step(self.t, self.speed)
        b = BatteryState()
        b.header.stamp.sec, b.header.stamp.nanosec = int(self.t), int((self.t % 1) * 1e9)
        b.header.frame_id = "SIMULATED"
        b.percentage = float(lvl / 100.0)
        b.present = True
        b.location = "SIMULATED"
        b.serial_number = "simulated-battery"
        b.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
        self.pub.publish(b)


def main():
    rclpy.init()
    n = BatterySim()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
