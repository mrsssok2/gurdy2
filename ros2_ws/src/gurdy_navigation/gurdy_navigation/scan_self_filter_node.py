"""scan_self_filter: remove the robot's own legs from the lidar scan.

    /scan_raw (Gazebo bridge)  ->  this node  ->  /scan  (slam_toolbox, AMCL, costmaps, collision_monitor)

The lidar plane is 0.05 m above base_link, inside the swing path of the femurs: while walking ~90 % of the raw scans
have returns at 0.15-0.29 m from the legs (docs/results/phase3_selfhit.md). Every return closer than the Nav2 footprint
(robot.yaml, grown by `margin`) is replaced by +inf (no return). Stateless, copies the input header (no clock use).
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from gurdy_core.robot import load_robot
from gurdy_navigation.footprint import beam_exit_radius, filter_self_hits, footprint_polygon


class ScanSelfFilter(Node):
    def __init__(self):
        super().__init__("scan_self_filter")
        self.declare_parameter("robot_yaml", "")
        self.declare_parameter("input_topic", "scan_raw")
        self.declare_parameter("output_topic", "scan")
        self.declare_parameter("margin", 0.02)       # extra metres beyond the Nav2 footprint
        robot = load_robot(self.get_parameter("robot_yaml").value or None)
        self.poly = footprint_polygon(robot)
        self.origin = tuple(robot.sensors["lidar"]["pos"][:2])
        self.margin = float(self.get_parameter("margin").value)
        self.exit_r, self.key = None, None
        self.pub = self.create_publisher(LaserScan, self.get_parameter("output_topic").value, qos_profile_sensor_data)
        self.create_subscription(LaserScan, self.get_parameter("input_topic").value, self.on_scan, qos_profile_sensor_data)

    def on_scan(self, m: LaserScan):
        key = (m.angle_min, m.angle_increment, len(m.ranges))
        if key != self.key:
            ang = m.angle_min + np.arange(len(m.ranges)) * m.angle_increment
            self.exit_r, self.key = beam_exit_radius(self.poly, ang, self.origin, self.margin), key
        out = LaserScan()
        out.header, out.angle_min, out.angle_max, out.angle_increment = m.header, m.angle_min, m.angle_max, m.angle_increment
        out.time_increment, out.scan_time, out.range_min, out.range_max = m.time_increment, m.scan_time, m.range_min, m.range_max
        out.ranges = filter_self_hits(m.ranges, self.exit_r).astype(np.float32).tolist()
        out.intensities = m.intensities
        self.pub.publish(out)


def main():
    rclpy.init()
    node = ScanSelfFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
