"""Tripod gait node.

    /cmd_vel (geometry_msgs/Twist: linear.x, linear.y, angular.z)
        -> gurdy_core.gait.TripodGait at robot.yaml policy_rate_hz (50 Hz, sim time)
        -> /joint_pd_controller/commands (std_msgs/Float64MultiArray, 18 joint targets)
    /gait/cmd (geometry_msgs/Twist): the shaped command actually executed (limits, stride cap,
        acceleration limits) — useful for debugging and tracking tests.

If /cmd_vel is silent for `cmd_timeout` seconds the command drops to zero (robot stops).
Everything above this node only speaks /cmd_vel, so teleop / Nav2 / the LLM agent are swappable.
"""
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from gurdy_core.gait import TripodGait
from gurdy_core.robot import load_robot


class GaitNode(Node):
    def __init__(self):
        super().__init__("gait_node")
        self.declare_parameter("robot_yaml", "")
        self.declare_parameter("cmd_timeout", 0.5)
        path = self.get_parameter("robot_yaml").value or None
        self.robot = load_robot(path)
        self.gait = TripodGait(self.robot)
        self.timeout = float(self.get_parameter("cmd_timeout").value)
        self.cmd = np.zeros(3)
        self.last_cmd_time = None
        self.pub = self.create_publisher(Float64MultiArray, "/joint_pd_controller/commands", 10)
        self.pub_cmd = self.create_publisher(Twist, "/gait/cmd", 10)
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd, 10)
        self.create_timer(self.gait.dt, self.tick)
        self.failures_reported = 0
        self.get_logger().info(
            f"tripod gait at {1 / self.gait.dt:.0f} Hz, cycle {self.gait.cycle} s, "
            f"limits vx={self.gait.cmd_limits[0]} vy={self.gait.cmd_limits[1]} wz={self.gait.cmd_limits[2]}")

    def on_cmd(self, msg: Twist):
        self.cmd = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
        self.last_cmd_time = self.get_clock().now()

    def tick(self):
        now = self.get_clock().now()
        if self.last_cmd_time is None or (now - self.last_cmd_time).nanoseconds * 1e-9 > self.timeout:
            self.cmd = np.zeros(3)
        q = self.gait.step(self.cmd)
        self.pub.publish(Float64MultiArray(data=q.tolist()))
        c = self.gait.cmd
        t = Twist()
        t.linear.x, t.linear.y, t.angular.z = float(c[0]), float(c[1]), float(c[2])
        self.pub_cmd.publish(t)
        if self.gait.ik_failures > self.failures_reported:
            self.get_logger().warn(f"IK failures so far: {self.gait.ik_failures} (leg held at last pose)")
            self.failures_reported = self.gait.ik_failures


def main():
    rclpy.init()
    node = GaitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
