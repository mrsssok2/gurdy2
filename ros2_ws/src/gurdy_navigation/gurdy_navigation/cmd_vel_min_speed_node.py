"""cmd_vel_min_speed: lift small non-zero Nav2 velocity commands to the minimum speed the policy can follow.

    /cmd_vel_smoothed (velocity_smoother)  ->  this node  ->  /cmd_vel_min  (collision_monitor input)  ->  /cmd_vel

MPPI starts from standstill with at most ax_max * model_dt = 0.04 m/s; the RL policy does not move below ~0.03 m/s from
rest. Zeroing such commands (smoother deadband) deadlocked the start, so (vx, vy) is scaled up to MIN_LINEAR_SPEED keeping
its direction and wz up to MIN_ANGULAR_SPEED (independently); below `epsilon` (normalised to the minimum) the command is an
exact zero. Teleop publishes /cmd_vel directly and is not affected. Pure function: gurdy_core.rl_command.lift_to_min_speed.
Stateless, no clock use (no use_sim_time needed); every message in -> one message out.
"""
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

from gurdy_core.rl_command import MIN_ANGULAR_SPEED, MIN_LINEAR_SPEED, MIN_SPEED_EPSILON, lift_to_min_speed


class CmdVelMinSpeed(Node):
    def __init__(self):
        super().__init__("cmd_vel_min_speed")
        self.declare_parameter("input_topic", "cmd_vel_smoothed")
        self.declare_parameter("output_topic", "cmd_vel_min")
        self.declare_parameter("min_linear", MIN_LINEAR_SPEED)
        self.declare_parameter("min_angular", MIN_ANGULAR_SPEED)
        self.declare_parameter("epsilon", MIN_SPEED_EPSILON)
        self.pub = self.create_publisher(Twist, self.get_parameter("output_topic").value, 10)
        self.create_subscription(Twist, self.get_parameter("input_topic").value, self.on_cmd, 10)

    def on_cmd(self, msg: Twist):
        c = lift_to_min_speed([msg.linear.x, msg.linear.y, msg.angular.z],
                              float(self.get_parameter("min_linear").value),
                              float(self.get_parameter("min_angular").value),
                              float(self.get_parameter("epsilon").value))
        out = Twist()
        out.linear.x, out.linear.y, out.angular.z = (float(v) for v in c)
        self.pub.publish(out)


def main():
    rclpy.init()
    node = CmdVelMinSpeed()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
