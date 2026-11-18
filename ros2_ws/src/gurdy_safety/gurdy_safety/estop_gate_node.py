"""/cmd_vel -> /cmd_vel_safe, zero while /estop (latched Bool) is true. Logic: gurdy_core.estop_gate.EstopGate (unit-tested).

While the stop is latched a zero Twist is also published at 20 Hz, so the controller never depends on its own command timeout.
Start it with `sim.launch.py estop_gate:=true` (that also remaps gait_node / policy_node to /cmd_vel_safe).
"""
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool

from gurdy_core.estop_gate import EstopGate


class EstopGateNode(Node):
    def __init__(self):
        super().__init__("estop_gate_node")
        self.gate = EstopGate()
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.pub = self.create_publisher(Twist, "/cmd_vel_safe", 10)
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd, 10)
        self.create_subscription(Bool, "/estop", self.on_estop, latched)
        self.create_timer(0.05, self.hold)

    def on_estop(self, m):
        was = self.gate.estop
        n = self.gate.set_estop(m.data)
        if m.data and not was:
            self.get_logger().warn("e-stop engaged: /cmd_vel_safe is zero")
        elif was and not m.data:
            self.get_logger().info(f"e-stop released ({n} non-zero commands were blocked)")
            self.hold(force=True)

    def _send(self, v):
        t = Twist()
        t.linear.x, t.linear.y, t.angular.z = v
        self.pub.publish(t)

    def on_cmd(self, m):
        self._send(self.gate.filter(m.linear.x, m.linear.y, m.angular.z))

    def hold(self, force=False):
        if self.gate.estop:
            self._send((0.0, 0.0, 0.0))


def main():
    rclpy.init()
    n = EstopGateNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
