"""Ground-truth odometry -> /odom + the odom -> base_link TF that Nav2 / slam_toolbox need.

    /odom/ground_truth (nav_msgs/Odometry, Gazebo OdometryPublisher)
        -> /odom (frame odom, child base_link) and TF odom -> base_link

This is the *stand-in* odometry (Phase 3 step 1). It is perfect, so SLAM quality measured with it
is an upper bound. Later it is replaced by robot_localization's EKF (IMU + leg odometry) that
publishes the same /odom + TF; nothing else changes (see docs/phase3_navigation.md).

Stamps are copied from the message (sim time), so this node does not need use_sim_time
(rclpy's /clock subscription at 1 kHz costs ~40 % CPU, see CLAUDE.md lessons).
"""
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


def to_transform(msg: Odometry, odom_frame: str, base_frame: str) -> TransformStamped:
    t = TransformStamped()
    t.header.stamp = msg.header.stamp
    t.header.frame_id = odom_frame
    t.child_frame_id = base_frame
    p, q = msg.pose.pose.position, msg.pose.pose.orientation
    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = p.x, p.y, p.z
    t.transform.rotation = q
    return t


class OdomTfNode(Node):
    def __init__(self):
        super().__init__("odom_tf_node")
        self.declare_parameter("input", "/odom/ground_truth")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.odom_frame = self.get_parameter("odom_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.pub = self.create_publisher(Odometry, "/odom", 10)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(Odometry, self.get_parameter("input").value, self.on_odom, 10)

    def on_odom(self, msg: Odometry):
        msg.header.frame_id = self.odom_frame
        msg.child_frame_id = self.base_frame
        self.pub.publish(msg)
        self.tf.sendTransform(to_transform(msg, self.odom_frame, self.base_frame))


def main():
    rclpy.init()
    node = OdomTfNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
