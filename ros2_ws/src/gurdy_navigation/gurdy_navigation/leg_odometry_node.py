"""Leg odometry node: /joint_states (+ /imu) -> /odom/legs, and an IMU copy with covariances for the EKF.

    /joint_states (sensor_msgs/JointState, 18 joints)        -> gurdy_core.leg_odometry
    /imu (sensor_msgs/Imu)  optional: orientation (stance detection under tilt) and gyro z
        -> /odom/legs   (nav_msgs/Odometry, frame odom, child base_link): twist = body-frame
                        (vx, vy, wz) from the stance feet with covariance; pose = dead-reckoned
                        (diagnostic only, the EKF fuses the twist)
        -> /imu/ekf     (sensor_msgs/Imu): /imu with orientation / angular-velocity covariances
                        filled in when the source leaves them 0 (Gazebo does), because
                        robot_localization treats a 0 covariance as "perfectly known".

Stamps are copied from the messages (sim time) so no use_sim_time is needed here (CLAUDE.md lessons).
Runs at the /joint_states rate (100 Hz in sim). Everything is the shared gurdy_core.leg_odometry code,
the same that the offline MuJoCo drift test uses.
"""
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, JointState

from gurdy_core.leg_odometry import LegOdomConfig, LegOdometry, quat_to_rotation
from gurdy_core.robot import load_robot

BIG = 1e3


def stamp_s(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def odometry_message(stamp, odom_frame, base_frame, x, y, yaw, est) -> Odometry:
    """Pure helper (unit-tested): fill an Odometry from a LegOdomEstimate and the dead-reckoned pose."""
    m = Odometry()
    m.header.stamp = stamp
    m.header.frame_id = odom_frame
    m.child_frame_id = base_frame
    m.pose.pose.position.x, m.pose.pose.position.y = float(x), float(y)
    m.pose.pose.orientation.z, m.pose.pose.orientation.w = float(np.sin(yaw / 2)), float(np.cos(yaw / 2))
    pc = [0.0] * 36
    for i in range(6):
        pc[i * 7] = BIG                       # the pose is dead-reckoned: unusable as a measurement
    m.pose.covariance = pc
    tc = [0.0] * 36
    for i in (2, 3, 4):
        tc[i * 7] = BIG                       # vz, wx, wy not estimated
    if est.valid:
        m.twist.twist.linear.x, m.twist.twist.linear.y = float(est.v[0]), float(est.v[1])
        m.twist.twist.angular.z = float(est.wz)
        tc[0], tc[7], tc[35] = (float(est.cov[0, 0]), float(est.cov[1, 1]), float(est.cov[2, 2]))
    else:
        tc[0] = tc[7] = tc[35] = BIG          # no stance foot: tell the EKF not to believe this
    m.twist.covariance = tc
    return m


def fill_imu_covariance(msg: Imu, orientation_var: float, gyro_var: float, accel_var: float) -> Imu:
    """Replace 'all zero' covariances (unknown / perfect) by realistic ones."""
    if msg.orientation_covariance[0] == 0.0:
        msg.orientation_covariance = [orientation_var, 0.0, 0.0, 0.0, orientation_var, 0.0, 0.0, 0.0, orientation_var]
    if msg.angular_velocity_covariance[0] == 0.0:
        msg.angular_velocity_covariance = [gyro_var, 0.0, 0.0, 0.0, gyro_var, 0.0, 0.0, 0.0, gyro_var]
    if msg.linear_acceleration_covariance[0] == 0.0:
        msg.linear_acceleration_covariance = [accel_var, 0.0, 0.0, 0.0, accel_var, 0.0, 0.0, 0.0, accel_var]
    return msg


class LegOdometryNode(Node):
    def __init__(self):
        super().__init__("leg_odometry_node")
        self.declare_parameter("robot_yaml", "")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("use_imu", True)               # orientation (tilt-aware stance) + gyro-assisted vx, vy
        self.declare_parameter("imu_max_age", 0.1)            # s: older IMU samples are ignored
        self.declare_parameter("imu_orientation_var", 0.01)   # rad^2, used when the IMU reports 0
        self.declare_parameter("imu_gyro_var", 2.5e-5)        # (rad/s)^2  (robot.yaml gyro_noise 0.005)
        self.declare_parameter("imu_accel_var", 4e-4)         # (m/s^2)^2  (robot.yaml accel_noise 0.02)
        path = self.get_parameter("robot_yaml").value or None
        self.robot = load_robot(path)
        self.odo = LegOdometry(self.robot, LegOdomConfig())
        self.odom_frame = self.get_parameter("odom_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.use_imu = bool(self.get_parameter("use_imu").value)
        self.imu_max_age = float(self.get_parameter("imu_max_age").value)
        self.joint_names = self.robot.joint_names
        self.index = None                      # message index of each of the 18 joints
        self.last_q = None
        self.last_t = None
        self.imu = None                        # (stamp s, R_wb, gyro_z)
        self.invalid_count = 0
        self.pub = self.create_publisher(Odometry, "/odom/legs", 10)
        self.pub_imu = self.create_publisher(Imu, "/imu/ekf", 20)       # reliable: matches either EKF QoS
        self.create_subscription(JointState, "/joint_states", self.on_joints, 10)
        self.create_subscription(Imu, "/imu", self.on_imu, qos_profile_sensor_data)
        self.get_logger().info(f"leg odometry started (imu {'on' if self.use_imu else 'off'})")

    def on_imu(self, msg: Imu):
        fill_imu_covariance(msg, float(self.get_parameter("imu_orientation_var").value),
                            float(self.get_parameter("imu_gyro_var").value),
                            float(self.get_parameter("imu_accel_var").value))
        self.pub_imu.publish(msg)
        o = msg.orientation
        self.imu = (stamp_s(msg.header.stamp), quat_to_rotation(o.w, o.x, o.y, o.z), msg.angular_velocity.z)

    def on_joints(self, msg: JointState):
        if self.index is None:
            try:
                self.index = [list(msg.name).index(n) for n in self.joint_names]
            except ValueError:
                self.get_logger().warn("joint_states lacks some of the 18 joints", throttle_duration_sec=5.0)
                return
        t = stamp_s(msg.header.stamp)
        q = np.array([msg.position[i] for i in self.index])
        if len(msg.velocity) == len(msg.name):
            qd = np.array([msg.velocity[i] for i in self.index])
        elif self.last_q is not None and t > self.last_t:       # no velocities published: differentiate
            qd = (q - self.last_q) / (t - self.last_t)
        else:
            qd = np.zeros(18)
        dt = 0.0 if self.last_t is None or t < self.last_t else t - self.last_t   # a sim reset restarts the pose
        if self.last_t is not None and t < self.last_t:
            self.odo.reset()
        self.last_q, self.last_t = q, t

        R_wb = gyro = None
        if self.use_imu and self.imu is not None and abs(t - self.imu[0]) < self.imu_max_age:
            R_wb, gyro = self.imu[1], self.imu[2]
        est = self.odo.estimate(q, qd, R_wb, gyro)
        if not est.valid:
            self.invalid_count += 1
            self.get_logger().warn(f"fewer than {self.odo.cfg.min_stance_feet} stance feet "
                                   f"({self.invalid_count} so far)", throttle_duration_sec=10.0)
        self.odo.update(est, dt, gyro)
        self.pub.publish(odometry_message(msg.header.stamp, self.odom_frame, self.base_frame,
                                          self.odo.x, self.odo.y, self.odo.yaw, est))


def main():
    rclpy.init()
    node = LegOdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
