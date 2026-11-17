"""Tripod gait node.

    /cmd_vel (geometry_msgs/Twist: linear.x, linear.y, angular.z)
        -> gurdy_core.gait.TripodGait at robot.yaml policy_rate_hz (50 Hz, sim time)
        -> /joint_pd_controller/commands (std_msgs/Float64MultiArray, 18 joint targets)
    /gait/cmd (geometry_msgs/Twist): the shaped command actually executed (limits, stride cap,
        acceleration limits) — useful for debugging and tracking tests.

Timing: the loop is driven by /joint_states (sim-time stamps, 100 Hz) and steps the gait whenever
sim time has advanced by one gait period. This avoids use_sim_time in rclpy, whose /clock
subscription (1 kHz from Gazebo) cost ~50 % CPU while the gait math itself costs ~2 %.

If /cmd_vel is silent for `cmd_timeout` seconds of sim time the command drops to zero (robot stops).
/estop (std_msgs/Bool, latched, same contract as policy_node): true -> ramp to the stand pose (joint speed limited by
`stand_speed`, no jump), ignore /cmd_vel, hold until /estop is false again; after a release the gait restarts from rest
and stands until a fresh /cmd_vel arrives.

Everything above this node only speaks /cmd_vel, so teleop / Nav2 / the LLM agent are swappable.
"""
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray

from gurdy_core.gait import TripodGait
from gurdy_core.robot import load_robot


class GaitNode(Node):
    def __init__(self):
        super().__init__("gait_node")
        self.declare_parameter("robot_yaml", "")
        self.declare_parameter("cmd_timeout", 0.5)
        self.declare_parameter("stand_speed", 1.0)      # rad/s joint speed limit while going to the stand pose (e-stop)
        path = self.get_parameter("robot_yaml").value or None
        self.robot = load_robot(path)
        self.gait = TripodGait(self.robot)
        self.timeout = float(self.get_parameter("cmd_timeout").value)
        self.cmd = np.zeros(3)
        self.last_cmd_time = None      # sim time (s) of the last /cmd_vel
        self.sim_now = None            # latest /joint_states stamp (s)
        self.last_tick = None
        self.estop = False
        self.q_stance = np.asarray(self.gait.joint_targets(), float).copy()   # 18 targets of the freshly reset gait = stand pose
        self.q_out = self.q_stance.copy()      # last published targets (the stand ramp starts here)
        self.pub = self.create_publisher(Float64MultiArray, "/joint_pd_controller/commands", 10)
        self.pub_cmd = self.create_publisher(Twist, "/gait/cmd", 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Bool, "/estop", self.on_estop, latched)
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd, 10)
        self.create_subscription(JointState, "/joint_states", self.on_joint_states, 10)
        self.failures_reported = 0
        self.late_ticks = 0            # gait periods that needed catch-up (diagnostic)
        self.get_logger().info(
            f"tripod gait at {1 / self.gait.dt:.0f} Hz, cycle {self.gait.cycle} s, "
            f"limits vx={self.gait.cmd_limits[0]} vy={self.gait.cmd_limits[1]} wz={self.gait.cmd_limits[2]}")

    def on_estop(self, msg: Bool):
        if bool(msg.data) != self.estop:
            self.get_logger().warn("E-STOP engaged: standing" if msg.data else "e-stop released")
            self.estop = bool(msg.data)
            self.cmd = np.zeros(3)
            self.last_cmd_time = None          # after a release: stand until a fresh /cmd_vel arrives
            self.gait.reset()

    def on_cmd(self, msg: Twist):
        if self.estop:
            return
        self.cmd = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
        self.last_cmd_time = self.sim_now

    def on_joint_states(self, msg: JointState):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_tick is not None and t < self.last_tick:      # sim reset: start over
            self.last_tick = None
        self.sim_now = t
        if self.last_tick is None:
            self.last_tick = t
            self.tick(1)
            return
        # Step once per elapsed gait period (catch up if a /joint_states message was late or
        # dropped), so the gait phase advances exactly with sim time.
        n = int((t - self.last_tick + 1e-6) // self.gait.dt)
        if n > 0:
            self.last_tick += n * self.gait.dt
            if n > 1:
                self.late_ticks += n - 1
                self.get_logger().info(f"catch-up gait periods so far: {self.late_ticks}",
                                       throttle_duration_sec=10.0)
            self.tick(min(n, 10))

    def tick(self, n_steps: int):
        if self.estop:                         # safe stand pose, rate-limited, no stride
            step = float(self.get_parameter("stand_speed").value) * self.gait.dt
            q = self.q_out + np.clip(self.q_stance - self.q_out, -step, step)
            self.q_out = q
            self.pub.publish(Float64MultiArray(data=q.tolist()))
            return
        if self.last_cmd_time is None or self.sim_now - self.last_cmd_time > self.timeout:
            self.cmd = np.zeros(3)
        for _ in range(n_steps):
            q = self.gait.step(self.cmd)
        self.q_out = q
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
