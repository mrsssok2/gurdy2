#!/usr/bin/env python3
"""LIVE test of the cmd_vel e-stop gate (sim running with estop_gate:=true). Keeps publishing /cmd_vel the whole time.
  phase 1 (walk 8 s)   -> /cmd_vel_safe follows /cmd_vel, odom speed > 0.03 m/s
  phase 2 (e-stop 8 s) -> /cmd_vel_safe is exactly zero (after 0.3 s), robot speed ~ 0, while /cmd_vel is still non-zero
  phase 3 (release 10 s)-> /cmd_vel_safe non-zero again, robot walks again
Writes JSON to argv[1]. Times are wall seconds (sim RTF ~1)."""
import json
import math
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool


def main():
    rclpy.init()
    n = Node("estop_gate_test")
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
    pub_cmd, pub_e = n.create_publisher(Twist, "/cmd_vel", 10), n.create_publisher(Bool, "/estop", latched)
    safe, speed, rec = [], [0.0], []
    n.create_subscription(Twist, "/cmd_vel_safe", lambda m: safe.append((time.time(), math.hypot(m.linear.x, m.linear.y) + abs(m.angular.z))), 10)
    n.create_subscription(Odometry, "/odom", lambda m: speed.__setitem__(0, math.hypot(m.twist.twist.linear.x, m.twist.twist.linear.y)), 10)
    ex = SingleThreadedExecutor(); ex.add_node(n)
    res = {}

    def phase(name, secs, estop):
        pub_e.publish(Bool(data=estop))
        t0 = time.time(); k0 = len(safe); sp = []
        while time.time() - t0 < secs:
            c = Twist(); c.linear.x = 0.12
            pub_cmd.publish(c)
            ex.spin_once(timeout_sec=0.05); time.sleep(0.05)
            sp.append((time.time() - t0, speed[0]))
        s = [v for t, v in safe[k0:] if t - t0 > 0.3]
        late = [v for t, v in sp if t > secs - 3]
        res[name] = {"safe_msgs": len(s), "safe_max": round(max(s), 4) if s else None, "safe_min": round(min(s), 4) if s else None,
                     "odom_speed_last3s_mean": round(sum(late) / len(late), 3), "odom_speed_max": round(max(v for _, v in sp), 3)}
        print(name, res[name], flush=True)
    for _ in range(20):
        ex.spin_once(timeout_sec=0.1)
    phase("1_walk", 8, False)
    phase("2_estop", 8, True)
    phase("3_released", 10, False)
    ok = (res["1_walk"]["safe_min"] is not None and res["1_walk"]["odom_speed_max"] > 0.03
          and res["2_estop"]["safe_max"] == 0.0 and res["2_estop"]["odom_speed_last3s_mean"] < 0.02
          and (res["3_released"]["safe_max"] or 0) > 0.05 and res["3_released"]["odom_speed_max"] > 0.03)
    res["pass"] = ok
    json.dump(res, open(sys.argv[1], "w"), indent=1)
    print("PASS" if ok else "FAIL", flush=True)
    pub_e.publish(Bool(data=False)); rclpy.shutdown()


main()
