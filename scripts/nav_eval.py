#!/usr/bin/env python3
"""Send Nav2 goals one after another (map frame) and report per goal: success (action status), sim time, GT path length,
final position error (ground truth vs goal), stall time (|v|<0.02 m/s for >5 s while the goal is active), recoveries.
    python3 scripts/nav_eval.py OUT.json "1.5,0.5,0.0" "4.0,0.5,0.0" ...
Needs ground-truth odom (/odom/ground_truth, map == world frame because the robot spawns at the world origin).
"""
import json
import math
import sys

import rclpy
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from rosgraph_msgs.msg import Clock


class Ev(Node):
    def __init__(self):
        super().__init__("nav_eval")
        self.ac = ActionClient(self, NavigateToPose, "/navigate_to_pose")
        self.pose, self.now, self.path_len, self.prev = None, None, 0.0, None
        self.create_subscription(Odometry, "/odom/ground_truth", self.on_odom, 20)
        self.create_subscription(Clock, "/clock", lambda m: setattr(self, "now", m.clock.sec + m.clock.nanosec * 1e-9), 10)
        self.rec = 0

    def on_odom(self, m):
        p = m.pose.pose.position
        self.pose = (p.x, p.y)
        if self.prev is not None:
            self.path_len += math.hypot(p.x - self.prev[0], p.y - self.prev[1])
        self.prev = self.pose

    def fb(self, msg):
        self.rec = msg.feedback.number_of_recoveries


def main():
    out, goals = sys.argv[1], [tuple(float(v) for v in g.split(",")) for g in sys.argv[2:]]
    rclpy.init()
    n = Ev()
    n.ac.wait_for_server()
    while n.pose is None or n.now is None:
        rclpy.spin_once(n, timeout_sec=0.1)
    results = []
    for (x, y, yaw) in goals:
        g = NavigateToPose.Goal()
        g.pose.header.frame_id = "map"
        g.pose.pose.position.x, g.pose.pose.position.y = x, y
        g.pose.pose.orientation.z, g.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        n.path_len, n.rec = 0.0, 0
        t0 = n.now
        fut = n.ac.send_goal_async(g, feedback_callback=n.fb)
        rclpy.spin_until_future_complete(n, fut)
        gh = fut.result()
        status, stall, still_since, max_stall = "REJECTED", 0.0, None, 0.0
        if gh.accepted:
            rf = gh.get_result_async()
            last_pos, last_t = n.pose, n.now
            while not rf.done() and n.now - t0 < 900:
                rclpy.spin_once(n, timeout_sec=0.1)
                if n.now - last_t >= 1.0:
                    v = math.hypot(n.pose[0] - last_pos[0], n.pose[1] - last_pos[1]) / (n.now - last_t)
                    if v < 0.02:
                        still_since = still_since if still_since is not None else last_t
                        max_stall = max(max_stall, n.now - still_since)
                    else:
                        still_since = None
                    last_pos, last_t = n.pose, n.now
            if rf.done():
                status = GoalStatus.STATUS_SUCCEEDED == rf.result().status and "SUCCEEDED" or f"status {rf.result().status}"
            else:
                status = "TIMEOUT"
                gh.cancel_goal_async()
        err = math.hypot(n.pose[0] - x, n.pose[1] - y)
        r = {"goal": (x, y, yaw), "status": status, "time_s": round(n.now - t0, 1), "path_m": round(n.path_len, 2),
             "final_err_m": round(err, 3), "recoveries": n.rec, "max_stall_s": round(max_stall, 1),
             "reached_0.25": status == "SUCCEEDED" and err < 0.25}
        print(r, flush=True)
        results.append(r)
    json.dump(results, open(out, "w"), indent=1)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
