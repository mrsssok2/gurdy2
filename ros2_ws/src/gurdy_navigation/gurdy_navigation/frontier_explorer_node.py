"""Frontier explorer: drive Nav2 to the best frontier of the SLAM map until none is left, then go back to the start.

    /map (slam_toolbox) -> gurdy_core.frontier.plan_frontiers -> NavigateToPose goal -> repeat

Loop (1 Hz, sim time from /clock):
  * no goal active: plan; nothing usable for `empty_cycles` consecutive cycles (>= `empty_seconds` sim s) -> done;
  * goal active: cancel it (no blacklist) when its frontier has been resolved; cancel + count a failure when it exceeds
    `goal_timeout` or makes no progress (`progress_window` s, < `progress_min` m closer);
  * a goal that failed `max_failures` times (Nav2 abort / timeout) is blacklisted (radius `blacklist_radius`); a goal that
    was reached `max_visits` times with a frontier still there is blacklisted too (unresolvable pocket, e.g. behind glass).
Then it navigates to the start pose (first map-frame pose seen), writes `log_path` (JSON, every goal) and touches `done_path`.
Odometry is taken from TF map->base_link only, so it works with ground-truth or EKF odometry alike.
Does not use use_sim_time (rclpy sim time costs CPU at 1 kHz); the clock comes from a /clock subscription.
"""
import json
import math
import os

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from tf2_ros import Buffer, TransformException, TransformListener

from gurdy_core import frontier as fr


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__("frontier_explorer")
        P = self.declare_parameter
        P("min_cells", 6)                # frontier cluster size filter (6 cells = 0.3 m)
        P("clearance", 0.40)             # m kept from walls/objects: goals outside Nav2's inflation band (inflation_radius 0.55, robot ~0.57 m wide)
        P("reach", 0.4)
        P("size_weight", 0.5)
        P("blacklist_radius", 0.6)
        P("goal_timeout", 150.0)         # sim s per goal
        P("progress_window", 40.0)       # sim s
        P("progress_min", 0.25)          # m
        P("arrive_radius", 0.35)         # m: within this of a frontier goal = visited (the robot sees the frontier from here)
        P("max_failures", 2)
        P("max_visits", 2)
        P("empty_cycles", 3)
        P("empty_seconds", 10.0)        # sim s with no frontier AND a map that stopped growing (slam updates every 2 s)
        P("min_goal_age", 4.0)           # sim s before "close enough" may cancel (lets the slam map refresh)
        P("max_time", 2400.0)            # sim s: give up exploring, still return to start
        P("return_to_start", True)
        P("log_path", "explore_log.json")
        P("done_path", "explore.done")
        g = lambda k: self.get_parameter(k).value
        self.p = {k: g(k) for k in ("min_cells", "clearance", "reach", "size_weight", "blacklist_radius", "goal_timeout",
                                    "progress_window", "progress_min", "arrive_radius", "max_failures", "max_visits", "empty_cycles",
                                    "empty_seconds", "min_goal_age", "max_time", "return_to_start", "log_path", "done_path")}
        self.tf = Buffer()
        self.tfl = TransformListener(self.tf, self)
        self.ac = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.now = None
        self.map = None
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid, "/map", self.on_map, qos)
        self.create_subscription(Clock, "/clock", self.on_clock, 10)
        self.t_start = None
        self.start_pose = None
        self.goal = None               # dict of the active goal
        self.gh = None
        self.cancelling = False
        self.pending = False
        self.goals = []                # finished goals (log)
        self.blacklist = []
        self.fail_count, self.visit_count = {}, {}
        self.empty = 0
        self.empty_since = None
        self.empty_known = 0
        self.returning = False
        self.finished = False
        self.reason = ""
        self.create_timer(1.0, self.tick)
        self.get_logger().info("frontier explorer up; waiting for /map, TF and navigate_to_pose")

    # ---- inputs --------------------------------------------------------------------------------------
    def on_clock(self, m):
        self.now = m.clock.sec + m.clock.nanosec * 1e-9

    def on_map(self, m):
        w, h = m.info.width, m.info.height
        self.map = (np.array(m.data, dtype=np.int8).reshape(h, w), m.info.resolution,
                    (m.info.origin.position.x, m.info.origin.position.y))

    def robot_xy(self):
        try:
            t = self.tf.lookup_transform("map", "base_link", rclpy.time.Time()).transform.translation
            return t.x, t.y
        except TransformException:
            return None

    # ---- main loop -----------------------------------------------------------------------------------
    def tick(self):
        if self.finished or self.now is None:
            return
        pose = self.robot_xy()
        if pose is None or self.map is None or not self.ac.server_is_ready():
            return
        if self.t_start is None:
            self.t_start, self.start_pose = self.now, pose
            self.get_logger().info(f"start pose {pose[0]:.2f},{pose[1]:.2f}; exploring")
        if self.gh is not None or self.pending or self.cancelling:
            self.monitor_goal(pose)
            return
        if self.returning:
            return
        if self.now - self.t_start > self.p["max_time"]:
            return self.begin_return("time limit")
        fs = self.plan(pose)
        if not fs:
            known = int((self.map[0] != -1).sum())
            if self.empty_since is None or known > self.empty_known:       # map still growing: restart the quiet period
                self.empty_since, self.empty_known, self.empty = self.now, known, 0
            self.empty += 1
            if self.empty >= self.p["empty_cycles"] and self.now - self.empty_since >= self.p["empty_seconds"]:
                self.begin_return("no frontiers left")
            return
        self.empty, self.empty_since = 0, None
        self.send(fs[0], pose)

    def plan(self, pose):
        grid, res, org = self.map
        return fr.plan_frontiers(grid, res, org, pose, min_cells=self.p["min_cells"], clearance_m=self.p["clearance"],
                                 reach_m=self.p["reach"], blacklist=self.blacklist,
                                 blacklist_radius_m=self.p["blacklist_radius"], size_weight=self.p["size_weight"])

    # ---- goals ---------------------------------------------------------------------------------------
    def send(self, f, pose, final=False, xy=None, yaw=None):
        xy = xy or f.goal_xy
        if yaw is None:
            yaw = math.atan2(xy[1] - pose[1], xy[0] - pose[0])
        g = NavigateToPose.Goal()
        g.pose.header.frame_id = "map"
        g.pose.pose.position.x, g.pose.pose.position.y = float(xy[0]), float(xy[1])
        g.pose.pose.orientation.z, g.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.goal = {"xy": [round(float(xy[0]), 3), round(float(xy[1]), 3)], "t0": self.now, "from": [round(pose[0], 2), round(pose[1], 2)],
                     "size_m": round(f.size_m, 2) if f else 0.0, "path_m": round(f.path_m, 2) if f else 0.0,
                     "final": final, "best_d": math.hypot(xy[0] - pose[0], xy[1] - pose[1]), "best_t": self.now,
                     "recoveries": 0, "cancel_reason": ""}
        self.gh, self.pending = None, True
        self.get_logger().info(f"goal {len(self.goals) + 1}: {self.goal['xy']} ({'return' if final else 'frontier'}), path {self.goal['path_m']} m, size {self.goal['size_m']} m")
        fut = self.ac.send_goal_async(g, feedback_callback=self.on_feedback)
        fut.add_done_callback(self.on_accept)

    def on_feedback(self, msg):
        if self.goal:
            self.goal["recoveries"] = max(self.goal["recoveries"], msg.feedback.number_of_recoveries)

    def on_accept(self, fut):
        h = fut.result()
        self.pending = False
        if not h.accepted:
            self.finish_goal("REJECTED")
            return
        self.gh = h
        h.get_result_async().add_done_callback(self.on_result)

    def on_result(self, fut):
        s = fut.result().status
        name = {GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED", GoalStatus.STATUS_CANCELED: "CANCELED",
                GoalStatus.STATUS_ABORTED: "ABORTED"}.get(s, f"status {s}")
        self.finish_goal(name)

    def monitor_goal(self, pose):
        gl = self.goal
        if gl is None or self.cancelling or self.pending:
            return
        d = math.hypot(gl["xy"][0] - pose[0], gl["xy"][1] - pose[1])
        if d < gl["best_d"] - self.p["progress_min"]:
            gl["best_d"], gl["best_t"] = d, self.now
        if gl["final"]:
            if self.now - gl["t0"] > 300.0:
                self.cancel("return timeout")
            return
        if d < self.p["arrive_radius"] and self.now - gl["t0"] >= self.p["min_goal_age"]:
            return self.cancel("close enough")
        if self.now - gl["t0"] > self.p["goal_timeout"]:
            return self.cancel("goal timeout")
        if self.now - gl["best_t"] > self.p["progress_window"]:
            return self.cancel("no progress")
        # target resolved while driving? (no valid frontier cluster within 0.8 m of the goal any more)
        grid, res, org = self.map
        near = False
        for c in fr.clean_frontier_mask(grid, self.p["min_cells"]):
            xy = [fr.cell_to_world(tuple(cell), res, org) for cell in c[:: max(1, len(c) // 20)]]
            if any(math.hypot(x - gl["xy"][0], y - gl["xy"][1]) < 0.8 for x, y in xy):
                near = True
                break
        if not near:
            self.cancel("frontier resolved")

    def cancel(self, reason):
        self.goal["cancel_reason"] = reason
        self.cancelling = True
        if self.gh is not None:
            self.gh.cancel_goal_async()

    def finish_goal(self, status):
        gl = self.goal
        pose = self.robot_xy() or (float("nan"), float("nan"))
        gl.update({"status": status, "t_end": self.now, "duration_s": round(self.now - gl["t0"], 1),
                   "end_pose": [round(pose[0], 2), round(pose[1], 2)], "t0": round(gl["t0"] - self.t_start, 1)})
        gl.pop("best_t"), gl.pop("best_d")
        self.goals.append(gl)
        reason = gl["cancel_reason"]
        self.get_logger().info(f"goal {len(self.goals)} {status} {reason} after {gl['duration_s']} s")
        self.goal, self.gh, self.cancelling, self.pending = None, None, False, False
        key = tuple(round(v / 0.5) for v in gl["xy"])
        if gl["final"]:
            self.finish()
            return
        bad = status in ("ABORTED", "REJECTED") or reason in ("goal timeout", "no progress")
        if bad:
            self.fail_count[key] = self.fail_count.get(key, 0) + 1
            if self.fail_count[key] >= self.p["max_failures"]:
                self.blacklist.append(tuple(gl["xy"]))
                gl["blacklisted"] = True
        elif status == "SUCCEEDED" or reason == "close enough":
            self.visit_count[key] = self.visit_count.get(key, 0) + 1
            if self.visit_count[key] >= self.p["max_visits"]:      # reached twice and the frontier is still there
                self.blacklist.append(tuple(gl["xy"]))
                gl["blacklisted"] = True

    # ---- end -----------------------------------------------------------------------------------------
    def begin_return(self, reason):
        self.reason = reason
        self.get_logger().info(f"exploration over: {reason}")
        pose = self.robot_xy()
        if self.p["return_to_start"] and self.start_pose is not None and pose is not None:
            self.returning = True
            self.send(None, pose, final=True, xy=self.start_pose, yaw=0.0)
        else:
            self.finish()

    def finish(self):
        self.finished = True
        pose = self.robot_xy()
        d = None
        if pose is not None and self.start_pose is not None:
            d = round(math.hypot(pose[0] - self.start_pose[0], pose[1] - self.start_pose[1]), 3)
        fr_goals = [g for g in self.goals if not g["final"]]
        out = {"reason": self.reason, "sim_time_s": round(self.now - self.t_start, 1), "start_pose": list(self.start_pose),
               "end_pose": list(pose) if pose else None, "end_to_start_m": d,
               "frontier_goals": len(fr_goals),
               "succeeded": sum(g["status"] == "SUCCEEDED" for g in fr_goals),
               "cancelled_resolved": sum(g.get("cancel_reason") == "frontier resolved" for g in fr_goals),
               "close_enough": sum(g.get("cancel_reason") == "close enough" for g in fr_goals),
               "failed": sum(g["status"] in ("ABORTED", "REJECTED") or g.get("cancel_reason") in ("goal timeout", "no progress")
                             for g in fr_goals),
               "blacklisted": [list(b) for b in self.blacklist],
               "return_status": next((g["status"] for g in self.goals if g["final"]), None),
               "goals": self.goals}
        os.makedirs(os.path.dirname(os.path.abspath(self.p["log_path"])), exist_ok=True)
        with open(self.p["log_path"], "w") as f:
            json.dump(out, f, indent=1)
        open(self.p["done_path"], "w").write(self.reason + "\n")
        self.get_logger().info(f"done: {self.reason}; log {self.p['log_path']}")


def main():
    rclpy.init()
    n = FrontierExplorer()
    try:
        while rclpy.ok() and not n.finished:
            rclpy.spin_once(n, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
