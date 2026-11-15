#!/usr/bin/env python3
"""Velocity-tracking protocol in MuJoCo through the DEPLOYMENT path (same as the ROS node, no ROS):
sim_mujoco model (the one matched to Gazebo), IMU + joint sensors -> gurdy_core.policy_runtime
(deadband, obs builder, ONNX, EMA) at 50 Hz, /cmd_vel at 20 Hz. Prints the same JSON as
`python3 -m sim_mujoco.experiments track` / scripts/gz_tracking.py, so compare_* tools apply.

    python3 scripts/policy_track_mujoco.py [--model models/dr_energy_ft_6M.onnx] [--controller tripod]
"""
import argparse
import json
import os
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gurdy_core"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gurdy_core.experiments import analyze_tracking, cmd_at, quat_to_rpy, tracking_protocol  # noqa: E402
from gurdy_core.policy_runtime import OnnxPolicy, PolicyController, load_meta  # noqa: E402
from gurdy_core.robot import load_robot  # noqa: E402
from sim_mujoco.mjcf import build_mjcf  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "..", "models", "dr_energy_ft_6M.onnx"))
    args = ap.parse_args()
    robot = load_robot()
    m = mujoco.MjModel.from_xml_string(build_mjcf(robot))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stance").id)
    qadr = np.array([m.joint(n).qposadr[0] for n in robot.joint_names])
    vadr = np.array([m.joint(n).dofadr[0] for n in robot.joint_names])
    gyro, quat = m.sensor("imu_gyro").adr[0], m.sensor("imu_quat").adr[0]
    ctl = PolicyController(robot, OnnxPolicy(args.model), load_meta(args.model))
    d.ctrl[:] = ctl.q_stance
    changes, _, total = tracking_protocol()
    decim = int(round(ctl.dt / m.opt.timestep))
    cmd_period = int(round(0.05 / m.opt.timestep))
    rows, k, cmd, wall, infer = [], 0, (0.0, 0.0, 0.0), time.perf_counter(), []
    while d.time < total - 1e-9:
        if k % cmd_period == 0:
            cmd = cmd_at(changes, d.time)
        if k % decim == 0:
            t0 = time.perf_counter()
            d.ctrl[:] = ctl.step(d.sensordata[quat:quat + 4], d.sensordata[gyro:gyro + 3],
                                 d.qpos[qadr], d.qvel[vadr], cmd)
            infer.append(time.perf_counter() - t0)
        mujoco.mj_step(m, d)
        k += 1
        if k % 10 == 0:
            rows.append((d.time, *d.qpos[:7]))
    wall = time.perf_counter() - wall
    a = np.array(rows)
    roll, pitch, yaw = quat_to_rpy(a[:, 4], a[:, 5], a[:, 6], a[:, 7])
    res = analyze_tracking(a[:, 0], a[:, 1], a[:, 2], yaw, a[:, 3], roll, pitch)
    res.update({"sim": "mujoco_policy_deployment", "rtf": total / wall, "model": os.path.basename(args.model),
                "tick_ms_mean": 1e3 * float(np.mean(infer)), "tick_ms_max": 1e3 * float(np.max(infer))})
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
