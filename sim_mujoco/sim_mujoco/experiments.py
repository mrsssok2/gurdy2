"""Run the shared step-response and standing/chatter protocols in MuJoCo (headless).

    python3 -m sim_mujoco.experiments step  > step_mujoco.json
    python3 -m sim_mujoco.experiments stand > stand_mujoco.json
"""
from __future__ import annotations

import json
import sys
import time

import mujoco
import numpy as np

from gurdy_core.experiments import (STAND_RECORD, STAND_SETTLE, analyze_stand, analyze_step,
                                    analyze_tracking, cmd_at, quat_to_rpy, stance_vector,
                                    step_protocol, tracking_protocol)
from gurdy_core.gait import TripodGait
from gurdy_core.robot import load_robot
from sim_mujoco.mjcf import build_mjcf


def _setup(robot, fixed_base):
    m = mujoco.MjModel.from_xml_string(build_mjcf(robot, fixed_base=fixed_base))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stance").id)
    qadr = np.array([m.joint(n).qposadr[0] for n in robot.joint_names])
    vadr = np.array([m.joint(n).dofadr[0] for n in robot.joint_names])
    return m, d, qadr, vadr


def _record(m, d, qadr, vadr, schedule, duration):
    """Step physics; ctrl follows the (time, q_ref) schedule. Samples every step (= PD rate)."""
    ts, qs, qds, taus = [], [], [], []
    ev = list(schedule)
    wall = time.perf_counter()
    while d.time < duration - 1e-9:
        while ev and ev[0][0] <= d.time + 1e-9:
            d.ctrl[:] = ev.pop(0)[1]
        mujoco.mj_step(m, d)
        ts.append(d.time)
        qs.append(d.qpos[qadr].copy())
        qds.append(d.qvel[vadr].copy())
        taus.append(d.actuator_force.copy())
    wall = time.perf_counter() - wall
    return np.array(ts), np.array(qs), np.array(qds), np.array(taus), duration / wall


def run_step(robot=None) -> dict:
    robot = robot or load_robot()
    events, _, total = step_protocol(robot)
    m, d, qadr, vadr = _setup(robot, fixed_base=True)
    t, q, qd, tau, rtf = _record(m, d, qadr, vadr, [(e.t, e.q_ref) for e in events], total)
    return {"sim": "mujoco", "rtf": rtf, "steps": analyze_step(robot, t, q, qd, tau)}


def run_stand(robot=None) -> dict:
    robot = robot or load_robot()
    m, d, qadr, vadr = _setup(robot, fixed_base=False)
    dur = STAND_SETTLE + STAND_RECORD
    t, q, qd, tau, rtf = _record(m, d, qadr, vadr, [(0.0, stance_vector(robot))], dur)
    res = analyze_stand(robot, t, q, qd, tau)
    res.update({"sim": "mujoco", "rtf": rtf})
    return res


def run_tracking(robot=None) -> dict:
    """Same /cmd_vel protocol as Gazebo: commands at 20 Hz, TripodGait at policy rate (50 Hz)."""
    robot = robot or load_robot()
    m, d, qadr, vadr = _setup(robot, fixed_base=False)
    gait = TripodGait(robot)
    changes, _, total = tracking_protocol()
    decim = int(round(gait.dt / m.opt.timestep))
    cmd_period = int(round(0.05 / m.opt.timestep))
    rows, k, cmd = [], 0, (0.0, 0.0, 0.0)
    wall = time.perf_counter()
    while d.time < total - 1e-9:
        if k % cmd_period == 0:
            cmd = cmd_at(changes, d.time)
        if k % decim == 0:
            d.ctrl[:] = gait.step(cmd)
        mujoco.mj_step(m, d)
        k += 1
        if k % 10 == 0:                           # 50 Hz pose log like Gazebo's odometry
            rows.append((d.time, *d.qpos[:7]))
    wall = time.perf_counter() - wall
    a = np.array(rows)
    roll, pitch, yaw = quat_to_rpy(a[:, 4], a[:, 5], a[:, 6], a[:, 7])
    res = analyze_tracking(a[:, 0], a[:, 1], a[:, 2], yaw, a[:, 3], roll, pitch)
    res.update({"sim": "mujoco", "rtf": total / wall, "ik_failures": gait.ik_failures})
    return res


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "step"
    out = {"step": run_step, "stand": run_stand, "track": run_tracking}[mode]()
    print(json.dumps(out, indent=2))
