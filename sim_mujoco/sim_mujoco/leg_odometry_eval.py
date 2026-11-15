"""Offline leg-odometry drift test: one MuJoCo walk (forward, strafe, turn), no Gazebo, no ROS.

    python3 -m sim_mujoco.leg_odometry_eval [--out results.json]

Records joint state + IMU at 100 Hz (like Gazebo's /joint_states) while the TripodGait walks the
command schedule, runs gurdy_core.leg_odometry on the recording, compares with the ground-truth
pose. Two variants:
  legs       vx, vy, wz all from the legs; pose dead-reckoned with the leg wz
  legs+gyro  vx, vy from the legs with the gyro as angular rate, heading integrated from the gyro
             (what the EKF effectively does with the IMU)
Metrics per segment (pose reset to ground truth at the segment start): position error as % of the
ground-truth path length, heading error in degrees, plus velocity RMSE vs the ground-truth body velocity.
"""
from __future__ import annotations

import argparse
import json

import mujoco
import numpy as np

from gurdy_core.gait import TripodGait
from gurdy_core.leg_odometry import LegOdometry, quat_to_rotation
from gurdy_core.robot import load_robot
from sim_mujoco.experiments import _setup

SAMPLE_HZ = 100.0
# (name, command (vx, vy, wz), duration s); each segment is followed by 2 s of standing.
SEGMENTS = [("forward", (0.10, 0.0, 0.0), 12.0), ("strafe", (0.0, 0.08, 0.0), 12.0),
            ("turn", (0.0, 0.0, 0.30), 12.0), ("arc", (0.08, 0.0, 0.20), 12.0)]
PAUSE = 2.0


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def yaw_of(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def record(robot, segments=SEGMENTS):
    m, d, qadr, vadr = _setup(robot, fixed_base=False)
    gait = TripodGait(robot)
    sched, t = [], 3.0                                   # 3 s standing first
    for name, cmd, dur in segments:
        sched.append((name, t, t + dur, cmd))
        t += dur + PAUSE
    total = t
    sid = {n: m.sensor(n).adr[0] for n in ("imu_gyro", "imu_quat")}
    decim = int(round(gait.dt / m.opt.timestep))
    samp = int(round(1.0 / SAMPLE_HZ / m.opt.timestep))
    rows, k = [], 0
    while d.time < total:
        cmd = (0.0, 0.0, 0.0)
        for _, t0, t1, c in sched:
            if t0 <= d.time < t1:
                cmd = c
        if k % decim == 0:
            d.ctrl[:] = gait.step(cmd)
        mujoco.mj_step(m, d)
        k += 1
        if k % samp == 0:
            rows.append(dict(t=d.time, q=d.qpos[qadr].copy(), qd=d.qvel[vadr].copy(),
                             gyro=d.sensordata[sid["imu_gyro"]:sid["imu_gyro"] + 3].copy(),
                             quat=d.sensordata[sid["imu_quat"]:sid["imu_quat"] + 4].copy(),
                             pos=d.qpos[:3].copy(), gt_quat=d.qpos[3:7].copy(),
                             gt_v_world=d.qvel[:3].copy()))
    return rows, sched


def run_variant(robot, rows, sched, use_gyro):
    """Dead-reckon per segment from the ground-truth pose at its start."""
    odo = LegOdometry(robot)
    out, vel_err = {}, {}
    ests = []
    for r in rows:
        R = quat_to_rotation(*r["quat"])
        ests.append(odo.estimate(r["q"], r["qd"], R, r["gyro"][2] if use_gyro else None))
    yaw_rate = (lambda i: rows[i]["gyro"][2]) if use_gyro else (lambda i: None)
    for name, t0, t1, _ in sched:
        idx = [i for i, r in enumerate(rows) if t0 <= r["t"] < t1 + PAUSE]   # include the stop-down
        odo.reset()
        p0, y0 = rows[idx[0]]["pos"][:2], yaw_of(rows[idx[0]]["gt_quat"])
        odo.x, odo.y, odo.yaw = p0[0], p0[1], y0
        path, ev = 0.0, []
        for a, b in zip(idx[:-1], idx[1:]):
            dt = rows[b]["t"] - rows[a]["t"]
            odo.update(ests[a], dt, yaw_rate(a))
            path += np.linalg.norm(rows[b]["pos"][:2] - rows[a]["pos"][:2])
            if t0 + 3.0 <= rows[a]["t"] < t1 and ests[a].valid:         # steady-state velocity error
                yaw = yaw_of(rows[a]["gt_quat"])
                c, s = np.cos(yaw), np.sin(yaw)
                vw = rows[a]["gt_v_world"]
                gt = np.array([c * vw[0] + s * vw[1], -s * vw[0] + c * vw[1]])
                ev.append([*(ests[a].v - gt), ests[a].wz - rows[a]["gyro"][2]])
        last = rows[idx[-1]]
        pos_err = np.linalg.norm([odo.x - last["pos"][0], odo.y - last["pos"][1]])
        ev = np.array(ev)
        out[name] = {"path_m": path, "pos_err_m": pos_err, "pos_err_pct": 100 * pos_err / max(path, 1e-9),
                     "yaw_gt_deg": float(np.degrees(wrap(yaw_of(last["gt_quat"]) - y0))),
                     "yaw_err_deg": float(np.degrees(wrap(odo.yaw - yaw_of(last["gt_quat"])))),
                     "v_rmse_mps": float(np.sqrt((ev[:, :2] ** 2).mean())),
                     "wz_rmse_rads": float(np.sqrt((ev[:, 2] ** 2).mean())),
                     "v_err_std_mps": [float(ev[:, 0].std()), float(ev[:, 1].std())],
                     "wz_err_std_rads": float(ev[:, 2].std()),
                     "mean_stance_feet": float(np.mean([len(e.stance) for e in ests[idx[0]:idx[-1]]])),
                     "valid_frac": float(np.mean([e.valid for e in ests[idx[0]:idx[-1]]]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    args = ap.parse_args()
    robot = load_robot()
    rows, sched = record(robot)
    res = {"legs": run_variant(robot, rows, sched, False), "legs+gyro": run_variant(robot, rows, sched, True)}
    for var, segs in res.items():
        print(f"== {var}")
        for n, r in segs.items():
            print(f"  {n:8s} path {r['path_m']:.2f} m  pos err {r['pos_err_m']:.3f} m = {r['pos_err_pct']:.1f} %  "
                  f"heading {r['yaw_gt_deg']:+6.1f} deg (err {r['yaw_err_deg']:+.1f})  "
                  f"v rmse {r['v_rmse_mps']:.3f} m/s  wz rmse {r['wz_rmse_rads']:.3f} rad/s  "
                  f"stance {r['mean_stance_feet']:.1f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
