#!/usr/bin/env python3
"""Compare Gazebo vs MuJoCo recordings (policy_diag.sh): timing, sensors, per-segment joint behaviour."""
import sys
import numpy as np
sys.path.insert(0, "gurdy_core")
from gurdy_core.experiments import tracking_protocol

g, m = (np.load(f"{sys.argv[1]}.{s}.npz") for s in ("gz", "mj"))
changes, _, total = tracking_protocol()
print("protocol changes:", [(round(float(t), 1), tuple(c)) for t, c in changes][:20], "total", total)
for name, d in (("gazebo", g), ("mujoco", m)):
    dj, dc, di = np.diff(d["js_t"]), np.diff(d["cmd_t"]), np.diff(d["imu_t"])
    print(f"\n[{name}] joint_states dt mean {dj.mean()*1e3:.1f} ms (max {dj.max()*1e3:.0f}); command dt mean {dc.mean()*1e3:.1f} "
          f"std {dc.std()*1e3:.1f} max {dc.max()*1e3:.0f} ms; imu dt mean {di.mean()*1e3:.1f} ms")
    # standing at start (first 1 s): gravity, gyro, qd
    s = d["imu_t"] < 1.0
    w, x, y, z = d["imu_quat"][s].T
    gz_ = -(1 - 2 * (x * x + y * y))
    print(f"  stand: gyro std {d['imu_gyro'][s].std(0).round(4)} mean {d['imu_gyro'][s].mean(0).round(4)}; qd std {d['js_qd'][d['js_t']<1].std():.4f}")
    # per segment joint stats (t window = last 60% of each segment)
    bounds = [t for t, _ in changes] + [total]
    print("  seg  | q-target rms (rad) | q amplitude p95-p5 per joint-type (yaw,hip,knee) | qd rms | gyro rms")
    for i in range(len(bounds) - 1):
        t0, t1 = bounds[i] + 0.4 * (bounds[i + 1] - bounds[i]), bounds[i + 1]
        sj = (d["js_t"] >= t0) & (d["js_t"] < t1)
        q, qd = d["js_q"][sj], d["js_qd"][sj]
        # target interpolated at js times (last published)
        k = np.searchsorted(d["cmd_t"], d["js_t"][sj], side="right") - 1
        tgt = d["cmd_q"][np.clip(k, 0, None)]
        err = np.sqrt(((q - tgt) ** 2).mean())
        amp = np.percentile(q, 95, axis=0) - np.percentile(q, 5, axis=0)
        si = (d["imu_t"] >= t0) & (d["imu_t"] < t1)
        print(f"  {i}  | {err:.4f} | {amp[0::3].mean():.3f} {amp[1::3].mean():.3f} {amp[2::3].mean():.3f} | "
              f"{np.sqrt((qd**2).mean()):.3f} | {np.sqrt((d['imu_gyro'][si]**2).mean(0)).round(3)}")
