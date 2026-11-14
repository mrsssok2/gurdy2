#!/usr/bin/env python3
"""Print a Gazebo vs MuJoCo comparison table (markdown) from parity_experiments.sh output."""
import json
import sys
from pathlib import Path

from gurdy_core.experiments import PARITY_TOL, parity_check_step

out, mode = Path(sys.argv[1]), sys.argv[2]
gz = json.loads((out / f"gazebo_{mode}.json").read_text())
mj = json.loads((out / f"mujoco_{mode}.json").read_text())
print(f"RTF: gazebo {gz['rtf']:.2f} (with 500 Hz recorder), mujoco {mj['rtf']:.0f}\n")
if mode == "step":
    checks = parity_check_step(gz["steps"], mj["steps"])
    print("| joint | step rad | rise ms gz / mj | overshoot % gz / mj | steady err mrad gz / mj | peak vel rad/s gz / mj | parity |")
    print("|---|---|---|---|---|---|---|")
    for a, b, c in zip(gz["steps"], mj["steps"], checks):
        print(f"| {a['joint']} | {a['amp']:.2f} | {a['rise_time_s']*1e3:.0f} / {b['rise_time_s']*1e3:.0f} "
              f"| {a['overshoot_pct']:.1f} / {b['overshoot_pct']:.1f} "
              f"| {a['steady_error_rad']*1e3:.2f} / {b['steady_error_rad']*1e3:.2f} "
              f"| {a['peak_velocity']:.2f} / {b['peak_velocity']:.2f} | {'PASS' if c['pass'] else 'FAIL'} |")
    print(f"\nTolerances: {PARITY_TOL}")
    print(f"STEP PARITY: {'PASS' if all(c['pass'] for c in checks) else 'FAIL'}")
else:
    print("| metric | gazebo | mujoco |\n|---|---|---|")
    for k in ("qd_rms", "q_p2p", "tau_std"):
        print(f"| worst {k} ({gz['worst_joint'][k]} / {mj['worst_joint'][k]}) | {gz['worst'][k]:.2e} | {mj['worst'][k]:.2e} |")
    print(f"| chatter pass (limits {gz['chatter_limits']}) | {gz['chatter_pass']} | {mj['chatter_pass']} |")
    for jt in ("hip_yaw", "hip_pitch", "knee"):
        print(f"| sag {jt} mrad | {gz['sag_rad_by_type'][jt]*1e3:.2f} | {mj['sag_rad_by_type'][jt]*1e3:.2f} |")
    print(f"| body height (FK) m | {gz['body_height_fk_m']:.4f} | {mj['body_height_fk_m']:.4f} |")
    print(f"| samples / duration s | {gz['samples']} / {gz['duration_s']:.1f} | {mj['samples']} / {mj['duration_s']:.1f} |")
    dh = abs(gz["body_height_fk_m"] - mj["body_height_fk_m"])
    ok = gz["chatter_pass"] and mj["chatter_pass"] and dh <= PARITY_TOL["body_height_m"]
    print(f"\nSTAND: chatter both pass = {gz['chatter_pass'] and mj['chatter_pass']}, "
          f"body height diff {dh*1e3:.2f} mm (tol {PARITY_TOL['body_height_m']*1e3:.0f} mm) -> {'PASS' if ok else 'FAIL'}")
