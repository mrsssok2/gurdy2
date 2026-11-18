#!/usr/bin/env python3
"""Offline leg-health calibration from a recorded walk (scripts/leg_health_record.py).
  python3 scripts/leg_health_analyze.py legs_walk.npz [more.npz ...]
Reports per-leg max/p99 of the monitor's EMA tracking error, the max ratio to the median of the others while walking, and which
(abs, ratio) settings would false-alarm; then a synthetic LOCKED-leg injection (measured joints frozen) to show detection time."""
import sys
import numpy as np
from gurdy_core.ops_monitors import LegHealth, LegHealthConfig
from gurdy_core.robot import LEG_NAMES

for path in sys.argv[1:]:
    d = np.load(path)["data"]
    t, ref, q = d[:, 0], d[:, 1:19], d[:, 19:37]
    print(f"{path}: {len(t)} samples, {t[-1]-t[0]:.0f} s sim, mean dt {np.mean(np.diff(t))*1000:.1f} ms")
    cfg = LegHealthConfig(abs_error_rad=9.9)               # never alarms: just collect the signal
    h = LegHealth(cfg); errs, ratios, walking = [], [], 0
    for i in range(len(t)):
        h.step(t[i], ref[i].tolist(), q[i].tolist())
        active = sum(1 for l in LEG_NAMES if h.moved[l] > cfg.excite_rad)
        if active >= cfg.min_active_legs:
            walking += 1
            e = np.array([h.err[l] for l in LEG_NAMES]); errs.append(e)
            r = [h.err[l] / max(sorted(h.err[o] for o in LEG_NAMES if o != l)[len(LEG_NAMES) // 2 - 1 + 0], 1e-3) for l in LEG_NAMES]
            ratios.append(r)
    E, R = np.array(errs), np.array(ratios)
    print(f"  judged (>=3 legs moving): {walking} samples = {walking*np.mean(np.diff(t)):.0f} s")
    print("  leg        mean   p99    max   max ratio-to-median-of-others")
    for k, l in enumerate(LEG_NAMES):
        print(f"  {l:8s} {E[:,k].mean():.3f} {np.percentile(E[:,k],99):.3f} {E[:,k].max():.3f}   {R[:,k].max():.2f}")
    print(f"  worst error any leg {E.max():.3f} rad; worst ratio {R.max():.2f}")
    # synthetic locked leg: from t_inj on, measured joints of that leg stay at their value at t_inj
    for leg in ("rm", "lf"):
        k = LEG_NAMES.index(leg); sl = slice(3 * k, 3 * k + 3); i0 = len(t) // 3
        for cfgx in (LegHealthConfig(), ):
            h = LegHealth(cfgx); qq = q.copy(); qq[i0:, sl] = q[i0, sl]; det = None
            for i in range(len(t)):
                if h.step(t[i], ref[i].tolist(), qq[i].tolist()) and det is None:
                    det = (t[i] - t[i0], i)
            print(f"  injected locked {leg}: " + (f"detected after {det[0]:.1f} s" if det else "NOT detected (injected at a point where the leg may be near its target)"))
    h = LegHealth(LegHealthConfig()); fa = []
    for i in range(len(t)):
        fa += h.step(t[i], ref[i].tolist(), q[i].tolist())
    print(f"  default thresholds on the healthy recording: false alarms = {fa}")
