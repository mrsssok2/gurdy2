#!/usr/bin/env python3
"""Gazebo recording: delay (sim time) from the /joint_states stamp that triggers a policy tick to the arrival
of that tick's /joint_pd_controller/commands message (recorder-side /clock, so +~1 ms measurement noise)."""
import sys
import numpy as np
d = np.load(sys.argv[1] + ".gz.npz")
js, c = d["js_t"], d["cmd_t"]
# tick js stamps: grid of 20 ms. Estimate grid by matching each command to the latest js stamp <= it,
# then take the most common phase of (stamp mod 20 ms).
k = np.searchsorted(js, c, side="right") - 1
last = js[np.clip(k, 0, None)]
lat_since_js = c - last
print(f"cmd arrival - latest /joint_states stamp: mean {lat_since_js.mean()*1e3:.1f} ms  p50 {np.percentile(lat_since_js,50)*1e3:.1f}  "
      f"p95 {np.percentile(lat_since_js,95)*1e3:.1f}  max {lat_since_js.max()*1e3:.1f}")
ph = np.round((js % 0.02) * 1e3).astype(int) % 20
print("js stamp phase mod 20 ms histogram (ms:count):", dict(zip(*np.unique(ph, return_counts=True))))
dj = np.diff(js) * 1e3
print(f"js gaps: p50 {np.median(dj):.1f} p95 {np.percentile(dj,95):.1f} p99 {np.percentile(dj,99):.1f} max {dj.max():.0f} ms; "
      f">15 ms: {100*(dj>15).mean():.1f} %")
dc = np.diff(c) * 1e3
print(f"cmd gaps: p50 {np.median(dc):.1f} p95 {np.percentile(dc,95):.1f} p99 {np.percentile(dc,99):.1f} max {dc.max():.0f} ms; "
      f">30 ms: {100*(dc>30).mean():.1f} %  (n={len(c)} in {c[-1]-c[0]:.0f} s => {len(c)/(c[-1]-c[0]):.1f} Hz)")
if "rx_imu" in d.files:
    ia = (d["rx_imu"] - d["imu_t"]) * 1e3
    ja = (d["rx_js"] - d["js_t"]) * 1e3
    print(f"receipt(/clock) - header stamp: imu mean {ia.mean():.1f} ms p95 {np.percentile(ia,95):.1f} max {ia.max():.0f}; "
          f"joint_states mean {ja.mean():.1f} p95 {np.percentile(ja,95):.1f} max {ja.max():.0f}")
