"""Side-by-side MP4 of the Phase 1 tracking protocol: tripod gait (left) vs policy (right).

    python3 rl/tools/make_video.py --run flat_bc_v1 --checkpoint latest [--out docs/videos/...mp4]

States are recorded with rl/eval.py's run_protocol (deterministic policy) and rendered offline with
MuJoCo's EGL renderer; the camera follows each robot. Every 2nd policy step -> 25 fps real time.
"""
import argparse
import os
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np
import yaml

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, HERE)
from eval import run_protocol           # noqa: E402
from sim_mujoco.mjcf import build_mjcf  # noqa: E402


def render(frames, w, h, label_rgb):
    # Checker floor (visual only, physics unchanged) so the motion is visible with a following camera.
    xml = build_mjcf().replace("<worldbody>", """<asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".78 .78 .78" rgb2=".58 .58 .58" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="50 50"/>
  </asset>
  <worldbody>""", 1).replace('rgba="0.7 0.7 0.7 1"', 'material="grid"', 1)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, h, w)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = 0.9, -25.0, 135.0
    out = []
    for q in frames[::2]:
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        cam.lookat[:] = [q[0], q[1], 0.08]
        r.update_scene(d, camera=cam)
        img = r.render().copy()
        img[:6, :] = label_rgb                      # colour bar: which controller
        out.append(img)
    r.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="latest")
    ap.add_argument("--out", default=None)
    ap.add_argument("--size", type=int, nargs=2, default=[480, 360])
    a = ap.parse_args()
    run = os.path.join(HERE, "runs", a.run)
    cfg = yaml.safe_load(open(os.path.join(run, "config.yaml")))
    ck = os.path.join(run, "checkpoints",
                      "ppo_latest.zip" if a.checkpoint == "latest" else f"ppo_{a.checkpoint}_steps.zip")
    (_, f_tri), (_, f_pol) = run_protocol(("tripod", cfg, True)), run_protocol((ck, cfg, True))
    n = min(len(f_tri), len(f_pol))
    w, h = a.size
    left = render(f_tri[:n], w, h, np.array([90, 90, 90], np.uint8))      # grey bar: tripod
    right = render(f_pol[:n], w, h, np.array([230, 120, 20], np.uint8))   # orange bar: policy
    out = a.out or os.path.join(run, "eval", f"protocol_tripod_vs_{a.run}_{a.checkpoint}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with imageio.get_writer(out, fps=25, codec="libx264", quality=6, macro_block_size=8) as wr:
        for l, rr in zip(left, right):
            wr.append_data(np.hstack([l, rr]))
    print(f"wrote {out}: {len(left)} frames, {os.path.getsize(out) / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
