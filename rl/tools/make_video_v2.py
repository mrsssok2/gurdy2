"""Comparison video v2: N controllers side by side (1280x720 per panel, 30 fps), MuJoCo nominal.

    python3 rl/tools/make_video_v2.py --controllers tripod flat_bc_v1:latest dr_energy_v1:latest \
        --out rl/runs/dr_energy_v1/eval/compare_3way.mp4

Segments (every controller runs the same commands from the same start state):
  1. "Command changes" - following camera: forward 0.10 m/s -> turn 0.3 rad/s -> sideways 0.08 m/s
     -> stop. Overlays: commanded vs actual velocity (1 s mean), tracking error %, energy used.
  2. "Fixed top camera" - forward 0.10 m/s for 10 s from a start line; 0.25 m floor markers make
     real distance and sideways drift visible. Overlay: distance, drift, energy.
  3. "Feet, 0.25x slow motion" - close side view of the feet, 2 s of walking rendered from 125 Hz
     states (every 4th 2 ms physics step) at 30 fps.
Title cards (1.5 s) between segments. Overlays are drawn with Pillow (DejaVu Sans).
"""
import argparse
import os
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, HERE)
from eval import Controller, _env, resolve        # noqa: E402
from sim_mujoco.mjcf import build_mjcf            # noqa: E402

FPS = 30
W, H = 1280, 720
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
COLORS = [(110, 110, 110), (40, 110, 220), (230, 120, 20), (40, 160, 70)]
LIM = np.array([0.15, 0.10, 0.60])

SEG_COMMANDS = [(0.0, (0.10, 0.0, 0.0)), (6.0, (0.0, 0.0, 0.3)), (12.0, (0.0, 0.08, 0.0)), (18.0, (0.0, 0.0, 0.0))]
SEG1_S, SEG2_S, SEG3_S = 20.0, 10.0, 2.0


def render_model():
    """Generated MJCF + visual-only extras: checker floor, start line, 0.25 m markers, big offscreen."""
    marks = "".join(
        f'<geom type="box" size="0.004 0.4 0.0005" pos="{x:.2f} 0 0.0006" rgba="0.1 0.1 0.1 1" contype="0" conaffinity="0"/>'
        for x in np.arange(0.25, 1.51, 0.25))
    xml = build_mjcf().replace("<worldbody>", f"""<asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".80 .80 .80" rgb2=".62 .62 .62" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="50 50"/>
  </asset>
  <worldbody>
    <geom type="box" size="0.006 0.4 0.0006" pos="0 0 0.0006" rgba="0.85 0.1 0.1 1" contype="0" conaffinity="0"/>
    <geom type="box" size="0.8 0.002 0.0005" pos="0.75 0 0.0005" rgba="0.2 0.4 0.9 1" contype="0" conaffinity="0"/>
    {marks}""", 1).replace('rgba="0.7 0.7 0.7 1"', 'material="grid"', 1)
    xml = xml.replace('<global offwidth="640" offheight="480"/>', f'<global offwidth="{W}" offheight="{H}"/>')
    return mujoco.MjModel.from_xml_string(xml)


def simulate(spec, cfg, commands, seconds, substeps=False):
    """Run one controller; returns per-policy-step dict lists (and 2 ms-substep qpos if asked)."""
    ctl = Controller(spec)
    env = _env({**cfg, "env": {**cfg["env"], "record_substeps": substeps}}, seconds + 1.0, fixed=True,
               obs_mode=ctl.obs_mode)
    obs, _ = env.reset(seed=0, options={"command": commands[0][1]})
    ctl.reset(env)
    out = {"t": [], "qpos": [], "cmd": [], "vel": [], "energy": [], "sub_qpos": []}
    t, e = 0.0, 0.0
    while t < seconds - 1e-9:
        cmd = [c for tc, c in commands if tc <= t + 1e-9][-1]
        env.command = np.array(cmd, float)
        obs, _, term, _, info = env.step(ctl.act(obs, env))
        t = env.steps * env.policy_dt
        e += info["energy_J"]
        out["t"].append(t)
        out["qpos"].append(env.data.qpos.copy())
        out["cmd"].append(np.array(cmd, float))
        out["vel"].append(np.array([*info["body_vel"][:2], info["yaw_rate"]]))
        out["energy"].append(e)
        if substeps:
            out["sub_qpos"] += [q for *_, q in info["substeps"]]
        if term:
            break
    return {k: np.array(v) for k, v in out.items()}


class Panel:
    def __init__(self, model, label, sub, color):
        self.m, self.d = model, mujoco.MjData(model)
        self.r = mujoco.Renderer(model, H, W)
        self.label, self.sub, self.color = label, sub, color
        self.f = ImageFont.truetype(FONT, 24)
        self.fb = ImageFont.truetype(FONT_B, 34)

    def scene(self, qpos, cam):
        self.d.qpos[:] = qpos
        mujoco.mj_forward(self.m, self.d)
        self.r.update_scene(self.d, camera=cam)
        return Image.fromarray(self.r.render())

    def decorate(self, img, lines):
        dr = ImageDraw.Draw(img, "RGBA")
        dr.rectangle([0, 0, W, 8], fill=self.color + (255,))
        dr.rectangle([16, 20, 16 + 760, 20 + 54 + 32 * len(lines)], fill=(255, 255, 255, 210))
        dr.text((28, 26), self.label, font=self.fb, fill=self.color + (255,))
        dr.text((28 + dr.textlength(self.label, font=self.fb) + 14, 34), self.sub, font=self.f, fill=(60, 60, 60, 255))
        for i, ln in enumerate(lines):
            dr.text((28, 72 + 32 * i), ln, font=self.f, fill=(20, 20, 20, 255))
        return img

    def close(self):
        self.r.close()


def follow_cam(qpos, dist=1.0, elev=-22.0, azim=135.0, z=0.08):
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.distance, c.elevation, c.azimuth = dist, elev, azim
    c.lookat[:] = [qpos[0], qpos[1], z]
    return c


def fixed_cam(lookat, dist, elev, azim):
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:], c.distance, c.elevation, c.azimuth = lookat, dist, elev, azim
    return c


def at_time(rec, t):
    return min(int(round(t / 0.02)) - 1, len(rec["t"]) - 1) if t > 0 else 0


def window_mean(rec, i, n=50):
    return rec["vel"][max(0, i - n + 1):i + 1].mean(axis=0)


def track_err(v, c):
    if not np.any(np.abs(c) > 1e-6):
        return None
    return 100 * np.linalg.norm((v - c) / LIM) / np.linalg.norm(c / LIM)


def title_card(text, sub, n_panels):
    img = Image.new("RGB", (W * n_panels, H), (24, 26, 30))
    dr = ImageDraw.Draw(img)
    f1, f2 = ImageFont.truetype(FONT_B, 64), ImageFont.truetype(FONT, 34)
    dr.text(((W * n_panels - dr.textlength(text, font=f1)) / 2, H / 2 - 70), text, font=f1, fill=(240, 240, 240))
    dr.text(((W * n_panels - dr.textlength(sub, font=f2)) / 2, H / 2 + 20), sub, font=f2, fill=(180, 180, 180))
    return np.array(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controllers", nargs="+", default=["tripod", "flat_bc_v1:latest"])
    ap.add_argument("--labels", nargs="+", help="panel labels (default derived)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crf", type=int, default=23)
    a = ap.parse_args()

    specs = []
    for c in a.controllers:
        if c == "tripod":
            specs.append(("tripod", "tripod", "Phase 1 gait (open loop)"))
        else:
            name, path, cfg = resolve(c)
            sub = "BC-init + PPO, flat" if "flat_bc" in name else "BC-init + PPO + DR + energy"
            specs.append((name.split(":")[0], path, sub))
    cfg = resolve(next(c for c in a.controllers if c != "tripod"))[2]
    labels = a.labels or [s[0] for s in specs]
    model = render_model()
    panels = [Panel(model, labels[i], specs[i][2], COLORS[i % len(COLORS)]) for i in range(len(specs))]
    n = len(panels)
    hold = int(1.5 * FPS)

    print("[video] simulating ...", flush=True)
    seg1 = [simulate(s[1], cfg, SEG_COMMANDS, SEG1_S) for s in specs]
    seg2 = [simulate(s[1], cfg, [(0.0, (0.10, 0.0, 0.0))], SEG2_S) for s in specs]
    seg3 = [simulate(s[1], cfg, [(0.0, (0.10, 0.0, 0.0))], 4.0, substeps=True) for s in specs]

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with imageio.get_writer(a.out, fps=FPS, codec="libx264", quality=None, macro_block_size=8,
                            ffmpeg_params=["-crf", str(a.crf), "-pix_fmt", "yuv420p"]) as wr:
        def put(frame, k=1):
            for _ in range(k):
                wr.append_data(frame)

        put(title_card("1 · Command changes", "forward 0.10 m/s  →  turn 0.3 rad/s  →  sideways 0.08 m/s  →  stop", n), hold)
        for k in range(int(SEG1_S * FPS)):
            t = (k + 1) / FPS
            row = []
            for p, rec in zip(panels, seg1):
                i = at_time(rec, t)
                c, v = rec["cmd"][i], window_mean(rec, i)
                err = track_err(v, c)
                lines = [f"t {t:5.1f} s   command  vx {c[0]:+.2f}  vy {c[1]:+.2f}  wz {c[2]:+.2f}",
                         f"actual (1 s mean)      vx {v[0]:+.2f}  vy {v[1]:+.2f}  wz {v[2]:+.2f}",
                         f"tracking error {err:5.1f} %" if err is not None else "tracking error  (stop)",
                         f"energy used {rec['energy'][i]:6.0f} J"]
                row.append(np.array(p.decorate(p.scene(rec["qpos"][i], follow_cam(rec["qpos"][i])), lines)))
            put(np.hstack(row))

        put(title_card("2 · Fixed top camera", "forward 0.10 m/s for 10 s from the red start line (markers every 0.25 m)", n), hold)
        cam = fixed_cam([0.55, 0.0, 0.0], 2.0, -62.0, 90.0)
        for k in range(int(SEG2_S * FPS)):
            t = (k + 1) / FPS
            row = []
            for p, rec in zip(panels, seg2):
                i = at_time(rec, t)
                q = rec["qpos"][i]
                lines = [f"t {t:5.1f} s   command vx +0.10 m/s",
                         f"distance {q[0]:.3f} m   (ideal {0.10 * t:.3f} m)",
                         f"sideways drift {q[1] * 100:+.1f} cm",
                         f"energy used {rec['energy'][i]:6.0f} J"]
                row.append(np.array(p.decorate(p.scene(q, cam), lines)))
            put(np.hstack(row))

        put(title_card("3 · Feet, 0.25x slow motion", "forward 0.10 m/s, close side view (2 s of walking)", n), hold)
        step = 4                                   # 2 ms substeps -> 125 Hz states -> 0.24x at 30 fps
        start = int(2.0 / 0.002)                   # skip the first 2 s (start-up)
        for k in range(int(SEG3_S / (step * 0.002))):
            row = []
            for p, rec in zip(panels, seg3):
                j = min(start + k * step, len(rec["sub_qpos"]) - 1)
                q = rec["sub_qpos"][j]
                cam = fixed_cam([q[0], q[1], 0.05], 0.55, -8.0, 90.0)
                lines = [f"real time {2.0 + j * 0.002 - start * 0.002:5.2f} s   (0.25x)"]
                row.append(np.array(p.decorate(p.scene(q, cam), lines)))
            put(np.hstack(row))
    for p in panels:
        p.close()
    print(f"[video] wrote {a.out} ({os.path.getsize(a.out) / 2**20:.1f} MiB)", flush=True)


if __name__ == "__main__":
    main()
