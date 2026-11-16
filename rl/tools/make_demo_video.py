"""Demo video (1280x720): tripod vs terrain_dr2 on a 12 deg slope and 4 cm steps (MuJoCo), then clips.

    python3 rl/tools/make_demo_video.py --out docs/videos/gurdy2_demo.mp4 --explore docs/results/phase3_explore_gt_timelapse.mp4
Two panels (640x720 each), same terrain (same seed) and command (vx 0.10 m/s); the slope seed is chosen so
uphill is roughly along +x. Overlays: distance, height gain, fall flag.
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

FPS, W, H, PW = 30, 1280, 720, 640
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
COL = [(110, 110, 110), (40, 160, 70)]


def candidate_seeds(cfg, terrain, want_uphill_x):
    """Seeds whose slope points uphill along +x (steps: any seed)."""
    if not want_uphill_x:
        yield from range(50)
        return
    env = _env(cfg, 5.0, fixed=True, terrain=terrain)
    for s in range(400):
        env.reset(seed=s)
        gx = env.tgen.height_at(env.heights, 1.0, 0.0) - env.tgen.height_at(env.heights, -1.0, 0.0)
        gy = env.tgen.height_at(env.heights, 0.0, 1.0) - env.tgen.height_at(env.heights, 0.0, -1.0)
        if gx > 0 and abs(gy) < 0.15 * gx:
            yield s


def rollout(spec, cfg, terrain, seed, seconds, cmd=(0.10, 0.0, 0.0)):
    ctl = Controller(spec)
    env = _env(cfg, seconds + 1.0, fixed=True, obs_mode=ctl.obs_mode, terrain=terrain)
    obs, _ = env.reset(seed=seed, options={"command": cmd})
    ctl.reset(env)
    q, fell = [env.data.qpos.copy()], []
    for _ in range(int(seconds / 0.02)):
        env.command = np.array(cmd, float)
        obs, _, term, _, _ = env.step(ctl.act(obs, env))
        q.append(env.data.qpos.copy())
        fell.append(term)
        if term:
            break
    return env, np.array(q), bool(any(fell))


def cam(q, dist=1.1, elev=-20.0, azim=125.0):
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:], c.distance, c.elevation, c.azimuth = [q[0], q[1], q[2]], dist, elev, azim
    return c


def terrain_scene(env_ref, ttype):
    m = env_ref.model
    m.geom_rgba[env_ref.floor_geom, 3] = 0.0
    m.geom_rgba[env_ref.terrain_geom] = [0.72, 0.66, 0.55, 1.0]
    m.vis.global_.offwidth, m.vis.global_.offheight = PW, H
    return m


def card(text, sub, n=2):
    img = Image.new("RGB", (W, H), (24, 26, 30))
    d = ImageDraw.Draw(img)
    f1, f2 = ImageFont.truetype(FONT_B, 60), ImageFont.truetype(FONT, 32)
    d.text(((W - d.textlength(text, font=f1)) / 2, H / 2 - 70), text, font=f1, fill=(240, 240, 240))
    d.text(((W - d.textlength(sub, font=f2)) / 2, H / 2 + 20), sub, font=f2, fill=(180, 180, 180))
    return np.array(img)


def terrain_segment(wr, title, ttype, value, seed_x, cfg_flat, cfg_dr2, ck_label, seconds=25.0):
    terrain = {"type": ttype, "value": value}
    specs = [("tripod", "tripod", cfg_dr2), (ck_label, resolve(ck_label)[1], cfg_dr2)]
    names = ["tripod gait", "terrain_dr2 (RL policy)"]
    # demo seed: first seed where the policy does not fall (eval success is 93 % at 12 deg, see exp_A); tripod outcome is whatever it is
    for seed in candidate_seeds(cfg_dr2, terrain, seed_x):
        recs = [rollout(s, c, terrain, seed, seconds) for _, s, c in specs]
        if not recs[1][2]:
            break
    print(title, "seed", seed, flush=True)
    model = terrain_scene(recs[0][0], ttype)
    data = mujoco.MjData(model)
    rend = mujoco.Renderer(model, H, PW)
    f, fb = ImageFont.truetype(FONT, 26), ImageFont.truetype(FONT_B, 34)
    n = int(seconds * FPS)
    for k in range(n):
        t = (k + 1) / FPS
        row = []
        for i, (env, q, fell) in enumerate(recs):
            j = min(int(round(t / 0.02)), len(q) - 1)
            data.qpos[:] = q[j]
            mujoco.mj_forward(model, data)
            rend.update_scene(data, camera=cam(q[j]))
            img = Image.fromarray(rend.render())
            dr = ImageDraw.Draw(img, "RGBA")
            dr.rectangle([0, 0, PW, 8], fill=COL[i] + (255,))
            dr.rectangle([12, 18, PW - 12, 128], fill=(255, 255, 255, 215))
            dr.text((24, 22), names[i], font=fb, fill=COL[i] + (255,))
            dr.text((24, 66), f"t {t:4.1f} s   distance {q[j][0] - q[0][0]:+.2f} m", font=f, fill=(20, 20, 20, 255))
            fallen = fell and j >= len(q) - 1
            dr.text((24, 94), "FELL - episode ended" if fallen else f"height gain {100 * (q[j][2] - q[0][2]):+.1f} cm",
                    font=f, fill=(200, 30, 30, 255) if fallen else (20, 20, 20, 255))
            row.append(np.array(img))
        fr = np.hstack(row)
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        d.rectangle([0, H - 44, W, H], fill=(0, 0, 0, 160))
        d.text((16, H - 38), title + "   |   command vx 0.10 m/s, MuJoCo", font=f, fill=(255, 255, 255, 255))
        wr.append_data(np.array(im))
    rend.close()
    return [(r[2], len(r[1])) for r in recs]


def clip_segment(wr, path, label, max_s=None, speed=1):
    rd = imageio.get_reader(path)
    fps = rd.get_meta_data()["fps"]
    f = ImageFont.truetype(FONT, 26)
    rep = max(1, int(round(FPS / (fps * speed))))      # source frames are shown for 1/fps (real time x speed)
    n_frames = rd.count_frames()
    for i, fr in enumerate(rd):
        if max_s and i / fps > max_s:
            break
        im = Image.fromarray(fr).convert("RGB")
        s = min(W / im.width, (H - 44) / im.height)
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        canvas.paste(im, ((W - im.width) // 2, 0))
        d = ImageDraw.Draw(canvas, "RGBA")
        d.rectangle([0, H - 44, W, H], fill=(0, 0, 0, 200))
        d.text((16, H - 38), label, font=f, fill=(255, 255, 255, 255))
        for _ in range(rep * (10 if i == n_frames - 1 else 1)):    # hold the final map ~2 s
            wr.append_data(np.array(canvas))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--explore", required=True)
    ap.add_argument("--explore-label", default="Phase 3: frontier exploration of flat_lab from an empty map (Gazebo, policy + Nav2 + slam_toolbox; sped up)")
    ap.add_argument("--policy", default="terrain_dr2:15000000")
    ap.add_argument("--nav-clip")
    a = ap.parse_args()
    cfg = resolve(a.policy)[2]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with imageio.get_writer(a.out, fps=FPS, codec="libx264", quality=None, macro_block_size=8,
                            ffmpeg_params=["-crf", "24", "-pix_fmt", "yuv420p"]) as wr:
        put = lambda fr, k=1: [wr.append_data(fr) for _ in range(k)]      # noqa: E731
        put(card("Gurdy 2.0", "18-DOF hexapod: RL gait on rough terrain + mapping and navigation"), 2 * FPS)
        put(card("Terrain: tripod vs RL policy", "12 deg slope, then 4 cm steps (MuJoCo, same terrain, same command)"), int(1.5 * FPS))
        r1 = terrain_segment(wr, "12 deg slope (uphill +x)", "slopes", 12.0, True, cfg, cfg, a.policy)
        r2 = terrain_segment(wr, "4 cm steps", "steps", 0.04, False, cfg, cfg, a.policy)
        print("falls/frames slope:", r1, "steps:", r2, flush=True)
        put(card("Mapping from scratch", "frontier exploration in Gazebo (RL policy + Nav2 + slam_toolbox)"), int(1.5 * FPS))
        clip_segment(wr, a.explore, a.explore_label)
        if a.nav_clip:
            clip_segment(wr, a.nav_clip, "Nav2 goal run")
    print("wrote", a.out, os.path.getsize(a.out) / 2**20, "MiB")


if __name__ == "__main__":
    main()
