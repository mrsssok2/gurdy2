"""Is the policy's extra energy jitter? Records q-dot and torque at 500 Hz on the Phase 1 protocol
for each controller and reports:
  * joint-velocity power by frequency band (gait ~1.25 Hz; > 5 Hz is not gait motion),
  * energy split: E_smooth = integral sum|tau_lp * qdot_lp| with both low-passed < 5 Hz (zero-phase
    FFT filter); jitter share = 1 - E_smooth / E,
  * action rate mean sum (a_t - a_{t-1})^2 and second difference sum (a_t - 2a_{t-1} + a_{t-2})^2.
    python3 rl/tools/smoothness.py --run flat_bc_v1 --checkpoint latest
"""
import argparse
import json
import os
import sys

import numpy as np
import yaml

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, HERE)
from eval import Controller, _env      # noqa: E402
from gurdy_core.experiments import cmd_at, tracking_protocol   # noqa: E402
from gurdy_core.robot import JOINT_TYPES                       # noqa: E402

BANDS = [(0, 3), (3, 5), (5, 10), (10, 25), (25, 250)]


def lowpass(x, fs, fc):
    X = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(len(x), 1 / fs)
    X[f > fc] = 0
    return np.fft.irfft(X, n=len(x), axis=0)


def record(spec, cfg):
    changes, _, total = tracking_protocol()
    ctl = Controller(spec)
    # nominal physics (DR off) and the controller's own observation format, as in eval.py
    env = _env({**cfg, "env": {**cfg["env"], "record_substeps": True}}, total + 1, fixed=True,
               obs_mode=ctl.obs_mode)
    obs, _ = env.reset(seed=0, options={"command": (0, 0, 0)})
    ctl.reset(env)
    qd, tau, acts, t = [], [], [], 0.0
    while t < total - 1e-9:
        env.command = np.array(cmd_at(changes, t), float)
        obs, _, term, _, info = env.step(ctl.act(obs, env))
        acts.append(env.mapper.a_filt.copy())
        for v, f, *_ in info["substeps"]:
            qd.append(v)
            tau.append(f)
        t = env.steps * env.policy_dt
        if term:
            break
    return np.array(qd), np.array(tau), np.array(acts), 1.0 / env.model.opt.timestep, changes, total


def analyse(qd, tau, acts, fs, changes, total):
    # analyse moving segments only (skip stops and the random tail): forward..combined
    segs = [(t0, t0 + 12.0) for t0, c in changes if any(abs(x) > 0 for x in c)][:7]
    out = {"bands_pct": {}, "per_joint_type_hf_pct": {}}
    E = Es = 0.0
    P = np.zeros(len(BANDS))
    Pj = np.zeros((3, 2))
    for t0, t1 in segs:
        i0, i1 = int((t0 + 2) * fs), int(t1 * fs)          # skip 2 s transient like eval
        v, f = qd[i0:i1], tau[i0:i1]
        E += np.sum(np.abs(f * v)) / fs
        Es += np.sum(np.abs(lowpass(f, fs, 5) * lowpass(v, fs, 5))) / fs
        V = np.abs(np.fft.rfft(v - v.mean(0), axis=0)) ** 2
        fr = np.fft.rfftfreq(len(v), 1 / fs)
        for b, (lo, hi) in enumerate(BANDS):
            P[b] += V[(fr >= lo) & (fr < hi)].sum()
        for j in range(3):
            Vj = V[:, j::3]
            Pj[j] += [Vj[fr >= 5].sum(), Vj.sum()]
    for b, (lo, hi) in enumerate(BANDS):
        out["bands_pct"][f"{lo}-{hi} Hz"] = float(100 * P[b] / P.sum())
    for j, t in enumerate(JOINT_TYPES):
        out["per_joint_type_hf_pct"][t] = float(100 * Pj[j, 0] / Pj[j, 1])
    # dominant stepping frequency (hip_pitch velocity, 0.3-5 Hz) and stride amplitude, forward segment
    t0 = segs[0][0]
    v = qd[int((t0 + 2) * fs):int((t0 + 12) * fs), 1::3]
    V = (np.abs(np.fft.rfft(v - v.mean(0), axis=0)) ** 2).sum(1)
    fr = np.fft.rfftfreq(len(v), 1 / fs)
    m = (fr > 0.3) & (fr < 5)
    out["step_freq_hz_forward"] = float(fr[m][np.argmax(V[m])])
    out["joint_speed_rms_by_type"] = {t: float(np.sqrt(np.mean(qd[:, j::3] ** 2))) for j, t in enumerate(JOINT_TYPES)}
    out["energy_J_moving"] = float(E)
    out["jitter_energy_share_pct"] = float(100 * (1 - Es / E))
    da = np.diff(acts, axis=0)
    out["action_rate"] = float(np.mean(np.sum(da ** 2, axis=1)))
    out["action_2nd_diff"] = float(np.mean(np.sum(np.diff(acts, n=2, axis=0) ** 2, axis=1)))
    out["joint_speed_rms"] = float(np.sqrt(np.mean(qd ** 2)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="latest")
    a = ap.parse_args()
    run = os.path.join(HERE, "runs", a.run)
    cfg = yaml.safe_load(open(os.path.join(run, "config.yaml")))
    ck = os.path.join(run, "checkpoints", "ppo_latest.zip" if a.checkpoint == "latest" else f"ppo_{a.checkpoint}_steps.zip")
    res = {}
    for name, spec in (("tripod", "tripod"), (f"{a.run}:{a.checkpoint}", ck)):
        res[name] = analyse(*record(spec, cfg))
    os.makedirs(os.path.join(run, "eval"), exist_ok=True)
    json.dump(res, open(os.path.join(run, "eval", "smoothness.json"), "w"), indent=2)
    names = list(res)
    print("| metric | " + " | ".join(names) + " |\n|---|---|---|")
    for k in res[names[0]]["bands_pct"]:
        print(f"| joint-velocity power {k} | " + " | ".join(f"{res[n]['bands_pct'][k]:.1f} %" for n in names) + " |")
    for k in res[names[0]]["per_joint_type_hf_pct"]:
        print(f"| {k}: power above 5 Hz | " + " | ".join(f"{res[n]['per_joint_type_hf_pct'][k]:.1f} %" for n in names) + " |")
    print("| stepping frequency, forward (Hz) | " + " | ".join(f"{res[n]['step_freq_hz_forward']:.2f}" for n in names) + " |")
    for t in ("hip_yaw", "hip_pitch", "knee"):
        print(f"| {t} speed RMS (rad/s) | " + " | ".join(f"{res[n]['joint_speed_rms_by_type'][t]:.2f}" for n in names) + " |")
    for k, lab in (("energy_J_moving", "energy, moving segments (J)"), ("jitter_energy_share_pct", "energy NOT explained by < 5 Hz motion"),
                   ("action_rate", "action rate sum (da)^2"), ("action_2nd_diff", "action 2nd difference"),
                   ("joint_speed_rms", "joint speed RMS (rad/s)")):
        print(f"| {lab} | " + " | ".join(f"{res[n][k]:.3g}" + (" %" if "share" in k else "") for n in names) + " |")


if __name__ == "__main__":
    main()
