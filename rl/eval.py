"""Step 2.5: deterministic evaluation of policies vs the tripod gait (MuJoCo, flat ground).

    python3 rl/eval.py --run flat_bc_v1 --checkpoints 10000000 30000000 latest
    python3 rl/eval.py --policies flat_bc_v1:latest dr_energy_v1:latest --scenarios all --episodes 50

Two protocols, identical for every controller:
 1. Phase 1 tracking protocol (gurdy_core.experiments.tracking_protocol, 123 s, commands at 20 Hz,
    analysed with the SAME analyze_tracking code as Phase 1): per-segment main-axis error, falls,
    mechanical energy (sum |tau*qdot| integrated every 2 ms substep) and cost of transport
    CoT = E / (m g d), d = horizontal path length.
 2. 100 episodes of 20 s with the training command distribution (fixed seeds 0..99): fall rate and
    per-command-window tracking error: mean body velocity over seconds 1-5 of each 5 s window vs
    the command, error = ||(v - c)/lim|| / ||c/lim|| (lim = command limits, mixes m/s and rad/s).
Policies act deterministically (mean action); observations normalised with the checkpoint's
VecNormalize statistics; each policy gets the observation format it was trained with (flat Box or
asymmetric Dict - the critic-only part is ignored by the actor). Results: JSON + markdown on stdout.

Step 2.7 --scenarios (protocol 2 under fixed domain-randomisation extremes, everything else nominal):
nominal, friction_low/high, heavy, weak_motors, latency_10ms, sensor_noise, pushes, all_extremes.

START GATE (2.7b, always run): from rest (zero command for 2 s + 0-0.7 s, i.e. 8 gait-clock phases),
apply each pure command (+-vx 0.10, +-vy 0.08, +-wz 0.3) for 8 s; a start fails if the main-axis speed
over the last 5 s is < 20 % of the command. A checkpoint qualifies only with 0 failed starts.
2.8 extends it to SMALL commands (20 % of the limits: +-vx 0.03, +-vy 0.02, +-wz 0.12), from rest
("s" prefix) and while walking ("w" prefix: a different command for 2 s + 0-0.7 s first): 144 starts.

Step 2.8 --terrain: protocol 2 on fixed terrain (type x difficulty grid TERRAIN_GRID, DR off), per cell:
success rate (no fall and episode mean window error < 50 %), fall rate, window error, CoT
(= energy / (m g path)). Terrain realisations depend only on the seed, so every controller walks the
same terrains.
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


SCENARIOS = {
    "nominal": None,
    "friction_low": {"friction": 0.5},
    "friction_high": {"friction": 1.25},
    "heavy": {"body_mass_scale": 1.15, "payload_kg": 0.3, "payload_offset": [0.02, 0.02],
              "link_mass_scale": [1.1] * 18},
    "weak_motors": {"effort_scale": 0.85, "kp_scale": 0.9},
    "latency_10ms": {"latency_substeps": 5},
    # step 2.9b (Gazebo-gap scenarios; also in the gate below when named explicitly)
    "latency_20ms": {"latency_substeps": 10},
    "target_offset": {"target_offset": [0.0, 0.06, 0.08] * 6},
    "soft_contact": {"contact_timeconst": 0.04, "contact_dampratio": 1.5},
    "sensor_noise": {"noise_scale": 1.0},
    "pushes": {"push": True},
    "all_extremes": {"friction": 0.5, "body_mass_scale": 1.15, "payload_kg": 0.3, "payload_offset": [0.02, 0.02],
                     "link_mass_scale": [1.1] * 18, "effort_scale": 0.85, "kp_scale": 0.9,
                     "damping_scale": 1.2, "joint_friction": 0.04, "latency_substeps": 5,
                     "noise_scale": 1.0, "push": True},
}


TERRAIN_GRID = {"rough": [0.005, 0.01, 0.02, 0.03, 0.04],         # peak-to-peak m
                "slopes": [5.0, 8.0, 10.0, 12.0, 15.0],         # deg
                "steps": [0.015, 0.02, 0.03, 0.04, 0.05]}        # m
TERRAIN_UNITS = {"rough": ("cm", 100), "slopes": ("deg", 1), "steps": ("cm", 100)}
SUCCESS_ERROR = 0.5


class Controller:
    """'tripod' or a policy checkpoint; .act(obs, env) -> action."""

    def __init__(self, spec):
        self.spec = spec
        self.obs_mode = "flat"
        if spec != "tripod":
            import torch
            from stable_baselines3 import PPO
            from train import vecnorm_for
            torch.set_num_threads(1)
            self.model = PPO.load(spec, device="cpu")
            with open(vecnorm_for(spec), "rb") as f:
                self.vn = pickle.load(f)
            import gymnasium as gym
            if isinstance(self.model.observation_space, gym.spaces.Dict):
                self.obs_mode = "asym"

    def reset(self, env):
        if self.spec == "tripod":
            from gurdy_core.gait import TripodGait
            self.gait = TripodGait(env.robot)

    def act(self, obs, env):
        if self.spec == "tripod":
            return (self.gait.step(env.command) - env.q_stance) / env.scale
        if isinstance(obs, dict):
            o = self.vn.normalize_obs({k: v[None].astype(np.float64) for k, v in obs.items()})
            o = {k: v.astype(np.float32) for k, v in o.items()}
        else:
            o = self.vn.normalize_obs(obs[None].astype(np.float64)).astype(np.float32)
        return self.model.policy.predict(o, deterministic=True)[0][0]


def _env(cfg, episode_s, fixed, obs_mode="flat", scenario=None, terrain=None):
    from envs.gurdy_env import GurdyEnv
    # Evaluation command distribution is FIXED (the original training distribution), whatever the run's
    # training config used: 2.7b's axis-aligned / rest-start sampling must not change the eval (bug found
    # 2026-11-15: dr_energy_ft was graded on its own harder command mix).
    env_cfg = {**cfg["env"], "episode_s": episode_s, "measure_energy": True, "obs_mode": obs_mode,
               "axis_aligned_prob": 0.0, "rest_start_prob": 0.0,
               # 2.8 additions are training-only too: standard = 15 % zero commands, no small-command
               # oversampling, no deadband, floor plane unless a terrain cell is evaluated
               "zero_command_prob": 0.15, "small_command_prob": 0.0, "command_deadband": False,
               "terrain": {"enabled": False}, "terrain_fixed": terrain,
               "domain_randomization": {**cfg["env"].get("domain_randomization", {}), "enabled": False},
               "dr_fixed": SCENARIOS.get(scenario) if scenario else None}
    if fixed:
        env_cfg["resample_command"] = False
    env = GurdyEnv(env_cfg, reward_config=cfg["reward"])
    env.penalty_scale = 1.0
    env.kappa = cfg["training"]["kappa_anneal"]["end"]
    return env


def run_protocol(args):
    """Phase 1 tracking protocol for one controller. Returns metrics + per-step trace."""
    spec, cfg, record = args
    from gurdy_core.experiments import analyze_tracking, cmd_at, quat_to_rpy, tracking_protocol
    changes, _, total = tracking_protocol()
    ctl = Controller(spec)
    env = _env(cfg, total + 1.0, fixed=True, obs_mode=ctl.obs_mode)
    obs, _ = env.reset(seed=0, options={"command": (0.0, 0.0, 0.0)})
    ctl.reset(env)
    rows, energy, path, prev_xy, frames = [], 0.0, 0.0, env.data.qpos[:2].copy(), []
    t = 0.0
    while t < total - 1e-9:
        env.command = np.array(cmd_at(changes, t), dtype=float)
        obs, _, term, _, info = env.step(ctl.act(obs, env))
        t = env.steps * env.policy_dt
        energy += info["energy_J"]
        xy = env.data.qpos[:2].copy()
        path += float(np.linalg.norm(xy - prev_xy))
        prev_xy = xy
        rows.append((t, *env.data.qpos[:7]))
        if record:
            frames.append(env.data.qpos.copy())
        if term:
            break
    a = np.array(rows)
    roll, pitch, yaw = quat_to_rpy(a[:, 4], a[:, 5], a[:, 6], a[:, 7])
    res = analyze_tracking(a[:, 0], a[:, 1], a[:, 2], yaw, a[:, 3], roll, pitch)
    m, g = env.robot.total_mass, env.robot.sim["gravity"]
    res.update({"controller": spec, "energy_J": energy, "path_m": path,
                "cost_of_transport": energy / (m * g * path) if path > 0 else float("nan"),
                "mean_power_W": energy / max(t, 1e-9)})
    return res, (np.array(frames) if record else None)


def run_episodes(args):
    """Training-distribution episodes (seeds) for one controller."""
    spec, cfg, seeds, scenario, terrain = (args + (None, None))[:5]
    ctl = Controller(spec)
    env = _env(cfg, cfg["env"]["episode_s"], fixed=False, obs_mode=ctl.obs_mode, scenario=scenario,
               terrain=terrain)
    lim = env.cmd_lim
    out = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        ctl.reset(env)
        vels, cmds, energy, done, term = [], [], 0.0, False, False
        path, prev_xy = 0.0, env.data.qpos[:2].copy()
        while not done:
            cmd = env.command.copy()
            obs, _, term, trunc, info = env.step(ctl.act(obs, env))
            vels.append([*info["body_vel"][:2], info["yaw_rate"]])
            cmds.append(cmd)
            energy += info["energy_J"]
            path += float(np.linalg.norm(env.data.qpos[:2] - prev_xy))
            prev_xy = env.data.qpos[:2].copy()
            done = term or trunc
        vels, cmds = np.array(vels), np.array(cmds)
        errs, wcmds = [], []
        w = env.cmd_steps
        for k in range(0, len(cmds) - w + 1, w):
            c = cmds[k]
            if not np.any(np.abs(c) > 1e-6) or not np.allclose(cmds[k:k + w], c):
                continue
            v = vels[k + int(1.0 / env.policy_dt):k + w].mean(axis=0)
            errs.append(float(np.linalg.norm((v - c) / lim) / np.linalg.norm(c / lim)))
            wcmds.append(c.tolist())
        out.append({"seed": seed, "fell": bool(term), "steps": len(cmds), "window_errors": errs,
                    "window_commands": wcmds,
                    "energy_J": energy, "path_m": path})
    return out


def _robot_mg():
    from gurdy_core.robot import load_robot
    r = load_robot()
    return r.total_mass * r.sim["gravity"]


ROBOT_MG = _robot_mg()


def summarize_episodes(ep):
    errs = np.concatenate([e["window_errors"] for e in ep]) if ep else np.array([np.nan])
    secs = sum(e["steps"] for e in ep) * 0.02
    per_ep = np.array([np.mean(e["window_errors"]) for e in ep if e["window_errors"]])
    return {"n": len(ep), "fall_rate": float(np.mean([e["fell"] for e in ep])),
            "window_error_mean": float(errs.mean()),
            # standard error across episodes (windows within an episode are correlated)
            "window_error_se": float(per_ep.std(ddof=1) / np.sqrt(len(per_ep))) if len(per_ep) > 1 else float("nan"),
            "window_error_median": float(np.median(errs)),
            "window_error_p90": float(np.percentile(errs, 90)),
            "windows_below_20pct": float(np.mean(errs < 0.2)), "n_windows": int(len(errs)),
            "success_rate": float(np.mean([not e["fell"] and (not e["window_errors"]
                                           or np.mean(e["window_errors"]) < SUCCESS_ERROR) for e in ep])),
            "cost_of_transport": float(sum(e["energy_J"] for e in ep)
                                       / (ROBOT_MG * max(sum(e.get("path_m", 0.0) for e in ep), 1e-9))),
            "mean_power_W": float(sum(e["energy_J"] for e in ep) / max(secs, 1e-9))}


def resolve(spec):
    """'run:checkpoint' -> (label, checkpoint path, run config)."""
    import yaml
    run, ck = spec.split(":")
    rdir = os.path.join(HERE, "runs", run)
    path = os.path.join(rdir, "checkpoints", "ppo_latest.zip" if ck == "latest" else f"ppo_{ck}_steps.zip")
    return f"{run}:{ck}", path, yaml.safe_load(open(os.path.join(rdir, "config.yaml")))


def main_scenarios(a):
    """Compare tripod + policies across DR scenarios (protocol 2) and the nominal tracking protocol."""
    pols = [resolve(p) for p in a.policies]
    ctrls = [("tripod", "tripod", pols[0][2])] + pols
    scen = list(SCENARIOS) if a.scenarios == ["all"] else a.scenarios
    out_dir = os.path.join(HERE, "runs", a.policies[-1].split(":")[0], "eval")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    with mp.get_context("forkserver").Pool(a.workers) as pool:
        prot = pool.map(run_protocol, [(path, cfg, False) for _, path, cfg in ctrls])
        jobs, keys = [], []
        for name, path, cfg in ctrls:
            for sc in scen:
                n = a.episodes if sc != "nominal" else max(a.episodes, 100)
                for ch in np.array_split(np.arange(n), max(1, a.workers // 2)):
                    jobs.append((path, cfg, [int(x) for x in ch], sc))
                    keys.append((name, sc))
        res = pool.map(run_episodes, jobs)
        starts = pool.map(run_start, start_gate_jobs([(path, cfg) for _, path, cfg in ctrls]))
    table = {}
    for (name, sc), lst in zip(keys, res):
        table.setdefault(name, {}).setdefault(sc, []).extend(lst)
    summary = {name: {"protocol": p, "scenarios": {sc: summarize_episodes(v) for sc, v in table[name].items()},
                      "start_gate": start_gate_summary(starts, path)}
               for (name, path, _), (p, _) in zip(ctrls, prot)}
    with open(os.path.join(out_dir, "eval_scenarios.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"evaluated {len(ctrls)} controllers x {len(scen)} scenarios in {time.time() - t0:.0f} s\n")
    print("Nominal tracking protocol (123 s):\n")
    seg_names = [s["segment"] for s in prot[0][0]["segments"]]
    print("| controller | " + " | ".join(seg_names) + " | falls | energy J | CoT | mean power W | start gate |")
    print("|---" * (len(seg_names) + 6) + "|")
    for name, s in summary.items():
        p, g = s["protocol"], s["start_gate"]
        print(f"| {name} | " + " | ".join(f"{x['main_axis_error_pct']:+.1f} %" for x in p["segments"])
              + f" | {int(p['fell'])} | {p['energy_J']:.0f} | {p['cost_of_transport']:.2f} | {p['mean_power_W']:.1f}"
              + f" | {'PASS' if g['pass'] else 'FAIL'} {g['failed_starts']}/{g['n_starts']} |")
    print("\nScenarios (fall rate / window error mean ± SE across episodes / mean power):\n")
    print("| scenario | " + " | ".join(summary) + " |\n|---" + "|---" * len(summary) + "|")
    for sc in scen:
        row = []
        for name in summary:
            e = summary[name]["scenarios"][sc]
            row.append(f"{100 * e['fall_rate']:.0f} % / {100 * e['window_error_mean']:.1f} ± {100 * e['window_error_se']:.1f} % / {e['mean_power_W']:.1f} W")
        print(f"| {sc} (n={summary[next(iter(summary))]['scenarios'][sc]['n']}) | " + " | ".join(row) + " |")


_LARGE = {"+vx": (0.10, 0, 0), "-vx": (-0.10, 0, 0), "+vy": (0, 0.08, 0), "-vy": (0, -0.08, 0),
          "+wz": (0, 0, 0.3), "-wz": (0, 0, -0.3)}
_SMALL = {"+vx": (0.03, 0, 0), "-vx": (-0.03, 0, 0), "+vy": (0, 0.02, 0), "-vy": (0, -0.02, 0),
          "+wz": (0, 0, 0.12), "-wz": (0, 0, -0.12)}
# name -> (command, command before the switch); "s" = small from rest, "w" = small while walking
def main_terrain(a):
    """Tripod + policies on every TERRAIN_GRID cell (protocol 2, DR off)."""
    pols = [resolve(p) for p in a.policies]
    ctrls = [("tripod", "tripod", pols[0][2])] + pols
    types = list(TERRAIN_GRID) if a.terrain == ["all"] else a.terrain
    out_dir = os.path.join(HERE, "runs", a.policies[-1].split(":")[0], "eval")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    jobs, keys = [], []
    for name, path, cfg in ctrls:
        for tt in types:
            for val in TERRAIN_GRID[tt]:
                for ch in np.array_split(np.arange(a.terrain_episodes), max(1, a.workers // 3)):
                    jobs.append((path, cfg, [int(x) for x in ch], None, {"type": tt, "value": val}))
                    keys.append((name, tt, val))
    with mp.get_context("forkserver").Pool(a.workers) as pool:
        res = pool.map(run_episodes, jobs)
    table = {}
    for (name, tt, val), lst in zip(keys, res):
        table.setdefault(name, {}).setdefault(tt, {}).setdefault(str(val), []).extend(lst)
    summary = {n: {tt: {v: summarize_episodes(ep) for v, ep in d.items()} for tt, d in t.items()}
               for n, t in table.items()}
    with open(os.path.join(out_dir, "eval_terrain.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"terrain eval: {len(ctrls)} controllers x {sum(len(TERRAIN_GRID[t]) for t in types)} cells x "
          f"{a.terrain_episodes} episodes in {time.time() - t0:.0f} s")
    print("cell = success % / fall % / window error mean % / CoT  (success: no fall and episode error < 50 %)\n")
    for tt in types:
        unit, k = TERRAIN_UNITS[tt]
        print(f"**{tt}**\n\n| {tt} ({unit}) | " + " | ".join(summary) + " |\n|---" + "|---" * len(summary) + "|")
        for val in TERRAIN_GRID[tt]:
            row = []
            for n in summary:
                e = summary[n][tt][str(val)]
                row.append(f"{100 * e['success_rate']:.0f} / {100 * e['fall_rate']:.0f} / "
                           f"{100 * e['window_error_mean']:.1f} / {e['cost_of_transport']:.1f}")
            print(f"| {val * k:g} | " + " | ".join(row) + " |")
        print()


START_CMDS = {**{k: (v, (0, 0, 0)) for k, v in _LARGE.items()},
              **{"s" + k: (v, (0, 0, 0)) for k, v in _SMALL.items()},
              **{"w" + k: (v, (0, 0, 0.3) if "vx" in k else (0.10, 0, 0)) for k, v in _SMALL.items()}}
START_DELAYS = np.arange(0.0, 0.71, 0.1)


def run_start(args):
    """Start-from-rest test for one controller and one pure command: speed ratio per start phase."""
    spec, cfg, name = args
    cmd, before = (np.array(c, float) for c in START_CMDS[name])
    axis = int(np.argmax(np.abs(cmd)))
    ctl = Controller(spec)
    ratios = []
    for d in START_DELAYS:
        env = _env(cfg, 12.0, fixed=True, obs_mode=ctl.obs_mode)
        obs, _ = env.reset(seed=0, options={"command": (0, 0, 0)})
        ctl.reset(env)
        vel = []
        while env.steps * env.policy_dt < 2.0 + d + 8.0 - 1e-9:
            t = env.steps * env.policy_dt
            env.command = cmd if t >= 2.0 + d - 1e-9 else before
            obs, _, term, _, info = env.step(ctl.act(obs, env))
            if t >= 2.0 + d + 3.0:
                vel.append([*info["body_vel"][:2], info["yaw_rate"]][axis])
            if term:
                break
        ratios.append(float(np.mean(vel) / cmd[axis]) if vel else 0.0)
    return spec, name, ratios


def start_gate_jobs(specs_cfgs):
    return [(path, cfg, k) for path, cfg in specs_cfgs for k in START_CMDS]


def start_gate_summary(results, path):
    per = {n: r for s, n, r in results if s == path}
    failed = {n: int(sum(v < 0.2 for v in r)) for n, r in per.items()}
    total = sum(failed.values())
    return {"failed_starts": total, "n_starts": len(START_CMDS) * len(START_DELAYS),
            "pass": total == 0,
            "failed_large": sum(v for n, v in failed.items() if n[0] in "+-"),
            "failed_small_rest": sum(v for n, v in failed.items() if n[0] == "s"),
            "failed_small_walking": sum(v for n, v in failed.items() if n[0] == "w"), "per_command_failed": failed,
            "per_command_speed_ratio": {n: float(np.mean(r)) for n, r in per.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", help="run:checkpoint ... (scenario mode)")
    ap.add_argument("--scenarios", nargs="+", default=["all"])
    ap.add_argument("--run")
    ap.add_argument("--checkpoints", nargs="+", default=["latest"])
    ap.add_argument("--config", default=None, help="defaults to the run's config.yaml")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--no-tripod", action="store_true")
    ap.add_argument("--terrain", nargs="+", help="2.8: terrain grid instead of DR scenarios: all | rough slopes steps")
    ap.add_argument("--terrain-episodes", type=int, default=30)
    a = ap.parse_args()
    if a.policies and a.terrain:
        return main_terrain(a)
    if a.policies:
        return main_scenarios(a)

    import yaml
    run = os.path.join(HERE, "runs", a.run)
    cfg = yaml.safe_load(open(a.config or os.path.join(run, "config.yaml")))
    out_dir = os.path.join(run, "eval")
    os.makedirs(out_dir, exist_ok=True)
    specs = [] if a.no_tripod else ["tripod"]
    for c in a.checkpoints:
        specs.append(os.path.join(run, "checkpoints", "ppo_latest.zip" if c == "latest" else f"ppo_{c}_steps.zip"))

    ctx = mp.get_context("forkserver")
    results = {}
    t0 = time.time()
    with ctx.Pool(a.workers) as pool:
        prot = pool.map(run_protocol, [(s, cfg, False) for s in specs])
        chunks = np.array_split(np.arange(a.episodes), a.workers)
        jobs = [(s, cfg, [int(x) for x in ch]) for s in specs for ch in chunks if len(ch)]
        eps = pool.map(run_episodes, jobs)
        starts = pool.map(run_start, start_gate_jobs([(s, cfg) for s in specs]))
    for s, (p, _) in zip(specs, prot):
        ep = [e for (spec, _, _), lst in zip(jobs, eps) if spec == s for e in lst]
        errs = np.concatenate([e["window_errors"] for e in ep])
        name = "tripod" if s == "tripod" else os.path.basename(s).replace(".zip", "")
        results[name] = {
            "protocol": p,
            "episodes": {"n": len(ep), "fall_rate": float(np.mean([e["fell"] for e in ep])),
                         "window_error_mean": float(errs.mean()), "window_error_median": float(np.median(errs)),
                         "window_error_p90": float(np.percentile(errs, 90)),
                         "windows_below_20pct": float(np.mean(errs < 0.2)), "n_windows": int(len(errs))},
            "start_gate": start_gate_summary(starts, s),
        }
    with open(os.path.join(out_dir, "eval.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"evaluated {len(specs)} controllers in {time.time() - t0:.0f} s\n")
    seg_names = [s["segment"] for s in next(iter(results.values()))["protocol"]["segments"]]
    print("| controller | " + " | ".join(seg_names) + " | falls (protocol) | energy J | CoT | 100-ep fall rate | window err mean / median / p90 | windows < 20 % | start gate |")
    print("|---" * (len(seg_names) + 8) + "|")
    for name, r in results.items():
        p, e = r["protocol"], r["episodes"]
        segs = " | ".join(f"{s['main_axis_error_pct']:+.1f} %" for s in p["segments"])
        print(f"| {name} | {segs} | {int(p['fell'])} | {p['energy_J']:.0f} | {p['cost_of_transport']:.2f} | "
              f"{100 * e['fall_rate']:.0f} % | {100 * e['window_error_mean']:.1f} / {100 * e['window_error_median']:.1f} / "
              f"{100 * e['window_error_p90']:.1f} % | {100 * e['windows_below_20pct']:.0f} % | "
              f"{'PASS' if r['start_gate']['pass'] else 'FAIL'} {r['start_gate']['failed_starts']}/{r['start_gate']['n_starts']} |")


if __name__ == "__main__":
    main()
