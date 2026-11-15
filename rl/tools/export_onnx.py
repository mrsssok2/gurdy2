"""Export the actor of a trained PPO checkpoint to ONNX (step 2.9).

    python3 rl/tools/export_onnx.py --checkpoint rl/runs/dr_energy_ft/checkpoints/ppo_6000000_steps.zip \
        --out models/dr_energy_ft_6M [--parity-samples 300]

Writes <out>.onnx (obs[1,195] float32 -> mean action[1,18]; VecNormalize mean/std/clip baked in as graph
constants, deterministic = no sampling, no action clip: ActionMapper clips) and <out>.json (metadata:
checkpoint, obs layout, rates, action scale/filter, deadband, normalisation) and, with --parity-samples,
<out>_parity.npz = recorded observations + the torch actions for the ONNX parity test (no torch needed).
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gurdy_core.rl_command import ANGULAR_DEADBAND, LINEAR_DEADBAND  # noqa: E402
from gurdy_core.rl_obs import FRAME_LAYOUT, HISTORY, OBS_DIM  # noqa: E402


class Actor(nn.Module):
    """obs -> normalise -> actor MLP -> mean action (what PPO.predict(deterministic=True) uses, unclipped)."""

    def __init__(self, policy, mean, var, eps, clip):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.sqrt(torch.as_tensor(var, dtype=torch.float32) + float(eps)))
        self.clip = float(clip)
        self.net, self.out = policy.mlp_extractor.policy_net, policy.action_net

    def forward(self, obs):
        x = torch.clamp((obs - self.mean) / self.std, -self.clip, self.clip)
        return self.out(self.net(x))


def build_actor(checkpoint):
    from stable_baselines3 import PPO
    from train import vecnorm_for
    model = PPO.load(checkpoint, device="cpu")
    with open(vecnorm_for(checkpoint), "rb") as f:
        vn = pickle.load(f)
    rms = vn.obs_rms["policy"] if isinstance(vn.obs_rms, dict) else vn.obs_rms
    actor = Actor(model.policy, rms.mean, rms.var, vn.epsilon, vn.clip_obs).eval()
    return model, vn, actor


def record_observations(checkpoint, cfg, n, seed=0):
    """Closed-loop rollout in the MuJoCo env (DR on, resampled commands) -> n actor observations."""
    from envs.gurdy_env import GurdyEnv
    from eval import Controller
    ctl = Controller(checkpoint)
    env = GurdyEnv({**cfg["env"], "obs_mode": ctl.obs_mode, "episode_s": 30.0})
    obs, _ = env.reset(seed=seed)
    out = []
    while len(out) < n:
        out.append((obs["policy"] if isinstance(obs, dict) else obs).copy())
        obs, _, term, trunc, _ = env.step(ctl.act(obs, env))
        if term or trunc:
            obs, _ = env.reset()
    return np.array(out, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True, help="path prefix; writes .onnx/.json[/_parity.npz]")
    ap.add_argument("--parity-samples", type=int, default=300)
    ap.add_argument("--config", default=None, help="run config.yaml (default: next to the checkpoint's run)")
    args = ap.parse_args()
    torch.set_num_threads(1)
    import yaml
    cfg_path = args.config or os.path.join(os.path.dirname(os.path.dirname(args.checkpoint)), "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    from envs.gurdy_env import DEFAULTS
    env_cfg = {**DEFAULTS, **cfg["env"]}
    model, vn, actor = build_actor(args.checkpoint)
    assert model.observation_space["policy"].shape == (OBS_DIM,) if hasattr(model.observation_space, "spaces") \
        else model.observation_space.shape == (OBS_DIM,)

    from gurdy_core.robot import load_gait
    from gurdy_core.robot import load_robot
    robot = load_robot()
    rate = float(robot.control["policy_rate_hz"])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    onnx_path = args.out + ".onnx"
    torch.onnx.export(actor, torch.zeros(1, OBS_DIM), onnx_path, input_names=["obs"],
                      output_names=["action_mean"], opset_version=17, dynamo=False)
    sha = hashlib.sha256(open(onnx_path, "rb").read()).hexdigest()
    meta = {
        "format_version": 1,
        "exported": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": os.path.relpath(args.checkpoint),
        "run_config": os.path.relpath(cfg_path),
        "onnx_sha256": sha,
        "input": {"name": "obs", "shape": [1, OBS_DIM], "dtype": "float32"},
        "output": {"name": "action_mean", "shape": [1, 18], "dtype": "float32",
                   "note": "deterministic mean, unclipped; ActionMapper clips to [-1, 1]"},
        "obs": {"history": HISTORY, "frame_layout": [[n, k] for n, k in FRAME_LAYOUT],
                "oldest_first": True, "source": "gurdy_core.rl_obs",
                "normalisation": "VecNormalize mean/std/clip baked into the graph (raw obs in)",
                "obs_clip": float(vn.clip_obs), "obs_epsilon": float(vn.epsilon)},
        "policy_rate_hz": rate,
        "clock_period_s": float(load_gait()["tripod"]["cycle_time"]),
        "action_scale": env_cfg["action_scale"],
        "action_filter_alpha": env_cfg["action_filter_alpha"],
        "action_mapping": "q_ref = clip(q_stance + scale * a_filt, limits); a_filt = alpha*clip(a,-1,1)"
                          " + (1-alpha)*a_filt_prev (gurdy_core.rl_action)",
        "command_deadband": {"linear_m_s": LINEAR_DEADBAND, "angular_rad_s": ANGULAR_DEADBAND,
                             "source": "gurdy_core.rl_command"},
        "trained_with_deadband": bool(env_cfg.get("command_deadband", False)),
    }
    if args.parity_samples:
        obs = record_observations(args.checkpoint, cfg, args.parity_samples)
        with torch.no_grad():
            act = actor(torch.as_tensor(obs)).numpy()
        np.savez_compressed(args.out + "_parity.npz", obs=obs, action=act)
        meta["parity_fixture"] = os.path.basename(args.out) + "_parity.npz"
    with open(args.out + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {onnx_path} ({os.path.getsize(onnx_path) / 1024:.0f} KiB), {args.out}.json")


if __name__ == "__main__":
    main()
