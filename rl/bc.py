"""Behaviour-cloning warm start from the tripod gait (plan C2). Results from a policy initialised
this way are labelled "BC-initialised from the tripod gait".

    python3 rl/bc.py --config rl/configs/flat_C2_bc.yaml --name c2_bc [--samples 1000000]
    python3 rl/train.py --config rl/configs/flat_C2_bc.yaml --name c2_bc \
        --init-from rl/runs/c2_bc/checkpoints/ppo_bc.zip

1. Demonstrations: the tripod gait (gurdy_core.gait) follows the training command distribution in
   the training env (same action filter). The EXECUTED action gets small Gaussian noise
   (--exec-noise) but the LABEL is the clean tripod action, so the clone also sees (and learns to
   recover from) slightly off-nominal states. Collected in parallel worker processes.
2. VecNormalize observation statistics are initialised from the demonstrations.
3. The actor mean is regressed onto the tripod actions (MSE); action std set to --bc-std.
4. The cloned policy is evaluated against the tripod (per-step reward, tracking, falls) and saved
   as checkpoints/ppo_bc.zip + ppo_vecnormalize_bc.pkl (a normal checkpoint for --init-from).
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _collect(args):
    """Worker: n samples of (obs, clean tripod action). No torch imported here."""
    env_cfg, reward_cfg, n, seed, exec_noise = args
    from envs.gurdy_env import GurdyEnv
    from gurdy_core.gait import TripodGait
    env = GurdyEnv(env_cfg, reward_config=reward_cfg)
    rng = np.random.default_rng(seed)
    obs_buf = np.empty((n, env.observation_space.shape[0]), np.float32)
    act_buf = np.empty((n, env.action_space.shape[0]), np.float32)
    obs, _ = env.reset(seed=seed)
    gait = TripodGait(env.robot)
    falls = 0
    for i in range(n):
        a = (gait.step(env.command) - env.q_stance) / env.scale
        obs_buf[i], act_buf[i] = obs, a
        obs, _, term, trunc, _ = env.step(a + rng.normal(0.0, exec_noise, a.shape))
        if term or trunc:
            falls += int(term)
            obs, _ = env.reset()
            gait = TripodGait(env.robot)
    return obs_buf, act_buf, falls


def evaluate(model, venv_norm, cfg, episodes=4, policy="bc"):
    """Per-step reward/tracking/falls of the deterministic BC policy (or the tripod) on the
    training command distribution."""
    import torch
    from envs.gurdy_env import GurdyEnv
    from gurdy_core.gait import TripodGait
    tot = {"reward": 0.0, "track": 0.0, "n": 0, "falls": 0}
    for ep in range(episodes):
        env = GurdyEnv(cfg["env"], reward_config=cfg["reward"])
        env.penalty_scale = 1.0
        obs, _ = env.reset(seed=1000 + ep)
        gait = TripodGait(env.robot)
        done = False
        while not done:
            if policy == "bc":
                o = venv_norm.normalize_obs(obs[None])
                with torch.no_grad():
                    a = model.policy.predict(o, deterministic=True)[0][0]
            else:
                a = (gait.step(env.command) - env.q_stance) / env.scale
            obs, r, term, trunc, info = env.step(a)
            tot["reward"] += r
            tot["track"] += info["reward_terms"]["track"]
            tot["n"] += 1
            done = term or trunc
        tot["falls"] += int(term)
    return {"reward_per_step": tot["reward"] / tot["n"], "track_per_step": tot["track"] / tot["n"],
            "falls": tot["falls"], "episodes": episodes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "flat_C2_bc.yaml"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--samples", type=int, default=1_000_000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--exec-noise", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bc-std", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from train import load_cfg, make_model, make_venv
    cfg = load_cfg(a.config)
    run = os.path.join(HERE, "runs", a.name)
    ckpt_dir = os.path.join(run, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    t0 = time.time()
    per = a.samples // a.workers
    jobs = [(cfg["env"], cfg["reward"], per, a.seed + 17 * k, a.exec_noise) for k in range(a.workers)]
    with mp.get_context("forkserver").Pool(a.workers) as pool:
        parts = pool.map(_collect, jobs)
    obs = np.concatenate([p[0] for p in parts])
    act = np.concatenate([p[1] for p in parts])
    demo_falls = sum(p[2] for p in parts)
    t_collect = time.time() - t0
    print(f"[bc] {len(obs)} demonstration samples in {t_collect:.0f} s ({demo_falls} falls), "
          f"|a| max {np.abs(act).max():.2f}", flush=True)

    import torch
    from stable_baselines3.common.vec_env import VecNormalize
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    venv = VecNormalize(make_venv(cfg, a.seed), norm_obs=True, norm_reward=cfg["ppo"]["normalize_reward"],
                        clip_obs=10.0, gamma=cfg["ppo"]["gamma"])
    venv.obs_rms.mean = obs.mean(axis=0).astype(np.float64)
    venv.obs_rms.var = obs.var(axis=0).astype(np.float64) + 1e-8
    venv.obs_rms.count = float(len(obs))
    model = make_model(cfg, venv, run, a.seed)
    pol = model.policy

    x = torch.as_tensor(venv.normalize_obs(obs), dtype=torch.float32)
    y = torch.as_tensor(act, dtype=torch.float32)
    perm = torch.randperm(len(x))
    n_val = len(x) // 10
    val, tr = perm[:n_val], perm[n_val:]
    params = list(pol.mlp_extractor.policy_net.parameters()) + list(pol.action_net.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)

    def mean_action(xb):
        return pol.action_net(pol.mlp_extractor.forward_actor(pol.extract_features(xb)))

    t1 = time.time()
    hist = []
    for ep in range(a.epochs):
        idx = tr[torch.randperm(len(tr))]
        for i in range(0, len(idx), 4096):
            b = idx[i:i + 4096]
            loss = torch.nn.functional.mse_loss(mean_action(x[b]), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            v = torch.nn.functional.mse_loss(mean_action(x[val]), y[val]).item()
        hist.append(v)
        print(f"[bc] epoch {ep + 1}/{a.epochs} val MSE {v:.5f} (label variance {y.var().item():.4f})", flush=True)
    with torch.no_grad():
        pol.log_std.fill_(float(np.log(a.bc_std)))
    t_train = time.time() - t1

    venv.training = False
    ev_bc = evaluate(model, venv, cfg, policy="bc")
    ev_tripod = evaluate(model, venv, cfg, policy="tripod")
    print(f"[bc] BC policy: {ev_bc} | tripod: {ev_tripod}", flush=True)
    model.save(os.path.join(ckpt_dir, "ppo_bc.zip"))
    venv.save(os.path.join(ckpt_dir, "ppo_vecnormalize_bc.pkl"))
    venv.close()
    report = {"samples": len(obs), "demo_falls": demo_falls, "collect_s": t_collect, "train_s": t_train,
              "val_mse": hist, "label_var": float(y.var()), "bc_std": a.bc_std, "exec_noise": a.exec_noise,
              "eval_bc": ev_bc, "eval_tripod": ev_tripod}
    with open(os.path.join(run, "bc_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[bc] saved {ckpt_dir}/ppo_bc.zip", flush=True)


if __name__ == "__main__":
    main()
