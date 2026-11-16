"""Train the Gurdy locomotion policy with SB3 PPO (CPU, BatchedSubprocVecEnv).

    python3 rl/train.py --name flat_v1                       # config rl/configs/flat.yaml
    python3 rl/train.py --name smoke --timesteps 2000000
    python3 rl/train.py --name flat_v1 --resume rl/runs/flat_v1/checkpoints/ppo_10000000_steps.zip
    python3 rl/train.py --name flat_v1 --resume latest       # newest checkpoint of that run

Run directory rl/runs/<name>/: config.yaml (+ git hash), tb/ (TensorBoard), checkpoints/
(ppo_<N>_steps.zip + ppo_vecnormalize_<N>_steps.pkl every training.checkpoint_every steps, and
ppo_latest.zip / ppo_vecnormalize_latest.pkl on exit or SIGTERM, e.g. `docker stop`).
"""
import argparse
import functools
import glob
import os
import re
import signal
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def make_env(env_cfg, reward_cfg):
    from envs.gurdy_env import GurdyEnv          # imported in the worker; no torch there
    return GurdyEnv(env_cfg, reward_config=reward_cfg)


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_cfg(path: str) -> dict:
    """YAML config with optional `_base: other.yaml` (relative path), deep-merged."""
    cfg = yaml.safe_load(open(path)) or {}
    base = cfg.pop("_base", None)
    if base:
        cfg = deep_merge(load_cfg(os.path.join(os.path.dirname(path), base)), cfg)
    return cfg


def git_hash():
    """Commit of the repo, read from .git directly (the rl image has no git binary)."""
    g = os.path.join(HERE, "..", ".git")
    try:
        head = open(os.path.join(g, "HEAD")).read().strip()
        if head.startswith("ref: "):
            ref = head[5:]
            path = os.path.join(g, ref)
            if os.path.exists(path):
                return open(path).read().strip()[:7]
            for line in open(os.path.join(g, "packed-refs")):
                if line.strip().endswith(ref):
                    return line.split()[0][:7]
        return head[:7]
    except OSError:
        return "unknown"


def schedule(step, a):
    """Linear ramp from a['start'] to a['end'] between from_step and to_step."""
    if step <= a["from_step"]:
        return a["start"]
    if step >= a["to_step"]:
        return a["end"]
    f = (step - a["from_step"]) / (a["to_step"] - a["from_step"])
    return a["start"] + f * (a["end"] - a["start"])


def tree_rss_mib():
    import psutil
    p = psutil.Process()
    return sum(c.memory_info().rss for c in [p] + p.children(recursive=True)) / 2**20


def find_latest(ckpt_dir):
    if os.path.exists(os.path.join(ckpt_dir, "ppo_latest.zip")):
        cands = [os.path.join(ckpt_dir, "ppo_latest.zip")]
    else:
        cands = []
    numbered = sorted(glob.glob(os.path.join(ckpt_dir, "ppo_*_steps.zip")),
                      key=lambda p: int(re.search(r"_(\d+)_steps", p).group(1)))
    cands += numbered[-1:]
    if not cands:
        raise FileNotFoundError(f"no checkpoint in {ckpt_dir}")
    return max(cands, key=os.path.getmtime)


def vecnorm_for(ckpt):
    """ppo_<tag>.zip -> ppo_vecnormalize_<tag>.pkl (tag: <N>_steps, latest, bc)."""
    tag = re.search(r"ppo_(.+)\.zip$", os.path.basename(ckpt)).group(1)
    return os.path.join(os.path.dirname(ckpt), f"ppo_vecnormalize_{tag}.pkl")


def make_model(cfg, venv, run, seed):
    from stable_baselines3 import PPO
    pc = cfg["ppo"]
    policy = "MlpPolicy"
    if pc.get("policy") == "asymmetric":
        from policies import AsymmetricActorCriticPolicy
        policy = AsymmetricActorCriticPolicy
    return PPO(policy, venv, n_steps=pc["n_steps"], batch_size=pc["batch_size"],
               n_epochs=pc["n_epochs"], learning_rate=pc["learning_rate"], gamma=pc["gamma"],
               gae_lambda=pc["gae_lambda"], clip_range=pc["clip_range"], ent_coef=pc["ent_coef"],
               vf_coef=pc["vf_coef"], max_grad_norm=pc["max_grad_norm"], target_kl=pc["target_kl"],
               use_sde=pc.get("use_sde", False), sde_sample_freq=pc.get("sde_sample_freq", -1),
               policy_kwargs=dict(net_arch=dict(pi=pc["net_arch"], vf=pc["net_arch"]),
                                  log_std_init=pc["log_std_init"]),
               tensorboard_log=os.path.join(run, "tb"), seed=seed, device="cpu", verbose=1)


def make_venv(cfg, seed):
    from vec_env import BatchedSubprocVecEnv
    pc = cfg["ppo"]
    venv = BatchedSubprocVecEnv(functools.partial(make_env, cfg["env"], cfg["reward"]),
                                pc["n_workers"], pc["envs_per_worker"])
    venv.seed(seed)
    return venv


def actor_parameters(policy):
    """Parameters that define the action distribution (frozen during critic warm-up)."""
    params = list(policy.action_net.parameters()) + [policy.log_std]
    params += list(policy.mlp_extractor.policy_net.parameters())
    return params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "flat.yaml"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--timesteps", type=int, help="total timesteps (overrides config)")
    ap.add_argument("--resume", help="checkpoint .zip, or 'latest'")
    ap.add_argument("--init-from", help="start a NEW run from these weights (e.g. BC checkpoint ppo_bc.zip)")
    ap.add_argument("--transfer-from", help="start a NEW asymmetric run from a FLAT checkpoint "
                                            "(actor + critic + VecNormalize['policy'] copied)")
    ap.add_argument("--extend-from", help="start a NEW asymmetric run from an asymmetric checkpoint whose critic "
                                          "has fewer privileged inputs (Experiment B: +8 failure values)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import VecNormalize
    cfg = load_cfg(a.config)
    pc, tc = cfg["ppo"], cfg["training"]
    total = a.timesteps or tc["total_timesteps"]
    run = os.path.join(HERE, "runs", a.name)
    ckpt_dir = os.path.join(run, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.set_num_threads(pc["torch_threads"])

    git = git_hash()
    with open(os.path.join(run, "config.yaml" if not a.resume else f"config_resume_{int(time.time())}.yaml"), "w") as f:
        yaml.safe_dump({**cfg, "_run": {"git": git, "total_timesteps": total, "seed": a.seed,
                                        "resume": a.resume, "started": time.strftime("%Y-%m-%d %H:%M:%S")}}, f)

    venv = make_venv(cfg, a.seed)
    n_envs = venv.num_envs
    warmup = 0

    if a.transfer_from:
        import pickle
        from policies import transfer_from_flat
        flat = PPO.load(a.transfer_from, device="cpu")
        with open(vecnorm_for(a.transfer_from), "rb") as f:
            flat_vn = pickle.load(f)
        venv = VecNormalize(venv, norm_obs=pc["normalize_obs"], norm_reward=pc["normalize_reward"],
                            clip_obs=10.0, gamma=pc["gamma"])
        venv.obs_rms["policy"] = flat_vn.obs_rms          # actor sees identically normalised inputs
        venv.ret_rms = flat_vn.ret_rms
        model = make_model(cfg, venv, run, a.seed)
        transfer_from_flat(model.policy, flat.policy)
        warmup = int(tc.get("critic_warmup_steps", 0))
        print(f"[train] new asymmetric run transferred from {a.transfer_from}; critic warm-up {warmup} steps",
              flush=True)
    elif a.extend_from:
        import pickle
        from policies import extend_privileged
        old = PPO.load(a.extend_from, device="cpu")
        with open(vecnorm_for(a.extend_from), "rb") as f:
            old_vn = pickle.load(f)
        venv = VecNormalize(venv, norm_obs=pc["normalize_obs"], norm_reward=pc["normalize_reward"],
                            clip_obs=10.0, gamma=pc["gamma"])
        venv.obs_rms["policy"] = old_vn.obs_rms["policy"]          # actor inputs normalised identically
        n_old = old_vn.obs_rms["privileged"].mean.shape[0]
        for k in ("mean", "var"):                                  # old privileged stats kept, new ones start at 0 / 1
            getattr(venv.obs_rms["privileged"], k)[:n_old] = getattr(old_vn.obs_rms["privileged"], k)
        venv.obs_rms["privileged"].count = old_vn.obs_rms["privileged"].count
        venv.ret_rms = old_vn.ret_rms
        model = make_model(cfg, venv, run, a.seed)
        extend_privileged(model.policy, old.policy)
        warmup = int(tc.get("critic_warmup_steps", 0))
        print(f"[train] new run extended from {a.extend_from} (privileged {n_old} -> "
              f"{venv.obs_rms['privileged'].mean.shape[0]}); critic warm-up {warmup} steps", flush=True)
    elif a.init_from:
        venv = VecNormalize.load(vecnorm_for(a.init_from), venv)
        venv.training = True
        model = PPO.load(a.init_from, env=venv, device="cpu", tensorboard_log=os.path.join(run, "tb"))
        model.num_timesteps = 0
        warmup = int(tc.get("critic_warmup_steps", 0))
        print(f"[train] new run initialised from {a.init_from}; critic warm-up {warmup} steps", flush=True)
    elif a.resume:
        ckpt = find_latest(ckpt_dir) if a.resume == "latest" else a.resume
        venv = VecNormalize.load(vecnorm_for(ckpt), venv)
        venv.training = True
        model = PPO.load(ckpt, env=venv, device="cpu", tensorboard_log=os.path.join(run, "tb"))
        print(f"[train] resumed {ckpt} at {model.num_timesteps} steps", flush=True)
    else:
        venv = VecNormalize(venv, norm_obs=pc["normalize_obs"], norm_reward=pc["normalize_reward"],
                            clip_obs=10.0, gamma=pc["gamma"])
        model = make_model(cfg, venv, run, a.seed)

    class Metrics(BaseCallback):
        """Per-rollout means of every reward term, fall rate, tracking error, kappa, std, RAM."""

        def _on_rollout_start(self):
            self.kappa = schedule(self.num_timesteps, tc["kappa_anneal"])
            self.pscale = schedule(self.num_timesteps, tc["penalty_curriculum"])
            self.training_env.set_attr("kappa", self.kappa)
            self.training_env.set_attr("penalty_scale", self.pscale)
            self.extra = {}
            for attr, sched in tc.get("schedules", {}).items():      # e.g. eff_progress, dr_scale
                self.extra[attr] = schedule(self.num_timesteps, sched)
                self.training_env.set_attr(attr, self.extra[attr])
            self.sums, self.n, self.falls, self.ends, self.clipped = {}, 0, 0, 0, 0
            self.track_err, self.t0 = [], time.time()
            self.tlev, self.tends = {}, {}                           # 2.8 terrain: level sums, (ends, falls)

        def _on_step(self):
            for info, done in zip(self.locals["infos"], self.locals["dones"]):
                for key, v in info.get("reward_terms", {}).items():
                    self.sums[key] = self.sums.get(key, 0.0) + v
                self.n += 1
                self.clipped += int(info.get("reward_clipped", False))
                cmd = info.get("command")
                if cmd is not None and np.any(np.abs(cmd) > 1e-6):
                    tv = info["track_vel"]
                    self.track_err.append(np.linalg.norm(tv - cmd) / np.linalg.norm(cmd))
                tt = info.get("terrain")
                if tt is not None:
                    s_, n_ = self.tlev.get(tt, (0.0, 0))
                    self.tlev[tt] = (s_ + info["terrain_level"], n_ + 1)
                if done:
                    self.ends += 1
                    fell = int(not info.get("TimeLimit.truncated", False))
                    self.falls += fell
                    if tt is not None:
                        e_, f_ = self.tends.get(tt, (0, 0))
                        self.tends[tt] = (e_ + 1, f_ + fell)
            return True

        def _on_rollout_end(self):
            lg = self.logger
            for key, v in self.sums.items():
                lg.record(f"reward_terms/{key}", v / max(self.n, 1))
            lg.record("gurdy/fall_rate", self.falls / max(self.ends, 1))
            lg.record("gurdy/episodes_ended", self.ends)
            lg.record("gurdy/track_rel_error", float(np.mean(self.track_err)) if self.track_err else 0.0)
            lg.record("gurdy/kappa", self.kappa)
            lg.record("gurdy/penalty_scale", self.pscale)
            for attr, v in self.extra.items():
                lg.record(f"gurdy/{attr}", v)
            for tt, (s_, n_) in self.tlev.items():
                lg.record(f"terrain/level_{tt}", s_ / max(n_, 1))
                lg.record(f"terrain/share_{tt}", n_ / max(self.n, 1))
            for tt, (e_, f_) in self.tends.items():
                lg.record(f"terrain/fall_rate_{tt}", f_ / max(e_, 1))
            lg.record("gurdy/reward_clipped_frac", self.clipped / max(self.n, 1))
            lg.record("gurdy/action_std", effective_std(self.model))
            lg.record("gurdy/ram_mib", tree_rss_mib())
            lg.record("gurdy/collect_samples_per_s", self.n / max(time.time() - self.t0, 1e-6))

    def effective_std(m, n=1024):
        """Mean action std of the policy distribution on recent rollout observations (works for
        plain Gaussian and gSDE, where log_std is a features x actions matrix)."""
        buf = m.rollout_buffer.observations
        if isinstance(buf, dict):
            flat = {k: v.reshape(-1, v.shape[-1]) for k, v in buf.items()}
            size = len(next(iter(flat.values())))
            idx = np.random.default_rng(0).choice(size, size=min(n, size), replace=False)
            obs = {k: torch.as_tensor(v[idx], dtype=torch.float32) for k, v in flat.items()}
        else:
            flat = buf.reshape(-1, buf.shape[-1])
            idx = np.random.default_rng(0).choice(len(flat), size=min(n, len(flat)), replace=False)
            obs = torch.as_tensor(flat[idx], dtype=torch.float32)
        with torch.no_grad():
            return float(m.policy.get_distribution(obs).distribution.stddev.mean())

    ckpt_cb = CheckpointCallback(save_freq=max(1, tc["checkpoint_every"] // n_envs), save_path=ckpt_dir,
                                 name_prefix="ppo", save_vecnormalize=True)

    def save_latest(*_):
        model.save(os.path.join(ckpt_dir, "ppo_latest.zip"))
        venv.save(os.path.join(ckpt_dir, "ppo_vecnormalize_latest.pkl"))
        print(f"[train] saved ppo_latest at {model.num_timesteps} steps", flush=True)

    def on_sigterm(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_sigterm)

    remaining = max(0, total - model.num_timesteps)
    print(f"[train] run {run} | {n_envs} envs | {remaining} steps to go (target {total}) | git {git}", flush=True)
    t = time.time()
    try:
        cbs = CallbackList([Metrics(), ckpt_cb])
        if warmup:
            # Critic warm-up: actor frozen (no gradient), so a random critic can't wreck the BC gait.
            for p in actor_parameters(model.policy):
                p.requires_grad_(False)
            model.learn(total_timesteps=warmup, callback=cbs, reset_num_timesteps=True,
                        tb_log_name="PPO", progress_bar=False)
            for p in actor_parameters(model.policy):
                p.requires_grad_(True)
            print(f"[train] critic warm-up done at {model.num_timesteps} steps; actor unfrozen", flush=True)
            remaining = max(0, total - model.num_timesteps)
        model.learn(total_timesteps=remaining, callback=cbs,
                    reset_num_timesteps=not (a.resume or warmup), tb_log_name="PPO", progress_bar=False)
    except KeyboardInterrupt:
        print("[train] interrupted", flush=True)
    finally:
        save_latest()
        print(f"[train] wall time {(time.time() - t) / 3600:.2f} h, steps {model.num_timesteps}", flush=True)
        venv.close()


if __name__ == "__main__":
    main()
