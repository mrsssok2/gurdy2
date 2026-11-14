# Phase 2 step 2.1 — RL throughput and RAM benchmark

Measured 2026-11-14 in the `rl` container (`gurdy2/rl:cpu`: Ubuntu 24.04, MuJoCo 3.3.0, torch 2.5.1
CPU, SB3 2.4.1, Gymnasium 1.0). Host: i5-13420H (4 P-cores + 4 E-cores, 12 threads), plugged in.
Host benchmark (single-thread MuJoCo stepping RTF) during the run: 65–70 (it reaches ~117 when the
CPU clocks high), so these are conservative. Environment: step-2.1 skeleton (50 Hz policy, 10 × 2 ms
physics steps per action, 195-dim observation, placeholder reward). Raw JSON: `phase2_bench.json`.
Reproduce: `docker compose run --rm rl python3 rl/bench.py`.

## Single environment
2,179 env steps/s (physics alone 3,871; Python overhead 44 %). Profiling showed the fall check
(Python loop over contacts) took 31 % of the step; it is now vectorised (overhead was 73 % before).

## Parallel environments

| Vectorisation | Envs | Env steps/s | RAM (process tree) |
|---|---|---|---|
| SB3 `SubprocVecEnv`, 1 worker | 1 | 790 | 703 MiB |
| SB3 `SubprocVecEnv`, 8 workers | 8 | 3,325 | 3,071 MiB |
| `BatchedSubprocVecEnv` 4 × 4 | 16 | 3,221 | 840 MiB |
| `BatchedSubprocVecEnv` 6 × 4 | 24 | 5,206 | 1,077 MiB |
| `BatchedSubprocVecEnv` 8 × 2 | 16 | 4,952 | 1,303 MiB |
| `BatchedSubprocVecEnv` 8 × 4 | 32 | 5,869 | 1,314 MiB |
| `BatchedSubprocVecEnv` 8 × 8 | 64 | 6,115 | 1,338 MiB |
| **`BatchedSubprocVecEnv` 10 × 4** | **40** | **6,797** | **1,551 MiB** |

SB3's `SubprocVecEnv` sends one pipe message per env per step and its worker imports torch
(~340 MiB per worker). `rl/vec_env.py` runs several envs per worker process (one message per worker
per step) and its worker (`rl/vec_worker.py`) imports no torch; it is tested to produce identical
observations, rewards, dones, episode stats and terminal observations to SB3's `DummyVecEnv`+Monitor.

## Full PPO loop (10 × 4 envs, VecNormalize, 2 × 256 actor/critic, ~16k samples per iteration)

| torch threads | Iteration | Collect | Update | Overall | RAM |
|---|---|---|---|---|---|
| **2** | 4.5 s | 3.1 s | 1.4 s | **3,639 samples/s** | 1,695 MiB |
| 4 | 5.9 s | 4.5 s | 1.4 s | 2,768 samples/s | 1,721 MiB |

More torch threads slow collection (they compete with the env workers). Container peak (docker
stats): **1.75 GiB** of the 6 GiB limit.

## Decision
`BatchedSubprocVecEnv` with **10 workers × 4 envs = 40 envs**, `n_steps` 512 (20,480 samples per
iteration, batch 5,120), torch threads 2. At the measured ~3.6k samples/s, 50M steps ≈ 3.8 h
(faster when the host clocks higher). This throughput feeds the go/no-go of the MJX spike (2.6).
