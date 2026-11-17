# Phase 5.1 — serving + VRAM / RAM / latency measurements (2026-11-17)

Machine: RTX 4050 6 GB (idle 0.15 GiB used), WSL 7.9 GiB RAM + 8 GiB swap (`.wslconfig` NOT changed), Windows RAM ~97 % used.
Scripts: `brain/bench/{llm_bench,llm_loop,load_gen,stt_tts_bench}.py`, `sample_resources.sh`, `scripts/phase5_setA_sim.sh`. Raw data: `docs/results/data/phase5/setA*/`.
Stack (outside the repo): Ollama 0.40.16 in `~/ollama` (**its `cuda_v13` backend crashes with "CUDA error: unknown" on driver 13.5 / WSL; moved aside, `cuda_v12` works**),
models `qwen3:1.7b` / `qwen3:0.6b` (Q4_K_M, pulled from the Ollama registry), faster-whisper 1.2.1 (CT2 float16 conversion of `~/hf_models/whisper-large-v3-turbo`), Piper 1.8 (en_US-lessac-medium), kokoro-onnx 0.6.1.

## Single components (warm, medians, 5 runs)
| component | VRAM | host RAM | latency |
|---|---|---|---|
| Qwen3-1.7B Q4_K_M, ctx 4096 | +1.70 GiB (1745 MiB) | ~1.0 GiB | decode 93–100 tok/s (74 under load); TTFT 0.03 s with cached prefix, 0.65–1.7 s with a new 2.5k-token prompt; load 12–22 s from cold disk |
| Qwen3-0.6B Q4_K_M, ctx 4096 | +0.98 GiB (1005 MiB) | ~1.0 GiB | decode 141–156 tok/s; TTFT 0.03 s cached / 0.2–1.3 s new prompt; load 2–15 s |
| Whisper large-v3-turbo CUDA int8_float16 | +1.05 GiB (1072 MiB) | ~0.5 GiB | 0.28 s (2.0 s clip) / 0.30 s (3.8 s clip); both transcripts correct but no punctuation |
| Whisper CUDA float16 | +2.05 GiB | – | 0.32 / 0.34 s → int8_float16 is the choice |
| Whisper CPU int8 (6 threads) | 0 | – | **5.7 s** (fixed 30 s window) → GPU needed |
| Piper CPU | 0 | small | first audio 0.08 s (33 chars) / 0.16 s (73 chars), RTF 0.04 |
| Kokoro-82M CPU | 0 | small | first audio 0.72 s / **1.34 s** |
| Kokoro-82M CUDA | +0.60 GiB (616 MiB) | small | 0.13 s / 0.18 s |
| Detector det_v2_s (CUDA, 15 Hz, alone) | ~0.3 GiB | ~0.5 GiB | 13–18 ms/frame |

**TTS decision:** Piper. Kokoro on CPU misses the ~1 s target for normal sentences; on GPU it is fast but costs 0.6 GiB VRAM for a voice nobody needs at 0.16 s. Piper (CPU, 0 VRAM) stays.
Piper's pip build is GPL-3: keep it a separate process like Ultralytics. The prompt (all 24 tool specs + state) is **2550 tokens** — the biggest latency lever; keep the prefix byte-identical so it is cached.

## Combined, set A (Gazebo semantic_lab headless with SLAM + Nav2 + policy + exploration, GPU detector 15 Hz, Whisper int8_float16 resident + a clip every 6 s, Qwen3-1.7B call every 5 s, 230 s)
| | measured |
|---|---|
| **peak GPU** | **3467 MiB (3.4 GiB)** total (idle 150) — under the 5.0 GiB limit, 2.6 GiB headroom |
| **peak WSL RAM used** | **4461 MiB of 7942** (min available 3481 MiB), swap unchanged (801 MiB, no growth) |
| LLM under load | TTFT median 0.084 s / max 0.47 s (shared prefix), decode 61 tok/s |
| Gazebo RTF | mean 0.54 (0.49–0.57) — earlier headless nav runs 0.6–1.0, so the GPU sharing / CPU load costs RTF |
| detector | 13.3 ms median, 3147 frames |
Not measured: Gazebo **GUI** on, the VLM (5.10), Piper running at the same time (CPU, small), Silero VAD.

**Is 8 GB WSL enough? Yes for set A — 4.4 GiB peak, no swapping — keep `.wslconfig` at 8 GB; the planned 10 GB is not needed.**

## Trap found: prompt-cache RAM
A first set-A run gave a false alarm (7.8 GiB used, 4.8 GiB swap, TTFT 1.8 s median): my benchmark put a unique prefix in every request, and the llama.cpp server keeps each ~2.5k-token prompt
(~0.3 GiB of KV for 1.7B) in a **host-RAM prompt cache** → +4 GiB in 3 minutes (`setA`, `setA2`, `setA4`: RAM climbs linearly only while the LLM loop runs). With one stable system prompt (real use) nothing grows (`setA5`).
Rules for the agent: keep system + tool block byte-identical across calls (also the fastest path), put the state block and the command after it; do not add per-turn data (timestamps, ids) before the tool block.
Still to check: a long session with many different state blocks (cache growth bound) — 5.3/5.6.

## Not done / open
- **GGUF conversion of the fine-tuned model:** Ollama's `create --quantize` accepts only int4/int8 (not Q4_K_M) in this version, so the fine-tuned Qwen3 needs llama.cpp's `convert_hf_to_gguf.py` + `llama-quantize` (the latter ships in the Ollama tarball). I downloaded the converter from llama.cpp master; running it was blocked by the permission classifier (downloaded code), so the base models came from the Ollama registry. Needs a decision before 5.5 (pin a llama.cpp release tag you approve, or use a reviewed copy).
- Whisper on real microphone audio (all STT numbers use Piper-synthesised speech; accuracy untested); Whisper on-demand load time (set C variant) not measured because set A fits.
