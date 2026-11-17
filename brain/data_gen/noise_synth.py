#!/usr/bin/env python3
"""Stage B: the noisy-STT round trip (design 6.1 step 5). Runs in the vision image (GPU Whisper, CPU Piper).

    docker run --rm --gpus all -v $PWD:/w -v ~/p5_pkgs:/pk -v ~/p5_tts:/t -v ~/p5_models:/m -e PYTHONPATH=/pk:/t -w /w \
        --entrypoint python3 gurdy2/vision:cuda brain/data_gen/noise_synth.py IN_candidates.json OUT_noisy.jsonl

For every unique utterance: synthesise with Piper under four increasingly harsh conditions (speed / pitch via resampling,
Piper noise scales, white noise at 25 / 8 / 3 / 6 dB SNR, phone band-limit, reverb + clipping, gain, leading silence), transcribe each with Whisper large-v3-turbo
(CT2 int8_float16, beam 1, English, no prompt, no VAD). Writes one JSON line per utterance with all transcripts.
Resumable: utterances already in OUT are skipped. Whisper's own punctuation / capitals are kept (that is what the agent will see).
Limitation: ONE Piper voice, so speaker variety is synthetic (speed / pitch / noise only); real microphone audio is harsher.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

CONDS = [  # name, length_scale range, noise_scale, noise_w, pitch/speed resample range, SNR dB, extra
    ("mild", (0.9, 1.1), 0.667, 0.8, (0.97, 1.03), 25, None),
    ("harsh", (0.7, 1.35), 1.1, 1.0, (0.85, 1.15), 8, None),
    ("phone", (0.6, 0.8), 1.0, 1.0, (0.9, 1.1), 3, "bandpass"),          # band-limited 300-3400 Hz, fast speech, SNR 3 dB
    ("room", (0.7, 0.9), 1.0, 1.0, (0.85, 1.15), 6, "reverb_clip"),     # light reverb + clipping, SNR 6 dB
]


def degrade(a, extra, rng):
    if extra == "bandpass":
        spec = np.fft.rfft(a)
        f = np.fft.rfftfreq(len(a), 1 / 16000)
        spec[(f < 300) | (f > 3400)] = 0
        return np.fft.irfft(spec, len(a)).astype(np.float32)
    if extra == "reverb_clip":
        ir = rng.normal(0, 0.5, 2400) * np.exp(-np.arange(2400) / 800.0)
        ir[0] = 1.0
        a = np.convolve(a, ir)[: len(a)].astype(np.float32)
        a = a / max(1e-3, float(np.abs(a).max()))
        return np.clip(a * 1.6, -1, 1).astype(np.float32)
    return a


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    texts = json.loads(Path(src).read_text())
    done = set()
    if Path(dst).exists():
        done = {json.loads(l)["say"] for l in open(dst)}
    todo = [t for t in texts if t not in done]
    print(f"{len(texts)} utterances, {len(done)} done, {len(todo)} to do", flush=True)
    from faster_whisper import WhisperModel
    from piper import PiperVoice, SynthesisConfig
    voice = PiperVoice.load("/m/tts/piper-lessac.onnx", use_cuda=False)
    sr = voice.config.sample_rate
    model = WhisperModel("/m/whisper-turbo-ct2", device="cuda", compute_type="int8_float16")
    rng = np.random.default_rng(0)
    t0 = time.time()
    with open(dst, "a") as out:
        for i, text in enumerate(todo):
            rec = {"say": text, "variants": []}
            for name, ls, ns, nw, rs, snr, extra in CONDS:
                cfg = SynthesisConfig(length_scale=float(rng.uniform(*ls)), noise_scale=ns, noise_w_scale=nw)
                a = np.concatenate([c.audio_float_array for c in voice.synthesize(text, syn_config=cfg)]).astype(np.float32)
                f = float(rng.uniform(*rs))                                  # resampling = pitch and speed shift together
                n_out = int(len(a) / f)
                a = np.interp(np.linspace(0, len(a) - 1, n_out), np.arange(len(a)), a)
                n16 = int(len(a) * 16000 / sr)
                a = np.interp(np.linspace(0, len(a) - 1, n16), np.arange(len(a)), a).astype(np.float32)
                a = degrade(a, extra, rng)
                a *= float(rng.uniform(0.4, 1.0)) / max(1e-3, float(np.abs(a).max()))
                sig = float(np.sqrt(np.mean(a ** 2)))
                a = a + rng.normal(0, sig / (10 ** (snr / 20)), a.shape).astype(np.float32)
                a = np.concatenate([np.zeros(int(16000 * rng.uniform(0.1, 0.5)), np.float32), a, np.zeros(int(16000 * 0.3), np.float32)])
                segs, _ = model.transcribe(a, language="en", beam_size=1, condition_on_previous_text=False, vad_filter=False,
                                           without_timestamps=True)
                rec["variants"].append({"cond": name, "text": "".join(s.text for s in segs).strip()})
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if (i + 1) % 100 == 0:
                el = time.time() - t0
                print(f"{len(done) + i + 1}/{len(texts)}  {el:.0f}s  eta {el / (i + 1) * (len(todo) - i - 1):.0f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
