"""Read-aloud voice evaluation.

  record : shows one sentence at a time from brain/eval/voice_prompts_v1.txt; Enter, say it, Enter. Same mic pipeline as the live
           loop (persistent WSLg mic, push-to-talk segmenter with pre-roll / post-roll / trim). Saves the WAV and the reference text
           to brain/runs/voice_eval/<timestamp>/ (meta.jsonl + NNN.wav). Keys at the prompt: Enter = start / send, `r` = redo the
           previous sentence, `s` = skip this one, `q` = finish.
  score  : scores a recorded folder OFFLINE with Whisper (same settings as push-to-talk: VAD filter off, question prompts not used):
           word error rate, exact-command accuracy (the transcript parses to the same command as the reference), per category,
           and a comparison across bias modes (none / prompt / full). Results: score_<time>.json + a printed table.

Examples (see docs/voice_eval.md and scripts/voice_eval.sh):
  scripts/voice_eval.sh record
  scripts/voice_eval.sh score brain/runs/voice_eval/20261119_101500 --bias none prompt full
"""
from __future__ import annotations

import argparse
import json
import queue
import re
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .audio import FRAME_N, SAMPLE_RATE, read_wav, write_wav

PROMPTS = Path(__file__).resolve().parents[1] / "eval" / "voice_prompts_v1.txt"
RUNS = Path("brain/runs/voice_eval")
DEFAULT_PLACES = ["home", "room A", "room B", "room D", "doorway"]        # what the reference / hypothesis are parsed against


# ====================================================================== prompts
def load_prompts(path: Path | str = PROMPTS) -> list[dict]:
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "|" not in line:
            continue
        cat, text = (x.strip() for x in line.split("|", 1))
        out.append({"id": len(out) + 1, "category": cat, "text": text})
    return out


# ====================================================================== scoring (pure functions, tested)
_NUM_WORDS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten"}


def normalize_words(text: str) -> list[str]:
    """Lower case, no punctuation, digits spelled as words ("2" == "two"), "okay" == "ok", "let's" kept as one word."""
    t = re.sub(r"[^a-z0-9' ]", " ", text.lower().replace("’", "'"))
    out = []
    for w in t.split():
        w = _NUM_WORDS.get(w, w)
        out.append("okay" if w == "ok" else w)
    return out


def wer(ref: str, hyp: str) -> tuple[int, int]:
    """(word edit distance, number of reference words)."""
    r, h = normalize_words(ref), normalize_words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)], len(r)


_NUM_WORDS_TO_DIGITS = {v: k for k, v in _NUM_WORDS.items() if k != "0"}


def parse_command(text: str, places: list[str] | None = None) -> str:
    """The command a transcript means, as a canonical string: the keyword planner's calls (tool + args), "YES" / "NO" for a bare
    answer, "NONE" when it parses to nothing. Room names go through the same normalisation as the agent (Alpha -> room A)."""
    from ..agent import rooms
    from ..agent.safety import is_clear_no, is_clear_yes
    from ..eval.keyword_parser import parse
    t = rooms.normalize(text)
    t = re.sub(r"[,;]?\s*\b(please|for me|now)\b", "", t, flags=re.I)                 # politeness is not a clause ("Head to room B, please.")
    t = re.sub(r"\b(" + "|".join(_NUM_WORDS_TO_DIGITS) + r")\b", lambda m: _NUM_WORDS_TO_DIGITS[m.group(1).lower()], t, flags=re.I)   # "two metres" == "2 meters"
    if is_clear_yes(t):
        return "YES"
    if is_clear_no(t):
        return "NO"
    out = parse(t, places or DEFAULT_PLACES)
    calls = out.get("calls")
    return json.dumps(calls, sort_keys=True) if calls else "NONE"


def score_item(ref: str, hyp: str, places: list[str] | None = None) -> dict:
    """Exact-command accuracy: the hypothesis parses to the same command as the reference. The parser is the keyword baseline, so
    phrasings it cannot parse (casual / long ones, ref -> "NONE") are compared as words instead (WER == 0): `cmd_basis` says which."""
    e, n = wer(ref, hyp)
    cmd_ref = parse_command(ref, places)
    cmd_hyp = parse_command(hyp, places) if hyp else "NONE"
    if cmd_ref == "NONE":
        basis, ok = "text", bool(hyp) and e == 0
    else:
        basis, ok = "command", bool(hyp) and cmd_hyp == cmd_ref
    return {"errors": e, "ref_words": n, "wer": e / max(1, n), "cmd_ref": cmd_ref, "cmd_hyp": cmd_hyp, "cmd_basis": basis, "command_ok": ok}


def summarize(items: list[dict]) -> dict:
    """items: [{category, errors, ref_words, command_ok, ...}] -> overall + per-category (WER = total errors / total reference words)."""
    def agg(xs):
        n = sum(x["ref_words"] for x in xs)
        return {"n": len(xs), "wer_pct": round(100 * sum(x["errors"] for x in xs) / max(1, n), 1),
                "command_acc_pct": round(100 * sum(x["command_ok"] for x in xs) / max(1, len(xs)), 1),
                "mean_confidence": round(float(np.mean([x["confidence"] for x in xs if x.get("confidence") is not None] or [float("nan")])), 2)}
    by = defaultdict(list)
    for x in items:
        by[x["category"]].append(x)
    return {"overall": agg(items), "by_category": {c: agg(v) for c, v in sorted(by.items())}}


def score_folder(folder: Path | str, stt_for_mode, modes: list[str], places: list[str] | None = None) -> dict:
    """Score every recorded clip for every bias mode. `stt_for_mode(mode)` -> an object with .transcribe(audio) (Transcript).
    Transcripts that the live loop would drop as hallucinations (repeated tokens, silence ...) count as "heard nothing"."""
    from .stt import reject_reason
    folder = Path(folder)
    meta = [json.loads(line) for line in (folder / "meta.jsonl").read_text().splitlines() if line.strip()]
    res: dict = {"folder": str(folder), "modes": {}, "items": {}}
    for mode in modes:
        stt = stt_for_mode(mode)
        items = []
        for m in meta:
            audio = read_wav(folder / m["wav"])
            tr = stt.transcribe(audio)
            why = reject_reason(tr, audio)
            hyp = "" if why else tr.text
            it = score_item(m["text"], hyp, places)
            items.append({**it, "id": m["id"], "category": m["category"], "ref": m["text"], "hyp": tr.text, "rejected": why,
                          "confidence": tr.confidence, "stt_s": round(tr.seconds, 2)})
        res["modes"][mode] = summarize(items)
        res["items"][mode] = items
    return res


def rescore_json(path: Path | str, places: list[str] | None = None) -> dict:
    """Re-score a saved score_*.json with the CURRENT parser (no Whisper): uses the stored transcripts (rejected ones count as heard nothing)."""
    old = json.loads(Path(path).read_text())
    res: dict = {"folder": old["folder"], "device": old.get("device"), "modes": {}, "items": {}}
    for mode, items in old["items"].items():
        new = []
        for x in items:
            hyp = "" if x.get("rejected") else x["hyp"]
            new.append({**x, **score_item(x["ref"], hyp, places)})
        res["items"][mode] = new
        res["modes"][mode] = summarize(new)
    return res


def format_report(res: dict) -> str:
    modes = list(res["modes"])
    lines = [f"Folder: {res['folder']}", "", "Overall (WER % lower is better, command accuracy % higher is better):"]
    lines.append(f"  {'mode':<8}{'n':>4}{'WER%':>8}{'cmd%':>8}{'conf':>7}")
    for m in modes:
        o = res["modes"][m]["overall"]
        lines.append(f"  {m:<8}{o['n']:>4}{o['wer_pct']:>8}{o['command_acc_pct']:>8}{o['mean_confidence']:>7}")
    cats = sorted({c for m in modes for c in res["modes"][m]["by_category"]})
    lines += ["", "Per category (WER% / command%):"]
    lines.append(f"  {'category':<10}{'n':>3}  " + "  ".join(f"{m:>12}" for m in modes))
    for c in cats:
        n = res["modes"][modes[0]]["by_category"][c]["n"]
        lines.append(f"  {c:<10}{n:>3}  " + "  ".join(
            f"{res['modes'][m]['by_category'][c]['wer_pct']:>5}/{res['modes'][m]['by_category'][c]['command_acc_pct']:<6}" for m in modes))
    lines += ["", "Command mismatches (first mode, reference -> heard):"]
    bad = [x for x in res["items"][modes[0]] if not x["command_ok"]]
    for x in bad:
        lines.append(f"  #{x['id']:>2} [{x['category']}] {x['ref']!r} -> {x['hyp']!r}" + (f"  (rejected: {x['rejected']})" if x["rejected"] else "")
                     + (f"  conf {x['confidence']:.2f}" if x["confidence"] is not None else ""))
    if not bad:
        lines.append("  none")
    return "\n".join(lines)


# ====================================================================== recording
def _pick_device(device: str) -> str:
    """auto: CUDA when CTranslate2 sees a GPU with >= 2 GiB free, else CPU."""
    if device != "auto":
        return device
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() < 1:
            return "cpu"
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        return "cuda" if out.strip() and float(out.split()[0]) >= 2048 else "cpu"
    except Exception:                                  # noqa: BLE001
        return "cpu"


def record(prompts: list[dict], out_dir: Path, source, ctl: "queue.Queue", out=print, cfg=None) -> list[dict]:
    """The recording loop. `source` yields 20 ms frames (the real mic or a test generator); `ctl` gets ("enter",) / ("redo",) / ("skip",) /
    ("quit",). Returns the metadata written to out_dir/meta.jsonl."""
    from .pipeline import VoiceConfig
    from .vad import Segmenter
    cfg = cfg or VoiceConfig(mode="ptt")
    seg = Segmenter(mode="ptt", partial_every_s=None, ptt_postroll_s=cfg.ptt_postroll_s)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta: list[dict] = []
    i, recording, saved_for = 0, False, {}

    def show():
        p = prompts[i]
        out(f"\n[{i + 1}/{len(prompts)}] ({p['category']})   Say:  \"{p['text']}\"      (Enter = start)")

    def save(audio):
        p = prompts[i]
        name = f"{p['id']:03d}.wav"
        write_wav(out_dir / name, audio)
        rec = {"id": p["id"], "category": p["category"], "text": p["text"], "wav": name, "audio_s": round(len(audio) / SAMPLE_RATE, 2),
               "rms": round(float(np.sqrt(np.mean(audio ** 2))), 4), "peak": round(float(np.abs(audio).max()), 3), "ts": time.strftime("%H:%M:%S")}
        meta[:] = [m for m in meta if m["id"] != p["id"]] + [rec]
        meta.sort(key=lambda m: m["id"])
        (out_dir / "meta.jsonl").write_text("".join(json.dumps(m) + "\n" for m in meta))
        out(f"  saved {name}  ({rec['audio_s']} s, rms {rec['rms']})")

    show()
    for frame in source:
        while True:
            try:
                c = ctl.get_nowait()
            except queue.Empty:
                break
            if c[0] == "quit":
                return meta
            if c[0] == "enter":
                if recording:
                    seg.release(); recording = False
                    out("  ■ sent  (finishing ...)")
                elif seg.finishing:
                    pass                                              # still in the post-roll of the previous clip: one at a time
                else:
                    seg.press(); recording = True
                    out("  ● recording ... say it, then press Enter")
            elif c[0] == "redo" and not recording and not seg.finishing:
                i = max(0, i - 1); show()
            elif c[0] == "skip" and not recording and not seg.finishing:
                i += 1
                if i >= len(prompts):
                    return meta
                show()
        for kind, audio in seg.feed(frame):
            if kind == "end":
                if audio is None or len(audio) < SAMPLE_RATE * 0.2:
                    out("  (too short, nothing saved - press Enter and try again)")
                    continue
                save(audio)
                i += 1
                if i >= len(prompts):
                    return meta
                show()
    return meta


def _stdin_ctl(ctl: "queue.Queue") -> None:
    def loop():
        for line in sys.stdin:
            line = line.strip().lower()
            ctl.put(("quit",) if line in ("q", "quit") else ("redo",) if line == "r" else ("skip",) if line == "s" else ("enter",))
    threading.Thread(target=loop, daemon=True).start()


def cmd_record(a) -> None:
    from .audio import ArecordSource
    prompts = load_prompts(a.prompts)
    out_dir = Path(a.out) if a.out else RUNS / time.strftime("%Y%m%d_%H%M%S")
    print(f"{len(prompts)} sentences -> {out_dir}\nopening the microphone (1.5 s warm-up) ...", flush=True)
    src = ArecordSource()
    src.start()
    if src.warmup_rms < 1e-4:
        print("  warning: the mic delivered silence while warming up (check `scripts/mic_test.sh list`)")
    ctl: queue.Queue = queue.Queue()
    _stdin_ctl(ctl)
    print("Enter = start recording, Enter = send.  r = redo the previous sentence, s = skip, q = finish.", flush=True)
    try:
        meta = record(prompts, out_dir, iter(src), ctl)
    except KeyboardInterrupt:
        meta = []
    finally:
        src.close()
    print(f"\ndone: {out_dir}  ({len(list((out_dir).glob('*.wav')))} clips)\nscore it:  scripts/voice_eval.sh score {out_dir} --bias none prompt full")


# ====================================================================== score command
def cmd_score(a) -> None:
    from .stt import FasterWhisperSTT
    folder = Path(a.folder)
    device = _pick_device(a.device)
    print(f"scoring {folder} on {device}, modes {a.bias} ...", flush=True)
    cache: dict = {}

    def stt_for_mode(mode):
        if "stt" not in cache:
            cache["stt"] = FasterWhisperSTT(a.stt_model, device, places=a.places, bias=mode, vad_filter=False)   # push-to-talk settings
        cache["stt"].set_vocabulary(a.places, mode)
        return cache["stt"]
    res = score_folder(folder, stt_for_mode, a.bias, a.places)
    res["device"] = device
    print("\n" + format_report(res))
    path = folder / f"score_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(res, indent=1))
    print(f"\nsaved {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="read the prompts aloud and save WAV + reference")
    r.add_argument("--prompts", default=str(PROMPTS))
    r.add_argument("--out", help="output folder (default brain/runs/voice_eval/<timestamp>)")
    rs = sub.add_parser("rescore", help="re-score a saved score_*.json with the current parser (no Whisper)")
    rs.add_argument("json")
    rs.add_argument("--places", nargs="*", default=DEFAULT_PLACES)
    s = sub.add_parser("score", help="score a recorded folder offline")
    s.add_argument("folder")
    s.add_argument("--bias", nargs="+", default=["prompt"], choices=["none", "prompt", "full"])
    s.add_argument("--device", default="auto", help="auto (GPU if free) | cpu | cuda")
    s.add_argument("--stt-model", default="/models/whisper-turbo-ct2")
    s.add_argument("--places", nargs="*", default=DEFAULT_PLACES, help="place names used to parse commands")
    a = ap.parse_args()
    if a.cmd == "rescore":
        res = rescore_json(a.json, a.places)
        print(format_report(res))
        out = Path(a.json).with_name(Path(a.json).stem + "_rescored.json")
        out.write_text(json.dumps(res, indent=1)); print(f"\nsaved {out}")
        return
    cmd_record(a) if a.cmd == "record" else cmd_score(a)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
