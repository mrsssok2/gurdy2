"""Speech service: Whisper (speech-to-text) and Piper (text-to-speech) behind a tiny HTTP API, so the web dashboard can talk to the robot from
a browser. Runs in the `voice_gpu` image (Whisper on the GPU, ~1 GiB VRAM; CPU fallback); the dashboard server only forwards bytes to it.

  POST /stt[?hint=<expected answers>&places=room%20A,room%20B]   body = audio (browser WebM/Opus or MP4/AAC, or 16-bit PCM WAV)
       -> {"text", "confidence", "seconds", "audio_s", "rms", "rejected": null | "too quiet" | "repetition" | "no speech" | "too short", "stop": bool}
  POST /tts   {"text": "..."} -> audio/wav (16-bit mono)
  GET  /health -> {"ok": true, "device": "cuda"|"cpu", "tts_rate": 22050}

The same safeguards as the live microphone pipeline are applied here: the vocabulary prompt (place names, objects), the question-specific `hint`
while the agent waits for a yes / no, the hallucination filter (`stt.reject_reason`: silence, repeated tokens, no-speech probability) and the
stop-word check (a stop is never discarded as noise). Nothing is stored on disk."""
from __future__ import annotations

import argparse
import io
import json
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import numpy as np

from .audio import SAMPLE_RATE, resample
from .stt import Transcript, reject_reason

MAX_AUDIO_BYTES = 4_000_000
MIN_AUDIO_S = 0.25


class SpeechEngine:
    def __init__(self, stt, tts, device: str = "cpu"):
        self.stt, self.tts, self.device = stt, tts, device
        self._lock = threading.Lock()                # one Whisper / Piper call at a time (the GPU model is not thread-safe)
        self._places: tuple | None = None

    # ------------------------------------------------------------------ audio in
    @staticmethod
    def decode(data: bytes) -> np.ndarray:
        """Browser recording or WAV -> 16 kHz mono float32."""
        if data[:4] == b"RIFF":
            with wave.open(io.BytesIO(data), "rb") as w:
                sr, ch, width, raw = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.readframes(w.getnframes())
            if width != 2:
                raise ValueError("only 16-bit PCM WAV is supported")
            a = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
            if ch > 1:
                a = a.reshape(-1, ch).mean(axis=1)
            return resample(a, sr, SAMPLE_RATE)
        from faster_whisper.audio import decode_audio        # PyAV: WebM/Opus (Chrome, Android), MP4/AAC (Safari)
        return np.asarray(decode_audio(io.BytesIO(data), sampling_rate=SAMPLE_RATE), np.float32)

    def stt_request(self, data: bytes, hint: str | None = None, places: list[str] | None = None) -> dict:
        from ..agent.stopwords import is_estop_phrase
        audio = self.decode(data)
        dur = len(audio) / SAMPLE_RATE
        rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
        if dur < MIN_AUDIO_S:
            return {"text": "", "confidence": None, "seconds": 0.0, "audio_s": round(dur, 2), "rms": rms, "rejected": "too short", "stop": False}
        with self._lock:
            if places is not None and hasattr(self.stt, "set_vocabulary") and tuple(places) != self._places:
                self.stt.set_vocabulary(places, getattr(self.stt, "bias", "prompt"))
                self._places = tuple(places)
            if hasattr(self.stt, "vad_filter"):
                self.stt.vad_filter = False             # push-to-talk: the button already marks the speech (same rule as the live pipeline)
            tr: Transcript = self.stt.transcribe(audio, prompt=hint) if hint else self.stt.transcribe(audio)
        stop = bool(tr.text) and is_estop_phrase(tr.text)
        why = reject_reason(tr, audio) if tr.text else None
        if stop:
            why = None                                   # a stop is never discarded
        return {"text": tr.text, "confidence": None if tr.confidence is None else round(float(tr.confidence), 3), "seconds": round(tr.seconds, 2),
                "audio_s": round(dur, 2), "rms": round(rms, 4), "rejected": why, "stop": stop}

    # ------------------------------------------------------------------ audio out
    def tts_wav(self, text: str) -> bytes:
        with self._lock:
            a = self.tts.synthesize(text)
        pcm = (np.clip(np.asarray(a, np.float32), -1, 1) * 32767).astype("<i2")
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(int(self.tts.sample_rate))
            w.writeframes(pcm.tobytes())
        return buf.getvalue()


def make_server(engine: SpeechEngine, port: int, bind: str = "0.0.0.0") -> ThreadingHTTPServer:
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):                     # quiet
            pass

        def _send(self, status: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status: int = 200) -> None:
            self._send(status, json.dumps(obj).encode())

        def _read(self) -> bytes | None:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_AUDIO_BYTES:
                self._json({"error": "bad body size"}, 413 if n > 0 else 400)
                return None
            return self.rfile.read(n)

        def do_GET(self):
            if urlsplit(self.path).path == "/health":
                return self._json({"ok": True, "device": engine.device, "tts_rate": int(engine.tts.sample_rate)})
            self._json({"error": "not found"}, 404)

        def do_POST(self):
            u = urlsplit(self.path)
            q = parse_qs(u.query)
            body = self._read()
            if body is None:
                return
            try:
                if u.path == "/stt":
                    places = q["places"][0].split(",") if q.get("places") else None
                    return self._json(engine.stt_request(body, (q.get("hint") or [None])[0], places))
                if u.path == "/tts":
                    text = str(json.loads(body).get("text", "")).strip()[:600]
                    if not text:
                        return self._json({"error": "empty text"}, 400)
                    return self._send(200, engine.tts_wav(text), "audio/wav")
            except Exception as e:                      # noqa: BLE001  (a bad recording must not kill the service)
                return self._json({"error": f"{type(e).__name__}: {e}"}, 400)
            self._json({"error": "not found"}, 404)

    return ThreadingHTTPServer((bind, port), H)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--stt-model", default="/models/whisper-turbo-ct2")
    ap.add_argument("--tts-model", default="/models/tts/piper-lessac.onnx")
    ap.add_argument("--device", default="auto", help="Whisper device: auto | cuda | cpu")
    a = ap.parse_args()
    from .stt import load_whisper
    from .tts import PiperTTS
    print("loading Whisper + Piper ...", flush=True)
    stt = load_whisper(a.stt_model, a.device, bias="prompt")
    tts = PiperTTS(a.tts_model)
    srv = make_server(SpeechEngine(stt, tts, stt.device), a.port)
    print(f"speech service on :{a.port} (Whisper on the {stt.device.upper()})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
