"""Voice for the dashboard: the browser records a push-to-talk clip, this module sends it to the speech service (Whisper + Piper,
`brain/voice/speech_service.py`) and feeds the transcript to the SAME paths as typed text:

  stop phrase   -> host.estop() at once (fast path, no LLM, no agent lock), then the text still goes to the agent (it answers from a template)
  noise / empty -> "Sorry, I didn't catch that." and nothing reaches the agent (hallucination filter in the speech service)
  question open -> the speech service is told the expected answers (`host.answer_hint()`), so a quiet "yes" is heard as "yes"
  otherwise     -> host.say(text, confidence), like a typed message

The transcript is shown as the user's chat bubble with its confidence. Replies are spoken by `POST /api/tts` (Piper), which the page calls for
robot messages while unmuted. Off in --demo mode (no GPU on a free host): the server is simply built without a speech client."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

MAX_AUDIO_BYTES = 2_000_000             # ~ 60 s of WebM/Opus; a push-to-talk command is a few seconds
MAX_TTS_CHARS = 400


class SpeechError(RuntimeError):
    pass


class SpeechClient:
    """HTTP client for brain.voice.speech_service (stdlib only: the dashboard also runs inside the ROS image)."""

    def __init__(self, url: str, timeout_s: float = 25.0, health_ttl_s: float = 5.0, clock=time.monotonic):
        self.url, self.timeout_s, self.health_ttl_s, self.clock = url.rstrip("/"), timeout_s, health_ttl_s, clock
        self._ok = False
        self._checked = -1e9

    def available(self) -> bool:
        if self.clock() - self._checked > self.health_ttl_s:
            self._checked = self.clock()
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=1.5) as r:
                    self._ok = bool(json.loads(r.read()).get("ok"))
            except (OSError, ValueError):
                self._ok = False
        return self._ok

    def _post(self, path: str, data: bytes, ctype: str) -> bytes:
        req = urllib.request.Request(self.url + path, data=data, method="POST", headers={"Content-Type": ctype})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            raise SpeechError(f"speech service: {e.code} {e.read()[:120]!r}") from e
        except OSError as e:
            raise SpeechError(f"speech service unreachable: {e}") from e

    def stt(self, audio: bytes, hint: str | None = None, places: list[str] | None = None) -> dict:
        q = {}
        if hint:
            q["hint"] = hint
        if places:
            q["places"] = ",".join(places)
        path = "/stt" + ("?" + urllib.parse.urlencode(q) if q else "")
        return json.loads(self._post(path, audio, "application/octet-stream"))

    def tts(self, text: str) -> bytes:
        return self._post("/tts", json.dumps({"text": text}).encode(), "application/json")


class VoiceGateway:
    def __init__(self, host, speech: SpeechClient, chat):
        self.host, self.speech, self.chat = host, speech, chat

    def available(self) -> bool:
        return self.speech.available()

    def handle_audio(self, audio: bytes) -> dict:
        """One recorded utterance -> transcript bubble + agent reply. Returns a small summary for the page."""
        try:
            places = list(self.host.places())
        except Exception:                                  # noqa: BLE001
            places = None
        hint = None
        try:
            hint = self.host.answer_hint()
        except Exception:                                  # noqa: BLE001
            pass
        tr = self.speech.stt(audio, hint, places)
        text = (tr.get("text") or "").strip()
        conf = tr.get("confidence")
        if tr.get("stop"):
            self.host.estop()                              # before anything else: the fast path never waits for the agent
        if not text:
            self.chat.add("system", "(heard nothing)")
            return {"heard": "", "rejected": tr.get("rejected") or "nothing heard"}
        if tr.get("rejected"):
            self.chat.add("system", f"(heard noise: {text!r}, {tr['rejected']}; ignored)")
            self.chat.add("robot", "Sorry, I didn't catch that.")
            return {"heard": text, "rejected": tr["rejected"], "confidence": conf}
        self.chat.add("user", text, voice=True, confidence=conf, stt_s=tr.get("seconds"))
        reply = self.host.say(text, conf)
        self.chat.add("robot", reply)
        return {"heard": text, "confidence": conf, "stop": bool(tr.get("stop")), "reply": reply}

    def speak(self, text: str) -> bytes:
        return self.speech.tts(text[:MAX_TTS_CHARS])
