"""Voice for the dashboard without audio hardware / models: speech engine on scripted STT + fake TTS, the HTTP speech service, the client,
the gateway rules (stop fast path, hallucination filter, confirmation hint, transcript bubble) and the web routes."""
import io
import json
import threading
import unittest
import wave

import numpy as np

from brain.voice.speech_service import SpeechEngine, make_server
from brain.voice.stt import ScriptedSTT
from web.api.server import GurdyWebServer
from web.api.telemetry import TrailBuffer
from web.api.voice import SpeechClient, SpeechError
from web.tests.test_api_server import TOKEN, FakeHost, request


def wav_bytes(seconds=1.0, amp=0.05, sr=16000, freq=220.0) -> bytes:
    t = np.arange(int(seconds * sr)) / sr
    pcm = (amp * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2")
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return b.getvalue()


class FakeTTS:
    sample_rate = 22050

    def synthesize(self, text):
        return np.zeros(2205, np.float32) + 0.1


class TrailTests(unittest.TestCase):
    def test_spinning_in_place_and_jitter_add_nothing(self):
        t = TrailBuffer()
        t.add(0.0, 0.0)
        for i in range(200):                      # a full-circle look-around with 2 cm of pose jitter
            t.add(0.02 * np.sin(i), 0.02 * np.cos(i))
        self.assertEqual(len(t), 1)

    def test_points_are_at_least_min_step_apart_and_only_last_metres_are_kept(self):
        t = TrailBuffer(min_step=0.12, keep_m=2.0)
        for i in range(400):
            t.add(i * 0.03, 0.0)                  # 12 m straight line
        pts = t.points()
        gaps = [b[0] - a[0] for a, b in zip(pts, pts[1:])]
        self.assertTrue(all(g >= 0.119 for g in gaps))
        self.assertLessEqual(pts[-1][0] - pts[0][0], 2.0 + 0.13)
        self.assertGreaterEqual(pts[-1][0] - pts[0][0], 1.8)

    def test_teleport_starts_a_new_trail(self):
        t = TrailBuffer()
        t.add(0, 0); t.add(0.5, 0); t.add(40, 40)
        self.assertEqual(len(t), 1)


class EngineTests(unittest.TestCase):
    def engine(self, *texts):
        return SpeechEngine(ScriptedSTT(list(texts)), FakeTTS())

    def test_normal_command(self):
        r = self.engine("go to room B").stt_request(wav_bytes())
        self.assertEqual((r["text"], r["rejected"], r["stop"]), ("go to room B", None, False))
        self.assertEqual(r["confidence"], 0.9)

    def test_silence_is_rejected_as_too_quiet(self):
        r = self.engine("Thank you.").stt_request(wav_bytes(amp=0.0005))
        self.assertEqual(r["rejected"], "too quiet")

    def test_hallucinated_repetition_is_rejected(self):
        r = self.engine("Roo. Roo. Roo.").stt_request(wav_bytes())
        self.assertEqual(r["rejected"], "repetition")

    def test_stop_is_never_rejected(self):
        r = self.engine("Stop. Stop. Stop.").stt_request(wav_bytes())
        self.assertTrue(r["stop"])
        self.assertIsNone(r["rejected"])
        r = self.engine("Stalk").stt_request(wav_bytes())                   # known mishearing of "stop"
        self.assertTrue(r["stop"])

    def test_too_short_never_reaches_the_stt(self):
        e = self.engine("go home")
        r = e.stt_request(wav_bytes(seconds=0.1))
        self.assertEqual(r["rejected"], "too short")
        self.assertEqual(e.stt.calls, [])

    def test_hint_is_passed_as_the_prompt(self):
        e = self.engine("yes")
        e.stt_request(wav_bytes(), hint="yes, no")
        self.assertEqual(e.stt.prompts, ["yes, no"])

    def test_wav_is_resampled_to_16k(self):
        a = SpeechEngine.decode(wav_bytes(seconds=1.0, sr=44100))
        self.assertEqual(len(a), 16000)

    def test_tts_returns_a_wav(self):
        w = wave.open(io.BytesIO(self.engine().tts_wav("hello")), "rb")
        self.assertEqual((w.getnchannels(), w.getframerate(), w.getnframes()), (1, 22050, 2205))


class ServiceTests(unittest.TestCase):
    def test_http_round_trip_through_the_client(self):
        srv = make_server(SpeechEngine(ScriptedSTT(["go to room B"]), FakeTTS(), "cpu"), 0, "127.0.0.1")
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            c = SpeechClient(f"http://127.0.0.1:{srv.server_address[1]}")
            self.assertTrue(c.available())
            self.assertEqual(c.stt(wav_bytes(), None, ["room A", "room B"])["text"], "go to room B")
            self.assertTrue(c.tts("hi").startswith(b"RIFF"))
            with self.assertRaises(SpeechError):
                c.stt(b"not audio at all, just bytes", None)
        finally:
            srv.shutdown()
            srv.server_close()

    def test_unreachable_service(self):
        c = SpeechClient("http://127.0.0.1:9", timeout_s=1)
        self.assertFalse(c.available())
        with self.assertRaises(SpeechError):
            c.tts("hi")


class FakeSpeech:
    def __init__(self):
        self.next = {"text": "go to room B", "confidence": 0.93, "seconds": 0.3, "rejected": None, "stop": False}
        self.calls, self.up = [], True

    def available(self):
        return self.up

    def stt(self, audio, hint=None, places=None):
        self.calls.append((len(audio), hint, places))
        return dict(self.next)

    def tts(self, text):
        return b"RIFF" + text.encode()


class VoiceHost(FakeHost):
    def __init__(self):
        super().__init__()
        self.hint = None
        self.said = []

    def places(self):
        return ["room A", "room B"]

    def answer_hint(self):
        return self.hint

    def say(self, text, conf=None, started_ago=None):
        self.said.append((text, conf))
        return super().say(text, conf, started_ago)


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.host, self.speech = VoiceHost(), FakeSpeech()
        self.srv = GurdyWebServer(self.host, token=TOKEN, static_dir=None, port=0, speech=self.speech).start()

    def tearDown(self):
        self.srv.close()

    def voice(self, data=b"\x1a\x45\xdf\xa3audio", token=TOKEN):
        return request(self.srv, "POST", "/api/voice", data, token=token, headers={"Content-Type": "audio/webm"})

    def test_transcript_bubble_has_confidence_and_agent_gets_the_text(self):
        st, r = self.voice()
        self.assertEqual(st, 200)
        self.assertEqual(r["heard"], "go to room B")
        self.assertEqual(self.host.said, [("go to room B", 0.93)])
        msgs = request(self.srv, "GET", "/api/messages?since=0")[1]["messages"]
        user = [m for m in msgs if m["role"] == "user"][0]
        self.assertEqual((user["text"], user["voice"], user["confidence"]), ("go to room B", True, 0.93))
        self.assertEqual(msgs[-1]["role"], "robot")
        self.assertEqual(self.speech.calls[0][2], ["room A", "room B"])        # vocabulary = the robot's places

    def test_pending_question_passes_the_expected_answers(self):
        self.host.hint = "yes or no"
        self.voice()
        self.assertEqual(self.speech.calls[0][1], "yes or no")

    def test_stop_phrase_fires_estop_first(self):
        self.speech.next = {"text": "Stop", "confidence": 0.8, "rejected": None, "stop": True}
        self.voice()
        self.assertEqual(self.host.estops, 1)
        self.assertEqual(self.host.said[0][0], "Stop")                          # the agent still answers (template)

    def test_noise_is_ignored_and_never_reaches_the_agent(self):
        self.speech.next = {"text": "Roo. Roo. Roo.", "confidence": 0.3, "rejected": "repetition", "stop": False}
        st, r = self.voice()
        self.assertEqual((st, r["rejected"]), (200, "repetition"))
        self.assertEqual(self.host.said, [])
        msgs = request(self.srv, "GET", "/api/messages?since=0")[1]["messages"]
        self.assertTrue(any("didn't catch" in m["text"] for m in msgs))
        self.assertFalse(any(m["role"] == "user" for m in msgs))

    def test_nothing_heard(self):
        self.speech.next = {"text": "", "rejected": "too short", "stop": False}
        self.assertEqual(self.voice()[1]["heard"], "")
        self.assertEqual(self.host.said, [])

    def test_auth_size_and_availability(self):
        self.assertEqual(self.voice(token="bad")[0], 401)
        import http.client
        c = http.client.HTTPConnection("127.0.0.1", self.srv.port, timeout=5)           # claims 3 MB, sends nothing: refused before reading
        c.putrequest("POST", "/api/voice")
        c.putheader("Authorization", f"Bearer {TOKEN}")
        c.putheader("Content-Length", "3000000")
        c.endheaders()
        self.assertEqual(c.getresponse().status, 413)
        c.close()
        self.assertEqual(request(self.srv, "POST", "/api/voice", b"", headers={"Content-Type": "audio/webm"})[0], 400)
        self.speech.up = False
        self.srv.voice.speech.up = False
        self.assertEqual(self.voice()[0], 503)
        self.assertFalse(request(self.srv, "GET", "/api/health", token=None)[1]["voice"])

    def test_health_and_hello_say_voice_is_on(self):
        self.assertTrue(request(self.srv, "GET", "/api/health", token=None)[1]["voice"])

    def test_tts_route(self):
        st, body = request(self.srv, "POST", "/api/tts", {"text": "I have arrived."})
        self.assertEqual((st, body), (200, b"RIFFI have arrived."))
        self.assertEqual(request(self.srv, "POST", "/api/tts", {"text": ""})[0], 400)
        self.assertEqual(request(self.srv, "POST", "/api/tts", {"text": "x"}, token="bad")[0], 401)

    def test_speech_service_error_is_a_502(self):
        def boom(*a, **k):
            raise SpeechError("down")
        self.speech.stt = boom
        self.assertEqual(self.voice()[0], 502)


class NoVoiceTests(unittest.TestCase):
    def test_server_without_speech_has_no_voice_routes(self):
        srv = GurdyWebServer(VoiceHost(), token=TOKEN, static_dir=None, port=0).start()
        try:
            self.assertEqual(request(srv, "POST", "/api/voice", b"abc")[0], 503)
            self.assertFalse(request(srv, "GET", "/api/health", token=None)[1]["voice"])
        finally:
            srv.close()

    def test_demo_never_has_voice_even_if_a_speech_client_is_passed(self):
        from brain.agent.graph import Agent
        from brain.agent.llm import KeywordLLM
        from brain.mock import MockBackend, gazebo_flat_lab
        from brain.tools import ToolLayer
        from brain.voice.host import AgentHost
        b = MockBackend(gazebo_flat_lab())
        host = AgentHost(Agent(ToolLayer(b), KeywordLLM(b.known_places, smart=True)))
        srv = GurdyWebServer(host, demo=True, static_dir=None, port=0, speech=FakeSpeech()).start()
        try:
            self.assertIsNone(srv.voice)
            self.assertEqual(request(srv, "POST", "/api/voice", b"abc", token=None)[0], 503)
            self.assertFalse(request(srv, "GET", "/api/health", token=None)[1]["voice"])
        finally:
            srv.close()


if __name__ == "__main__":
    unittest.main()
