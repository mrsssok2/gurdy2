"""python3 -m web.api --mock [--port 8080] [--bind 127.0.0.1] [--ollama MODEL]   private: mock world, token required
python3 -m web.api --demo [--demo-reset-min 10] [--trust-proxy]                  public demo: mock only, no token, rate limited, auto-reset"""
import argparse
import os
import threading

from .server import GurdyWebServer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true", help="serve the mock-world agent (the only source so far)")
    ap.add_argument("--demo", action="store_true", help="public demo: MockBackend only, no token, per-IP rate limits, 200-char messages, periodic world reset")
    ap.add_argument("--demo-reset-min", type=float, default=10.0, help="demo: reset the mock world every N minutes (0 = never)")
    ap.add_argument("--trust-proxy", action="store_true", help="behind ONE reverse proxy: take the client IP from the last X-Forwarded-For entry (otherwise ignored)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8080)), help="default: $PORT or 8080 (hosting platforms set PORT)")
    ap.add_argument("--bind", default="127.0.0.1", help="127.0.0.1 (default) or 0.0.0.0 for the LAN / a container")
    ap.add_argument("--speech-url", default=os.environ.get("GURDY_SPEECH_URL") or None, help="speech service (Whisper + Piper, brain.voice.speech_service) -> mic button + spoken replies; env GURDY_SPEECH_URL; never in --demo")
    ap.add_argument("--tls-port", type=int, default=int(os.environ.get("GURDY_TLS_PORT") or 0) or None, help="also serve https on this port (needs --tls-cert and --tls-key; the phone's microphone needs https)")
    ap.add_argument("--tls-cert", default=os.environ.get("GURDY_TLS_CERT") or None)
    ap.add_argument("--tls-key", default=os.environ.get("GURDY_TLS_KEY") or None)
    ap.add_argument("--ca", default=os.environ.get("GURDY_TLS_CA") or None, help="public CA certificate file served at /ca.crt (for the phone to trust)")
    ap.add_argument("--ollama", default=None, help="LLM model name (needs OLLAMA_URL); default: keyword planner (not allowed with --demo)")
    a = ap.parse_args()
    if a.demo and a.ollama:
        ap.error("--demo uses the keyword planner only (no LLM on a public page)")
    if not (a.mock or a.demo):
        ap.error("pass --mock (private) or --demo (public)")
    if a.demo:
        from .demo import build_host, make_agent
        host = build_host()
        host.start()
        srv = GurdyWebServer(host, bind=a.bind, port=a.port, demo=True, agent_factory=make_agent, reset_every_s=a.demo_reset_min * 60,
                             trust_proxy=a.trust_proxy).start()
        print(f"DEMO mode on http://{a.bind}:{srv.port}/  (simulated robot only, no token, resets every {a.demo_reset_min:g} min)", flush=True)
    else:
        from brain.voice.cli import build_local_host
        host, _ = build_local_host(a.ollama, lab=True)
        host.start()
        from .voice import SpeechClient
        speech = SpeechClient(a.speech_url) if a.speech_url else None
        srv = GurdyWebServer(host, bind=a.bind, port=a.port, trust_proxy=a.trust_proxy, speech=speech, tls_port=a.tls_port, tls_cert=a.tls_cert,
                             tls_key=a.tls_key, ca_file=a.ca).start()
        print(f"mission control on http://{a.bind}:{srv.port}/  (mock world)" + (f", https on :{srv.tls_port}" if srv.tls_port else "")
              + (f", voice via {a.speech_url}" if speech else ""))
        print(f"access token: {srv.token}   (set GURDY_WEB_TOKEN to choose your own; it is not stored anywhere)", flush=True)
    threading.Event().wait()


if __name__ == "__main__":
    main()
