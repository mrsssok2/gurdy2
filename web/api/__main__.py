"""python3 -m web.api --mock [--port 8080] [--bind 127.0.0.1] [--ollama MODEL]   (mock world + keyword planner unless --ollama)"""
import argparse
import threading

from .server import GurdyWebServer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mock", action="store_true", help="serve the mock-world agent (the only source in step 1)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--bind", default="127.0.0.1", help="127.0.0.1 (default) or 0.0.0.0 for the LAN / phone")
    ap.add_argument("--ollama", default=None, help="LLM model name (needs OLLAMA_URL); default: keyword planner")
    a = ap.parse_args()
    if not a.mock:
        ap.error("step 1 serves the mock world only: pass --mock")
    from ..voice.cli import build_local_host
    host, _ = build_local_host(a.ollama)
    host.start()
    srv = GurdyWebServer(host, bind=a.bind, port=a.port).start()
    print(f"mission control on http://{a.bind}:{srv.port}/  (mock world)")
    print(f"access token: {srv.token}   (set GURDY_WEB_TOKEN to choose your own; it is not stored anywhere)")
    threading.Event().wait()


if __name__ == "__main__":
    main()
