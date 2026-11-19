"""A language model that cannot be reached becomes one plain sentence, not a raw URLError; details go to the agent log."""
import socket

from brain.agent.graph import Agent
from brain.agent.llm import LLMUnavailable, ScriptedLLM
from brain.agent.ollama_llm import OllamaLLM
from brain.mock import MockBackend
from brain.tools import ToolLayer
from brain.voice.host import AgentHost, _error_text


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def agent_with(llm):
    return Agent(ToolLayer(MockBackend()), llm)


def test_connection_refused_is_a_plain_sentence(capsys):
    llm = OllamaLLM("gurdy-sft-400", host=f"http://127.0.0.1:{free_port()}", timeout=2)      # nothing listens: Errno 111
    agent = agent_with(llm)
    r = agent.say("go to room B")
    assert r.reply == "My language model is not running right now."
    assert "URLError" not in r.reply and "Errno" not in r.reply
    assert agent.errors and agent.errors[0].startswith("LLMUnavailable")
    assert "language model unavailable" in capsys.readouterr().err                           # the detail is in the log


def test_agent_recovers_when_the_model_comes_back():
    calls = {"n": 0}

    def plan(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise LLMUnavailable("down")
        return {"reply": "Hello."}
    agent = agent_with(ScriptedLLM(plans=[plan, plan]))
    assert agent.say("hi").reply == "My language model is not running right now."
    assert agent.say("hi").reply == "Hello."


def test_unknown_model_says_not_installed():
    import http.server
    import threading

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error":"model not found"}')

        def log_message(self, *a):
            pass
    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        llm = OllamaLLM("nope", host=f"http://127.0.0.1:{srv.server_address[1]}", timeout=2)
        assert agent_with(llm).say("go home").reply == "My language model is not installed right now."
    finally:
        srv.shutdown()


def test_host_error_text_uses_the_plain_sentence_and_keeps_generic_errors():
    assert _error_text(LLMUnavailable("x")) == "My language model is not running right now."
    assert _error_text(RuntimeError("boom")) == "Something went wrong: RuntimeError: boom"
