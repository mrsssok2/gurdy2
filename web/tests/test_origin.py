"""Cross-origin access for the static demo's "Connect to my robot": only the configured pages, never a wildcard, the token is still required."""
import base64
import http.client
import json
import os
import socket
import unittest

from web.api.server import DEFAULT_ALLOWED_ORIGINS, GurdyWebServer, parse_origins
from web.tests.test_api_server import TOKEN, FakeHost

PAGES = "https://mrsssok2.github.io"
EVIL = "https://evil.example"


def raw(srv, method, path, headers=None, body=None):
    c = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=5)
    c.request(method, path, body, headers or {})
    r = c.getresponse()
    data = r.read()
    out = (r.status, {k.lower(): v for k, v in r.getheaders()}, data)
    c.close()
    return out


def ws_status(srv, origin=None, token=TOKEN):
    s = socket.create_connection(("127.0.0.1", srv.port), timeout=5)
    key = base64.b64encode(os.urandom(16)).decode()
    extra = f"Origin: {origin}\r\n" if origin else ""
    s.sendall((f"GET /ws?token={token} HTTP/1.1\r\nHost: robot.example\r\n{extra}Upgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
    status = int(s.makefile("rb").readline().split()[1])
    s.close()
    return status


class Base(unittest.TestCase):
    def make(self, **kw):
        self.host = FakeHost()
        self.srv = GurdyWebServer(self.host, token=TOKEN, port=0, **kw).start()
        return self.srv

    def tearDown(self):
        getattr(self, "srv", None) and self.srv.close()


class OriginTests(Base):
    def test_default_allows_only_the_github_pages_site(self):
        self.assertEqual(DEFAULT_ALLOWED_ORIGINS, (PAGES,))
        srv = self.make()
        st, h, _ = raw(srv, "GET", "/api/health", {"Origin": PAGES})
        self.assertEqual((st, h["access-control-allow-origin"], h["vary"]), (200, PAGES, "Origin"))
        st, h, _ = raw(srv, "GET", "/api/health", {"Origin": EVIL})
        self.assertEqual(st, 403)
        self.assertNotIn("access-control-allow-origin", h)                 # nothing a browser could read
        for lookalike in (PAGES + ".evil.com", "http://mrsssok2.github.io", PAGES + ":444", "https://github.io"):
            self.assertEqual(raw(srv, "GET", "/api/health", {"Origin": lookalike})[0], 403, lookalike)

    def test_never_a_wildcard(self):
        srv = self.make()
        for origin in (PAGES, EVIL, "null"):
            _, h, _ = raw(srv, "GET", "/api/health", {"Origin": origin})
            self.assertNotEqual(h.get("access-control-allow-origin"), "*")
        st, h, _ = raw(srv, "OPTIONS", "/api/say", {"Origin": PAGES, "Access-Control-Request-Method": "POST"})
        self.assertNotIn("*", h["access-control-allow-origin"] + h["access-control-allow-headers"] + h["access-control-allow-methods"])

    def test_preflight_needs_no_token_but_a_listed_origin(self):
        srv = self.make()
        st, h, _ = raw(srv, "OPTIONS", "/api/say", {"Origin": PAGES, "Access-Control-Request-Method": "POST",
                                                   "Access-Control-Request-Headers": "authorization,content-type",
                                                   "Access-Control-Request-Private-Network": "true"})
        self.assertEqual(st, 204)
        self.assertEqual(h["access-control-allow-origin"], PAGES)
        self.assertIn("Authorization", h["access-control-allow-headers"])
        self.assertIn("POST", h["access-control-allow-methods"])
        self.assertEqual(h["access-control-allow-private-network"], "true")      # public https page -> tailnet address
        self.assertEqual(raw(srv, "OPTIONS", "/api/say", {"Origin": EVIL, "Access-Control-Request-Method": "POST"})[0], 403)
        self.assertEqual(raw(srv, "OPTIONS", "/api/say")[0], 403)                # not a CORS preflight

    def test_token_is_still_required_cross_origin(self):
        srv = self.make()
        st, h, body = raw(srv, "GET", "/api/state", {"Origin": PAGES})
        self.assertEqual(st, 401)
        self.assertEqual(h["access-control-allow-origin"], PAGES)                # the page can read "wrong token" and tell the user
        self.assertEqual(raw(srv, "GET", "/api/state", {"Origin": PAGES, "Authorization": "Bearer nope"})[0], 401)
        self.assertEqual(raw(srv, "GET", "/api/state", {"Origin": PAGES, "Authorization": f"Bearer {TOKEN}"})[0], 200)
        st, _, _ = raw(srv, "POST", "/api/say", {"Origin": PAGES, "Content-Type": "application/json"}, json.dumps({"text": "hi"}))
        self.assertEqual(st, 401)
        self.assertEqual(self.host.estops, 0)

    def test_disallowed_origin_cannot_command_even_with_the_token(self):
        srv = self.make()
        st, _, _ = raw(srv, "POST", "/api/estop", {"Origin": EVIL, "Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}, "{}")
        self.assertEqual(st, 403)
        st, _, _ = raw(srv, "POST", "/api/say", {"Origin": EVIL, "Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}, json.dumps({"text": "walk"}))
        self.assertEqual(st, 403)
        self.assertEqual(self.host.estops, 0)
        self.assertEqual(self.host.agent.layer.backend.calls if hasattr(self.host.agent.layer.backend, "calls") else [], [])

    def test_websocket_origin_is_checked(self):
        srv = self.make()
        self.assertEqual(ws_status(srv, PAGES), 101)
        self.assertEqual(ws_status(srv, EVIL), 403)
        self.assertEqual(ws_status(srv, None), 101)                              # non-browser client: the token decides
        self.assertEqual(ws_status(srv, PAGES, token="wrong"), 401)
        self.assertEqual(ws_status(srv, "https://robot.example"), 101)           # the page served by this very host (Host header) is same-origin
        self.assertEqual(ws_status(srv, "http://robot.example"), 101)

    def test_own_pages_always_work(self):
        srv = self.make(allowed_origins=())                                      # nothing else allowed
        own = f"http://127.0.0.1:{srv.port}"
        st, h, _ = raw(srv, "GET", "/api/health", {"Origin": own})
        self.assertEqual((st, h.get("access-control-allow-origin")), (200, own))
        self.assertEqual(raw(srv, "GET", "/api/health", {"Origin": PAGES})[0], 403)
        self.assertEqual(raw(srv, "GET", "/api/health")[0], 200)

    def test_configurable_list_and_env(self):
        srv = self.make(allowed_origins=["https://me.example", "http://localhost:8000"])
        self.assertEqual(raw(srv, "GET", "/api/health", {"Origin": "https://me.example"})[0], 200)
        self.assertEqual(raw(srv, "GET", "/api/health", {"Origin": "http://localhost:8000"})[0], 200)
        self.assertEqual(raw(srv, "GET", "/api/health", {"Origin": PAGES})[0], 403)    # replacing the list replaces the default
        self.srv.close()
        os.environ["GURDY_ALLOWED_ORIGINS"] = "https://from-env.example,https://two.example/"
        try:
            srv = self.make()
            self.assertEqual(srv.allowed_origins, ("https://from-env.example", "https://two.example"))
            os.environ["GURDY_ALLOWED_ORIGINS"] = ""
            self.srv.close()
            self.assertEqual(self.make().allowed_origins, DEFAULT_ALLOWED_ORIGINS)       # empty variable = default, not "allow nothing"
        finally:
            os.environ.pop("GURDY_ALLOWED_ORIGINS", None)

    def test_parse_origins(self):
        self.assertEqual(parse_origins(None), DEFAULT_ALLOWED_ORIGINS)
        self.assertEqual(parse_origins("none"), ())
        self.assertEqual(parse_origins(" https://a.io/ , http://b:81 "), ("https://a.io", "http://b:81"))
        for bad in ("*", "https://a.io/path", "ftp://a.io", "a.io", "https://*.io"):
            with self.assertRaises(ValueError, msg=bad):
                parse_origins(bad)


if __name__ == "__main__":
    unittest.main()
