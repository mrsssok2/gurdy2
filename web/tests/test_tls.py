"""https on a second port with a certificate from scripts/make_web_cert.sh, verified against the local CA (stdlib + openssl only)."""
import http.client
import os
import shutil
import ssl
import subprocess
import tempfile
import unittest
from pathlib import Path

from web.api.server import GurdyWebServer
from web.tests.test_api_server import TOKEN, FakeHost

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "make_web_cert.sh"


@unittest.skipUnless(shutil.which("openssl") and shutil.which("bash"), "openssl / bash not available")
class TlsTests(unittest.TestCase):
    def test_https_with_the_local_ca_and_public_ca_download(self):
        with tempfile.TemporaryDirectory() as home:
            subprocess.run(["bash", str(SCRIPT), "192.168.137.1"], env={**os.environ, "HOME": home}, check=True, capture_output=True)
            tls = Path(home) / ".gurdy" / "tls"
            srv = GurdyWebServer(FakeHost(), token=TOKEN, static_dir=None, port=0, tls_port=0, tls_cert=str(tls / "gurdy.crt"),
                                 tls_key=str(tls / "gurdy.key"), ca_file=str(tls / "ca.crt")).start()
            try:
                self.assertNotEqual(srv.tls_port, srv.port)
                ctx = ssl.create_default_context(cafile=str(tls / "ca.crt"))
                c = http.client.HTTPSConnection("127.0.0.1", srv.tls_port, context=ctx, timeout=5)
                c.request("GET", "/api/health")
                self.assertEqual(c.getresponse().status, 200)
                c.close()
                c = http.client.HTTPSConnection("localhost", srv.tls_port, context=ctx, timeout=5)
                c.request("GET", "/ca.crt")
                r = c.getresponse()
                self.assertEqual((r.status, r.read()), (200, (tls / "ca.crt").read_bytes()))
                c.close()
                c = http.client.HTTPSConnection("127.0.0.1", srv.tls_port, context=ssl.create_default_context(), timeout=5)   # system CAs only
                with self.assertRaises(ssl.SSLError):
                    c.request("GET", "/api/health")
                c.close()
                c = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=5)           # plain http still works on the first port
                c.request("GET", "/api/health")
                self.assertEqual(c.getresponse().status, 200)
                c.close()
                self.assertNotIn(b"PRIVATE", (tls / "ca.crt").read_bytes())                  # only the public certificate is ever served
            finally:
                srv.close()


if __name__ == "__main__":
    unittest.main()
