"""scripts/lib_tailscale.sh against a stub `tailscale`: the exact serve command (tailnet-only Serve, never Funnel), states, failures."""
import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUB = """#!/bin/bash
echo "$@" >> "$TS_LOG"
case "$1" in
  status) printf '{"BackendState":"%s","Self":{"DNSName":"my-pc.tail1234.ts.net."}}\\r\\n' "${TS_STATE:-Running}" ;;
  serve) [ -n "$TS_FAIL" ] && { echo "HTTPS not enabled"; exit 1; }; echo ok ;;
esac
"""


@unittest.skipUnless(shutil.which("bash") and shutil.which("python3"), "bash / python3 needed")
class TailscaleLibTests(unittest.TestCase):
    def run_lib(self, script, **env):
        with tempfile.TemporaryDirectory() as d:
            stub = Path(d) / "ts"
            stub.write_text(STUB)
            stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
            log = Path(d) / "log"
            e = {**os.environ, "TAILSCALE_BIN": str(stub), "TS_LOG": str(log), **env}
            r = subprocess.run(["bash", "-c", f"cd {ROOT}; source scripts/lib_tailscale.sh; {script}"], env=e, capture_output=True, text=True)
            return r, (log.read_text().splitlines() if log.exists() else [])

    def test_start_uses_serve_on_443_to_localhost_and_never_funnel(self):
        r, calls = self.run_lib('ts_check && ts_start 8080 && echo "URL=$TS_URL"; ts_stop')
        self.assertIn("URL=https://my-pc.tail1234.ts.net/", r.stdout)
        self.assertEqual(calls, ["status --json", "serve --bg --https=443 http://127.0.0.1:8080", "serve --https=443 off"])
        self.assertFalse(any("funnel" in c for c in calls))

    def test_not_logged_in_is_reported_and_serve_is_not_started(self):
        r, calls = self.run_lib("ts_check; echo rc=$?", TS_STATE="NeedsLogin")
        self.assertIn("NeedsLogin", r.stdout)
        self.assertIn("rc=1", r.stdout)
        self.assertEqual(calls, ["status --json"])

    def test_serve_failure_points_to_the_doc(self):
        r, _ = self.run_lib("ts_check; ts_start 8080; echo rc=$?", TS_FAIL="1")
        self.assertIn("rc=1", r.stdout)
        self.assertIn("docs/phase7_remote.md", r.stdout)

    def test_missing_client(self):
        r = subprocess.run(["bash", "-c", f"cd {ROOT}; source scripts/lib_tailscale.sh; ts_check; echo rc=$?"],
                           env={"PATH": "/usr/bin:/bin", "TAILSCALE_WIN_PATH": "/nonexistent/tailscale.exe"}, capture_output=True, text=True)
        self.assertIn("Install it on Windows", r.stdout)

    def test_start_scripts_never_publish_to_the_lan_for_tailscale_alone(self):
        for name in ("start_web.sh", "start_web_robot.sh"):
            text = (ROOT / "scripts" / name).read_text()
            self.assertIn("tailscale)", text)
            self.assertNotIn("funnel", text.lower().replace("never funnel", ""))


if __name__ == "__main__":
    unittest.main()
