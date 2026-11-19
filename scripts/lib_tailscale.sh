# Source me. Private remote access through Tailscale Serve (HTTPS with a valid *.ts.net certificate, tailnet only, NEVER Funnel).
# The Tailscale client runs on WINDOWS (docs/phase7_remote.md explains why); from WSL this calls tailscale.exe through Windows interop.
#   ts_check          -> finds the client, checks it is logged in and running, sets TS_HOST (this PC's name on the tailnet); returns 1 with a hint otherwise
#   ts_start [port]   -> `serve --bg --https=443 http://127.0.0.1:<port>` (default 8080), sets TS_URL (https://<host>/)
#   ts_stop           -> `serve --https=443 off` (only our serve entry; other serve settings stay)
# Override the client with TAILSCALE_BIN=... (tests use a stub).
_ts() { "$TS_BIN" "$@" 2>&1 | tr -d '\r'; return "${PIPESTATUS[0]}"; }

ts_find() {
  if [ -n "${TAILSCALE_BIN:-}" ]; then TS_BIN=$TAILSCALE_BIN
  elif command -v tailscale.exe >/dev/null 2>&1; then TS_BIN=$(command -v tailscale.exe)
  elif [ -x "/mnt/c/Program Files/Tailscale/tailscale.exe" ]; then TS_BIN="/mnt/c/Program Files/Tailscale/tailscale.exe"
  elif command -v tailscale >/dev/null 2>&1; then TS_BIN=$(command -v tailscale)
  else return 1; fi
}

ts_check() {
  ts_find || { echo "Tailscale not found. Install it on Windows first (docs/phase7_remote.md, step 2)."; return 1; }
  local js
  js=$(_ts status --json) || { echo "tailscale did not answer: $js"; echo "Is the Tailscale app running on Windows and logged in? (docs/phase7_remote.md)"; return 1; }
  TS_HOST=$(printf '%s' "$js" | python3 -c '
import sys, json
d = json.load(sys.stdin)
if d.get("BackendState") != "Running":
    print("STATE:" + str(d.get("BackendState")))
else:
    print((d.get("Self") or {}).get("DNSName", "").rstrip("."))') || { echo "could not read tailscale status"; return 1; }
  case "$TS_HOST" in
    STATE:*) echo "Tailscale is not connected (${TS_HOST#STATE:}). Open the Tailscale tray icon on Windows, log in / Connect."; return 1 ;;
    "")      echo "Tailscale gave no machine name. Enable MagicDNS + HTTPS certificates (docs/phase7_remote.md, step 3)."; return 1 ;;
  esac
}

ts_start() {
  local port=${1:-8080} out
  out=$(_ts serve --bg --https=443 "http://127.0.0.1:$port") || {
    echo "tailscale serve failed:"; echo "$out"
    echo "If it mentions HTTPS certificates: enable them in the admin console (docs/phase7_remote.md, step 3)."; return 1; }
  # the first run may print a link asking you to enable HTTPS: show anything that is not the normal confirmation
  printf '%s\n' "$out" | grep -qi "https://login.tailscale.com\|not enabled\|enable" && { echo "$out"; }
  TS_URL="https://$TS_HOST/"
}

ts_stop() { [ -n "${TS_BIN:-}" ] && _ts serve --https=443 off >/dev/null 2>&1 || true; }
