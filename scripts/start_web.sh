#!/usr/bin/env bash
# Mission control on the mock world. Usage: scripts/start_web.sh [lan]
#   (no arg) reachable from this PC only: http://localhost:8080
#   lan      reachable from the Wi-Fi (phone) -> also needs the Windows firewall rule, docs/phase7_phone.md
# Token: $GURDY_WEB_TOKEN if set, else a random one generated here and printed (never written to a file).
set -euo pipefail
cd "$(dirname "$0")/.."
export GURDY_WEB_TOKEN="${GURDY_WEB_TOKEN:-$(python3 -c 'import secrets;print(secrets.token_urlsafe(24))')}"
[ "${1:-}" = lan ] && export WEB_PUBLISH=0.0.0.0 || export WEB_PUBLISH=127.0.0.1
docker rm -f gurdy2-web >/dev/null 2>&1 || true
echo "open:   http://localhost:8080/#token=$GURDY_WEB_TOKEN"
[ "$WEB_PUBLISH" = 0.0.0.0 ] && echo "phone:  http://<PC Wi-Fi IPv4>:8080/#token=$GURDY_WEB_TOKEN"
echo "stop:   Ctrl+C (container gurdy2-web is removed)"
exec docker compose run --rm --service-ports --name gurdy2-web web
