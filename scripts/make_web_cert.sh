#!/bin/bash
# Local HTTPS for the dashboard (the phone's microphone needs a secure context). Makes a small private CA once and a server certificate for
# the addresses you give. Everything stays in ~/.gurdy/tls (never in git).
#   scripts/make_web_cert.sh 192.168.137.1 192.168.1.23     # IPs / host names the phone will use (laptop Wi-Fi / hotspot address)
# Re-run it when the laptop's address changes: the CA stays the same, so the phone keeps trusting it; only gurdy.crt is replaced.
# Files: ca.crt (PUBLIC, install it on the phone), ca.key (private, never copy it), gurdy.crt + gurdy.key (the server).
set -euo pipefail
D="$HOME/.gurdy/tls"; mkdir -p "$D"; chmod 700 "$D"; cd "$D"
if [ ! -f ca.key ]; then
  openssl genrsa -out ca.key 3072 2>/dev/null; chmod 600 ca.key
  openssl req -x509 -new -nodes -key ca.key -sha256 -days 3650 -subj "/CN=Gurdy local CA" \
    -addext "basicConstraints=critical,CA:TRUE" -addext "keyUsage=critical,keyCertSign,cRLSign" -out ca.crt
  echo "created the local CA: $D/ca.crt"
fi
SAN="DNS:localhost,IP:127.0.0.1"
for a in "$@"; do
  if [[ "$a" =~ ^[0-9.]+$ ]]; then SAN="$SAN,IP:$a"; else SAN="$SAN,DNS:$a"; fi
done
openssl genrsa -out gurdy.key 2048 2>/dev/null; chmod 644 gurdy.key      # the container user must be able to read it; the CA key stays 600
openssl req -new -key gurdy.key -subj "/CN=gurdy" -out gurdy.csr
printf "subjectAltName=%s\nbasicConstraints=CA:FALSE\nkeyUsage=digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\n" "$SAN" > gurdy.ext
openssl x509 -req -in gurdy.csr -CA ca.crt -CAkey ca.key -CAcreateserial -sha256 -days 397 -extfile gurdy.ext -out gurdy.crt 2>/dev/null
rm -f gurdy.csr gurdy.ext
echo "server certificate for: $SAN  (valid 397 days)"
echo "phone: install $D/ca.crt (docs/phase7_voice.md)"
