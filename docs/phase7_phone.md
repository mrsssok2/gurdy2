# Opening mission control on your phone (same Wi-Fi) — and "Add to Home Screen"

Status (2026-11-18): **written from documentation and knowledge of WSL2 / Docker Desktop; NOT tested on a real phone yet** (step 6 of `docs/phase7_design.md` is the test). If a command differs on your Windows build, tell me and I fix this file.

You need: phone and PC on the **same Wi-Fi**, Windows 11, the server running, and its access token (printed when the server starts).

## 1. Start the server so the LAN can reach it

Run it on `0.0.0.0` (all interfaces), not the default `127.0.0.1`.

**Recommended (container, Docker Desktop publishes the port on Windows for you):**
```bash
docker compose run --rm --service-ports -p 8080:8080 brain python3 -m web.api --mock --bind 0.0.0.0
```
Docker Desktop listens on the Windows side of port 8080 itself, so you only need the firewall rule in section 3 (no port forwarding).
Note the printed `access token: ...`.

*(Once step 4 lands there will be a `web` compose profile and `scripts/start_web.sh`; same idea.)*

**If you run the server directly inside WSL** (not in Docker), WSL2 needs one of the two options in section 2.

## 2. Only for a server running directly in WSL2 (skip if you used Docker above)

### Option A — port forwarding (works on every Windows 11 / WSL2 setup)
In WSL: `hostname -I` → the first address is the WSL IP (it **changes after every WSL restart**).
In **PowerShell as Administrator** (replace `172.x.x.x`):
```powershell
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=8080 connectaddress=172.x.x.x connectport=8080
netsh interface portproxy show v4tov4
```
After a WSL restart: `netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=8080` and add it again with the new IP.

### Option B — mirrored networking (WSL 2.0+, Windows 11 22H2+): no forwarding, no changing IP
Edit `C:\Users\<you>\.wslconfig` and add under `[wsl2]` (keep your memory / swap lines):
```ini
[wsl2]
networkingMode=mirrored
```
Then in PowerShell: `wsl --shutdown`, reopen WSL. Allow inbound traffic to WSL through the Hyper-V firewall (Administrator PowerShell, once):
```powershell
Set-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B0B0A4D0B0A}' -DefaultInboundAction Allow
```
Risk: mirrored mode changes how WSL networking behaves (VPNs, Docker Desktop). If something else breaks, remove the line and `wsl --shutdown`.

## 3. Windows firewall rule (needed in every case)
Administrator PowerShell — allows port 8080 **on private networks only**:
```powershell
New-NetFirewallRule -DisplayName "Gurdy web 8080" -Direction Inbound -Protocol TCP -LocalPort 8080 -Action Allow -Profile Private
```
Your Wi-Fi must be classified **Private** (Settings → Network & internet → Wi-Fi → your network → Network profile type). A "Public" profile blocks the rule on purpose.
Remove it when you are done: `Remove-NetFirewallRule -DisplayName "Gurdy web 8080"` (and the portproxy: `netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=8080`).

## 4. Open it on the phone
1. PC's Wi-Fi address: PowerShell `ipconfig` → "Wireless LAN adapter Wi-Fi" → IPv4 (e.g. `192.168.1.23`).
2. First check from the PC: `http://localhost:8080/api/health` shows `{"ok": true, ...}`.
3. Phone browser: `http://192.168.1.23:8080/api/health` — same JSON means the network path works.
4. Then `http://192.168.1.23:8080/#token=YOUR_TOKEN` (the part after `#` is never sent to the server and the page removes it from the address bar after saving it). Or open without it and paste the token in the login box.

Nothing loads? In order: (a) Wi-Fi profile is Private; (b) the firewall rule exists; (c) the router has no "AP / client isolation" (guest networks usually do); (d) the portproxy points at the *current* WSL IP; (e) the server was started with `--bind 0.0.0.0`.

## 5. Add to Home Screen
- **iPhone (Safari):** Share button → *Add to Home Screen* → Add. It opens full-screen with the Gurdy icon and the dark status bar. Must be Safari, not Chrome.
- **Android (Chrome):** ⋮ menu → *Add to Home screen* (or *Install app* when offered).

**Plain http on a LAN is not a "secure context"**, so browsers will not run the service worker there (and Chrome may add only a shortcut that opens in a tab). Consequences: on iPhone the home-screen app still works (full-screen), but there is **no offline start**; the page needs the PC to be reachable to open. For the full installable app + offline start you need https: either
- Android Chrome only: `chrome://flags/#unsafely-treat-insecure-origin-as-secure` → add `http://192.168.1.23:8080` → enable → relaunch; or
- a tunnel with https (see `docs/phase7_hosting.md`).

Also: the PC's Wi-Fi IP is part of the app's identity. If the router hands out a new address the home-screen app points at the old one — set a **DHCP reservation** for the PC in the router, or re-add the shortcut.

## 6. Safety notes
- The token travels in clear text over plain http. Fine on your home Wi-Fi for a simulated robot; do not expose this port to the internet. Stop the server and remove the firewall rule when you are not using it.
- The big red **E-STOP** is in the top bar on every tab. With a real robot it is a convenience; the voice stop and the physical power switch stay the primary stops.
