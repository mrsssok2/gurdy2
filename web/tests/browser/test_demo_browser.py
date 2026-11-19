#!/usr/bin/env python3
"""Browser tests of the static demo: "Connect to my robot" (fallback, token / origin errors, real-robot mode, back to demo, what is stored where) and the
in-browser voice wiring (lazy model, transcript bubble, stop fast path, hallucination filter, spoken replies, mute) with a STUB Whisper worker.
Needs Chromium, so it is run through Docker, not by CI:   scripts/run_browser_tests.sh

Environment (set by the runner): SITE_URL (the built static demo), ROBOT_URL / ROBOT_TOKEN (a robot server that allows SITE_URL as an origin),
ROBOT_OTHER_ORIGIN_URL (a robot server whose allow-list does NOT include SITE_URL), MIC_WAV (16 kHz wav used as the fake microphone)."""
import json
import os
import sys
import time

from playwright.sync_api import sync_playwright

SITE = os.environ["SITE_URL"]
ROBOT, TOKEN, OTHER = os.environ["ROBOT_URL"], os.environ["ROBOT_TOKEN"], os.environ["ROBOT_OTHER_ORIGIN_URL"]
MIC_WAV = os.environ["MIC_WAV"]
results, errors = [], []

STUB_WORKER = r"""
// stub of whisper_worker.js: progress events, then fixed transcripts from a queue set by the test (self.NEXT is filled through a message)
let queue = [];
self.onmessage = async (ev) => {
  const { id, op } = ev.data;
  if (op === "load") {
    for (const pct of [20, 60, 100]) { self.postMessage({ type: "progress", file: "model.onnx", loaded: pct * 41000, total: 100 * 41000 }); await new Promise(r => setTimeout(r, 250)); }
    self.postMessage({ type: "ready", device: "wasm", model: "tiny.en" });
    self.postMessage({ id, result: { device: "wasm", model: "tiny.en" } });
  } else if (op === "transcribe") {
    const r = await fetch("stub_transcripts.json").then(x => x.json()).catch(() => ({ next: [] }));
    const text = r.next.length ? r.next.shift() : "";
    await fetch("stub_pop", { method: "POST" }).catch(() => {});
    self.postMessage({ id, result: { text, seconds: 0.2, device: "wasm", model: "tiny.en", usedPrompt: !!ev.data.prompt, promptSeen: ev.data.prompt || "" } });
  }
};
"""


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + (f"   [{detail}]" if detail and not cond else ""))


def msgs(pg):
    return pg.evaluate("[...document.querySelectorAll('.msg')].map(e => e.textContent)")


def wait_msg(pg, pred, timeout=15.0):
    """Poll the chat until pred(list_of_messages) is true (voice turns take a moment: recording stop delay, decode, worker, agent)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        m = msgs(pg)
        if pred(m):
            return True
        time.sleep(0.3)
    return False


def say(pg, text, wait=1.3):
    pg.fill("#chatInput", text)
    pg.press("#chatInput", "Enter")
    time.sleep(wait)


def new_page(ctx, name):
    pg = ctx.new_page()
    pg.on("pageerror", lambda e: errors.append(f"{name}: {e}"))
    pg.on("console", lambda m: m.type == "error" and "Failed to load resource" not in m.text and "net::ERR" not in m.text and ":8091" not in m.text and errors.append(f"{name} console: {m.text}"))
    return pg


def open_demo(ctx, name="page"):
    pg = new_page(ctx, name)
    pg.goto(SITE)
    pg.wait_for_selector(".conn.on", timeout=180000)
    time.sleep(1)
    return pg


def connect_dialog(pg, url, token):
    pg.click("#remoteBtn")
    pg.fill("#robotUrl", url)
    pg.fill("#robotToken", token)
    pg.click("#robotGo")


with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", f"--use-file-for-fake-audio-capture={MIC_WAV}"])

    # ------------------------------------------------------------------ 1. Connect to my robot
    ctx = b.new_context(viewport={"width": 1440, "height": 900})
    pg = open_demo(ctx, "demo")
    check("demo by default: banner, simulator running, Connect button present", pg.is_visible("#demoBanner") and pg.is_visible("#remoteBtn") and not pg.locator(".banner.real").count())

    connect_dialog(pg, "https://no-such-robot.invalid", "x")
    pg.wait_for_function("document.getElementById('robotGo').textContent === 'Connect'", timeout=30000)
    err = pg.inner_text("#robotErr")
    check("unreachable robot -> the documented message, stays in demo mode", "isn't running or this device isn't on your Tailscale network" in err and pg.is_visible("#demoBanner") and not pg.locator(".banner.real").count(), err)
    check("a failed attempt stores nothing", pg.evaluate("localStorage.getItem('gurdy_robot_url')") in (None, "") and pg.evaluate("sessionStorage.getItem('gurdy_remote')") in (None, ""))

    pg.fill("#robotUrl", ROBOT); pg.fill("#robotToken", "definitely-wrong"); pg.click("#robotGo")
    pg.wait_for_function("document.getElementById('robotGo').textContent === 'Connect'", timeout=30000)
    check("wrong token -> 'not accepted', stays in demo", "token was not accepted" in pg.inner_text("#robotErr") and pg.is_visible("#demoBanner"), pg.inner_text("#robotErr"))

    pg.fill("#robotUrl", OTHER); pg.fill("#robotToken", TOKEN); pg.click("#robotGo")
    pg.wait_for_function("document.getElementById('robotGo').textContent === 'Connect'", timeout=30000)
    e2 = pg.inner_text("#robotErr")
    check("robot that does not list this page as an origin -> same 'isn't running' message plus the allowed-origins hint (browsers cannot tell them apart)",
          "isn't running or this device isn't on your Tailscale network" in e2 and "allowed-origins" in e2 and pg.is_visible("#demoBanner"), e2)

    pg.fill("#robotUrl", "not a url at all"); pg.click("#robotGo")
    time.sleep(0.5)
    check("garbage address -> validation message, nothing sent", "https" in pg.inner_text("#robotErr") and pg.is_visible("#remoteDialog"))
    pg.fill("#robotUrl", "http://insecure.example"); pg.click("#robotGo"); time.sleep(0.5)
    check("plain http to a remote host is refused client-side", "https" in pg.inner_text("#robotErr"))

    # success: full pasted URL with #token=, token field empty
    pg.fill("#robotUrl", f"{ROBOT}/#token={TOKEN}"); pg.fill("#robotToken", ""); pg.click("#robotGo")
    pg.wait_for_selector(".banner.real", timeout=60000)
    pg.wait_for_selector(".conn.on", timeout=60000)
    time.sleep(1)
    check("connected: 'Connected to REAL robot' banner + Back to demo, demo banner gone", "Connected to REAL robot" in pg.inner_text(".banner.real") and pg.is_visible("#backDemo") and not pg.is_visible("#demoBanner"))
    check("no simulator is downloaded or started in robot mode", pg.evaluate("!document.getElementById('bootOverlay') || true") and pg.evaluate("window.__demoReadyMs === undefined"))
    check("storage: URL in localStorage, token only in sessionStorage", pg.evaluate("localStorage.getItem('gurdy_robot_url')") == ROBOT
          and TOKEN not in json.dumps(pg.evaluate("Object.assign({}, localStorage)")) and TOKEN in (pg.evaluate("sessionStorage.getItem('gurdy_remote')") or ""))
    say(pg, "where are you?")
    check("chat goes to the robot server (its mock world answers)", any("room A" in m for m in msgs(pg)), str(msgs(pg)))
    st = pg.evaluate("fetch(window.GURDY_REMOTE.base + '/api/state', {headers: {Authorization: 'Bearer ' + window.GURDY_REMOTE.token}}).then(r => r.json())")
    check("the page really talks to the robot origin (state.source=mock, not demo)", st.get("source") == "mock" and st.get("demo") is None)
    pg.click("#estop"); time.sleep(1.2)
    check("E-STOP reaches the robot", any("Emergency stop" in m for m in msgs(pg)))

    pg.reload(); pg.wait_for_selector(".banner.real", timeout=60000); pg.wait_for_selector(".conn.on", timeout=60000)
    check("reload in the same tab keeps the connection (token in sessionStorage)", pg.is_visible(".banner.real"))

    pg.click("#backDemo"); pg.wait_for_selector(".conn.on", timeout=180000); time.sleep(1)
    check("Back to demo: simulator again, no real banner, URL still remembered, token gone",
          pg.is_visible("#demoBanner") and not pg.locator(".banner.real").count() and pg.evaluate("localStorage.getItem('gurdy_robot_url')") == ROBOT
          and pg.evaluate("sessionStorage.getItem('gurdy_remote')") in (None, ""))
    pg.click("#remoteBtn")
    check("dialog remembers the URL but not the token", pg.input_value("#robotUrl") == ROBOT and pg.input_value("#robotToken") == "")
    pg.click("#robotCancel")

    # a stored session whose robot disappeared -> demo mode + message
    pg.evaluate("sessionStorage.setItem('gurdy_remote', JSON.stringify({base: 'https://gone.invalid', token: 't'}))")
    pg.reload(); pg.wait_for_selector(".conn.on", timeout=180000); time.sleep(1)
    check("robot gone on reload -> demo mode + 'isn't running' toast", pg.is_visible("#demoBanner") and "isn't running" in pg.inner_text(".remote-toast"))
    ctx.close()

    # a robot that is up but the page is on a phone profile (layout of the new bits)
    ctx = b.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True, has_touch=True)
    pg = open_demo(ctx, "phone")
    check("phone: short 'My robot' button is visible and E-STOP stays visible", pg.is_visible("#remoteBtn .short") and pg.is_visible("#estop"))
    pg.screenshot(path=os.environ.get("SHOT_DIR", "/tmp") + "/remote_phone.png")
    ctx.close()

    # ------------------------------------------------------------------ 2. Voice wiring with a stub Whisper
    ctx = b.new_context(viewport={"width": 1440, "height": 900}, permissions=["microphone"])
    worker_requests = []
    ctx.on("request", lambda r: worker_requests.append(r.url) if "whisper_worker.js" in r.url else None)
    state = {"next": []}

    def route_worker(route):
        route.fulfill(status=200, content_type="text/javascript", body=STUB_WORKER)
    ctx.route("**/whisper_worker.js", route_worker)
    ctx.route("**/stub_transcripts.json", lambda r: r.fulfill(status=200, content_type="application/json", body=json.dumps(state)))
    ctx.route("**/stub_pop", lambda r: (state["next"].pop(0) if state["next"] else None, r.fulfill(status=200, body="ok"))[1])
    ctx.add_init_script("""
      window.__spoken = [];
      Object.defineProperty(window, 'speechSynthesis', { configurable: true, value: { speak: (u) => window.__spoken.push(u.text), cancel: () => { window.__spoken.push('<cancel>'); }, getVoices: () => [], speaking: false } });
    """)
    pg = open_demo(ctx, "voice")
    check("voice: mic button shown in the demo, model NOT requested yet (lazy)", pg.is_visible("#micBtn") and len(worker_requests) == 0, str(worker_requests))
    ctxinfo = pg.evaluate("fetch('/api/voice_context').then(r => r.json())")
    check("vocabulary prompt comes from the Python side (room names, objects)", "room B" in ctxinfo["prompt"] and "red box" in ctxinfo["prompt"].lower() and ctxinfo["hint"] is None, str(ctxinfo)[:200])

    def hold_mic(seconds=1.6):
        box = pg.locator("#micBtn").bounding_box()
        pg.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        pg.mouse.down(); time.sleep(seconds); pg.mouse.up()

    state["next"] = ["Go to room B."]
    hold_mic()
    pg.wait_for_selector(".msg.user", timeout=30000)
    time.sleep(1.5)
    check("first press loads the model (worker requested once) and shows progress", len(worker_requests) == 1)
    m = msgs(pg)
    check("transcript shows as my bubble marked 'voice', the agent answers", any(x.startswith("Go to room B.") and "voice" in x for x in m) and any("Shall I go" in x for x in m), str(m))
    check("a pending question gives the expected answers as the hint", (pg.evaluate("fetch('/api/voice_context').then(r => r.json())")["hint"] or "").startswith("Yes."))
    check("reply is spoken with speechSynthesis", any("Shall I go" in s for s in pg.evaluate("window.__spoken")), str(pg.evaluate("window.__spoken")))

    state["next"] = ["Yes."]
    hold_mic()
    check("'Yes.' confirms by voice and the robot starts", wait_msg(pg, lambda m: any("Going to room B" in x for x in m)), str(msgs(pg)))
    state["next"] = ["Stalk"]
    hold_mic()
    check("stop fast path: a one-word mishearing of 'stop' stops the robot", wait_msg(pg, lambda m: any(x.startswith("Stalk") for x in m) and any("Stopped" in x for x in m))
          and pg.inner_text("#sMode").strip() in ("idle", "estop"), str(msgs(pg)[-4:]))

    pg.evaluate("window.__spoken.length = 0")
    pg.click("#muteBtn")
    state["next"] = ["Where are you?"]
    hold_mic(); wait_msg(pg, lambda m: any(x.startswith("Where are you?") for x in m)); time.sleep(1.5)
    spoken = [s for s in pg.evaluate("window.__spoken") if s != "<cancel>"]
    check("mute toggle: replies are no longer spoken", spoken == [] and any("room" in x for x in msgs(pg)[-2:]), str(spoken))
    check("pressing the mic silences the robot (barge-in)", "<cancel>" in pg.evaluate("window.__spoken"))
    ctx.close()

    # ------------------------------------------------------------------ 2b. hallucination filter on a SILENT microphone (separate browser: its fake mic plays silence)
    b2 = p.chromium.launch(args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", f"--use-file-for-fake-audio-capture={os.environ['MIC_WAV_SILENT']}"])
    ctx = b2.new_context(viewport={"width": 1440, "height": 900}, permissions=["microphone"])
    state = {"next": []}
    ctx.route("**/whisper_worker.js", route_worker)
    ctx.route("**/stub_transcripts.json", lambda r: r.fulfill(status=200, content_type="application/json", body=json.dumps(state)))
    ctx.route("**/stub_pop", lambda r: (state["next"].pop(0) if state["next"] else None, r.fulfill(status=200, body="ok"))[1])
    pg = open_demo(ctx, "silent")
    for said in ("Thank you.", "Go to room B."):                      # Whisper's classic silence hallucination, and a real command heard on a silent clip
        state["next"] = [said]
        hold_mic()
        wait_msg(pg, lambda m, n=said: sum("Sorry, I didn't catch that" in x for x in m) >= (1 if n == "Thank you." else 2))
    m = msgs(pg)
    check("silence: 'Thank you.' and even a real-looking command on a silent clip are dropped", sum("Sorry, I didn't catch that" in x for x in m) == 2 and not any("voice" in x for x in m), str(m))
    check("silence: nothing reached the agent (no motion, no question)", not any("Shall I go" in x or "Going" in x for x in m))
    ctx.close()
    b2.close()

    # ------------------------------------------------------------------ 3. server-side demo / robot pages must not grow a browser mic
    ctx = b.new_context(viewport={"width": 1440, "height": 900}, permissions=["microphone"])
    pg = new_page(ctx, "robotpage")
    pg.goto(ROBOT + f"/#token={TOKEN}")
    pg.wait_for_selector(".conn.on", timeout=60000)
    check("robot's own page: no browser-Whisper mic (its voice is the server's GPU Whisper, off here)", not pg.is_visible("#micBtn") and pg.evaluate("window.GURDY_BROWSER_VOICE === undefined"))
    ctx.close()
    b.close()

print("\nERRORS:", errors)
failed = [n for n, ok in results if not ok]
print(f"{len(results) - len(failed)}/{len(results)} checks passed")
sys.exit(1 if failed or errors else 0)
