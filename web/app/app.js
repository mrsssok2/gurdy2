import { makeView, fmtTime, fmtNum, batteryClass, legSummary, newAlerts, wsUrl, backoff, prettyName, colorHex, objectLabel, placeLabels, angDiff } from "./lib.js";

const $ = (id) => document.getElementById(id);
const C = { bg: "#0B0E13", room: "#121821", wall: "#3A4552", obstacle: "#4B5563", zone: "#D93036", accent: "#2DD4BF", text: "#E6EDF3", muted: "#8B98A5" };

const S = { demo: false, token: "", ws: null, attempt: 0, timer: null, map: null, state: null, lastSeq: 0, seen: new Set(), history: [], tab: "map", disp: null };

function store(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode: fine */ } }
function load(k) { try { return localStorage.getItem(k) || ""; } catch { return ""; } }

// ---------------------------------------------------------------- connection
function setConn(kind, text) { $("conn").className = "conn " + kind; $("connText").textContent = text; }

async function api(path, opts = {}) {
  const auth = S.demo ? {} : { Authorization: "Bearer " + S.token };      // the public demo has no token
  const r = await fetch(path, { ...opts, headers: { ...auth, "Content-Type": "application/json", ...(opts.headers || {}) } });
  if (r.status === 401) throw Object.assign(new Error("unauthorized"), { code: 401 });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
  return r.json();
}

async function connect() {
  clearTimeout(S.timer);
  try {                                                // public, needs no token: tells us whether this is the demo server
    S.demo = !!(await (await fetch("/api/health")).json()).demo;
    $("demoBanner").hidden = !S.demo;
  } catch { return retry("server not reachable"); }
  if (!S.token && !S.demo) return askToken();
  try {
    S.map = await api("/api/map");                     // also validates the token (a WebSocket cannot report a 401 to the page)
  } catch (e) {
    if (e.code === 401) return askToken("That token was not accepted.");
    return retry("server not reachable");
  }
  const ws = new WebSocket(wsUrl(location, S.demo ? "" : S.token));
  S.ws = ws;
  ws.onopen = () => { S.attempt = 0; setConn("on", "connected"); };
  ws.onmessage = (ev) => { try { onMessage(JSON.parse(ev.data)); } catch { /* ignore malformed */ } };
  ws.onclose = () => { if (S.ws === ws) { S.ws = null; retry("disconnected"); } };
  ws.onerror = () => ws.close();
}

function retry(text) {
  setConn("off", text);
  S.timer = setTimeout(connect, backoff(S.attempt++));
}

function onMessage(m) {
  if (m.type === "hello") { S.map = m.map; S.lastSeq = 0; S.seen.clear(); S.history = []; $("chatList").replaceChildren(); m.chat.forEach(addChat); }   // also sent after a demo reset
  else if (m.type === "state") onState(m);
  else if (m.type === "chat") addChat(m);
  else if (m.type === "error") addChat({ seq: -1, role: "system", text: m.error });
}

function send(obj) {
  if (S.ws && S.ws.readyState === WebSocket.OPEN) { S.ws.send(JSON.stringify(obj)); return true; }
  return false;
}

// ---------------------------------------------------------------- e-stop and chat
async function estop() {
  if (send({ op: "estop" })) return;
  try { await api("/api/estop", { method: "POST", body: "{}" }); } catch (e) { addChat({ seq: -1, role: "system", text: "E-stop failed to send: " + e.message }); }
}

async function say(text) {
  if (send({ op: "say", text })) return;
  try { await api("/api/say", { method: "POST", body: JSON.stringify({ text }) }); await pollChat(); } catch (e) { addChat({ seq: -1, role: "system", text: "Not sent: " + e.message }); }
}

async function pollChat() {
  const r = await api("/api/messages?since=" + S.lastSeq);
  r.messages.forEach(addChat);
}

function addChat(m) {
  if (m.seq > 0) { if (m.seq <= S.lastSeq) return; S.lastSeq = m.seq; }
  const el = document.createElement("div");
  el.className = "msg " + (["user", "robot", "system"].includes(m.role) ? m.role : "robot");
  el.textContent = m.text;                              // textContent only: robot / alert text is never parsed as HTML
  const list = $("chatList");
  const atEnd = list.scrollHeight - list.scrollTop - list.clientHeight < 80;
  list.appendChild(el);
  while (list.childElementCount > 300) list.firstChild.remove();
  if (atEnd || m.role === "user") list.scrollTop = list.scrollHeight;
}

// ---------------------------------------------------------------- state -> UI
function onState(st) {
  S.state = st;
  if (st.connected === false) { setConn("off", st.error ? "robot source error" : "no robot"); return; }
  setConn(st.stale ? "stale" : "on", st.stale ? "busy" : "connected");
  $("staleBanner").hidden = !st.stale;
  $("estopBanner").hidden = !st.estop;
  const p = st.pose || {};
  $("px").textContent = fmtNum(p.x, 2, " m"); $("py").textContent = fmtNum(p.y, 2, " m");
  $("ph").textContent = fmtNum(p.heading_deg, 0, "°"); $("pplace").textContent = p.place || "–";
  $("sMode").textContent = st.mode || "–";
  $("sTask").textContent = st.task ? st.task.tool : "none";
  $("sSpeed").textContent = fmtNum(st.speed_mps, 2, " m/s");
  $("sTime").textContent = fmtTime(st.sim_time_s);
  const b = st.battery_pct;
  $("sBat").textContent = fmtNum(b, 0, " %");
  $("sBatBar").style.width = Math.max(0, Math.min(100, b || 0)) + "%";
  $("sBatBar").className = batteryClass(b);
  $("sSim").hidden = !st.battery_simulated;
  $("sLegSummary").textContent = legSummary(st.legs);
  $("tBat").textContent = fmtNum(b, 0, "%"); $("tBatBar").style.width = Math.max(0, Math.min(100, b || 0)) + "%"; $("tBatBar").className = batteryClass(b); $("tSim").hidden = !st.battery_simulated;
  $("tLegText").textContent = legSummary(st.legs);
  renderLegs(st.legs || {});
  renderObjects(st.objects || []);
  renderAlerts(st.alerts || []);
  kick();
}

function renderLegs(legs) {
  const box = $("legs");
  box.replaceChildren(...Object.entries(legs).map(([name, state]) => {
    const d = document.createElement("div"); d.className = "leg " + state;
    const dot = document.createElement("span"); dot.className = "dot";
    const n = document.createElement("span"); n.textContent = prettyName(name);
    const s = document.createElement("span"); s.className = "st"; s.textContent = state;
    d.append(dot, n, s); return d;
  }));
  $("tLegs").replaceChildren(...Object.entries(legs).map(([name, state]) => {
    const d = document.createElement("span"); d.className = "dot " + state; d.title = `${prettyName(name)}: ${state}`; return d;
  }));
}

function renderObjects(objs) {
  $("sObjCount").textContent = String(objs.length);
  const ul = $("objList");
  if (!objs.length) { ul.replaceChildren(emptyLi("Nothing seen yet.")); return; }
  ul.replaceChildren(...objs.map((o) => {
    const li = document.createElement("li");
    const sw = document.createElement("span"); sw.className = "sw"; sw.style.background = colorHex(o.color);
    const t = document.createElement("span"); t.textContent = `${objectLabel(o)}${o.place ? " · " + o.place : ""}`;
    const w = document.createElement("span"); w.className = "when"; w.textContent = o.stale ? "stale" : fmtTime(o.age_s) + " ago";
    li.append(sw, t, w); return li;
  }));
}

function emptyLi(text) { const li = document.createElement("li"); li.className = "empty"; li.textContent = text; return li; }

function renderAlerts(active) {
  for (const t of [...S.seen]) if (!active.includes(t)) S.seen.delete(t);       // a cleared alert may fire again later
  S.history.unshift(...newAlerts(active, S.seen, new Date()));
  if (S.history.length > 50) S.history.length = 50;
  $("aCount").textContent = String(active.length);
  const badge = $("alertBadge"); badge.hidden = active.length === 0; badge.textContent = String(active.length);
  const item = (a, when) => {
    const li = document.createElement("li"); if (/fail|fell|fall|critical|stuck/i.test(a.text)) li.className = "crit";
    const t = document.createElement("span"); t.textContent = a.text;
    const w = document.createElement("span"); w.className = "when"; w.textContent = when;
    li.append(t, w); return li;
  };
  const hhmm = (d) => d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  $("alertNow").replaceChildren(...(active.length ? active.map((t) => item({ text: t }, "now")) : [emptyLi("No active alerts.")]));
  const old = S.history.filter((a) => !active.includes(a.text));
  $("alertHist").replaceChildren(...(old.length ? old.map((a) => item(a, hhmm(a.at))) : [emptyLi("Nothing earlier this session.")]));
}

// ---------------------------------------------------------------- map
const FONT = "13px Inter, system-ui, sans-serif";

/** Small shape per object class, centred on (x, y), in the object's own colour. `s` = half size in px. */
function drawObject(g, o, x, y, s) {
  const col = colorHex(o.color);
  g.save();
  g.globalAlpha = o.stale ? 0.45 : 1;
  g.fillStyle = col; g.strokeStyle = "#0B0E13"; g.lineWidth = 1.5;
  switch (o.class) {
    case "barrel": g.beginPath(); g.arc(x, y, s, 0, 6.2832); g.fill(); g.stroke(); g.beginPath(); g.arc(x, y, s * 0.5, 0, 6.2832); g.strokeStyle = "#0B0E1388"; g.stroke(); break;
    case "traffic_cone": g.beginPath(); g.moveTo(x, y - s * 1.1); g.lineTo(x + s, y + s * 0.8); g.lineTo(x - s, y + s * 0.8); g.closePath(); g.fill(); g.stroke(); break;
    case "fire_extinguisher": g.beginPath(); g.roundRect(x - s * 0.55, y - s, s * 1.1, s * 2, s * 0.4); g.fill(); g.stroke(); break;
    case "toolbox": g.beginPath(); g.roundRect(x - s * 1.2, y - s * 0.65, s * 2.4, s * 1.3, 3); g.fill(); g.stroke(); break;
    case "mannequin": g.beginPath(); g.arc(x, y - s * 0.55, s * 0.5, 0, 6.2832); g.fill(); g.stroke(); g.beginPath(); g.roundRect(x - s * 0.7, y, s * 1.4, s * 1.1, 4); g.fill(); g.stroke(); break;
    case "chair": g.beginPath(); g.roundRect(x - s * 0.8, y - s * 0.2, s * 1.6, s * 1.0, 3); g.fill(); g.stroke(); g.fillRect(x - s * 0.8, y - s * 1.0, s * 1.6, s * 0.4); break;
    case "box": default: g.beginPath(); g.roundRect(x - s, y - s, s * 2, s * 2, 3); g.fill(); g.stroke();
  }
  if (o.stale) { g.setLineDash([3, 3]); g.strokeStyle = col; g.strokeRect(x - s - 3, y - s - 3, 2 * s + 6, 2 * s + 6); }
  g.restore();
}

function drawRobot(g, x, y, th, R) {
  g.save();
  g.translate(x, y);
  // soft glow ring
  const grad = g.createRadialGradient(0, 0, R * 0.4, 0, 0, R * 3.2);
  grad.addColorStop(0, "#2DD4BF66"); grad.addColorStop(1, "#2DD4BF00");
  g.fillStyle = grad; g.beginPath(); g.arc(0, 0, R * 3.2, 0, 6.2832); g.fill();
  g.rotate(th);
  g.shadowColor = "#2DD4BF"; g.shadowBlur = 14;
  g.fillStyle = "#2DD4BF"; g.strokeStyle = "#062B27"; g.lineWidth = 2;
  g.beginPath(); g.moveTo(R * 1.5, 0); g.lineTo(-R * 0.95, R * 0.95); g.lineTo(-R * 0.45, 0); g.lineTo(-R * 0.95, -R * 0.95); g.closePath(); g.fill();
  g.shadowBlur = 0; g.stroke();
  g.restore();
}

function drawMap() {
  S.raf = 0;
  const cv = $("map"), dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (!w || !h || !S.map) return;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.fillStyle = C.bg; g.fillRect(0, 0, w, h);
  const pad = Math.max(14, Math.min(w, h) * 0.04);
  const v = makeView(S.map.bounds, w, h, pad, true);
  const rect = v.rect;
  g.font = FONT; g.textBaseline = "middle";
  const measure = (t) => g.measureText(t).width;
  const LH = 17;
  const taken = [], labels = [];
  for (const [name, r] of Object.entries(S.map.rooms || {})) {          // floor + room titles (placed first, so they win their corner)
    g.fillStyle = C.room; g.fillRect(...rect(r));
    g.strokeStyle = "#1B2430"; g.lineWidth = 1; g.strokeRect(...rect(r));
    const t = prettyName(name).toUpperCase(), rr = rect(r), box = { x: rr[0] + 10, y: rr[1] + 8, w: measure(t) + 2, h: LH };
    taken.push(box); labels.push({ t, box, room: true });
  }
  g.fillStyle = C.wall; for (const r of S.map.walls || []) g.fillRect(...rect(r));
  g.fillStyle = C.obstacle; for (const r of S.map.obstacles || []) g.fillRect(...rect(r));
  const st = S.state || {};
  for (const z of st.zones || []) {                                   // no-go zones
    const zp = v.P(z.x, z.y); g.beginPath(); g.arc(zp.x, zp.y, z.r * v.s, 0, 2 * Math.PI);
    g.fillStyle = "#D9303622"; g.fill(); g.setLineDash([6, 4]); g.strokeStyle = C.zone; g.lineWidth = 1.5; g.stroke(); g.setLineDash([]);
  }
  // trail (fades towards the old end), then planned path (dashed, with a goal ring)
  const tr = st.trail || [];
  if (tr.length > 1) {
    g.lineWidth = 3; g.lineCap = "round"; g.lineJoin = "round";
    const n = tr.length, step = Math.max(1, Math.floor(n / 40));
    for (let i = step; i < n; i += step) {
      const a = v.P(...tr[i - step]), b = v.P(...tr[i]); g.beginPath(); g.moveTo(a.x, a.y); g.lineTo(b.x, b.y);
      g.strokeStyle = `rgba(45,212,191,${0.15 + 0.6 * (i / n)})`; g.stroke();
    }
  }
  const D = S.disp, path = st.path || [];
  if (path.length > 1) {
    const p0 = D ? v.P(D.x, D.y) : v.P(...path[0]);
    g.beginPath(); g.moveTo(p0.x, p0.y);
    for (let i = 1; i < path.length; i++) { const q = v.P(...path[i]); g.lineTo(q.x, q.y); }
    g.setLineDash([8, 6]); g.lineWidth = 2.5; g.strokeStyle = "#FACC15CC"; g.stroke(); g.setLineDash([]);
    const e = v.P(...path[path.length - 1]);
    g.beginPath(); g.arc(e.x, e.y, 9, 0, 6.2832); g.strokeStyle = "#FACC15"; g.lineWidth = 2; g.stroke();
    g.beginPath(); g.arc(e.x, e.y, 3, 0, 6.2832); g.fillStyle = "#FACC15"; g.fill();
  }
  // markers first, then labels laid out so none overlap and all stay inside the map
  const items = [];
  const placesSeen = new Set(Object.keys(S.map.rooms || {}).map((n) => n.toLowerCase()));
  for (const [name, p] of Object.entries(S.map.places || {})) {
    if (name === "doorway") continue;
    const pp = v.P(p.x, p.y);
    g.fillStyle = C.muted; g.beginPath(); g.arc(pp.x, pp.y, 3, 0, 6.2832); g.fill();
    if (!placesSeen.has(name.toLowerCase())) items.push({ text: prettyName(name), ax: pp.x, ay: pp.y, r: 3, kind: "place" });
  }
  const sz = Math.max(6, Math.min(11, 0.16 * v.s));
  for (const o of st.objects || []) {
    const op = v.P(o.x, o.y);
    drawObject(g, o, op.x, op.y, sz);
    items.push({ text: objectLabel(o), ax: op.x, ay: op.y, r: sz, kind: "obj" });
  }
  const RR = Math.max(13, 0.2 * v.s), rp = D ? v.P(D.x, D.y) : null;
  if (D) items.push({ text: "Gurdy", ax: rp.x, ay: rp.y, r: RR * 1.9, kind: "robot" });
  for (const l of labels) { g.fillStyle = "#6B7886"; g.textAlign = "left"; g.font = "600 12px 'Space Grotesk', Inter, sans-serif"; g.fillText(l.t, l.box.x, l.box.y + LH / 2); }
  g.font = FONT;
  const markers = items.map((i) => { const r = i.kind === "robot" ? RR * 1.6 : i.r; return { x: i.ax - r, y: i.ay - r, w: 2 * r, h: 2 * r }; });
  for (const l of placeLabels(items, w, h, measure, LH, taken.concat(markers))) {
    const kind = items.find((i) => i.text === l.text && i.ax === l.ax)?.kind;
    g.textAlign = "left";
    g.fillStyle = "#0B0E13BB"; g.beginPath(); g.roundRect(l.x - 3, l.y - 1, l.w + 6, l.h + 2, 5); g.fill();
    g.fillStyle = kind === "robot" ? C.accent : kind === "obj" ? C.text : C.muted;
    g.fillText(l.text, l.x, l.y + l.h / 2);
  }
  if (D) drawRobot(g, rp.x, rp.y, v.ang(D.h), RR);
}

/** Smooth motion: state arrives at 2 Hz, the marker eases towards the newest pose every animation frame. */
function frame() {
  const st = S.state;
  if (st && st.pose) {
    const T = { x: st.pose.x, y: st.pose.y, h: (st.pose.heading_deg || 0) * Math.PI / 180 };
    const D = S.disp || (S.disp = { ...T });
    const dx = T.x - D.x, dy = T.y - D.y, dh = angDiff(D.h, T.h);
    if (Math.hypot(dx, dy) > 3) { D.x = T.x; D.y = T.y; } else { D.x += dx * 0.18; D.y += dy * 0.18; }
    D.h += dh * 0.2;
    S.settled = Math.hypot(dx, dy) < 0.002 && Math.abs(dh) < 0.002;
  }
  drawMap();
  S.raf = S.settled && !S.dirty ? 0 : requestAnimationFrame(frame);
  S.dirty = false;
}
function kick() { S.dirty = true; S.settled = false; if (S.tab === "map" || isDesktop()) { if (!S.raf) S.raf = requestAnimationFrame(frame); } }
const isDesktop = () => window.matchMedia("(min-width: 900px)").matches;

// ---------------------------------------------------------------- UI wiring
function showTab(name) {
  S.tab = name;
  for (const v of document.querySelectorAll(".view")) v.classList.toggle("active", v.id === "view-" + name);
  for (const t of document.querySelectorAll(".tab")) t.classList.toggle("active", t.dataset.view === name);
  if (name === "map") kick();
  if (name === "chat") { const l = $("chatList"); l.scrollTop = l.scrollHeight; }
}

function askToken(err) {
  const d = $("login");
  $("loginErr").hidden = !err; $("loginErr").textContent = err || "";
  if (!d.open) d.showModal();
}

document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => showTab(t.dataset.view)));
$("estop").addEventListener("click", estop);
$("chatForm").addEventListener("submit", (e) => { e.preventDefault(); const i = $("chatInput"), t = i.value.trim(); if (t) { i.value = ""; say(t); } });
$("loginForm").addEventListener("submit", () => { S.token = $("tokenInput").value.trim(); $("tokenInput").value = ""; store("gurdy_token", S.token); connect(); });
window.addEventListener("resize", kick);
new ResizeObserver(kick).observe($("map"));

const fromUrl = new URLSearchParams(location.hash.slice(1)).get("token");      // http://host:8080/#token=...  (a fragment never reaches server logs)
if (fromUrl) { store("gurdy_token", fromUrl); history.replaceState(null, "", location.pathname); }
S.token = load("gurdy_token");
connect();

// Offline start: the service worker caches the app shell only (never /api or the WebSocket). It needs a secure context
// (https or localhost); over plain http on the LAN the browser simply refuses and the page works online as before.
if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(() => {});
