import { makeView, fmtTime, fmtNum, batteryClass, legSummary, newAlerts, wsUrl, backoff } from "./lib.js";

const $ = (id) => document.getElementById(id);
const C = { bg: "#0B0E13", room: "#121821", wall: "#3A4552", obstacle: "#4B5563", zone: "#D93036", accent: "#2DD4BF", text: "#E6EDF3", muted: "#8B98A5" };

const S = { token: "", ws: null, attempt: 0, timer: null, map: null, state: null, lastSeq: 0, seen: new Set(), history: [], tab: "map" };

function store(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode: fine */ } }
function load(k) { try { return localStorage.getItem(k) || ""; } catch { return ""; } }

// ---------------------------------------------------------------- connection
function setConn(kind, text) { $("conn").className = "conn " + kind; $("connText").textContent = text; }

async function api(path, opts = {}) {
  const r = await fetch(path, { ...opts, headers: { Authorization: "Bearer " + S.token, "Content-Type": "application/json", ...(opts.headers || {}) } });
  if (r.status === 401) throw Object.assign(new Error("unauthorized"), { code: 401 });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
  return r.json();
}

async function connect() {
  clearTimeout(S.timer);
  if (!S.token) return askToken();
  try {
    S.map = await api("/api/map");                     // also validates the token (a WebSocket cannot report a 401 to the page)
  } catch (e) {
    if (e.code === 401) return askToken("That token was not accepted.");
    return retry("server not reachable");
  }
  const ws = new WebSocket(wsUrl(location, S.token));
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
  if (m.type === "hello") { S.map = m.map; S.lastSeq = 0; $("chatList").replaceChildren(); m.chat.forEach(addChat); }
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
  renderLegs(st.legs || {});
  renderObjects(st.objects || []);
  renderAlerts(st.alerts || []);
  if (S.tab === "map") drawMap();
}

function renderLegs(legs) {
  const box = $("legs");
  box.replaceChildren(...Object.entries(legs).map(([name, state]) => {
    const d = document.createElement("div"); d.className = "leg " + state;
    const dot = document.createElement("span"); dot.className = "dot";
    const n = document.createElement("span"); n.textContent = name.replace("_", " ");
    const s = document.createElement("span"); s.className = "st"; s.textContent = state;
    d.append(dot, n, s); return d;
  }));
}

function renderObjects(objs) {
  $("sObjCount").textContent = String(objs.length);
  const ul = $("objList");
  if (!objs.length) { ul.replaceChildren(emptyLi("Nothing seen yet.")); return; }
  ul.replaceChildren(...objs.map((o) => {
    const li = document.createElement("li");
    const t = document.createElement("span"); t.textContent = `${o.color ? o.color + " " : ""}${o.class}${o.place ? " · " + o.place : ""}`;
    const w = document.createElement("span"); w.className = "when"; w.textContent = o.stale ? "stale" : fmtTime(o.age_s) + " ago";
    li.append(t, w); return li;
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
function drawMap() {
  const cv = $("map"), dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (!w || !h || !S.map) return;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.fillStyle = C.bg; g.fillRect(0, 0, w, h);
  const v = makeView(S.map.bounds, w, h);
  const rect = (r) => [v.X(r.x0), v.Y(r.y1), (r.x1 - r.x0) * v.s, (r.y1 - r.y0) * v.s];
  g.font = "12px Inter, system-ui, sans-serif"; g.textBaseline = "middle";
  for (const [name, r] of Object.entries(S.map.rooms || {})) {
    g.fillStyle = C.room; g.fillRect(...rect(r));
    g.fillStyle = C.muted; g.textAlign = "left"; g.fillText(name, v.X(r.x0) + 8, v.Y(r.y1) + 14);
  }
  g.fillStyle = C.wall; for (const r of S.map.walls || []) g.fillRect(...rect(r));
  g.fillStyle = C.obstacle; for (const r of S.map.obstacles || []) g.fillRect(...rect(r));
  const st = S.state || {};
  for (const z of st.zones || []) {                                   // no-go zones
    g.beginPath(); g.arc(v.X(z.x), v.Y(z.y), z.r * v.s, 0, 2 * Math.PI);
    g.fillStyle = "#D9303622"; g.fill(); g.setLineDash([6, 4]); g.strokeStyle = C.zone; g.lineWidth = 1.5; g.stroke(); g.setLineDash([]);
  }
  g.fillStyle = C.muted; g.textAlign = "center";
  for (const [name, p] of Object.entries(S.map.places || {})) { g.fillRect(v.X(p.x) - 2, v.Y(p.y) - 2, 4, 4); if (name !== "doorway") g.fillText(name, v.X(p.x), v.Y(p.y) + 12); }
  for (const o of st.objects || []) {                                 // what the robot remembers (not ground truth)
    g.beginPath(); g.arc(v.X(o.x), v.Y(o.y), 5, 0, 2 * Math.PI); g.fillStyle = o.stale ? C.muted : C.accent; g.fill();
    g.fillStyle = C.text; g.textAlign = "left"; g.fillText(`${o.color ? o.color + " " : ""}${o.class}`, v.X(o.x) + 8, v.Y(o.y));
  }
  const tr = st.trail || [];
  if (tr.length > 1) { g.beginPath(); tr.forEach(([x, y], i) => (i ? g.lineTo(v.X(x), v.Y(y)) : g.moveTo(v.X(x), v.Y(y)))); g.strokeStyle = "#2DD4BF99"; g.lineWidth = 2; g.stroke(); }
  if (st.pose) {                                                      // robot: triangle pointing along the heading
    const th = -(st.pose.heading_deg || 0) * Math.PI / 180, L = Math.max(9, 0.3 * v.s), cx = v.X(st.pose.x), cy = v.Y(st.pose.y);
    g.beginPath();
    g.moveTo(cx + L * Math.cos(th), cy + L * Math.sin(th));
    g.lineTo(cx + L * 0.7 * Math.cos(th + 2.5), cy + L * 0.7 * Math.sin(th + 2.5));
    g.lineTo(cx + L * 0.7 * Math.cos(th - 2.5), cy + L * 0.7 * Math.sin(th - 2.5));
    g.closePath(); g.fillStyle = C.accent; g.fill(); g.strokeStyle = "#062B27"; g.lineWidth = 1.5; g.stroke();
  }
}

// ---------------------------------------------------------------- UI wiring
function showTab(name) {
  S.tab = name;
  for (const v of document.querySelectorAll(".view")) v.hidden = v.id !== "view-" + name;
  for (const t of document.querySelectorAll(".tab")) t.classList.toggle("active", t.dataset.view === name);
  if (name === "map") requestAnimationFrame(drawMap);
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
window.addEventListener("resize", () => S.tab === "map" && drawMap());

const fromUrl = new URLSearchParams(location.hash.slice(1)).get("token");      // http://host:8080/#token=...  (a fragment never reaches server logs)
if (fromUrl) { store("gurdy_token", fromUrl); history.replaceState(null, "", location.pathname); }
S.token = load("gurdy_token");
connect();
