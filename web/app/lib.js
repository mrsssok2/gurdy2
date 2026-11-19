// Pure helpers (no DOM): kept separate so they can be unit-tested under node in CI (Phase 7 step 7).

/**
 * World -> canvas transform for map bounds `b` ({x0,y0,x1,y1}) inside w x h pixels. World y points up; canvas y points down.
 * `rot`: rotate the map 90 deg counter-clockwise (world +x points up the screen). `autoRot` rotates only when that makes the map bigger
 * (a landscape map in a portrait phone screen). Returns {s, rot, P(x, y) -> {x, y}, rect(r) -> [x, y, w, h], ang(heading_rad) -> canvas angle}.
 */
export function makeView(b, w, h, pad = 12, autoRot = false) {
  const bw = b.x1 - b.x0, bh = b.y1 - b.y0;
  const fit = (mw, mh) => Math.min((w - 2 * pad) / mw, (h - 2 * pad) / mh);
  const rot = autoRot && fit(bh, bw) > fit(bw, bh) * 1.05;
  const s = Math.max(1e-6, rot ? fit(bh, bw) : fit(bw, bh));
  const pw = (rot ? bh : bw) * s, ph = (rot ? bw : bh) * s, ox = (w - pw) / 2, oy = (h - ph) / 2;
  const P = rot ? (x, y) => ({ x: ox + (b.y1 - y) * s, y: oy + (b.x1 - x) * s })
                : (x, y) => ({ x: ox + (x - b.x0) * s, y: h - (oy + (y - b.y0) * s) });
  const rect = (r) => { const p = P(r.x0, r.y0), q = P(r.x1, r.y1); return [Math.min(p.x, q.x), Math.min(p.y, q.y), Math.abs(q.x - p.x), Math.abs(q.y - p.y)]; };
  return { s, rot, P, rect, ang: (hd) => (rot ? -hd - Math.PI / 2 : -hd) };
}

export function fmtTime(sec) {
  if (sec == null || !isFinite(sec)) return "–";
  const t = Math.floor(sec), m = Math.floor(t / 60), s = t % 60;
  return m >= 60 ? `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, "0")}m` : `${m}:${String(s).padStart(2, "0")}`;
}

export function fmtNum(v, digits = 2, unit = "") {
  return v == null || !isFinite(v) ? "–" : `${Number(v).toFixed(digits)}${unit}`;
}

export function batteryClass(pct) { return pct < 10 ? "crit" : pct < 25 ? "low" : ""; }

export function legSummary(legs) {
  const vals = Object.values(legs || {});
  if (!vals.length) return "–";
  const bad = vals.filter((v) => v !== "ok").length;
  return bad ? `${bad} of ${vals.length} not ok` : "all ok";
}

/** New alert strings (not in `seen`), plus a severity guess; `seen` is mutated. Alerts are plain strings from the robot. */
export function newAlerts(active, seen, now) {
  const out = [];
  for (const text of active || []) {
    if (!seen.has(text)) { seen.add(text); out.push({ text, at: now, crit: /fail|fell|fall|critical|stuck|estop/i.test(text) }); }
  }
  return out;
}

export function wsUrl(loc, token, sid) {
  const q = [token ? "token=" + encodeURIComponent(token) : "", sid ? "sid=" + encodeURIComponent(sid) : ""].filter(Boolean).join("&");
  return `${loc.protocol === "https:" ? "wss" : "ws"}://${loc.host}/ws${q ? "?" + q : ""}`;
}

/** Reconnect delay: 0.5 s doubling to 8 s. */
export function backoff(attempt) { return Math.min(8000, 500 * 2 ** Math.min(attempt, 5)); }

/** "fire_extinguisher" -> "fire extinguisher" */
export function prettyName(s) { return String(s ?? "").replace(/_/g, " ").trim(); }

/** Object colour name -> hex (the robot's own colour label, so a green barrel is drawn green). Unknown / missing -> neutral grey. */
export const COLOR_HEX = { red: "#F25C5C", green: "#4ADE80", blue: "#5B9DFF", yellow: "#FACC15", orange: "#FB923C", white: "#F1F5F9", black: "#4B5563", grey: "#9CA3AF", gray: "#9CA3AF", purple: "#A78BFA", brown: "#A16207" };
export function colorHex(name) { return COLOR_HEX[String(name || "").toLowerCase()] || "#B8C2CC"; }

/** Label text for a remembered object: "green barrel", "fire extinguisher". */
export function objectLabel(o) { return `${o.color ? o.color + " " : ""}${prettyName(o.class)}`; }

const overlaps = (a, b, gap) => a.x < b.x + b.w + gap && b.x < a.x + a.w + gap && a.y < b.y + b.h + gap && b.y < a.y + a.h + gap;

/**
 * Greedy label layout. items: [{text, ax, ay, r}] (anchor point in px, marker radius r) in priority order; `measure(text)` -> width px;
 * lineH = text height. `taken` = rectangles already used (e.g. room titles). Every label is tried at 8 positions around its anchor and the first
 * one that is inside [0,w]x[0,h] and overlaps nothing placed so far wins; if none is free the least-overlapping inside position is used, so a
 * label is ALWAYS inside the map. Returns [{text, x, y, w, h, ax, ay, leader}] with (x,y) the top-left of the text box; `leader` = draw a thin line to the anchor.
 */
export function placeLabels(items, w, h, measure, lineH = 16, taken = [], margin = 4) {
  const placed = taken.map((t) => ({ ...t }));
  const out = [];
  const dirs = [[1, 0], [-1, 0], [0, -1], [0, 1], [0.7, -0.7], [-0.7, -0.7], [0.7, 0.7], [-0.7, 0.7]];
  for (const it of items) {
    const tw = measure(it.text), r = (it.r ?? 4) + 4;
    let best = null, bestScore = Infinity;
    search: for (const ring of [0, 16, 36, 64, 100]) {            // close first, then further away (a leader line joins label and marker)
      for (const [dx, dy] of dirs) {
        const ox = dx * (r + ring), oy = dy * (r + ring);
        let x = it.ax + ox - (dx < 0 ? tw : dx > 0 ? 0 : tw / 2), y = it.ay + oy - (dy < 0 ? lineH : dy > 0 ? 0 : lineH / 2);
        const cx = Math.min(Math.max(x, margin), Math.max(margin, w - tw - margin)), cy = Math.min(Math.max(y, margin), Math.max(margin, h - lineH - margin));
        const clamped = Math.abs(cx - x) + Math.abs(cy - y);
        const box = { x: cx, y: cy, w: tw, h: lineH };
        const hits = placed.reduce((n, p) => n + (overlaps(box, p, 2) ? 1 : 0), 0);
        const score = hits * 1000 + clamped + ring * 0.01;
        if (score < bestScore) { bestScore = score; best = { ...box, leader: ring > 0 || clamped > 0 }; if (hits === 0 && clamped === 0) break search; }
      }
    }
    placed.push({ x: best.x, y: best.y, w: best.w, h: best.h });
    out.push({ text: it.text, ...best, ax: it.ax, ay: it.ay });
  }
  return out;
}

/** Shortest signed angle difference b - a in radians. */
export function angDiff(a, b) { return Math.atan2(Math.sin(b - a), Math.cos(b - a)); }

/** WebSocket address of a robot on another host: https://host[:port] -> wss://host[:port]/ws?token=...&sid=... (http -> ws, only for localhost). */
export function remoteWsUrl(base, token, sid) {
  const u = new URL(base);
  const q = [token ? "token=" + encodeURIComponent(token) : "", sid ? "sid=" + encodeURIComponent(sid) : ""].filter(Boolean).join("&");
  return `${u.protocol === "https:" ? "wss" : "ws"}://${u.host}/ws${q ? "?" + q : ""}`;
}
