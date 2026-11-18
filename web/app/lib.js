// Pure helpers (no DOM): kept separate so they can be unit-tested under node in CI (Phase 7 step 7).

/** World -> canvas transform for map bounds `b` ({x0,y0,x1,y1}) inside w x h pixels. World y points up; canvas y points down. */
export function makeView(b, w, h, pad = 12) {
  const s = Math.max(1e-6, Math.min((w - 2 * pad) / (b.x1 - b.x0), (h - 2 * pad) / (b.y1 - b.y0)));
  const ox = (w - s * (b.x1 - b.x0)) / 2, oy = (h - s * (b.y1 - b.y0)) / 2;
  return { s, X: (x) => ox + (x - b.x0) * s, Y: (y) => h - (oy + (y - b.y0) * s) };
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

export function wsUrl(loc, token) {
  return `${loc.protocol === "https:" ? "wss" : "ws"}://${loc.host}/ws?token=${encodeURIComponent(token)}`;
}

/** Reconnect delay: 0.5 s doubling to 8 s. */
export function backoff(attempt) { return Math.min(8000, 500 * 2 ** Math.min(attempt, 5)); }
