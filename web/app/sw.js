// Service worker: caches the app SHELL (html, css, js, fonts, icons) so the page opens with no connection.
// It NEVER caches robot data: /api/* and /ws are not handled here at all (the browser goes straight to the network), and only
// same-origin GET requests for the listed shell files are ever stored. Bump VERSION when the shell file list changes.
const VERSION = "gurdy-shell-v4";
const SHELL = [
  "./", "index.html", "style.css", "app.js", "lib.js", "manifest.json",
  "icons/icon-192.png", "icons/icon-512.png", "icons/icon-maskable-512.png", "icons/apple-touch-icon.png",
  "fonts/inter-latin-400-normal.woff2",
  "fonts/space-grotesk-latin-600-normal.woff2", "fonts/space-grotesk-latin-700-normal.woff2",
  "fonts/jetbrains-mono-latin-400-normal.woff2", "fonts/jetbrains-mono-latin-700-normal.woff2",
];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k)))).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request, url = new URL(req.url);
  if (req.method !== "GET" || url.origin !== self.location.origin) return;               // other origins / writes: untouched
  if (url.pathname.startsWith("/api/") || url.pathname === "/ws") return;               // robot data: never cached, never intercepted
  const shellPath = url.pathname === "/" ? "./" : url.pathname.replace(/^\//, "");
  if (!SHELL.includes(shellPath) && req.mode !== "navigate") return;                    // only the known shell
  // Network first (so a new version shows up as soon as the server is reachable), cached copy when offline.
  event.respondWith(
    fetch(req).then((res) => {
      if (res.ok && SHELL.includes(shellPath)) { const copy = res.clone(); caches.open(VERSION).then((c) => c.put(req, copy)); }
      return res;
    }).catch(() => caches.match(req).then((hit) => hit || (req.mode === "navigate" ? caches.match("./") : Response.error())))
  );
});
