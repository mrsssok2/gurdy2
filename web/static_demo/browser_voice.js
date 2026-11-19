// Hold-to-talk speech recognition for the STATIC demo, entirely in the browser: Whisper (transformers.js) in a worker, the model downloaded lazily on the first
// mic press, replies spoken with the browser's speechSynthesis. Audio never leaves the device. app.js uses window.GURDY_BROWSER_VOICE when the page is the static demo.
(() => {
  "use strict";
  const params = new URLSearchParams(location.search);
  const MODEL = ["tiny.en", "base.en"].includes(params.get("whisper")) ? params.get("whisper") : "base.en";      // base.en by default (tiny.en heard "go to room B" as "Go to Ruby."); ?whisper=tiny.en for the small one
  const APPROX_MB = { "tiny.en": 50, "base.en": 85 };                                                          // total first download (weights + tokenizer + runtime), measured in docs/phase7_demo_deploy.md
  let worker = null, loadPromise = null, nextId = 1;
  const pending = new Map(), files = new Map(), listeners = new Set();
  const api = { supported: typeof Worker !== "undefined" && typeof WebAssembly === "object" && !!(window.AudioContext || window.webkitAudioContext),
                model: MODEL, approxMb: APPROX_MB[MODEL], state: "idle", device: null, loadedMs: null };

  function progress() {
    let loaded = 0, total = 0;
    for (const f of files.values()) { loaded += f.loaded; total += f.total; }
    return { loaded, total, pct: total ? Math.min(99, Math.round(100 * loaded / total)) : 0 };
  }
  const notify = (extra) => { const p = { ...progress(), state: api.state, ...extra }; listeners.forEach((fn) => fn(p)); };
  api.onProgress = (fn) => { listeners.add(fn); return () => listeners.delete(fn); };

  function startWorker() {
    worker = new Worker("whisper_worker.js", { type: "module" });
    worker.onmessage = (ev) => {
      const m = ev.data;
      if (m.type === "progress") {
        if (m.file && m.total) { files.set(m.file, { loaded: m.loaded || 0, total: m.total }); notify(); }
      } else if (m.type === "ready") { api.device = m.device; }
      else if (pending.has(m.id)) { const p = pending.get(m.id); pending.delete(m.id); m.error ? p.rej(new Error(m.error)) : p.res(m.result); }
    };
    worker.onerror = (e) => { const err = new Error(e.message || "speech worker error"); pending.forEach((p) => p.rej(err)); pending.clear(); api.state = "error"; notify({ error: err.message }); };
  }
  const call = (msg, transfer) => new Promise((res, rej) => { const id = nextId++; pending.set(id, { res, rej }); worker.postMessage({ id, ...msg }, transfer || []); });

  /** Downloads + starts the model on first use (idempotent). */
  api.ensureLoaded = () => {
    if (loadPromise) return loadPromise;
    const t0 = performance.now();
    api.state = "loading"; notify();
    startWorker();
    loadPromise = call({ op: "load", model: MODEL }).then((r) => { api.state = "ready"; api.device = r.device; api.loadedMs = Math.round(performance.now() - t0); notify(); return r; })
      .catch((e) => { api.state = "error"; loadPromise = null; notify({ error: e.message }); throw e; });
    return loadPromise;
  };

  async function decode16k(blob) {
    const AC = window.AudioContext || window.webkitAudioContext;
    let ctx;
    try { ctx = new AC({ sampleRate: 16000 }); } catch { ctx = new AC(); }
    const buf = await ctx.decodeAudioData(await blob.arrayBuffer());
    ctx.close && ctx.close();
    const n = buf.length, mono = new Float32Array(n);
    for (let c = 0; c < buf.numberOfChannels; c++) { const d = buf.getChannelData(c); for (let i = 0; i < n; i++) mono[i] += d[i] / buf.numberOfChannels; }
    if (buf.sampleRate === 16000) return mono;
    const out = new Float32Array(Math.max(1, Math.round(n * 16000 / buf.sampleRate)));            // browsers that ignore the requested rate: linear resample
    for (let i = 0; i < out.length; i++) { const x = i * buf.sampleRate / 16000, i0 = Math.floor(x), f = x - i0; out[i] = mono[i0] * (1 - f) + (mono[Math.min(n - 1, i0 + 1)] * f); }
    return out;
  }
  const rmsOf = (a) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return Math.sqrt(s / Math.max(1, a.length)); };

  /** blob -> {text, rms, seconds, device, usedPrompt}. `ctx` = {prompt, hint} from the Python side (vocabulary prompt; expected answers while a question is open). */
  api.transcribe = async (blob, ctx = {}) => {
    const audio = await decode16k(blob);
    const rms = rmsOf(audio), audioSeconds = audio.length / 16000;           // before the buffer is transferred to the worker
    await api.ensureLoaded();
    const prompt = ctx.hint ? ctx.hint : ctx.prompt;                    // same rule as the live pipeline: the expected answers replace the command prompt
    const r = await call({ op: "transcribe", audio, prompt }, [audio.buffer]);
    return { ...r, rms, audioSeconds };
  };

  // ---- spoken replies (the browser's own voices; nothing is sent anywhere)
  api.canSpeak = "speechSynthesis" in window;
  api.speak = (text) => {
    if (!api.canSpeak) return;
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "en-US"; u.rate = 1.02;
    const voices = speechSynthesis.getVoices().filter((v) => /^en/i.test(v.lang));
    if (voices.length) u.voice = voices.find((v) => v.default) || voices[0];
    speechSynthesis.speak(u);
  };
  api.stopSpeaking = () => { if (api.canSpeak) speechSynthesis.cancel(); };

  window.GURDY_BROWSER_VOICE = api;
})();
