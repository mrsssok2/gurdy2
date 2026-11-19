// Whisper in the browser (transformers.js / ONNX Runtime Web), in a module Web Worker. Loaded lazily: nothing here runs until the visitor first presses the mic.
// The microphone audio is transcribed on this device and never uploaded; only the model files are downloaded (from the Hugging Face hub, cached by the browser).
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

env.allowLocalModels = false;
const MODELS = { "tiny.en": "onnx-community/whisper-tiny.en", "base.en": "onnx-community/whisper-base.en" };
let asr = null, device = "wasm", modelName = "tiny.en", promptOk = true;
const post = (m) => self.postMessage(m);

async function build(name, dev) {
  // WASM: 8-bit quantised weights (tiny.en: ~41 MB). WebGPU: encoder fp32 + 4-bit decoder (the configuration of the transformers.js Whisper WebGPU demo, larger download).
  const opts = dev === "webgpu" ? { device: "webgpu", dtype: { encoder_model: "fp32", decoder_model_merged: "q4" } } : { device: "wasm", dtype: "q8" };
  return pipeline("automatic-speech-recognition", MODELS[name], { ...opts, progress_callback: (p) => post({ type: "progress", ...p }) });
}

async function load(name) {
  modelName = MODELS[name] ? name : "tiny.en";
  let gpu = false;
  try { gpu = !!(self.navigator && navigator.gpu && (await navigator.gpu.requestAdapter())); } catch { gpu = false; }
  const order = gpu ? ["webgpu", "wasm"] : ["wasm"];
  let last = null;
  for (const dev of order) {
    try { asr = await build(modelName, dev); device = dev; post({ type: "ready", device, model: modelName }); return; }
    catch (e) { last = e; post({ type: "note", text: `${dev} unavailable: ${String((e && e.message) || e).slice(0, 120)}` }); }
  }
  throw last;
}

function promptIds(prompt) {
  // Whisper "initial prompt": <|startofprev|> + the text (a vocabulary the model should prefer). Best effort: if the tokenizer API differs we simply go without.
  try {
    const tok = asr.tokenizer, sop = tok.model.convert_tokens_to_ids ? tok.model.convert_tokens_to_ids(["<|startofprev|>"])[0] : tok.convert_tokens_to_ids(["<|startofprev|>"])[0];
    const ids = Array.from(tok.encode(" " + prompt.trim()));
    return sop != null && ids.length ? [sop, ...ids] : null;
  } catch { return null; }
}

async function transcribe(audio, prompt) {
  const t0 = performance.now();
  const run = (withPrompt) => asr(audio, { max_new_tokens: 48, ...(withPrompt ? { generate_kwargs: { prompt_ids: promptIds(prompt) } } : {}) });
  let out, usedPrompt = false;
  const ids = prompt && promptOk ? promptIds(prompt) : null;
  try { out = ids ? await run(true) : await run(false); usedPrompt = !!ids; }
  catch (e) {
    if (ids) { promptOk = false; out = await run(false); }                 // the prompt path failed once: never try it again, transcribe without
    else if (device === "webgpu") { asr = await build(modelName, "wasm"); device = "wasm"; out = await run(false); }   // a WebGPU run failed: fall back to WASM
    else throw e;
  }
  let text = (Array.isArray(out) ? out.map((o) => o.text).join(" ") : out.text || "").trim();
  if (usedPrompt && prompt) {                                               // some builds echo the prompt text; strip it
    const p = prompt.trim();
    if (text.startsWith(p)) text = text.slice(p.length).trim();
  }
  return { text, seconds: (performance.now() - t0) / 1000, device, model: modelName, usedPrompt };
}

self.onmessage = async (ev) => {
  const { id, op } = ev.data;
  try {
    if (op === "load") { await load(ev.data.model); post({ id, result: { device, model: modelName } }); }
    else if (op === "transcribe") post({ id, result: await transcribe(ev.data.audio, ev.data.prompt) });
  } catch (e) { post({ id, error: String((e && e.message) || e) }); }
};
