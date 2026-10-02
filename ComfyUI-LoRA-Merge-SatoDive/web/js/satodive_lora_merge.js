// LoRA Merge Studio - SatoDive  (v2)
// Themed nodes, LoRA cards with Civitai pictures, info dialog, LoRA browser,
// multi-LoRA mixer with per-block sliders, run / save buttons.
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const T = {
  setup: "SatoDiveLoRAStudioSetup",
  slot: "SatoDiveLoRASlot",
  merge: "SatoDiveLoRAMergeStudioV2",
};
const SLOT_TYPES = [T.slot];
const MERGE_TYPES = [T.merge];
const LETTERS = "ABCDEFGHIJ";
const COLORS = ["#22d3ee", "#f472b6", "#facc15", "#4ade80", "#fb923c", "#60a5fa", "#e879f9", "#2dd4bf", "#f87171", "#a3e635"];
const DARK = ["#0b3d47", "#4b1437", "#4a3d07", "#0f3d22", "#4a2508", "#10284d", "#43124a", "#0b3b36", "#4a1414", "#2c3d0a"];
const THEME = {
  [T.setup]: { color: "#3b2f0f", bgcolor: "#17150f", acc: "#fbbf24" },
  [T.merge]: { color: "#2d1d63", bgcolor: "#141122", acc: "#a78bfa" },
};
const EVT = "satodive:lora-changed";
const API = "/satodive/lora_merge";

// ------------------------------------------------------------------------------------
// Styles
// ------------------------------------------------------------------------------------
const CSS = `
.sato-card{--acc:#a78bfa;box-sizing:border-box;width:100%;height:100%;overflow:hidden;color:#e5e7eb;
  font:500 11.5px/1.4 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
  background:linear-gradient(160deg,rgba(255,255,255,.055),rgba(255,255,255,.012));
  border:1px solid rgba(255,255,255,.09);border-radius:12px;padding:9px 10px;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.05),0 6px 18px rgba(0,0,0,.25);display:flex;flex-direction:column;gap:7px}
.sato-card *,.sato-modal *{box-sizing:border-box}
.sato-card>*{flex-shrink:0}
.sato-row-h{display:flex;align-items:center;gap:7px;min-width:0}
.sato-title{font-weight:700;font-size:12.5px;color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1;min-width:0}
.sato-sub{opacity:.6;font-size:10.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sato-dot{width:8px;height:8px;border-radius:50%;background:var(--acc);box-shadow:0 0 10px var(--acc);flex:none}
.sato-badge{flex:none;padding:1px 8px;border-radius:999px;font-size:10px;font-weight:700;letter-spacing:.02em;white-space:nowrap;
  color:var(--acc);background:color-mix(in srgb,var(--acc) 16%,transparent);border:1px solid color-mix(in srgb,var(--acc) 42%,transparent)}
.sato-badge.warn{--acc:#fbbf24}.sato-badge.bad{--acc:#f87171}.sato-badge.ok{--acc:#4ade80}.sato-badge.dim{--acc:#94a3b8}
.sato-letter{flex:none;width:22px;height:22px;border-radius:7px;display:grid;place-items:center;font-weight:900;font-size:12px;color:#0b0b10;background:var(--acc);box-shadow:0 0 12px color-mix(in srgb,var(--acc) 60%,transparent)}
.sato-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:5px}
.sato-stat{background:rgba(0,0,0,.28);border:1px solid rgba(255,255,255,.05);border-radius:8px;padding:4px 7px;min-width:0}
.sato-stat b{display:block;font-size:12px;color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sato-stat b.ok{color:#4ade80}.sato-stat b.bad{color:#f87171}
.sato-stat span{display:block;opacity:.55;font-size:9px;text-transform:uppercase;letter-spacing:.08em}
.sato-thumb{flex:none;width:76px;height:76px;border-radius:10px;overflow:hidden;background:rgba(0,0,0,.35);border:1px solid rgba(255,255,255,.08);
  display:grid;place-items:center;font-weight:900;font-size:22px;color:var(--acc);cursor:pointer;position:relative}
.sato-thumb img,.sato-thumb video{width:100%;height:100%;object-fit:cover;display:block}
.sato-chips{display:flex;flex-wrap:wrap;gap:4px;max-height:42px;overflow:hidden}
.sato-chip{cursor:pointer;padding:2px 7px;border-radius:6px;font-size:10.5px;background:rgba(255,255,255,.06);
  border:1px solid rgba(255,255,255,.08);color:#e5e7eb;transition:all .15s;white-space:nowrap}
.sato-chip:hover{background:color-mix(in srgb,var(--acc) 25%,transparent);border-color:var(--acc)}
.sato-chip i{font-style:normal;opacity:.6;margin-left:4px}
.sato-msg{font-size:10.5px;opacity:.85}
.sato-msg.warn{color:#fbbf24}.sato-msg.bad{color:#f87171}.sato-msg.ok{color:#4ade80}
.sato-kv{display:grid;grid-template-columns:auto 1fr;gap:3px 10px;font-size:10.5px}
.sato-kv span{opacity:.55}.sato-kv b{font-weight:600;color:#f1f5f9;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sato-btns{display:flex;gap:5px;flex-wrap:wrap}
.sato-btn{cursor:pointer;user-select:none;padding:4px 9px;border-radius:8px;font-size:10.5px;font-weight:700;color:#e5e7eb;
  background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.11);transition:all .12s;white-space:nowrap}
.sato-btn:hover{background:color-mix(in srgb,var(--acc) 22%,transparent);border-color:var(--acc);color:#fff}
.sato-btn.primary{background:color-mix(in srgb,var(--acc) 30%,transparent);border-color:color-mix(in srgb,var(--acc) 70%,transparent);color:#fff}
.sato-btn.on{background:var(--acc);color:#0b0b10;border-color:var(--acc)}
.sato-btn.grow{flex:1;text-align:center}
.sato-guide{border-radius:10px;padding:7px 9px;background:rgba(167,139,250,.08);border:1px solid rgba(167,139,250,.25);font-size:10.8px}
.sato-guide b.t{display:block;color:#fff;font-size:11.5px;margin-bottom:2px}
.sato-guide .pick{color:#c4b5fd;margin-top:3px}
.sato-guide .cost{opacity:.6;margin-top:2px}
.sato-sec-t{font-size:10px;font-weight:800;letter-spacing:.09em;text-transform:uppercase;opacity:.65;display:flex;align-items:center;gap:6px}
.sato-sec-t:after{content:"";flex:1;height:1px;background:rgba(255,255,255,.08)}
.sato-mixrow{display:grid;grid-template-columns:22px minmax(0,1fr) minmax(80px,1.2fr) 34px 24px 24px;gap:6px;align-items:center}
.sato-mixrow .nm{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:10.8px}
.sato-mixrow.muted{opacity:.4}
.sato-ms{cursor:pointer;width:24px;height:20px;border-radius:6px;display:grid;place-items:center;font-size:10px;font-weight:800;
  background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1)}
.sato-ms.m.on{background:#f87171;color:#111;border-color:#f87171}.sato-ms.s.on{background:#facc15;color:#111;border-color:#facc15}
.sato-tabs{display:flex;gap:4px;flex-wrap:wrap}
.sato-tab{cursor:pointer;padding:3px 10px;border-radius:7px;font-weight:800;font-size:11px;border:1px solid color-mix(in srgb,var(--c) 50%,transparent);
  color:var(--c);background:rgba(0,0,0,.25)}
.sato-tab.on{background:var(--c);color:#0b0b10}
.sato-scroll{overflow-y:auto;overflow-x:hidden;padding-right:4px;border-radius:8px;background:rgba(0,0,0,.18);
  border:1px solid rgba(255,255,255,.05);scrollbar-width:thin;scrollbar-color:#4c3d86 transparent}
.sato-sec.closed .sato-rows{display:none}
.sato-bh,.sato-br{display:grid;grid-template-columns:14px minmax(70px,1fr) minmax(90px,2fr) 36px;gap:7px;align-items:center;padding:3px 7px}
.sato-bh{font-weight:700;cursor:pointer;background:rgba(255,255,255,.035);position:sticky;top:0;z-index:1;backdrop-filter:blur(6px)}
.sato-bh .car{opacity:.6;transition:transform .15s}.sato-sec.closed .car{transform:rotate(-90deg)}
.sato-br{font-size:10.5px}.sato-br:hover{background:rgba(255,255,255,.03)}
.sato-br .lbl,.sato-bh .lbl{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sato-br.none{opacity:.35}
.sato-val{text-align:right;font-variant-numeric:tabular-nums;font-size:10.5px;cursor:pointer}
.sato-range{-webkit-appearance:none;appearance:none;width:100%;height:5px;border-radius:999px;outline:none;margin:0;cursor:pointer;
  background:linear-gradient(90deg,var(--c) 0%,var(--c) var(--p),rgba(255,255,255,.1) var(--p))}
.sato-range::-webkit-slider-thumb{-webkit-appearance:none;width:12px;height:12px;border-radius:50%;background:#fff;border:2px solid var(--c);box-shadow:0 0 8px var(--c)}
.sato-range::-moz-range-thumb{width:10px;height:10px;border-radius:50%;background:#fff;border:2px solid var(--c)}
.sato-range:disabled{opacity:.25;cursor:not-allowed}
.sato-report{display:flex;flex-wrap:wrap;gap:4px}
.sato-empty{padding:14px 8px;text-align:center;opacity:.65;font-size:11px}
.sato-help{font-size:10.3px;opacity:.7}
.sato-details summary{cursor:pointer;font-size:10.5px;opacity:.75}
.sato-details ul{margin:4px 0 0 0;padding-left:16px;font-size:10.3px;opacity:.85}
/* ---- modal ---- */
.sato-ov{position:fixed;inset:0;z-index:100000;background:rgba(5,6,12,.72);backdrop-filter:blur(3px);display:flex;align-items:center;justify-content:center;padding:24px}
.sato-modal{--acc:#a78bfa;width:min(1100px,96vw);max-height:92vh;display:flex;flex-direction:column;color:#e5e7eb;
  font:500 12.5px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;background:#16161e;
  border:1px solid rgba(255,255,255,.1);border-radius:16px;box-shadow:0 30px 80px rgba(0,0,0,.6);overflow:hidden}
.sato-mh{display:flex;align-items:center;gap:10px;padding:16px 20px 10px}
.sato-mh h2{margin:0;font-size:19px;color:#fff;flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sato-mb{overflow:auto;padding:4px 20px 18px;display:flex;flex-direction:column;gap:14px}
.sato-mf{display:flex;justify-content:flex-end;gap:8px;padding:10px 20px;border-top:1px solid rgba(255,255,255,.07)}
.sato-table{width:100%;border-collapse:collapse;font-size:12px}
.sato-table td{border:1px solid rgba(255,255,255,.08);padding:6px 10px;vertical-align:middle}
.sato-table td:first-child{width:150px;color:#cbd5e1;background:rgba(255,255,255,.03);font-weight:600}
.sato-table input,.sato-table textarea,.sato-key input{width:100%;background:rgba(0,0,0,.35);color:#fff;border:1px solid rgba(255,255,255,.12);border-radius:6px;padding:4px 7px;font:inherit}
.sato-table textarea{min-height:46px;resize:vertical}
.sato-table a,.sato-modal a{color:#93c5fd}
.sato-gal{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px}
.sato-gal .it{position:relative;border-radius:10px;overflow:hidden;background:#000;aspect-ratio:3/4;cursor:zoom-in;border:1px solid rgba(255,255,255,.08)}
.sato-gal .it img,.sato-gal .it video{width:100%;height:100%;object-fit:cover;display:block;transition:transform .25s}
.sato-gal .it:hover img,.sato-gal .it:hover video{transform:scale(1.04)}
.sato-gal .it.nsfw img,.sato-gal .it.nsfw video{filter:blur(22px)}
.sato-gal .it .tag{position:absolute;left:6px;top:6px;font-size:10px;font-weight:800;background:rgba(0,0,0,.6);padding:1px 6px;border-radius:5px}
.sato-desc{max-height:220px;overflow:auto;font-size:12px;opacity:.9;background:rgba(0,0,0,.2);padding:8px 12px;border-radius:8px}
.sato-desc img{max-width:100%}
.sato-light{position:fixed;inset:0;z-index:100001;background:rgba(0,0,0,.88);display:flex;gap:16px;padding:24px;align-items:center;justify-content:center}
.sato-light img,.sato-light video{max-width:70vw;max-height:90vh;border-radius:10px}
.sato-light .meta{width:min(380px,28vw);max-height:90vh;overflow:auto;background:#16161e;border-radius:12px;padding:14px;font-size:12px;color:#e5e7eb}
.sato-light .meta pre{white-space:pre-wrap;word-break:break-word;background:rgba(0,0,0,.35);padding:8px;border-radius:8px;font:12px/1.4 ui-monospace,monospace}
.sato-spin{width:26px;height:26px;border-radius:50%;border:3px solid rgba(255,255,255,.15);border-top-color:var(--acc);animation:satospin 0.8s linear infinite;margin:30px auto}
@keyframes satospin{to{transform:rotate(360deg)}}
.sato-bgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:10px}
.sato-bcard{position:relative;border-radius:12px;overflow:hidden;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);cursor:pointer;transition:all .15s}
.sato-bcard:hover{border-color:var(--acc);transform:translateY(-2px)}
.sato-bcard.sel{border-color:#4ade80;box-shadow:0 0 0 2px #4ade80}
.sato-bcard .im{aspect-ratio:1/1;background:rgba(0,0,0,.4);display:grid;place-items:center;font-size:28px;font-weight:900;color:var(--acc);overflow:hidden}
.sato-bcard .im img,.sato-bcard .im video{width:100%;height:100%;object-fit:cover}
.sato-bcard .tx{padding:6px 8px}
.sato-bcard .tx b{display:block;font-size:11.5px;color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sato-bcard .tx span{font-size:10px;opacity:.6;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;display:block}
.sato-bcard .inf{position:absolute;right:6px;top:6px;width:22px;height:22px;border-radius:6px;background:rgba(0,0,0,.65);display:grid;place-items:center;font-size:12px}
.sato-tools{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.sato-tools input[type=search]{flex:1;min-width:180px;background:rgba(0,0,0,.35);color:#fff;border:1px solid rgba(255,255,255,.12);border-radius:8px;padding:6px 10px;font:inherit}
.sato-tools label{display:flex;gap:5px;align-items:center;font-size:12px;opacity:.85;cursor:pointer}
.sato-toast{position:fixed;bottom:22px;left:50%;transform:translateX(-50%);z-index:100002;background:#1f1f2b;color:#fff;border:1px solid rgba(255,255,255,.15);border-radius:10px;padding:8px 14px;font:600 12px Inter,system-ui,sans-serif;box-shadow:0 10px 30px rgba(0,0,0,.5)}
`;

function injectCSS() {
  if (document.getElementById("sato-lora-merge-css")) return;
  const s = document.createElement("style");
  s.id = "sato-lora-merge-css";
  s.textContent = CSS;
  document.head.appendChild(s);
}

// ------------------------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------------------------
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const W = (node, name) => node?.widgets?.find((w) => w.name === name);
const baseName = (s) => String(s || "").split(/[\\/]/).pop().replace(/\.(safetensors|ckpt|pt|pth|bin)$/i, "");
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const fmtN = (n) => (n == null ? "-" : n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e3 ? (n / 1e3).toFixed(1) + "k" : String(n));
const colorOf = (letter) => COLORS[Math.max(0, LETTERS.indexOf(letter))] || "#a78bfa";
const h = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };

function toast(msg) {
  const t = h("div", "sato-toast", esc(msg));
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 1800);
}
function copy(text) {
  try { navigator.clipboard?.writeText(text); toast("Copied: " + String(text).slice(0, 60)); } catch (_) {}
}
function lsGet(k, d) { try { const v = localStorage.getItem("satodive." + k); return v == null ? d : JSON.parse(v); } catch (_) { return d; } }
function lsSet(k, v) { try { localStorage.setItem("satodive." + k, JSON.stringify(v)); } catch (_) {} }

let PRESETS = null;
async function getPresets() {
  if (PRESETS) return PRESETS;
  try {
    PRESETS = await (await api.fetchApi(API + "/presets")).json();
  } catch (e) {
    console.warn("[SatoDive] presets unavailable", e);
    PRESETS = { presets: {}, auto: "Auto-detect / Other", help: { methods: [], method_help: {}, tips: {}, block_help: "" } };
  }
  return PRESETS;
}
async function getJSON(url) { return (await api.fetchApi(url)).json(); }
async function postJSON(url, body) {
  return (await api.fetchApi(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).json();
}
const familyOfArch = (arch) => PRESETS?.presets?.[arch]?.family || "auto";

// ---- graph helpers -----------------------------------------------------------------
function linkById(graph, id) {
  const L = graph?.links;
  if (!L || id == null) return null;
  return typeof L.get === "function" ? L.get(id) : L[id];
}
function inputLink(node, slot) {
  if (slot < 0 || !node.inputs?.[slot]) return null;
  if (typeof node.getInputLink === "function") { try { return node.getInputLink(slot); } catch (_) {} }
  return linkById(node.graph ?? app.graph, node.inputs[slot].link);
}
function upstream(node, inputName) {
  const g = node.graph ?? app.graph;
  let cur = node;
  let slot = typeof inputName === "number" ? inputName : (node.inputs || []).findIndex((i) => i.name === inputName);
  for (let guard = 0; guard < 16; guard++) {
    const link = inputLink(cur, slot);
    if (!link) return null;
    const n = g.getNodeById(link.origin_id);
    if (!n) return null;
    if (n.type === "Reroute" || n.comfyClass === "Reroute") { cur = n; slot = 0; continue; }
    return n;
  }
  return null;
}
function setupFor(node) {
  const n = upstream(node, "pipe");
  return n && n.comfyClass === T.setup ? n : null;
}
function archFor(node) {
  const s = setupFor(node);
  return (s && W(s, "architecture")?.value) || PRESETS?.auto || "Auto-detect / Other";
}
// which merge input(s) a slot feeds: returns the first letter found
function slotLetter(node) {
  const g = node.graph ?? app.graph;
  const seen = new Set();
  const walk = (n, depth) => {
    for (const out of n.outputs || []) {
      for (const id of out.links || []) {
        const l = linkById(g, id);
        if (!l || seen.has(id)) continue;
        seen.add(id);
        const t = g.getNodeById(l.target_id);
        if (!t) continue;
        if (t.type === "Reroute" && depth < 8) { const r = walk(t, depth + 1); if (r) return r; continue; }
        const inp = t.inputs?.[l.target_slot];
        if (!inp) continue;
        const m = /lora_([A-J])$/.exec(inp.name || "");
        if (m) return m[1];
      }
    }
    return null;
  };
  return walk(node, 0);
}
// LoRAs connected to a merge node: [{letter, input, node, name, merged}]
function mergeSources(node) {
  const out = [];
  (node.inputs || []).forEach((inp, i) => {
    const m = /lora_([A-J])$/.exec(inp.name || "");
    if (!m) return;
    const letter = m[1];
    const src = upstream(node, i);
    if (!src) return;
    if (SLOT_TYPES.includes(src.comfyClass)) out.push({ letter, node: src, name: W(src, "lora_name")?.value || "" });
    else if (MERGE_TYPES.includes(src.comfyClass)) out.push({ letter, node: src, name: "⟲ " + (src.title || "Merge Studio"), merged: true });
  });
  return out;
}

async function queueNode(node) {
  try { await app.queuePrompt(0, 1, [String(node.id)]); return; } catch (e) { console.warn("[SatoDive] partial queue failed", e); }
  try { app.canvas.selectNode?.(node, false); await app.extensionManager.command.execute("Comfy.QueueSelectedOutputNodes"); }
  catch (e) { await app.queuePrompt(0, 1); }
}

function addDOM(node, name, el, height) {
  const hf = typeof height === "function" ? height : () => height;
  const w = node.addDOMWidget(name, "sato_" + name, el, {
    serialize: false, hideOnZoom: false, getMinHeight: () => hf(), getMaxHeight: () => hf(), getHeight: () => hf(),
  });
  w.serialize = false;
  w.computeSize = (width) => [width, hf() + 6];
  for (const ev of ["wheel"]) el.addEventListener(ev, (e) => e.stopPropagation(), { passive: true });
  return w;
}
// The frontend adds the output-image preview lazily (first draw) without growing the node,
// so grow our nodes whenever their content needs more room than they have.
function keepFitted(node) {
  const odf = node.onDrawForeground;
  node.onDrawForeground = function (ctx) {
    const r = odf?.apply(this, arguments);
    try {
      const need = this.computeSize()[1];
      if (!this.flags?.collapsed && need > this.size[1] + 1) this.setSize([this.size[0], need]);
    } catch (_) {}
    return r;
  };
}
function fitNode(node, minW = 0) {
  const sz = node.computeSize();
  node.setSize([Math.max(node.size[0], sz[0], minW), Math.max(sz[1], 40)]);
  node.setDirtyCanvas?.(true, true);
  app.graph.setDirtyCanvas(true, true);
}
function hideWidget(node, name) {
  const w = W(node, name);
  if (!w) return;
  w.hidden = true;
  w.computeSize = () => [0, -4];
}
function addToPrompt(node, word) {
  const s = setupFor(node);
  const pw = W(s, "prompt");
  if (!pw) return toast("Connect a LoRA Studio Setup first");
  const v = String(pw.value || "");
  if (v.toLowerCase().includes(String(word).toLowerCase())) return toast("Already in the prompt");
  pw.value = v.trim() ? `${word}, ${v}` : word;
  app.graph.setDirtyCanvas(true, true);
  toast(`Added "${word}" to the preview prompt`);
}

// Civitai pictures load through ComfyUI (cached on disk, survive offline / ad-blockers)
const cimg = (url) => (url && /^https:\/\/image\.civitai\.com\//.test(url) ? `${API}/img?u=${encodeURIComponent(url)}` : url);

function mediaEl(url, type, cls) {
  url = cimg(url);
  if (type === "video") {
    const v = document.createElement("video");
    Object.assign(v, { src: url, muted: true, loop: true, autoplay: true, playsInline: true });
    v.setAttribute("muted", "");
    if (cls) v.className = cls;
    return v;
  }
  const i = document.createElement("img");
  i.loading = "lazy";
  i.src = url;
  if (cls) i.className = cls;
  return i;
}

function sanitize(html) {
  const tpl = document.createElement("template");
  tpl.innerHTML = String(html || "");
  tpl.content.querySelectorAll("script,style,iframe,object,embed,form,input,button").forEach((e) => e.remove());
  tpl.content.querySelectorAll("*").forEach((e) => {
    for (const a of [...e.attributes]) {
      if (/^on/i.test(a.name) || (/^(href|src)$/i.test(a.name) && /^\s*javascript:/i.test(a.value))) e.removeAttribute(a.name);
    }
    if (e.tagName === "A") { e.target = "_blank"; e.rel = "noopener noreferrer"; }
  });
  return tpl.innerHTML;
}

// ------------------------------------------------------------------------------------
// Civitai info cache (frontend)
// ------------------------------------------------------------------------------------
const INFO = new Map(); // name -> response
async function civitaiInfo(name, { offline = false, refresh = false } = {}) {
  if (!name) return null;
  if (!refresh && INFO.has(name)) {
    const c = INFO.get(name);
    if (c.civitai || offline) return c;
  }
  const r = await getJSON(`${API}/civitai?name=${encodeURIComponent(name)}&offline=${offline ? 1 : 0}&refresh=${refresh ? 1 : 0}`);
  INFO.set(name, r);
  return r;
}
function firstThumb(info, allowNsfw) {
  const ims = [...(info?.civitai?.images || [])].sort((a, b) => (a.type === "video") - (b.type === "video"));
  const im = ims.find((i) => allowNsfw || (i.nsfw || 1) <= 1);
  return im ? { url: im.thumb, type: im.type } : null;
}

// ------------------------------------------------------------------------------------
// Info dialog (Civitai)
// ------------------------------------------------------------------------------------
function closeOnEsc(ov) {
  const f = (e) => { if (e.key === "Escape") { ov.remove(); document.removeEventListener("keydown", f); } };
  document.addEventListener("keydown", f);
}

function lightbox(im, node) {
  const lb = h("div", "sato-light");
  const media = mediaEl(im.large || im.url, im.type === "video" ? "video" : "image");
  if (im.type === "video") media.controls = true;
  const meta = h("div", "meta");
  meta.innerHTML = im.prompt
    ? `<b>Prompt</b><pre>${esc(im.prompt)}</pre>${im.negative ? `<b>Negative</b><pre>${esc(im.negative)}</pre>` : ""}
       <div class="sato-sub">${[im.steps && "steps " + im.steps, im.cfg && "cfg " + im.cfg, im.sampler, im.seed && "seed " + im.seed].filter(Boolean).map(esc).join(" · ")}</div>`
    : `<div class="sato-sub">No prompt shared for this image.</div>`;
  if (im.prompt) {
    const b = h("div", "sato-btns");
    const c = h("div", "sato-btn", "📋 Copy prompt"); c.onclick = (e) => { e.stopPropagation(); copy(im.prompt); };
    b.appendChild(c);
    const s = setupFor(node);
    if (s) {
      const u = h("div", "sato-btn primary", "Use as preview prompt");
      u.onclick = (e) => { e.stopPropagation(); W(s, "prompt").value = im.prompt; app.graph.setDirtyCanvas(true, true); toast("Preview prompt updated"); };
      b.appendChild(u);
    }
    meta.appendChild(b);
  }
  lb.append(media, meta);
  lb.onclick = (e) => { if (e.target === lb) lb.remove(); };
  meta.onclick = (e) => e.stopPropagation();
  document.body.appendChild(lb);
}

async function openInfo(name, node) {
  injectCSS();
  await getPresets();
  const ov = h("div", "sato-ov");
  const md = h("div", "sato-modal");
  const accent = node?.__satoAcc || "#a78bfa";
  md.style.setProperty("--acc", accent);
  ov.appendChild(md);
  ov.onclick = (e) => { if (e.target === ov) ov.remove(); };
  closeOnEsc(ov);
  document.body.appendChild(ov);

  const render = async (refresh) => {
    md.innerHTML = `<div class="sato-mh"><h2>${esc(baseName(name))}</h2></div>
      <div class="sato-mb"><div class="sato-spin"></div><div class="sato-empty">Reading the file hash and asking Civitai…<br>(first time on a big file can take a few seconds)</div></div>`;
    let r;
    try { r = await civitaiInfo(name, { refresh }); }
    catch (e) { r = { error: String(e) }; }
    const cv = r.civitai || {};
    const found = cv.found;
    const user = r.user || {};
    const arch = node ? archFor(node) : null;
    const famArch = arch ? familyOfArch(arch) : "auto";
    const famMismatch = found && r.family && r.family !== "other" && famArch !== "auto" && r.family !== famArch;
    const title = user.name || cv.model_name || baseName(name);
    md.innerHTML = "";
    const head = h("div", "sato-mh");
    head.innerHTML = `<h2 title="${esc(title)}">${esc(title)}</h2>`;
    md.appendChild(head);
    const body = h("div", "sato-mb");
    md.appendChild(body);

    const badges = h("div", "sato-btns");
    badges.innerHTML = `<span class="sato-badge" style="--acc:#c084fc">${esc(cv.type || "LORA")}</span>
      ${cv.base_model ? `<span class="sato-badge ${famMismatch ? "warn" : "ok"}">${esc(cv.base_model)}</span>` : ""}
      ${famMismatch ? `<span class="sato-badge warn">⚠ your setup is ${esc(arch)}</span>` : ""}
      ${r.local_preview ? `<span class="sato-badge dim">local preview image</span>` : ""}
      ${cv.source ? `<span class="sato-badge dim">${esc(cv.source)}</span>` : ""}`;
    body.appendChild(badges);
    if (r.error) body.appendChild(h("div", "sato-msg warn", "⚠ " + esc(r.error)));

    const tbl = h("table", "sato-table");
    const row = (k, v, tip) => { const tr = h("tr"); tr.innerHTML = `<td title="${esc(tip || "")}">${esc(k)}${tip ? " ⓘ" : ""}</td><td></td>`; if (typeof v === "string") tr.children[1].innerHTML = v; else tr.children[1].appendChild(v); tbl.appendChild(tr); return tr; };
    row("File", `${esc(r.file || name)} <span class="sato-sub">(${r.size_mb ?? "?"} MB)</span>`);
    const hashTd = row("Hash (sha256)", `<span style="font-family:ui-monospace,monospace;cursor:pointer" title="click to copy">${esc(r.sha256 || "-")}</span>`);
    hashTd.children[1].onclick = () => r.sha256 && copy(r.sha256);
    row("Civitai", found ? `<a href="${esc(cv.url)}" target="_blank" rel="noopener">View on Civitai ↗</a> <span class="sato-sub">${esc(cv.version_name || "")}</span>`
      : `<span class="sato-sub">${cv && cv.found === false ? "Not found on Civitai (private / renamed / trained locally)." : "Not checked yet."}</span>`);

    const editable = (field, value, placeholder, multiline) => {
      const el = document.createElement(multiline ? "textarea" : "input");
      el.value = value ?? "";
      el.placeholder = placeholder;
      el.onchange = async () => {
        const res = await postJSON(API + "/notes", { name, [field]: el.value });
        r.user = res.user || {};
        INFO.set(name, r);
        toast("Saved");
        node && renderSlot(node);
      };
      return el;
    };
    row("Name", editable("name", user.name, cv.model_name || baseName(name)), "Your own display name for this LoRA (saved locally).");
    if (found) row("Base Model", `${esc(cv.base_model)}`);
    const words = h("div", "sato-chips");
    words.style.maxHeight = "none";
    for (const w of cv.trained_words || []) {
      const c = h("span", "sato-chip", `${esc(w)}<i>${node ? "⊕" : "⧉"}</i>`);
      c.title = node ? "Click: add to the preview prompt. Right-click: copy" : "Click to copy";
      c.onclick = () => (node ? addToPrompt(node, w) : copy(w));
      c.oncontextmenu = (e) => { e.preventDefault(); copy(w); };
      words.appendChild(c);
    }
    if (!(cv.trained_words || []).length) words.innerHTML = `<span class="sato-sub">none listed</span>`;
    row("Trained Words", words, "Words the LoRA was trained with. Put them in your prompt.");
    if (cv.tags?.length) row("Tags", cv.tags.map((t) => `<span class="sato-chip">${esc(t)}</span>`).join(" "));
    if (found) row("Creator / stats", `${esc(cv.creator || "-")} · ⬇ ${fmtN(cv.downloads)} · 👍 ${fmtN(cv.likes)}`);
    const sRow = h("div", "sato-btns");
    sRow.style.alignItems = "center";
    const smin = editable("strength_min", user.strength_min, "e.g. 0.6");
    const smax = editable("strength_max", user.strength_max, "e.g. 1.0");
    smin.style.width = smax.style.width = "110px";
    sRow.append("min", smin, "max", smax);
    if (node && W(node, "strength_model")) {
      const b = h("div", "sato-btn", "Use middle on this slot");
      b.onclick = () => {
        const a = parseFloat(smin.value), z = parseFloat(smax.value);
        if (isNaN(a) && isNaN(z)) return toast("Fill min / max first");
        const v = Math.round(((isNaN(a) ? z : a) + (isNaN(z) ? a : z)) / 2 * 100) / 100;
        W(node, "strength_model").value = v;
        app.graph.setDirtyCanvas(true, true);
        toast("Slot strength = " + v);
      };
      sRow.appendChild(b);
    }
    row("Strength range", sRow, "Your own note of the strength range that works well (saved locally, shown on the slot).");
    row("Additional notes", editable("notes", user.notes, "Anything you want to remember about this LoRA…", true));
    body.appendChild(tbl);

    // gallery
    const ims = cv.images || [];
    if (ims.length) {
      let showNsfw = lsGet("nsfw", false);
      const top = h("div", "sato-tools");
      top.innerHTML = `<span class="sato-sec-t" style="flex:1">Example images (${ims.length})</span>`;
      const lab = h("label", null, `<input type="checkbox" ${showNsfw ? "checked" : ""}> show NSFW`);
      top.appendChild(lab);
      body.appendChild(top);
      const gal = h("div", "sato-gal");
      const fill = () => {
        gal.innerHTML = "";
        for (const im of ims) {
          const it = h("div", "it" + (!showNsfw && im.nsfw > 1 ? " nsfw" : ""));
          it.appendChild(mediaEl(im.thumb, im.type));
          if (im.type === "video") it.appendChild(h("span", "tag", "▶ video"));
          else if (im.prompt) it.appendChild(h("span", "tag", "prompt"));
          it.onclick = () => lightbox(im, node);
          gal.appendChild(it);
        }
      };
      lab.querySelector("input").onchange = (e) => { showNsfw = e.target.checked; lsSet("nsfw", showNsfw); fill(); };
      fill();
      body.appendChild(gal);
    }
    if (cv.description || cv.version_description) {
      const det = h("details", "sato-details");
      det.innerHTML = `<summary>About this LoRA (from Civitai)</summary>`;
      const d = h("div", "sato-desc", sanitize(cv.version_description ? cv.version_description + "<hr>" + cv.description : cv.description));
      det.appendChild(d);
      body.appendChild(det);
    }
    // api key
    const key = h("details", "sato-details sato-key");
    key.innerHTML = `<summary>Civitai API key (optional - only needed for login-restricted models)</summary>
      <div class="sato-tools" style="margin-top:6px"><input type="password" placeholder="paste your Civitai API key"><div class="sato-btn">Save key</div></div>`;
    key.querySelector(".sato-btn").onclick = async () => { await postJSON(API + "/civitai_key", { key: key.querySelector("input").value }); toast("API key saved"); };
    body.appendChild(key);

    const foot = h("div", "sato-mf");
    const rf = h("div", "sato-btn", "↻ Refresh from Civitai"); rf.onclick = () => render(true);
    const cl = h("div", "sato-btn primary", "Close"); cl.onclick = () => ov.remove();
    foot.append(rf);
    if (found) { const op = h("div", "sato-btn", "Open on Civitai ↗"); op.onclick = () => window.open(cv.url, "_blank", "noopener"); foot.append(op); }
    foot.append(cl);
    md.appendChild(foot);
    if (node) renderSlot(node);
  };
  render(false);
}

// ------------------------------------------------------------------------------------
// LoRA browser
// ------------------------------------------------------------------------------------
function familyGuess(entry) {
  const b = (entry.base_model || "").toLowerCase().replace(/[\s._-]/g, "");
  const n = (entry.name || "").toLowerCase().replace(/[\s._-]/g, "");
  const s = b || n;
  if (/krea/.test(s) && !/flux1/.test(b)) return "krea2";
  if (/klein/.test(s)) return "klein";
  if (/zimage|zit\b/.test(s)) return "zimage";
  if (/qwen/.test(s)) return "qwen21";
  return b ? "other" : null;
}

async function openBrowser(node) {
  injectCSS();
  await getPresets();
  const fam = familyOfArch(archFor(node));
  const current = W(node, "lora_name")?.value;
  const ov = h("div", "sato-ov");
  const md = h("div", "sato-modal");
  md.style.setProperty("--acc", node.__satoAcc || "#a78bfa");
  md.style.width = "min(1250px,96vw)";
  ov.appendChild(md);
  ov.onclick = (e) => { if (e.target === ov) ov.remove(); };
  closeOnEsc(ov);
  document.body.appendChild(ov);
  md.innerHTML = `<div class="sato-mh"><h2>Pick a LoRA</h2></div><div class="sato-mb"><div class="sato-spin"></div></div>`;
  let list = [];
  try { list = (await getJSON(API + "/lora_list")).loras || []; } catch (e) { md.querySelector(".sato-mb").innerHTML = `<div class="sato-msg bad">${esc(e)}</div>`; return; }
  const body = md.querySelector(".sato-mb");
  body.innerHTML = "";
  const tools = h("div", "sato-tools");
  tools.innerHTML = `<input type="search" placeholder="Search by file name, title or base model…">
    <label><input type="checkbox" class="f" ${fam !== "auto" && lsGet("browseFilter", true) ? "checked" : ""} ${fam === "auto" ? "disabled" : ""}> only ${esc(archFor(node))} LoRAs</label>
    <label><input type="checkbox" class="n" ${lsGet("nsfw", false) ? "checked" : ""}> show NSFW</label>`;
  const fetchAll = h("div", "sato-btn");
  tools.appendChild(fetchAll);
  body.appendChild(tools);
  const hint = h("div", "sato-help", "Pictures come from Civitai (press “Fetch Civitai info”) or from an image saved next to the LoRA file (name.png / name.preview.png). Click a card to use it, ⓘ for details.");
  body.appendChild(hint);
  const grid = h("div", "sato-bgrid");
  body.appendChild(grid);
  const foot = h("div", "sato-mf");
  const cl = h("div", "sato-btn primary", "Close"); cl.onclick = () => ov.remove();
  foot.appendChild(cl);
  md.appendChild(foot);

  const q = tools.querySelector("input[type=search]");
  const fbox = tools.querySelector(".f");
  const nbox = tools.querySelector(".n");
  const updFetch = () => { const miss = list.filter((e) => !e.checked).length; fetchAll.textContent = miss ? `☁ Fetch Civitai info (${miss} not checked)` : "✓ All checked on Civitai"; };
  const fill = () => {
    const term = q.value.trim().toLowerCase();
    const showN = nbox.checked;
    grid.innerHTML = "";
    let shown = 0;
    for (const e of list) {
      const f = familyGuess(e);
      if (fbox.checked && fam !== "auto" && f && f !== fam) continue;
      const hay = `${e.name} ${e.title} ${e.base_model} ${e.user?.name || ""}`.toLowerCase();
      if (term && !hay.includes(term)) continue;
      shown++;
      const c = h("div", "sato-bcard" + (e.name === current ? " sel" : ""));
      const im = h("div", "im");
      const thumb = e.thumb && (showN || !e.thumb.nsfw) ? e.thumb : null;
      if (thumb) im.appendChild(mediaEl(thumb.url, thumb.type));
      else if (e.local_preview) im.appendChild(mediaEl(`${API}/local_preview?name=${encodeURIComponent(e.name)}`, "image"));
      else im.textContent = baseName(e.name).slice(0, 2).toUpperCase();
      c.appendChild(im);
      const tx = h("div", "tx", `<b title="${esc(e.name)}">${esc(e.user?.name || e.title || baseName(e.name))}</b><span>${esc(e.base_model || (e.checked ? (e.found ? "" : "not on Civitai") : "not checked"))} · ${esc(baseName(e.name))}</span>`);
      c.appendChild(tx);
      const inf = h("div", "inf", "ⓘ"); inf.title = "Details";
      inf.onclick = (ev) => { ev.stopPropagation(); openInfo(e.name, node); };
      c.appendChild(inf);
      c.onclick = () => {
        const lw = W(node, "lora_name");
        if (lw) { lw.value = e.name; lw.callback?.(e.name); }
        ov.remove();
        toast("Selected " + baseName(e.name));
      };
      grid.appendChild(c);
    }
    if (!shown) grid.innerHTML = `<div class="sato-empty">No LoRA matches. ${fbox.checked ? "Untick the architecture filter to see all." : ""}</div>`;
    updFetch();
  };
  q.oninput = fill;
  fbox.onchange = () => { lsSet("browseFilter", fbox.checked); fill(); };
  nbox.onchange = () => { lsSet("nsfw", nbox.checked); fill(); };
  fetchAll.onclick = async () => {
    const todo = list.filter((e) => !e.checked);
    let i = 0;
    for (const e of todo) {
      i++;
      fetchAll.textContent = `☁ Fetching ${i}/${todo.length}…`;
      try {
        const r = await civitaiInfo(e.name);
        const cv = r.civitai || {};
        e.checked = true; e.found = !!cv.found; e.title = cv.model_name || ""; e.base_model = cv.base_model || "";
        const t = firstThumb(r, true);
        if (t) e.thumb = { ...t, nsfw: !firstThumb(r, false) };
      } catch (_) {}
      if (i % 4 === 0) fill();
    }
    fill();
  };
  fill();
  q.focus();
}

// ------------------------------------------------------------------------------------
// Setup node
// ------------------------------------------------------------------------------------
// ---- live check of the native loaders feeding the Setup node (before you even run) ----
const FAM_RX = { krea2: /krea/i, zimage: /z[-_ ]?image/i, klein: /klein/i, qwen21: /qwen[-_ ]?image[-_ ]?2[._]?1/i };
const FAM_LABEL = { krea2: "Krea2", zimage: "Z-Image", klein: "Flux.2 Klein", qwen21: "Qwen-Image 2.1" };
function famFromName(n) { for (const [f, rx] of Object.entries(FAM_RX)) if (rx.test(n || "")) return f; return null; }
function liveIssues(node) {
  const arch = W(node, "architecture")?.value;
  const p = PRESETS?.presets?.[arch] || {};
  const fam = p.family || "auto";
  const out = [];
  if (fam === "auto") return out;
  const unet = upstream(node, "model");
  const un = W(unet, "unet_name")?.value;
  const uf = famFromName(un);
  if (un && uf && uf !== fam) out.push(`Load Diffusion Model has <b>${esc(un)}</b> - that looks like a <b>${FAM_LABEL[uf]}</b> model, not ${esc(arch)}.`);
  const clip = upstream(node, "clip");
  const ct = W(clip, "type")?.value;
  if (clip && ct && p.clip_type && ct !== p.clip_type) out.push(`Load CLIP type is <b>${esc(ct)}</b> - ${esc(arch)} needs type <b>${esc(p.clip_type)}</b>.`);
  const cn = W(clip, "clip_name")?.value;
  const cf = famFromName(cn);
  if (cn && cf && cf !== fam) out.push(`Load CLIP has <b>${esc(cn)}</b> - that looks like a ${FAM_LABEL[cf]} text encoder.`);
  return out;
}
function liveSig(node) {
  const u = upstream(node, "model"), c = upstream(node, "clip");
  return [W(node, "architecture")?.value, W(u, "unet_name")?.value, W(c, "clip_name")?.value, W(c, "type")?.value].join("|");
}

function renderSetupCard(node) {
  const el = node.__satoCard;
  if (!el) return;
  const arch = W(node, "architecture")?.value;
  const p = PRESETS?.presets?.[arch] || {};
  const run = node.__satoRun;
  const issues = PRESETS ? liveIssues(node) : [];
  let status = issues.length ? `<div class="sato-msg bad">⚠ ${issues.join("<br>⚠ ")}</div>` : "";
  if (run && !issues.length) {
    if (run.warnings?.length) status = `<div class="sato-msg warn">⚠ ${esc(run.warnings.join(" "))}</div>`;
    else status = `<div class="sato-msg ok">✓ Model <b>${esc(run.model_class)}</b> matches · CLIP ${run.has_clip ? "✓" : "✗ (previews off)"} · VAE ${run.has_vae ? "✓" : "✗"}${run.has_reference ? " · reference image ✓" : ""}</div>`;
  }
  el.innerHTML = `
    <div class="sato-row-h"><div class="sato-dot"></div><div class="sato-title">${esc(arch)}</div>
      <span class="sato-badge">${esc(p.steps ?? "-")} steps · CFG ${esc(p.cfg ?? "-")}</span></div>
    <div class="sato-kv">
      <span>UNET</span><b title="${esc(p.unet_hint)}">${esc(p.unet_hint || "-")}</b>
      <span>Text enc.</span><b title="${esc(p.te_hint)}">${esc(p.te_hint || "-")}</b>
      <span>VAE</span><b title="${esc(p.vae_hint)}">${esc(p.vae_hint || "-")}</b>
      <span>Sampler</span><b>${esc(p.sampler || "-")} · ${esc(p.scheduler || "-")}${p.shift ? " · shift " + p.shift : ""}</b>
    </div>
    <div class="sato-msg">${esc(p.notes || "")}</div>${status}`;
}

function applyArchDefaults(node, arch) {
  const p = PRESETS?.presets?.[arch];
  if (!p) return;
  for (const [name, v] of [["steps", p.steps], ["cfg", p.cfg], ["sampler", p.sampler], ["scheduler", p.scheduler], ["width", p.width], ["height", p.height], ["shift", p.shift]]) {
    const w = W(node, name);
    if (!w || v == null) continue;
    if (w.options?.values && !w.options.values.includes(v)) continue;
    w.value = v;
  }
  app.graph.setDirtyCanvas(true, true);
}

function initSetup(node) {
  const card = h("div", "sato-card");
  card.style.setProperty("--acc", THEME[T.setup].acc);
  node.__satoCard = card;
  addDOM(node, "arch_card", card, () => (PRESETS && liveIssues(node).length ? 158 + 22 * liveIssues(node).length : 158));
  const aw = W(node, "architecture");
  if (aw) {
    const orig = aw.callback;
    aw.callback = function (v, ...rest) {
      const r = orig?.apply(this, [v, ...rest]);
      applyArchDefaults(node, v);
      node.__satoRun = null;
      renderSetupCard(node);
      window.dispatchEvent(new CustomEvent(EVT));
      window.dispatchEvent(new CustomEvent(EVT + ":arch"));
      return r;
    };
  }
  getPresets().then(() => renderSetupCard(node));
  node.__satoTimer = setInterval(() => {
    if (!node.graph) return clearInterval(node.__satoTimer);
    const sig = liveSig(node);
    if (sig !== node.__satoSig) { node.__satoSig = sig; node.__satoRun = null; renderSetupCard(node); requestAnimationFrame(() => fitNode(node)); }
  }, 1200);
  requestAnimationFrame(() => fitNode(node, 400));
}

// ------------------------------------------------------------------------------------
// LoRA Slot nodes
// ------------------------------------------------------------------------------------
function slotAccent(node) {
  const L = slotLetter(node);
  const acc = L ? colorOf(L) : "#94a3b8";
  node.__satoLetter = L;
  node.__satoAcc = acc;
  node.color = L ? DARK[LETTERS.indexOf(L)] : "#2a2d36";
  node.bgcolor = "#121318";
  return { L, acc };
}

function renderSlot(node) {
  const el = node.__satoCard;
  if (!el) return;
  const { L, acc } = slotAccent(node);
  el.style.setProperty("--acc", acc);
  const name = W(node, "lora_name")?.value;
  const info = node.__satoInfo;
  const run = node.__satoRunInfo && node.__satoRunInfo.name === name ? node.__satoRunInfo : null;
  const civ = INFO.get(name);
  const cv = civ?.civitai || {};
  const user = civ?.user || {};
  el.innerHTML = "";

  // top: thumbnail + title
  const top = h("div", "sato-row-h");
  top.style.alignItems = "stretch";
  const th = h("div", "sato-thumb");
  th.title = "LoRA info / Civitai pictures";
  const t = firstThumb(civ, lsGet("nsfw", false));
  if (t) th.appendChild(mediaEl(t.url, t.type));
  else if (civ?.local_preview) th.appendChild(mediaEl(`${API}/local_preview?name=${encodeURIComponent(name)}`, "image"));
  else th.textContent = L || "?";
  th.onclick = () => openInfo(name, node);
  top.appendChild(th);
  const right = h("div");
  right.style.cssText = "flex:1;min-width:0;display:flex;flex-direction:column;gap:4px";
  const title = user.name || cv.model_name || info?.title || baseName(name) || "No LoRA";
  const rowT = h("div", "sato-row-h", `${L ? `<div class="sato-letter">${L}</div>` : ""}<div class="sato-title" title="${esc(name)}">${esc(title)}</div>`);
  right.appendChild(rowT);
  right.appendChild(h("div", "sato-sub", esc(baseName(name))));
  const auto = PRESETS?.auto;
  const arch = archFor(node);
  const famArch = familyOfArch(arch);
  const famLora = civ?.family || (info?.detected_arch && info.detected_arch !== auto ? familyOfArch(info.detected_arch) : null);
  const mismatch = famLora && famLora !== "other" && famArch !== "auto" && famLora !== famArch;
  const bad = h("div", "sato-btns");
  if (cv.base_model) bad.innerHTML += `<span class="sato-badge ${mismatch ? "warn" : "ok"}" title="Civitai base model">${esc(cv.base_model)}</span>`;
  else if (info?.detected_arch) bad.innerHTML += `<span class="sato-badge ${mismatch ? "warn" : ""}">${esc(info.detected_arch === auto ? "unknown arch" : info.detected_arch)}</span>`;
  if (info?.format) bad.innerHTML += `<span class="sato-badge dim">${esc(info.format)}</span>`;
  if (user.strength_min != null || user.strength_max != null)
    bad.innerHTML += `<span class="sato-badge" title="your saved strength range">str ${user.strength_min ?? "?"}–${user.strength_max ?? "?"}</span>`;
  right.appendChild(bad);
  top.appendChild(right);
  el.appendChild(top);

  if (!name) { el.appendChild(h("div", "sato-msg bad", "Pick a LoRA file (or 🔎 Browse).")); }
  else if (info?.error) { el.appendChild(h("div", "sato-msg bad", esc(info.error))); }
  else if (info) {
    const ranks = info.ranks?.length ? (info.ranks.length > 2 ? `${info.ranks[0]}–${info.ranks[info.ranks.length - 1]}` : info.ranks.join(", ")) : "-";
    let applied = `<b title="Run the node to check">–</b>`;
    if (run) applied = run.patched > 0 ? `<b class="ok" title="weights changed by this LoRA on the connected model">✓ ${run.patched} weights</b>` : `<b class="bad">✗ none</b>`;
    el.appendChild(h("div", "sato-grid", `
      <div class="sato-stat"><span>Rank</span><b>${esc(ranks)}</b></div>
      <div class="sato-stat"><span>Size</span><b>${info.size_mb ?? "-"} MB</b></div>
      <div class="sato-stat"><span>Text enc.</span><b>${info.te_modules ? "yes (" + info.te_modules + ")" : "no"}</b></div>
      <div class="sato-stat"><span>Modules</span><b>${info.modules}</b></div>
      <div class="sato-stat"><span>Applied</span>${applied}</div>
      <div class="sato-stat"><span>Preview</span><b>${run?.previewed ? "✓ this LoRA" : "–"}</b></div>`));
    const words = (cv.trained_words?.length ? cv.trained_words : info.trigger_words) || [];
    if (words.length) {
      const chips = h("div", "sato-chips");
      for (const w of words.slice(0, 10)) {
        const c = h("span", "sato-chip", `${esc(w)}<i>⊕</i>`);
        c.title = "Click: add to the preview prompt · right-click: copy";
        c.onclick = () => addToPrompt(node, w);
        c.oncontextmenu = (e) => { e.preventDefault(); copy(w); };
        chips.appendChild(c);
      }
      el.appendChild(chips);
    }
    let msg = "";
    if (mismatch) msg = `<div class="sato-msg warn">⚠ This looks like a ${esc(cv.base_model || info.detected_arch)} LoRA but the setup is ${esc(arch)}.</div>`;
    else if (run && run.matched === 0) msg = `<div class="sato-msg bad">✗ No weights matched the connected model - wrong architecture?</div>`;
    else if (!L && node.comfyClass === T.slot) msg = `<div class="sato-msg">Tip: connect <b>lora</b> to a Merge Studio input to give this slot a letter.</div>`;
    if (info.satodive_recipe) msg += `<div class="sato-msg ok">✦ SatoDive merge</div>`;
    if (msg) el.appendChild(h("div", null, msg));
  } else {
    el.appendChild(h("div", "sato-empty", "Analysing…"));
  }
  // toolbar
  const tb = h("div", "sato-btns");
  const b1 = h("div", "sato-btn primary grow", "▶ Preview this LoRA");
  b1.title = "Runs only this node (and what it needs) and shows the image below.";
  b1.onclick = () => {
    const pw = W(node, "preview");
    if (pw && String(pw.value).toLowerCase().startsWith("off")) { pw.value = pw.options.values[0]; toast("Preview was Off - switched to " + pw.value); }
    queueNode(node);
  };
  const b2 = h("div", "sato-btn", "ⓘ Info");
  b2.title = "Civitai pictures, trigger words, base model, your notes";
  b2.onclick = () => openInfo(name, node);
  const b3 = h("div", "sato-btn", "🔎 Browse");
  b3.title = "Pick a LoRA from a gallery";
  b3.onclick = () => openBrowser(node);
  tb.append(b1, b2, b3);
  el.appendChild(tb);
  requestAnimationFrame(() => {
    // card height follows its content (trigger words, warnings...)
    const scale = app.canvas?.ds?.scale || 1;
    let need = 22;
    for (const c of el.children) need += c.getBoundingClientRect().height / scale + 7;
    need = Math.ceil(clamp(need, 150, 420));
    if (el.getBoundingClientRect().height > 0 && Math.abs((node.__satoH || 0) - need) > 4) { node.__satoH = need; fitNode(node, 380); }
  });
  app.graph.setDirtyCanvas(true, false);
}

async function refreshSlot(node) {
  const name = W(node, "lora_name")?.value;
  node.__satoInfo = null;
  renderSlot(node);
  if (!name) return;
  try {
    await getPresets();
    const [info] = await Promise.all([
      getJSON(`${API}/info?name=${encodeURIComponent(name)}&arch=${encodeURIComponent(archFor(node))}`),
      civitaiInfo(name, { offline: true }).catch(() => null),
    ]);
    node.__satoInfo = info;
  } catch (e) {
    node.__satoInfo = { error: String(e) };
  }
  renderSlot(node);
}

function initSlot(node) {
  const card = h("div", "sato-card");
  node.__satoCard = card;
  node.__satoH = 238;
  addDOM(node, "lora_card", card, () => node.__satoH || 238);
  const lw = W(node, "lora_name");
  if (lw) {
    const orig = lw.callback;
    lw.callback = function (...args) {
      const r = orig?.apply(this, args);
      node.__satoRunInfo = null;
      refreshSlot(node);
      window.dispatchEvent(new CustomEvent(EVT));
      return r;
    };
  }
  node.__satoListener = () => refreshSlot(node);
  window.addEventListener(EVT + ":arch", node.__satoListener);
  const occ = node.onConnectionsChange;
  node.onConnectionsChange = function (...a) {
    const r = occ?.apply(this, a);
    setTimeout(() => renderSlot(node), 30);
    return r;
  };
  setTimeout(() => refreshSlot(node), 50);
  requestAnimationFrame(() => fitNode(node, 380));
}

// ------------------------------------------------------------------------------------
// Merge Studio (2-10 LoRAs, mixer, per-block sliders)
// ------------------------------------------------------------------------------------
const RMIN = -1, RMAX = 2;

const MIX_WIDGET = "mix_settings";

function mergeState(node) {
  if (!node.__sato) {
    let mix = {};
    try { mix = JSON.parse(W(node, MIX_WIDGET)?.value || "{}") || {}; } catch (_) {}
    node.__sato = { mix, groups: [], sources: [], tab: null, closed: node.properties?.satoClosed || {}, report: null };
  }
  return node.__sato;
}
function blockVal(node, letter, g) { const st = mergeState(node); return st.mix[letter]?.[g] ?? 1; }
function setBlockVal(node, letter, g, v) {
  const st = mergeState(node); const k = letter;
  st.mix[k] = st.mix[k] || {};
  if (Math.abs(v - 1) < 1e-6) delete st.mix[k][g]; else st.mix[k][g] = Math.round(v * 1000) / 1000;
}
function saveMix(node) {
  const st = mergeState(node);
  for (const k of Object.keys(st.mix)) if (st.mix[k] && typeof st.mix[k] === "object" && !Array.isArray(st.mix[k]) && !Object.keys(st.mix[k]).length) delete st.mix[k];
  if (Array.isArray(st.mix._mute) && !st.mix._mute.length) delete st.mix._mute;
  if (Array.isArray(st.mix._solo) && !st.mix._solo.length) delete st.mix._solo;
  const w = W(node, MIX_WIDGET);
  if (w) w.value = JSON.stringify(st.mix);
  app.graph.setDirtyCanvas(true, false);
}
function paintRange(r) { r.style.setProperty("--p", ((parseFloat(r.value) - parseFloat(r.min)) / (parseFloat(r.max) - parseFloat(r.min))) * 100 + "%"); }
function mkRange(color, value, disabled, oninput, min = RMIN, max = RMAX) {
  const r = document.createElement("input");
  r.type = "range"; r.className = "sato-range"; r.min = min; r.max = max; r.step = 0.05; r.value = value; r.disabled = !!disabled;
  r.style.setProperty("--c", color);
  paintRange(r);
  r.addEventListener("input", () => { paintRange(r); oninput(parseFloat(r.value)); });
  r.addEventListener("pointerdown", (e) => e.stopPropagation());
  return r;
}
function sectionsOf(groups) {
  const secs = [], idx = {};
  for (const g of groups) {
    if (!(g.section in idx)) { idx[g.section] = secs.length; secs.push({ name: g.section, groups: [] }); }
    secs[idx[g.section]].groups.push(g);
  }
  return secs;
}

function applyBlockPreset(node, letter, kind) {
  const st = mergeState(node);
  const blocks = st.groups.filter((g) => g.id !== "text_encoder" && g.id !== "other");
  const n = Math.max(1, blocks.length - 1);
  const pos = {};
  blocks.forEach((g, i) => (pos[g.id] = i / n));
  for (const g of st.groups) {
    const p = pos[g.id];
    let v = blockVal(node, letter, g.id);
    switch (kind) {
      case "full": v = 1; break;
      case "off": v = 0; break;
      case "comp": if (p != null) v = p <= 0.45 ? 1 : p <= 0.6 ? 0.5 : 0.1; break;
      case "style": if (p != null) v = p >= 0.55 ? 1 : p >= 0.4 ? 0.5 : 0.1; break;
      case "up": if (p != null) v = 0.2 + 0.8 * p; break;
      case "down": if (p != null) v = 1 - 0.8 * p; break;
      case "teoff": if (g.id === "text_encoder") v = 0; break;
    }
    setBlockVal(node, letter, g.id, Math.round(v * 100) / 100);
  }
  saveMix(node);
  renderMerge(node);
}

function renderMerge(node) {
  const el = node.__satoCard;
  if (!el) return;
  const st = mergeState(node);
  const help = PRESETS?.help || {};
  const scrollTop = el.querySelector(".sato-scroll")?.scrollTop || 0;
  el.innerHTML = "";
  const srcs = st.sources;
  const letters = srcs.map((s) => s.letter);
  if (!st.tab || !letters.includes(st.tab)) st.tab = letters[0] || null;

  // ---- action buttons
  const acts = h("div", "sato-btns");
  const pv = h("div", "sato-btn primary grow", "▶ Preview merge");
  pv.title = "Runs only this node (and the LoRA slots it needs) and shows the result below.";
  pv.onclick = () => queueNode(node);
  const sv = h("div", "sato-btn grow", "💾 Merge & Save LoRA");
  sv.title = "Merges and writes models/loras/SatoDive/<filename>_###.safetensors";
  sv.onclick = async () => {
    const sw = W(node, "save_lora");
    const prev = sw?.value;
    if (sw) sw.value = true;
    try { await queueNode(node); toast("Merging & saving…"); } finally { if (sw) sw.value = prev; app.graph.setDirtyCanvas(true, false); }
  };
  acts.append(pv, sv);
  el.appendChild(acts);

  // ---- method guide
  {
    const m = W(node, "method")?.value;
    const mh = help.method_help?.[m];
    if (mh) el.appendChild(h("div", "sato-guide", `<b class="t">${esc(m)}</b>${esc(mh.what)}<div class="pick">👉 ${esc(mh.pick)}</div><div class="cost">${esc(mh.cost)}</div>`));
    const det = h("details", "sato-details");
    det.open = !!node.properties?.satoHelpOpen;
    det.ontoggle = () => { node.properties = node.properties || {}; node.properties.satoHelpOpen = det.open; requestAnimationFrame(() => layoutMerge(node)); };
    det.innerHTML = `<summary>What do the other settings do?</summary><ul>${Object.entries(help.tips || {}).map(([k, v]) => `<li><b>${esc(k)}</b>: ${esc(v)}</li>`).join("")}</ul>`;
    el.appendChild(det);
  }

  // ---- mixer
  el.appendChild(h("div", "sato-sec-t", "Mixer - how much of each LoRA"));
  if (!srcs.length) {
    el.appendChild(h("div", "sato-empty",
      "Connect two or more <b>LoRA Slot</b> nodes to <b>lora_A</b>, <b>lora_B</b>… - a new input appears every time you connect one."));
  }
  const mute = new Set(st.mix._mute || []), solo = new Set(st.mix._solo || []);
  for (const s of srcs) {
    const c = colorOf(s.letter);
    const muted = mute.has(s.letter) || (solo.size && !solo.has(s.letter));
    const row = h("div", "sato-mixrow" + (muted ? " muted" : ""));
    row.style.setProperty("--acc", c);
    const lt = h("div", "sato-letter", s.letter);
    const civ = INFO.get(s.name);
    const nm = h("div", "nm", esc(civ?.user?.name || civ?.civitai?.model_name || baseName(s.name)));
    nm.title = s.name;
    row.append(lt, nm);
    {
      const gain = st.mix._mix?.[s.letter] ?? 1;
      const val = h("div", "sato-val", Number(gain).toFixed(2));
      val.title = "double-click: reset to 1.00";
      const r = mkRange(c, gain, false, (v) => {
        st.mix._mix = st.mix._mix || {};
        if (Math.abs(v - 1) < 1e-6) delete st.mix._mix[s.letter]; else st.mix._mix[s.letter] = Math.round(v * 100) / 100;
        if (!Object.keys(st.mix._mix).length) delete st.mix._mix;
        val.textContent = v.toFixed(2); saveMix(node);
      }, 0, 2);
      val.ondblclick = () => { r.value = 1; paintRange(r); r.dispatchEvent(new Event("input")); };
      const M = h("div", "sato-ms m" + (mute.has(s.letter) ? " on" : ""), "M"); M.title = "Mute this LoRA";
      const S = h("div", "sato-ms s" + (solo.has(s.letter) ? " on" : ""), "S"); S.title = "Solo: only the soloed LoRAs are used";
      M.onclick = () => { mute.has(s.letter) ? mute.delete(s.letter) : mute.add(s.letter); st.mix._mute = [...mute]; saveMix(node); renderMerge(node); };
      S.onclick = () => { solo.has(s.letter) ? solo.delete(s.letter) : solo.add(s.letter); st.mix._solo = [...solo]; saveMix(node); renderMerge(node); };
      row.append(r, val, M, S);
    }
    el.appendChild(row);
  }

  // ---- per-block fine tune
  if (srcs.length && st.groups.length) {
    el.appendChild(h("div", "sato-sec-t", "Fine-tune per block"));
    const tabs = h("div", "sato-tabs");
    for (const L of letters) {
      const t = h("div", "sato-tab" + (L === st.tab ? " on" : ""), "LoRA " + L);
      t.style.setProperty("--c", colorOf(L));
      t.onclick = () => { st.tab = L; renderMerge(node); };
      tabs.appendChild(t);
    }
    el.appendChild(tabs);
    const L = st.tab;
    const c = colorOf(L);
    const pr = h("div", "sato-btns");
    pr.style.setProperty("--acc", c);
    for (const [k, lab, tip] of [["full", "Full", "Every block at 1.0"], ["comp", "Composition", "Keep this LoRA's layout / pose / shapes (early blocks)"],
      ["style", "Style & detail", "Keep this LoRA's colors, textures, look (late blocks)"], ["up", "Fade in", "Weak early → strong late"],
      ["down", "Fade out", "Strong early → weak late"], ["teoff", "No text-enc.", "Ignore this LoRA's text-encoder part"], ["off", "Off", "Every block at 0"]]) {
      const b = h("div", "sato-btn", lab); b.title = tip;
      b.onclick = () => applyBlockPreset(node, L, k);
      pr.appendChild(b);
    }
    el.appendChild(pr);
    if (help.block_help) el.appendChild(h("div", "sato-help", "ⓘ " + esc(help.block_help)));
    const sc = h("div", "sato-scroll");
    sc.style.height = node.__satoScrollH + "px";
    for (const sec of sectionsOf(st.groups)) {
      const box = h("div", "sato-sec" + (st.closed[sec.name] ? " closed" : ""));
      const has = (g) => (g.counts?.[L] ?? 0) > 0;
      const anyHas = sec.groups.some(has);
      const avg = sec.groups.reduce((t, g) => t + blockVal(node, L, g.id), 0) / sec.groups.length;
      const head = h("div", "sato-bh");
      const car = h("div", "car", "▾");
      const lbl = h("div", "lbl", `${esc(sec.name)}${sec.groups.length > 1 ? " <span class='sato-sub'>×" + sec.groups.length + "</span>" : ""}`);
      const hv = h("div", "sato-val", avg.toFixed(2));
      const rowRefs = [];
      const hr = mkRange(c, avg, !anyHas, (v) => {
        for (const g of sec.groups) setBlockVal(node, L, g.id, v);
        hv.textContent = v.toFixed(2);
        for (const rr of rowRefs) { rr.r.value = v; paintRange(rr.r); rr.v.textContent = v.toFixed(2); }
        saveMix(node);
      });
      hr.addEventListener("click", (e) => e.stopPropagation());
      head.append(car, lbl, hr, hv);
      head.onclick = (e) => {
        if (e.target.tagName === "INPUT") return;
        st.closed[sec.name] = !st.closed[sec.name];
        node.properties = node.properties || {};
        node.properties.satoClosed = st.closed;
        box.classList.toggle("closed");
      };
      box.appendChild(head);
      if (sec.groups.length > 1 || sec.groups[0].label !== sec.name) {
        const rows = h("div", "sato-rows");
        for (const g of sec.groups) {
          const row = h("div", "sato-br" + (has(g) ? "" : " none"));
          const who = letters.filter((x) => (g.counts?.[x] ?? 0) > 0);
          const dot = h("div", null, `<span title="used by: ${who.join(", ") || "-"}" style="display:inline-block;width:7px;height:7px;border-radius:50%;background:${has(g) ? c : "#555"}"></span>`);
          const l = h("div", "lbl", esc(g.label));
          l.title = `${g.id} · used by LoRA ${who.join(", ") || "-"}`;
          const v0 = blockVal(node, L, g.id);
          const v = h("div", "sato-val", has(g) ? v0.toFixed(2) : "–");
          v.title = "double-click: reset to 1.00";
          const r = mkRange(c, v0, !has(g), (x) => { setBlockVal(node, L, g.id, x); v.textContent = x.toFixed(2); saveMix(node); });
          v.ondblclick = () => { setBlockVal(node, L, g.id, 1); r.value = 1; paintRange(r); v.textContent = "1.00"; saveMix(node); };
          rowRefs.push({ r, v });
          row.append(dot, l, r, v);
          rows.appendChild(row);
        }
        box.appendChild(rows);
      }
      sc.appendChild(box);
    }
    el.appendChild(sc);
    requestAnimationFrame(() => { sc.scrollTop = scrollTop; });
  } else if (srcs.length && !st.groups.length) {
    el.appendChild(h("div", "sato-empty", "Block sliders appear once the LoRAs are analysed (or after the first preview)."));
  }

  // ---- report
  const rep = h("div", "sato-report");
  const r = st.report;
  if (r) {
    const rk = r.ranks?.length ? (r.ranks.length > 1 ? r.ranks[0] + "–" + r.ranks[r.ranks.length - 1] : r.ranks[0]) : "-";
    rep.innerHTML = `<span class="sato-badge ok">${r.modules} weights merged</span>
      <span class="sato-badge" title="LoRA rank of the result">rank ${esc(rk)}</span>
      <span class="sato-badge ${r.energy_kept < 90 ? "warn" : "ok"}" title="How much of the mix survived compression (100% = exact)">kept ${r.energy_kept}%</span>
      ${r.size_mb ? `<span class="sato-badge dim">${r.size_mb} MB</span>` : ""}
      ${r.saved ? `<span class="sato-badge ok" title="${esc(r.saved)}">💾 saved ${esc(baseName(r.saved))}</span>` : ""}`;
  } else {
    rep.innerHTML = `<span class="sato-badge dim">Adjust, then ▶ Preview merge - or 💾 save</span>`;
  }
  el.appendChild(rep);
  requestAnimationFrame(() => layoutMerge(node));
}

function layoutMerge(node) {
  // panel height = everything except the scroll area + scroll area sized to its rows
  const el = node.__satoCard;
  if (!el) return;
  const st = mergeState(node);
  const secs = sectionsOf(st.groups);
  const rowsH = secs.reduce((t, s) => t + 25 + (st.closed[s.name] ? 0 : (s.groups.length > 1 || s.groups[0].label !== s.name ? s.groups.length * 22 : 0)), 0);
  const want = clamp(rowsH + 10, 70, 330);
  const sc = el.querySelector(".sato-scroll");
  if (sc) sc.style.height = want + "px";
  node.__satoScrollH = want;
  let total = 18;
  for (const c of el.children) total += c.getBoundingClientRect().height / (app.canvas?.ds?.scale || 1) + 7;
  if (sc) total = total - sc.getBoundingClientRect().height / (app.canvas?.ds?.scale || 1) + want;
  const newH = Math.ceil(clamp(total, 160, 1400));
  if (Math.abs((node.__satoH || 0) - newH) > 4) {
    node.__satoH = newH;
    fitNode(node, 540);
  }
}

async function refreshMerge(node) {
  const st = mergeState(node);
  st.sources = mergeSources(node);
  renderMerge(node);
  await getPresets();
  const files = st.sources.map((s) => (s.merged ? "" : s.name));
  await Promise.all(st.sources.filter((s) => !s.merged && s.name).map((s) => civitaiInfo(s.name, { offline: true }).catch(() => null)));
  let groups = [];
  if (files.some(Boolean)) {
    try {
      const res = await getJSON(`${API}/groups?names=${encodeURIComponent(files.join("|"))}&arch=${encodeURIComponent(archFor(node))}`);
      groups = (res.groups || []).map((g) => ({ ...g, counts: Object.fromEntries(st.sources.map((s, i) => [s.letter, g.counts?.[i] ?? 0])) }));
    } catch (e) { console.warn("[SatoDive] group detection failed", e); }
  }
  // chained merges: use the groups reported by the upstream merge run
  const byId = new Map(groups.map((g) => [g.id, g]));
  for (const s of st.sources) {
    if (!s.merged) continue;
    for (const g of s.node.__sato?.groups || []) {
      const cur = byId.get(g.id) || { ...g, counts: {} };
      cur.counts = { ...cur.counts, [s.letter]: Object.values(g.counts || {}).reduce((a, b) => a + b, 0) || 1 };
      byId.set(g.id, cur);
    }
  }
  st.groups = [...byId.values()];
  renderMerge(node);
}

function initMerge(node) {
  hideWidget(node, MIX_WIDGET);
  node.__satoScrollH = 120;
  const card = h("div", "sato-card");
  card.style.setProperty("--acc", THEME[node.comfyClass].acc);
  node.__satoCard = card;
  addDOM(node, "studio", card, () => node.__satoH || 420);
  const mw = W(node, "method");
  if (mw) {
    const orig = mw.callback;
    mw.callback = function (...a) { const r = orig?.apply(this, a); renderMerge(node); return r; };
  }
  node.__satoListener = () => { clearTimeout(node.__satoT); node.__satoT = setTimeout(() => refreshMerge(node), 120); };
  window.addEventListener(EVT, node.__satoListener);
  const occ = node.onConnectionsChange;
  node.onConnectionsChange = function (...args) {
    const r = occ?.apply(this, args);
    node.__satoListener();
    // re-letter the slots
    setTimeout(() => (node.graph ?? app.graph)._nodes?.forEach((n) => SLOT_TYPES.includes(n.comfyClass) && renderSlot(n)), 60);
    return r;
  };
  setTimeout(() => refreshMerge(node), 80);
  requestAnimationFrame(() => fitNode(node, 540));
}

// ------------------------------------------------------------------------------------
// Registration
// ------------------------------------------------------------------------------------
app.registerExtension({
  name: "SatoDive.LoRAMergeStudio",

  async setup() {
    injectCSS();
    getPresets();
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!Object.values(T).includes(nodeData.name)) return;
    injectCSS();

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      const th = THEME[nodeData.name];
      if (th) { this.color = th.color; this.bgcolor = th.bgcolor; }
      keepFitted(this);
      if (nodeData.name === T.setup) initSetup(this);
      else if (SLOT_TYPES.includes(nodeData.name)) initSlot(this);
      else if (MERGE_TYPES.includes(nodeData.name)) initMerge(this);
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      const th = THEME[nodeData.name];
      if (th) { this.color = th.color; this.bgcolor = th.bgcolor; }
      if (MERGE_TYPES.includes(nodeData.name)) {
        this.__sato = null;
        mergeState(this);
        setTimeout(() => refreshMerge(this), 200);
      } else if (SLOT_TYPES.includes(nodeData.name)) {
        setTimeout(() => refreshSlot(this), 200);
      } else if (nodeData.name === T.setup) {
        setTimeout(() => renderSetupCard(this), 150);
      }
      return r;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (msg) {
      const r = onExecuted?.apply(this, arguments);
      const last = (a) => (Array.isArray(a) && a.length ? a[a.length - 1] : null);
      if (nodeData.name === T.setup && last(msg?.sato_setup)) {
        this.__satoRun = last(msg.sato_setup);
        renderSetupCard(this);
      }
      if (SLOT_TYPES.includes(nodeData.name) && last(msg?.sato_info)) {
        const info = last(msg.sato_info);
        this.__satoRunInfo = { name: W(this, "lora_name")?.value, ...info };
        this.__satoInfo = { ...(this.__satoInfo || {}), ...info };
        renderSlot(this);
      }
      if (MERGE_TYPES.includes(nodeData.name)) {
        const st = mergeState(this);
        if (last(msg?.sato_report)) st.report = last(msg.sato_report);
        if (Array.isArray(msg?.sato_groups) && msg.sato_groups.length) {
          const letters = msg.sato_letters || ["A", "B"];
          const byId = new Map(st.groups.map((g) => [g.id, g]));
          for (const g of msg.sato_groups) {
            if (!g || typeof g.id !== "string") continue;
            const counts = g.by || (g.counts ? Object.fromEntries(letters.map((L, i) => [L, g.counts[i] ?? 0])) : {});
            byId.set(g.id, { ...(byId.get(g.id) || {}), ...g, counts });
          }
          st.groups = [...byId.values()];
        }
        renderMerge(this);
      }
      return r;
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      if (this.__satoTimer) clearInterval(this.__satoTimer);
      if (this.__satoListener) {
        window.removeEventListener(EVT, this.__satoListener);
        window.removeEventListener(EVT + ":arch", this.__satoListener);
      }
      return onRemoved?.apply(this, arguments);
    };
  },
});
