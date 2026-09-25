// LoRA Merge Studio - SatoDive
// Frontend: themed nodes, LoRA info cards, per-block A/B slider studio, run / save buttons.
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const T = {
  setup: "SatoDiveLoRAStudioSetup",
  a: "SatoDiveLoRASlotA",
  b: "SatoDiveLoRASlotB",
  merge: "SatoDiveLoRAMergeStudio",
};

const THEME = {
  [T.setup]: { color: "#3b2f0f", bgcolor: "#17150f", acc: "#fbbf24" },
  [T.a]: { color: "#0b3d47", bgcolor: "#0e181b", acc: "#22d3ee" },
  [T.b]: { color: "#4b1437", bgcolor: "#1b0f16", acc: "#f472b6" },
  [T.merge]: { color: "#2d1d63", bgcolor: "#141122", acc: "#a78bfa" },
};

const ACC_A = "#22d3ee";
const ACC_B = "#f472b6";
const EVT = "satodive:lora-changed";

// ------------------------------------------------------------------------------------
// Styles
// ------------------------------------------------------------------------------------
const CSS = `
.sato-card{--acc:#a78bfa;box-sizing:border-box;width:100%;height:100%;overflow:hidden;color:#e5e7eb;
  font:500 11.5px/1.35 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
  background:linear-gradient(160deg,rgba(255,255,255,.055),rgba(255,255,255,.012));
  border:1px solid rgba(255,255,255,.09);border-radius:12px;padding:9px 11px;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.05),0 6px 18px rgba(0,0,0,.25)}
.sato-card *{box-sizing:border-box}
.sato-head{display:flex;align-items:center;gap:7px;margin-bottom:7px;min-width:0}
.sato-title{font-weight:700;font-size:12.5px;color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1;min-width:0}
.sato-dot{width:8px;height:8px;border-radius:50%;background:var(--acc);box-shadow:0 0 10px var(--acc);flex:none}
.sato-badge{flex:none;padding:1px 8px;border-radius:999px;font-size:10px;font-weight:700;letter-spacing:.02em;
  color:var(--acc);background:color-mix(in srgb,var(--acc) 16%,transparent);border:1px solid color-mix(in srgb,var(--acc) 42%,transparent)}
.sato-badge.warn{--acc:#fbbf24}.sato-badge.bad{--acc:#f87171}.sato-badge.ok{--acc:#4ade80}.sato-badge.dim{--acc:#94a3b8}
.sato-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:5px}
.sato-stat{background:rgba(0,0,0,.28);border:1px solid rgba(255,255,255,.05);border-radius:8px;padding:5px 7px;min-width:0}
.sato-stat b{display:block;font-size:12.5px;color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sato-stat span{display:block;opacity:.55;font-size:9px;text-transform:uppercase;letter-spacing:.08em}
.sato-chips{display:flex;flex-wrap:wrap;gap:4px;margin-top:7px;max-height:44px;overflow:hidden}
.sato-chip{cursor:pointer;padding:2px 7px;border-radius:6px;font-size:10.5px;background:rgba(255,255,255,.06);
  border:1px solid rgba(255,255,255,.08);color:#e5e7eb;transition:all .15s}
.sato-chip:hover{background:color-mix(in srgb,var(--acc) 25%,transparent);border-color:var(--acc)}
.sato-msg{margin-top:6px;font-size:10.5px;opacity:.85}
.sato-msg.warn{color:#fbbf24}.sato-msg.bad{color:#f87171}.sato-msg.ok{color:#4ade80}
.sato-kv{display:grid;grid-template-columns:auto 1fr;gap:3px 10px;font-size:10.5px}
.sato-kv span{opacity:.55}.sato-kv b{font-weight:600;color:#f1f5f9;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sato-panel{display:flex;flex-direction:column;gap:6px}
.sato-ab{display:flex;gap:6px;min-width:0}
.sato-ab .sato-pill{flex:1;min-width:0;display:flex;gap:6px;align-items:center;padding:4px 8px;border-radius:8px;
  background:rgba(0,0,0,.3);border:1px solid color-mix(in srgb,var(--c) 40%,transparent)}
.sato-ab .sato-pill i{font-style:normal;font-weight:800;color:var(--c)}
.sato-ab .sato-pill em{font-style:normal;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;opacity:.9}
.sato-presets{display:flex;flex-wrap:wrap;gap:4px}
.sato-btn{cursor:pointer;user-select:none;padding:3px 8px;border-radius:7px;font-size:10.5px;font-weight:600;color:#e5e7eb;
  background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);transition:all .12s}
.sato-btn:hover{background:rgba(167,139,250,.22);border-color:#a78bfa;color:#fff}
.sato-scroll{overflow-y:auto;overflow-x:hidden;padding-right:4px;border-radius:8px;background:rgba(0,0,0,.18);
  border:1px solid rgba(255,255,255,.05);scrollbar-width:thin;scrollbar-color:#4c3d86 transparent}
.sato-sec{border-bottom:1px solid rgba(255,255,255,.05)}
.sato-sec-h{display:grid;grid-template-columns:14px minmax(60px,1fr) minmax(0,1.3fr) 34px minmax(0,1.3fr) 34px;gap:6px;align-items:center;
  padding:5px 6px;font-weight:700;cursor:pointer;background:rgba(255,255,255,.035);position:sticky;top:0;z-index:1;backdrop-filter:blur(6px)}
.sato-sec-h .car{opacity:.6;transition:transform .15s}.sato-sec.closed .car{transform:rotate(-90deg)}
.sato-sec.closed .sato-rows{display:none}
.sato-row{display:grid;grid-template-columns:14px minmax(60px,1fr) minmax(0,1.3fr) 34px minmax(0,1.3fr) 34px;gap:6px;align-items:center;padding:2px 6px;font-size:10.5px}
.sato-row:hover{background:rgba(255,255,255,.03)}
.sato-row .lbl,.sato-sec-h .lbl{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sato-val{text-align:right;font-variant-numeric:tabular-nums;font-size:10.5px;cursor:pointer;opacity:.9}
.sato-val.off{opacity:.3}
.sato-range{-webkit-appearance:none;appearance:none;width:100%;height:5px;border-radius:999px;outline:none;margin:0;cursor:pointer;
  background:linear-gradient(90deg,var(--c) 0%,var(--c) var(--p),rgba(255,255,255,.1) var(--p))}
.sato-range::-webkit-slider-thumb{-webkit-appearance:none;width:12px;height:12px;border-radius:50%;background:#fff;
  border:2px solid var(--c);box-shadow:0 0 8px var(--c)}
.sato-range::-moz-range-thumb{width:10px;height:10px;border-radius:50%;background:#fff;border:2px solid var(--c)}
.sato-range:disabled{opacity:.22;cursor:not-allowed}
.sato-report{font-size:10.5px;display:flex;flex-wrap:wrap;gap:4px}
.sato-empty{padding:18px 8px;text-align:center;opacity:.6}
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
const W = (node, name) => node.widgets?.find((w) => w.name === name);
const baseName = (s) => String(s || "").split(/[\\/]/).pop().replace(/\.(safetensors|ckpt|pt|pth|bin)$/i, "");

let PRESETS = null;
async function getPresets() {
  if (PRESETS) return PRESETS;
  try {
    const r = await api.fetchApi("/satodive/lora_merge/presets");
    PRESETS = await r.json();
  } catch (e) {
    console.warn("[SatoDive] presets unavailable", e);
    PRESETS = { presets: {}, auto: "Auto-detect / Other" };
  }
  return PRESETS;
}

async function getJSON(url) {
  const r = await api.fetchApi(url);
  return r.json();
}

function linkById(graph, id) {
  const L = graph?.links;
  if (!L) return null;
  return typeof L.get === "function" ? L.get(id) : L[id];
}

function inputLink(node, slot) {
  if (slot < 0 || !node.inputs?.[slot]) return null;
  if (typeof node.getInputLink === "function") {
    try { return node.getInputLink(slot); } catch (_) {}
  }
  const id = node.inputs[slot].link;
  return id == null ? null : linkById(node.graph ?? app.graph, id);
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
    if (n.type === "Reroute" || n.comfyClass === "Reroute") {
      cur = n;
      slot = 0;
      continue;
    }
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

async function queueNode(node) {
  try {
    await app.queuePrompt(0, 1, [String(node.id)]);
    return;
  } catch (e) {
    console.warn("[SatoDive] partial queue failed, trying command", e);
  }
  try {
    app.canvas.selectNode?.(node, false);
    await app.extensionManager.command.execute("Comfy.QueueSelectedOutputNodes");
  } catch (e) {
    await app.queuePrompt(0, 1);
  }
}

function addButton(node, label, cb) {
  const w = node.addWidget("button", label, null, cb, { serialize: false });
  w.serialize = false;
  return w;
}

function addDOM(node, name, el, height) {
  const h = typeof height === "function" ? height : () => height;
  const w = node.addDOMWidget(name, "sato_" + name, el, {
    serialize: false,
    hideOnZoom: false,
    getMinHeight: () => h(),
    getMaxHeight: () => h(),
    getHeight: () => h(),
  });
  w.serialize = false;
  w.computeSize = (width) => [width, h() + 6];
  el.addEventListener("wheel", (e) => e.stopPropagation(), { passive: true });
  return w;
}

function fitNode(node, minW = 0) {
  const sz = node.computeSize();
  node.setSize([Math.max(node.size[0], sz[0], minW), Math.max(sz[1], 40)]);
  node.setDirtyCanvas?.(true, true);
  app.graph.setDirtyCanvas(true, true);
}

function theme(node) {
  const t = THEME[node.comfyClass];
  if (!t) return;
  node.color = t.color;
  node.bgcolor = t.bgcolor;
}

function copy(text) {
  try {
    navigator.clipboard?.writeText(text);
  } catch (_) {}
}

// ------------------------------------------------------------------------------------
// Setup node
// ------------------------------------------------------------------------------------
function renderSetupCard(node) {
  const el = node.__satoCard;
  if (!el) return;
  const arch = W(node, "architecture")?.value;
  const p = PRESETS?.presets?.[arch] || {};
  const run = node.__satoRun;
  let status = "";
  if (run) {
    if (run.warnings?.length) status = `<div class="sato-msg warn">⚠ ${esc(run.warnings.join(" "))}</div>`;
    else status = `<div class="sato-msg ok">✓ Model class <b>${esc(run.model_class)}</b> matches · CLIP ${run.has_clip ? "✓" : "✗"} · VAE ${run.has_vae ? "✓" : "✗"}${run.has_reference ? " · reference image ✓" : ""}</div>`;
  }
  el.innerHTML = `
    <div class="sato-head"><div class="sato-dot"></div><div class="sato-title">${esc(arch)}</div>
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
  const set = (name, v) => {
    const w = W(node, name);
    if (w && v !== undefined && v !== null) {
      if (w.options?.values && !w.options.values.includes(v)) return;
      w.value = v;
    }
  };
  set("steps", p.steps);
  set("cfg", p.cfg);
  set("sampler", p.sampler);
  set("scheduler", p.scheduler);
  set("width", p.width);
  set("height", p.height);
  set("shift", p.shift);
  app.graph.setDirtyCanvas(true, true);
}

function initSetup(node) {
  const card = document.createElement("div");
  card.className = "sato-card";
  card.style.setProperty("--acc", THEME[T.setup].acc);
  node.__satoCard = card;
  addDOM(node, "arch_card", card, 148);

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
  requestAnimationFrame(() => fitNode(node, 400));
}

// ------------------------------------------------------------------------------------
// Slot nodes (A / B)
// ------------------------------------------------------------------------------------
function renderSlotCard(node, info, loading) {
  const el = node.__satoCard;
  if (!el) return;
  const slot = node.comfyClass === T.a ? "A" : "B";
  const name = W(node, "lora_name")?.value;
  if (loading) {
    el.innerHTML = `<div class="sato-head"><div class="sato-dot"></div><div class="sato-title">${esc(baseName(name))}</div><span class="sato-badge">LoRA ${slot}</span></div><div class="sato-empty">Analysing…</div>`;
    return;
  }
  if (!info || info.error) {
    el.innerHTML = `<div class="sato-head"><div class="sato-dot"></div><div class="sato-title">${esc(baseName(name) || "No LoRA")}</div><span class="sato-badge">LoRA ${slot}</span></div><div class="sato-msg bad">${esc(info?.error || "Pick a LoRA file.")}</div>`;
    return;
  }
  const arch = archFor(node);
  const auto = PRESETS?.auto;
  const mismatch = arch && arch !== auto && info.detected_arch && info.detected_arch !== auto && info.detected_arch !== arch
    && !(String(arch).startsWith("Krea2") && String(info.detected_arch).startsWith("Krea2"));
  const ranks = info.ranks?.length ? (info.ranks.length > 2 ? `${info.ranks[0]}–${info.ranks[info.ranks.length - 1]}` : info.ranks.join(", ")) : "-";
  const alphas = info.alphas?.length ? (info.alphas.length > 2 ? `${info.alphas[0]}–${info.alphas[info.alphas.length - 1]}` : info.alphas.join(", ")) : "-";
  const matched = info.matched !== undefined ? `${info.matched}${info.unmatched ? ` <small style="opacity:.6">(${info.unmatched} skipped)</small>` : ""}` : "–";
  const trig = (info.trigger_words || []).map((t) => `<span class="sato-chip" data-t="${esc(t)}" title="click to copy">${esc(t)}</span>`).join("");
  let msg = "";
  if (mismatch) msg = `<div class="sato-msg warn">⚠ Looks like a <b>${esc(info.detected_arch)}</b> LoRA, but the setup is <b>${esc(arch)}</b>.</div>`;
  else if (info.matched === 0) msg = `<div class="sato-msg bad">✗ No keys matched the connected model.</div>`;
  else if (info.base_model_meta) msg = `<div class="sato-msg">Trained on: ${esc(info.base_model_meta)}</div>`;
  if (info.satodive_recipe) msg += `<div class="sato-msg ok">✦ SatoDive merge of ${esc(baseName(info.satodive_recipe.a))} + ${esc(baseName(info.satodive_recipe.b))}</div>`;

  el.innerHTML = `
    <div class="sato-head"><div class="sato-dot"></div><div class="sato-title" title="${esc(name)}">${esc(info.title || baseName(name))}</div>
      <span class="sato-badge ${mismatch ? "warn" : ""}">${esc(info.detected_arch === auto ? "unknown arch" : info.detected_arch)}</span>
      <span class="sato-badge dim">${esc(info.format)}</span></div>
    <div class="sato-grid">
      <div class="sato-stat"><span>Rank</span><b>${esc(ranks)}</b></div>
      <div class="sato-stat"><span>Alpha</span><b>${esc(alphas)}</b></div>
      <div class="sato-stat"><span>Size</span><b>${info.size_mb ?? "-"} MB</b></div>
      <div class="sato-stat"><span>Modules</span><b>${info.modules}</b></div>
      <div class="sato-stat"><span>Text enc.</span><b>${info.te_modules ? info.te_modules + " mods" : "none"}</b></div>
      <div class="sato-stat"><span>Matched</span><b>${matched}</b></div>
    </div>
    ${trig ? `<div class="sato-chips">${trig}</div>` : ""}${msg}`;
  el.querySelectorAll(".sato-chip").forEach((c) => c.addEventListener("click", () => {
    copy(c.dataset.t);
    c.textContent = "copied ✓";
    setTimeout(() => (c.textContent = c.dataset.t), 900);
  }));
}

async function refreshSlot(node) {
  const name = W(node, "lora_name")?.value;
  if (!name) return renderSlotCard(node, null);
  renderSlotCard(node, null, true);
  try {
    await getPresets();
    const info = await getJSON(`/satodive/lora_merge/info?name=${encodeURIComponent(name)}&arch=${encodeURIComponent(archFor(node))}`);
    if (node.__satoRunInfo && node.__satoRunInfo.name === name) Object.assign(info, { matched: node.__satoRunInfo.matched, unmatched: node.__satoRunInfo.unmatched });
    node.__satoInfo = info;
    renderSlotCard(node, info);
  } catch (e) {
    renderSlotCard(node, { error: String(e) });
  }
}

function initSlot(node) {
  const card = document.createElement("div");
  card.className = "sato-card";
  card.style.setProperty("--acc", THEME[node.comfyClass].acc);
  node.__satoCard = card;
  addButton(node, "▶  Run this LoRA preview", () => queueNode(node));
  addDOM(node, "lora_card", card, 176);

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
  setTimeout(() => refreshSlot(node), 50);
  requestAnimationFrame(() => fitNode(node, 360));
}

// ------------------------------------------------------------------------------------
// Merge studio
// ------------------------------------------------------------------------------------
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const RMIN = -1, RMAX = 2;

function mergeState(node) {
  if (!node.__sato) {
    let w = { a: {}, b: {} };
    try {
      const j = JSON.parse(W(node, "block_weights")?.value || "{}");
      w = { a: j.a || {}, b: j.b || {} };
    } catch (_) {}
    node.__sato = { groups: [], weights: w, closed: node.properties?.satoClosed || {}, names: {}, report: null };
  }
  return node.__sato;
}

function saveWeights(node) {
  const st = mergeState(node);
  const out = { a: {}, b: {} };
  for (const s of ["a", "b"]) for (const [k, v] of Object.entries(st.weights[s])) if (Math.abs(v - 1) > 1e-6) out[s][k] = Math.round(v * 1000) / 1000;
  const w = W(node, "block_weights");
  if (w) w.value = JSON.stringify(out);
  app.graph.setDirtyCanvas(true, false);
}

function paintRange(r) {
  const p = ((parseFloat(r.value) - RMIN) / (RMAX - RMIN)) * 100;
  r.style.setProperty("--p", p + "%");
}

function mkRange(color, value, disabled, oninput) {
  const r = document.createElement("input");
  r.type = "range";
  r.className = "sato-range";
  r.min = RMIN; r.max = RMAX; r.step = 0.05;
  r.value = value;
  r.disabled = !!disabled;
  r.style.setProperty("--c", color);
  paintRange(r);
  r.addEventListener("input", () => { paintRange(r); oninput(parseFloat(r.value)); });
  r.addEventListener("pointerdown", (e) => e.stopPropagation());
  return r;
}

function mkVal(v, off) {
  const s = document.createElement("div");
  s.className = "sato-val" + (off ? " off" : "");
  s.textContent = Number(v).toFixed(2);
  s.title = "double-click: reset to 1.0";
  return s;
}

function sectionsOf(groups) {
  const secs = [];
  const idx = {};
  for (const g of groups) {
    if (!(g.section in idx)) { idx[g.section] = secs.length; secs.push({ name: g.section, groups: [] }); }
    secs[idx[g.section]].groups.push(g);
  }
  return secs;
}

function applyPreset(node, kind) {
  const st = mergeState(node);
  const blocks = st.groups.filter((g) => g.id !== "text_encoder" && g.id !== "other");
  const n = Math.max(1, blocks.length - 1);
  const pos = {};
  blocks.forEach((g, i) => (pos[g.id] = i / n));
  for (const g of st.groups) {
    const p = pos[g.id] ?? 0.5;
    let a = st.weights.a[g.id] ?? 1, b = st.weights.b[g.id] ?? 1;
    switch (kind) {
      case "flat": a = 1; b = 1; break;
      case "aonly": a = 1; b = 0; break;
      case "bonly": a = 0; b = 1; break;
      case "half": a = 0.5; b = 0.5; break;
      case "ramp_ab": if (g.id in pos) { a = 1 - p; b = p; } break;
      case "ramp_ba": if (g.id in pos) { a = p; b = 1 - p; } break;
      case "swap": [a, b] = [b, a]; break;
      case "te_off": if (g.id === "text_encoder") { a = 0; b = 0; } break;
    }
    st.weights.a[g.id] = Math.round(a * 100) / 100;
    st.weights.b[g.id] = Math.round(b * 100) / 100;
  }
  saveWeights(node);
  renderMergePanel(node);
}

function renderMergePanel(node) {
  const el = node.__satoCard;
  if (!el) return;
  const st = mergeState(node);
  const scrollTop = el.querySelector(".sato-scroll")?.scrollTop || 0;
  el.innerHTML = "";

  // A / B header
  const ab = document.createElement("div");
  ab.className = "sato-ab";
  ab.innerHTML = `
    <div class="sato-pill" style="--c:${ACC_A}"><i>A</i><em title="${esc(st.names.a)}">${esc(baseName(st.names.a) || "not connected")}</em></div>
    <div class="sato-pill" style="--c:${ACC_B}"><i>B</i><em title="${esc(st.names.b)}">${esc(baseName(st.names.b) || "not connected")}</em></div>`;
  el.appendChild(ab);

  // presets
  const pr = document.createElement("div");
  pr.className = "sato-presets";
  const P = [["flat", "Flat 1.0"], ["half", "50 / 50"], ["aonly", "A only"], ["bonly", "B only"], ["ramp_ab", "Ramp A→B"],
    ["ramp_ba", "Ramp B→A"], ["swap", "Swap A↔B"], ["te_off", "TE off"], ["refresh", "↻ Detect"]];
  for (const [k, lab] of P) {
    const b = document.createElement("div");
    b.className = "sato-btn";
    b.textContent = lab;
    b.title = k === "refresh" ? "Re-detect block groups from the connected LoRAs" : "Apply to all block sliders";
    b.addEventListener("click", () => (k === "refresh" ? refreshMerge(node, true) : applyPreset(node, k)));
    pr.appendChild(b);
  }
  el.appendChild(pr);

  // sliders (panel height follows the number of detected groups)
  const secs = sectionsOf(st.groups);
  const rowsH = secs.reduce((t, s) => t + 28 + (s.groups.length > 1 || s.groups[0].label !== s.name ? s.groups.length * 21 : 0), 0);
  const want = clamp(rowsH + 12, 90, 400);
  if (want !== node.__satoScrollH) {
    node.__satoScrollH = want;
    requestAnimationFrame(() => fitNode(node, 520));
  }
  const sc = document.createElement("div");
  sc.className = "sato-scroll";
  sc.style.height = node.__satoScrollH + "px";
  if (!st.groups.length) {
    sc.innerHTML = `<div class="sato-empty">Connect <b style="color:${ACC_A}">Slot A</b> and <b style="color:${ACC_B}">Slot B</b> — block sliders are detected from both LoRAs.</div>`;
  }
  for (const sec of secs) {
    const box = document.createElement("div");
    box.className = "sato-sec" + (st.closed[sec.name] ? " closed" : "");
    const head = document.createElement("div");
    head.className = "sato-sec-h";
    const car = document.createElement("div"); car.className = "car"; car.textContent = "▾";
    const lbl = document.createElement("div"); lbl.className = "lbl";
    lbl.textContent = `${sec.name}${sec.groups.length > 1 ? " ×" + sec.groups.length : ""}`;
    const avg = (s) => sec.groups.reduce((t, g) => t + (st.weights[s][g.id] ?? 1), 0) / sec.groups.length;
    const anyA = sec.groups.some((g) => g.a > 0), anyB = sec.groups.some((g) => g.b > 0);
    const va = mkVal(avg("a"), !anyA), vb = mkVal(avg("b"), !anyB);
    const rows = document.createElement("div"); rows.className = "sato-rows";
    const rowRefs = [];
    const master = (s, vEl) => (v) => {
      for (const g of sec.groups) st.weights[s][g.id] = v;
      vEl.textContent = v.toFixed(2);
      for (const rr of rowRefs) if (rr.s === s) { rr.r.value = v; paintRange(rr.r); rr.v.textContent = v.toFixed(2); }
      saveWeights(node);
    };
    const ra = mkRange(ACC_A, avg("a"), !anyA, master("a", va));
    const rb = mkRange(ACC_B, avg("b"), !anyB, master("b", vb));
    [ra, rb].forEach((r) => r.addEventListener("click", (e) => e.stopPropagation()));
    head.append(car, lbl, ra, va, rb, vb);
    head.addEventListener("click", (e) => {
      if (e.target.tagName === "INPUT") return;
      st.closed[sec.name] = !st.closed[sec.name];
      node.properties = node.properties || {};
      node.properties.satoClosed = st.closed;
      box.classList.toggle("closed");
    });
    box.appendChild(head);

    if (sec.groups.length > 1 || sec.groups[0].label !== sec.name) {
      for (const g of sec.groups) {
        const row = document.createElement("div");
        row.className = "sato-row";
        const dot = document.createElement("div");
        const tag = g.a && g.b ? "A+B" : g.a ? "A" : "B";
        dot.innerHTML = `<span title="${tag}" style="display:inline-block;width:7px;height:7px;border-radius:50%;background:${g.a && g.b ? "linear-gradient(90deg," + ACC_A + " 50%," + ACC_B + " 50%)" : g.a ? ACC_A : ACC_B}"></span>`;
        const l = document.createElement("div"); l.className = "lbl"; l.textContent = g.label; l.title = `${g.id}  ·  A: ${g.a} modules · B: ${g.b} modules`;
        const wa = st.weights.a[g.id] ?? 1, wb = st.weights.b[g.id] ?? 1;
        const vA = mkVal(wa, !g.a), vB = mkVal(wb, !g.b);
        const rA = mkRange(ACC_A, wa, !g.a, (v) => { st.weights.a[g.id] = v; vA.textContent = v.toFixed(2); saveWeights(node); });
        const rB = mkRange(ACC_B, wb, !g.b, (v) => { st.weights.b[g.id] = v; vB.textContent = v.toFixed(2); saveWeights(node); });
        const reset = (s, r, v) => () => { st.weights[s][g.id] = 1; r.value = 1; paintRange(r); v.textContent = "1.00"; saveWeights(node); };
        vA.addEventListener("dblclick", reset("a", rA, vA));
        vB.addEventListener("dblclick", reset("b", rB, vB));
        rowRefs.push({ s: "a", r: rA, v: vA }, { s: "b", r: rB, v: vB });
        row.append(dot, l, rA, vA, rB, vB);
        rows.appendChild(row);
      }
      box.appendChild(rows);
    }
    sc.appendChild(box);
  }
  el.appendChild(sc);

  // report
  const rep = document.createElement("div");
  rep.className = "sato-report";
  const r = st.report;
  if (r) {
    rep.innerHTML = `
      <span class="sato-badge ok">${r.modules} modules</span>
      <span class="sato-badge">rank ${esc(r.ranks?.length ? (r.ranks.length > 1 ? r.ranks[0] + "–" + r.ranks[r.ranks.length - 1] : r.ranks[0]) : "-")}</span>
      <span class="sato-badge ${r.energy_kept < 90 ? "warn" : "ok"}" title="Spectral energy kept by rank resizing">energy ${r.energy_kept}%</span>
      <span class="sato-badge dim">A ${r.a_matched} · B ${r.b_matched} matched</span>
      ${r.size_mb ? `<span class="sato-badge dim">${r.size_mb} MB</span>` : ""}
      ${r.saved ? `<span class="sato-badge ok" title="${esc(r.saved)}">💾 ${esc(baseName(r.saved))}</span>` : ""}`;
  } else {
    rep.innerHTML = `<span class="sato-badge dim">Drag sliders, then ▶ Preview or 💾 Save</span>`;
  }
  el.appendChild(rep);
  const s2 = el.querySelector(".sato-scroll");
  if (s2) s2.scrollTop = scrollTop;
}

function upstreamLoraName(node, input) {
  const n = upstream(node, input);
  if (!n) return { name: "", node: null };
  if (n.comfyClass === T.a || n.comfyClass === T.b) return { name: W(n, "lora_name")?.value || "", node: n };
  if (n.comfyClass === T.merge) return { name: "⟲ merged (" + (n.title || "Merge Studio") + ")", node: n, merged: true };
  return { name: "", node: n };
}

function mergeGroups(oldG, newG) {
  const map = new Map(oldG.map((g) => [g.id, g]));
  for (const g of newG) map.set(g.id, g);
  return [...map.values()];
}

async function refreshMerge(node, force) {
  const st = mergeState(node);
  const A = upstreamLoraName(node, "lora_a");
  const B = upstreamLoraName(node, "lora_b");
  st.names = { a: A.name, b: B.name };
  const key = [A.name, B.name, archFor(node)].join("|");
  if (!force && key === st.lastKey) return renderMergePanel(node);
  st.lastKey = key;
  await getPresets();
  const fileA = A.merged ? "" : A.name, fileB = B.merged ? "" : B.name;
  let groups = [];
  if (fileA || fileB) {
    try {
      const res = await getJSON(`/satodive/lora_merge/groups?a=${encodeURIComponent(fileA)}&b=${encodeURIComponent(fileB)}&arch=${encodeURIComponent(archFor(node))}`);
      groups = res.groups || [];
    } catch (e) {
      console.warn("[SatoDive] group detection failed", e);
    }
  }
  // chained merges: reuse the groups reported by the upstream Merge Studio run
  const byId = new Map(groups.map((g) => [g.id, { ...g }]));
  for (const [up, side] of [[A, "a"], [B, "b"]]) {
    if (!up.merged || !up.node?.__sato?.groups?.length) continue;
    for (const g of up.node.__sato.groups) {
      const cur = byId.get(g.id) || { ...g, a: 0, b: 0 };
      cur[side] = Math.max(cur[side] || 0, (g.a || 0) + (g.b || 0));
      byId.set(g.id, cur);
    }
  }
  groups = [...byId.values()];
  st.groups = groups;
  renderMergePanel(node);
}

function initMerge(node) {
  const bw = W(node, "block_weights");
  if (bw) { bw.hidden = true; bw.computeSize = () => [0, -4]; }
  node.__satoScrollH = 120;

  addButton(node, "▶  Preview merge", () => queueNode(node));
  addButton(node, "💾  Merge & Save LoRA", async () => {
    const sw = W(node, "save_lora");
    const prev = sw?.value;
    if (sw) sw.value = true;
    try { await queueNode(node); } finally { if (sw) sw.value = prev; app.graph.setDirtyCanvas(true, false); }
  });

  const card = document.createElement("div");
  card.className = "sato-card sato-panel";
  card.style.setProperty("--acc", THEME[T.merge].acc);
  node.__satoCard = card;
  addDOM(node, "block_studio", card, () => node.__satoScrollH + 132);

  node.__satoListener = () => { clearTimeout(node.__satoT); node.__satoT = setTimeout(() => refreshMerge(node, true), 120); };
  window.addEventListener(EVT, node.__satoListener);

  const occ = node.onConnectionsChange;
  node.onConnectionsChange = function (...args) {
    const r = occ?.apply(this, args);
    node.__satoListener();
    return r;
  };
  setTimeout(() => refreshMerge(node, true), 80);
  requestAnimationFrame(() => fitNode(node, 520));
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
      theme(this);
      if (nodeData.name === T.setup) initSetup(this);
      else if (nodeData.name === T.a || nodeData.name === T.b) initSlot(this);
      else if (nodeData.name === T.merge) initMerge(this);
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      theme(this);
      if (nodeData.name === T.merge) {
        this.__sato = null;
        const st = mergeState(this);
        st.lastKey = null;
        setTimeout(() => refreshMerge(this, true), 150);
      } else if (nodeData.name === T.a || nodeData.name === T.b) {
        setTimeout(() => refreshSlot(this), 150);
      } else if (nodeData.name === T.setup) {
        setTimeout(() => renderSetupCard(this), 150);
      }
      return r;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (msg) {
      const r = onExecuted?.apply(this, arguments);
      if (nodeData.name === T.setup && msg?.sato_setup?.[0]) {
        this.__satoRun = msg.sato_setup[0];
        renderSetupCard(this);
      }
      if ((nodeData.name === T.a || nodeData.name === T.b) && msg?.sato_info?.[0]) {
        const info = msg.sato_info[0];
        this.__satoRunInfo = { name: W(this, "lora_name")?.value, matched: info.matched, unmatched: info.unmatched };
        this.__satoInfo = info;
        renderSlotCard(this, info);
      }
      if (nodeData.name === T.merge) {
        const st = mergeState(this);
        if (msg?.sato_report?.[0]) st.report = msg.sato_report[0];
        if (Array.isArray(msg?.sato_groups) && msg.sato_groups.length) st.groups = mergeGroups(st.groups, msg.sato_groups);
        renderMergePanel(this);
      }
      return r;
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      if (this.__satoListener) {
        window.removeEventListener(EVT, this.__satoListener);
        window.removeEventListener(EVT + ":arch", this.__satoListener);
      }
      return onRemoved?.apply(this, arguments);
    };
  },
});
