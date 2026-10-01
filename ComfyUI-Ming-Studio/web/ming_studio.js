import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const MING_NODES = [
  "MingCanvasSize", "MingDesignPrompt", "MingTransparentTrigger", "MingEnhancerPrompts", "MingPromptCleanup",
  "MingLayerDecomposeEncode", "MingEditEncode", "MingDecodeLayers", "MingLayerStudio", "MingSavePSD",
  "MingRGBAComposite", "MingJoinAlpha", "MingWebsiteDesign",
];
const BLENDS = ["normal", "multiply", "screen", "overlay", "darken", "lighten", "color-dodge", "color-burn",
  "hard-light", "soft-light", "difference", "exclusion", "hue", "saturation", "color", "luminosity"];

// ------------------------------------------------------------------ styles
const CSS = `
.ming-root{--m-bg:#0f0d16;--m-panel:#17141f;--m-panel2:#1f1b2b;--m-line:#2c2640;--m-text:#ece9f5;--m-dim:#9a93b3;
  --m-acc:#8b5cf6;--m-acc2:#22d3ee;--m-warn:#f59e0b;font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
  color:var(--m-text);background:var(--m-bg);border:1px solid var(--m-line);border-radius:14px;overflow:hidden;
  display:flex;flex-direction:column;width:100%;height:100%;box-sizing:border-box;font-size:12px;user-select:none}
.ming-root *{box-sizing:border-box}
.ming-head{display:flex;align-items:center;gap:8px;padding:8px 12px;background:linear-gradient(90deg,#1d1530,#121a2a);border-bottom:1px solid var(--m-line)}
.ming-logo{font-weight:800;letter-spacing:.12em;font-size:11px;background:linear-gradient(90deg,#c4b5fd,#67e8f9);-webkit-background-clip:text;background-clip:text;color:transparent}
.ming-chip{padding:2px 8px;border-radius:999px;background:var(--m-panel2);border:1px solid var(--m-line);color:var(--m-dim);font-size:10.5px;white-space:nowrap}
.ming-sp{flex:1}
.ming-body{flex:1;display:flex;min-height:0}
.ming-stagewrap{flex:1;display:flex;flex-direction:column;min-width:0;border-right:1px solid var(--m-line)}
.ming-tools{display:flex;gap:4px;padding:6px 8px;border-bottom:1px solid var(--m-line);background:var(--m-panel);flex-wrap:wrap}
.ming-btn{appearance:none;border:1px solid var(--m-line);background:var(--m-panel2);color:var(--m-text);border-radius:8px;padding:4px 9px;
  font-size:11px;cursor:pointer;transition:all .15s;display:inline-flex;align-items:center;gap:5px;white-space:nowrap}
.ming-btn:hover{border-color:var(--m-acc);background:#2a2340}
.ming-btn.on{background:linear-gradient(135deg,#6d28d9,#7c3aed);border-color:#a78bfa}
.ming-btn.primary{background:linear-gradient(135deg,#7c3aed,#0891b2);border:none;font-weight:600}
.ming-btn.primary:hover{filter:brightness(1.15)}
.ming-btn:disabled{opacity:.5;cursor:wait}
.ming-stage{flex:1;position:relative;overflow:hidden;min-height:0;cursor:crosshair}
.ming-stage canvas{position:absolute;inset:0;margin:auto;max-width:100%;max-height:100%;image-rendering:auto;box-shadow:0 10px 40px rgba(0,0,0,.5)}
.ming-bg-checker{background:repeating-conic-gradient(#2a2638 0 25%,#1d1a28 0 50%) 0 0/20px 20px}
.ming-bg-white{background:#f5f5f7}.ming-bg-dark{background:#0a0a0c}
.ming-empty{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;color:var(--m-dim);gap:8px;text-align:center;padding:20px}
.ming-empty b{color:var(--m-text);font-size:14px}
.ming-side{width:290px;display:flex;flex-direction:column;background:var(--m-panel);min-height:0}
.ming-list{flex:1;overflow-y:auto;padding:6px}
.ming-list::-webkit-scrollbar{width:8px}.ming-list::-webkit-scrollbar-thumb{background:#2f2945;border-radius:8px}
.ming-item{display:flex;align-items:center;gap:8px;padding:6px;border-radius:10px;border:1px solid transparent;margin-bottom:4px;cursor:pointer;background:var(--m-panel2);transition:background .12s,border-color .12s}
.ming-item:hover{border-color:#3b3356}
.ming-item.sel{border-color:var(--m-acc);background:#251d3a;box-shadow:0 0 0 1px rgba(139,92,246,.35) inset}
.ming-item.hidden .ming-thumb,.ming-item.hidden .ming-name{opacity:.38}
.ming-item.drop{border-top:2px solid var(--m-acc2)}
.ming-grip{color:#5b5375;cursor:grab;font-size:13px;padding:0 2px}
.ming-thumb{width:44px;height:44px;border-radius:7px;flex:none;background:repeating-conic-gradient(#3a3550 0 25%,#262234 0 50%) 0 0/10px 10px;overflow:hidden;display:flex;align-items:center;justify-content:center;border:1px solid #332d48}
.ming-thumb img{max-width:100%;max-height:100%}
.ming-meta{flex:1;min-width:0}
.ming-name{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ming-sub{color:var(--m-dim);font-size:10px;margin-top:2px}
.ming-name input{width:100%;background:#0f0d16;border:1px solid var(--m-acc);color:var(--m-text);border-radius:5px;padding:2px 4px;font:inherit}
.ming-ico{width:26px;height:26px;border-radius:7px;border:1px solid var(--m-line);background:#141120;color:var(--m-text);cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:13px;flex:none}
.ming-ico:hover{border-color:var(--m-acc)}
.ming-ico.off{color:#5b5375}
.ming-ico.solo{color:var(--m-acc2);border-color:var(--m-acc2)}
.ming-props{border-top:1px solid var(--m-line);padding:8px 10px;display:grid;grid-template-columns:62px 1fr 34px;gap:6px 8px;align-items:center}
.ming-props label{color:var(--m-dim)}
.ming-props input[type=range]{width:100%;accent-color:var(--m-acc)}
.ming-props select{grid-column:2 / span 2;background:#0f0d16;color:var(--m-text);border:1px solid var(--m-line);border-radius:6px;padding:3px}
.ming-foot{display:grid;grid-template-columns:1fr 1fr;gap:6px;padding:8px;border-top:1px solid var(--m-line);background:#130f1c}
.ming-foot .ming-btn{justify-content:center;padding:7px}
.ming-foot .wide{grid-column:1 / span 2}
.ming-toast{position:absolute;left:50%;bottom:14px;transform:translateX(-50%);background:#1f1b2b;border:1px solid var(--m-acc);padding:6px 12px;border-radius:999px;color:var(--m-text);font-size:11px;pointer-events:none;opacity:0;transition:opacity .2s}
.ming-toast.show{opacity:1}
.ming-card{padding:10px 12px;display:flex;flex-direction:column;gap:8px;overflow:auto}
.ming-card h4{margin:0;font-size:13px}
.ming-swatches{display:flex;gap:4px;flex-wrap:wrap}
.ming-sw{width:46px;height:34px;border-radius:7px;border:1px solid rgba(255,255,255,.12);display:flex;align-items:flex-end;justify-content:center;font-size:8.5px;padding-bottom:2px;cursor:pointer}
.ming-tags{display:flex;gap:4px;flex-wrap:wrap}
.ming-warn{color:var(--m-warn);font-size:11px}
.ming-dl{padding:10px;display:flex;flex-direction:column;gap:6px;align-items:stretch}
`;

function injectCss() {
  if (document.getElementById("ming-studio-css")) return;
  const s = document.createElement("style");
  s.id = "ming-studio-css";
  s.textContent = CSS;
  document.head.appendChild(s);
}

const el = (tag, cls, html) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
};

const viewUrl = (f) => api.apiURL(`/view?filename=${encodeURIComponent(f.filename)}&subfolder=${encodeURIComponent(f.subfolder || "")}&type=${f.type || "temp"}&r=${Date.now() % 1e7}`);

function download(f) {
  const a = document.createElement("a");
  a.href = viewUrl(f);
  a.download = f.filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

function hideWidget(node, name) {
  const w = node.widgets?.find((x) => x.name === name);
  if (!w) return null;
  w.hidden = true;
  w.type = "hidden";
  w.computeSize = () => [0, -4];
  return w;
}

// ------------------------------------------------------------------ Layer Studio
class LayerStudio {
  constructor(node) {
    this.node = node;
    this.layers = []; // {file, img, canvas, bbox, ...state}
    this.order = [];  // indices top-first
    this.sel = -1;
    this.solo = -1;
    this.bg = "checker";
    this.outline = true;
    this.compare = false;
    this.composite = null;
    this.size = [0, 0];
    this.build();
  }

  build() {
    const r = (this.root = el("div", "ming-root"));
    const head = el("div", "ming-head");
    head.append(el("span", "ming-logo", "◆ MING LAYER STUDIO"));
    this.chipCount = el("span", "ming-chip", "no layers");
    this.chipSize = el("span", "ming-chip", "—");
    head.append(this.chipCount, this.chipSize, el("span", "ming-sp"));
    this.chipHint = el("span", "ming-chip", "click canvas = pick · alt+click = hide");
    head.append(this.chipHint);
    r.append(head);

    const body = el("div", "ming-body");
    const wrap = el("div", "ming-stagewrap");
    const tools = el("div", "ming-tools");
    const mk = (label, title, fn) => {
      const b = el("button", "ming-btn", label);
      b.title = title;
      b.onclick = (e) => { e.stopPropagation(); fn(b); };
      tools.append(b);
      return b;
    };
    this.bgBtns = {};
    for (const [k, lbl] of [["checker", "▦"], ["white", "☐ Light"], ["dark", "■ Dark"]]) {
      this.bgBtns[k] = mk(lbl, "Stage background", () => { this.bg = k; this.paint(); this.syncTools(); });
    }
    mk("👁 All", "Show all layers", () => { this.layers.forEach((l) => (l.visible = true)); this.solo = -1; this.commit(); });
    mk("⦸ None", "Hide all layers", () => { this.layers.forEach((l) => (l.visible = false)); this.solo = -1; this.commit(); });
    mk("⇄ Invert", "Invert visibility", () => { this.layers.forEach((l) => (l.visible = !l.visible)); this.solo = -1; this.commit(); });
    this.outlineBtn = mk("⬚ Bounds", "Outline the selected layer", () => { this.outline = !this.outline; this.paint(); this.syncTools(); });
    this.compareBtn = mk("◐ Original", "Hold to view the original composite", () => {});
    this.compareBtn.onpointerdown = () => { this.compare = true; this.paint(); };
    this.compareBtn.onpointerup = this.compareBtn.onpointerleave = () => { this.compare = false; this.paint(); };
    mk("↺ Reset", "Reset order, names, opacity and blend", () => { this.resetState(); });
    wrap.append(tools);

    this.stage = el("div", "ming-stage ming-bg-checker");
    this.canvas = document.createElement("canvas");
    this.stage.append(this.canvas);
    this.empty = el("div", "ming-empty", "<b>No layers yet</b><span>Queue the workflow. Decomposed RGBA layers appear here.<br>Pick, hide, solo, reorder, rename, then export straight to Photoshop.</span>");
    this.stage.append(this.empty);
    this.toast = el("div", "ming-toast");
    this.stage.append(this.toast);
    this.stage.addEventListener("pointerdown", (e) => this.onStageClick(e));
    wrap.append(this.stage);
    body.append(wrap);

    const side = el("div", "ming-side");
    this.list = el("div", "ming-list");
    side.append(this.list);
    this.props = el("div", "ming-props");
    this.opLabel = el("span", "", "100");
    this.opRange = el("input");
    this.opRange.type = "range"; this.opRange.min = 0; this.opRange.max = 100;
    this.opRange.oninput = () => { const l = this.layers[this.sel]; if (!l) return; l.opacity = this.opRange.value / 100; this.opLabel.textContent = this.opRange.value; this.paint(); };
    this.opRange.onchange = () => this.commit();
    this.blendSel = el("select");
    BLENDS.forEach((b) => { const o = el("option", "", b); o.value = b; this.blendSel.append(o); });
    this.blendSel.onchange = () => { const l = this.layers[this.sel]; if (!l) return; l.blend = this.blendSel.value; this.commit(); };
    this.props.append(el("label", "", "Opacity"), this.opRange, this.opLabel, el("label", "", "Blend"), this.blendSel);
    side.append(this.props);

    const foot = el("div", "ming-foot");
    const ex = (label, fmt, cls = "") => {
      const b = el("button", "ming-btn " + cls, label);
      b.onclick = (e) => { e.stopPropagation(); this.exportAs(fmt, b); };
      foot.append(b);
      return b;
    };
    ex("⬇ Export PSD (Photoshop)", "psd", "primary wide");
    ex("🗂 Layers ZIP", "zip");
    ex("🖼 Flatten PNG", "png");
    const selBtn = el("button", "ming-btn wide", "✂ Selected layer PNG");
    selBtn.onclick = (e) => { e.stopPropagation(); this.exportAs("selected", selBtn); };
    foot.append(selBtn);
    side.append(foot);
    body.append(side);
    r.append(body);

    // keep litegraph from stealing interactions
    for (const evt of ["pointerdown", "mousedown", "wheel", "keydown", "dblclick"]) {
      side.addEventListener(evt, (e) => e.stopPropagation());
      tools.addEventListener(evt, (e) => e.stopPropagation());
    }
    new ResizeObserver(() => this.paint()).observe(this.stage);
    this.syncTools();
  }

  syncTools() {
    for (const [k, b] of Object.entries(this.bgBtns)) b.classList.toggle("on", this.bg === k);
    this.outlineBtn.classList.toggle("on", this.outline);
    this.stage.className = "ming-stage ming-bg-" + (this.bg === "checker" ? "checker" : this.bg);
  }

  flash(msg) {
    this.toast.textContent = msg;
    this.toast.classList.add("show");
    clearTimeout(this._t);
    this._t = setTimeout(() => this.toast.classList.remove("show"), 1800);
  }

  async load(files, state, size, composite) {
    this.size = size || this.size;
    const st = state?.layers || [];
    const loaded = await Promise.all(files.map((f) => new Promise((res) => {
      const img = new Image();
      img.onload = () => res(img);
      img.onerror = () => res(null);
      img.src = viewUrl(f);
    })));
    if (loaded.some((x) => !x)) {
      this.empty.style.display = "flex";
      this.empty.innerHTML = "<b>Layers expired</b><span>Temporary files were cleared. Queue the workflow again.</span>";
      return;
    }
    this.layers = files.map((f, i) => {
      const s = st.find((x) => +x.index === i) || {};
      const c = document.createElement("canvas");
      c.width = loaded[i].naturalWidth; c.height = loaded[i].naturalHeight;
      const ctx = c.getContext("2d", { willReadFrequently: true });
      ctx.drawImage(loaded[i], 0, 0);
      return { index: i, file: f, img: loaded[i], canvas: c, ctx, bbox: this.computeBBox(ctx, c.width, c.height),
        name: s.name || `Layer ${i + 1}`, visible: s.visible !== false, opacity: s.opacity ?? 1, blend: s.blend || "normal" };
    });
    this.order = st.length === files.length ? st.map((x) => +x.index) : files.map((_, i) => i);
    this.composite = null;
    if (composite) {
      const img = new Image();
      img.onload = () => (this.composite = img);
      img.src = viewUrl(composite);
    }
    if (!this.size[0] && this.layers[0]) this.size = [this.layers[0].canvas.width, this.layers[0].canvas.height];
    this.sel = this.layers.length ? this.order[0] : -1;
    this.solo = -1;
    this.empty.style.display = this.layers.length ? "none" : "flex";
    this.chipCount.textContent = `${this.layers.length} layers`;
    this.chipSize.textContent = `${this.size[0]}×${this.size[1]}`;
    this.renderList();
    this.paint();
  }

  computeBBox(ctx, w, h) {
    const step = Math.max(1, Math.floor(Math.max(w, h) / 256));
    const d = ctx.getImageData(0, 0, w, h).data;
    let x0 = w, y0 = h, x1 = -1, y1 = -1, px = 0;
    for (let y = 0; y < h; y += step) for (let x = 0; x < w; x += step) {
      if (d[(y * w + x) * 4 + 3] > 12) { px++; if (x < x0) x0 = x; if (y < y0) y0 = y; if (x > x1) x1 = x; if (y > y1) y1 = y; }
    }
    if (x1 < 0) return null;
    const total = Math.ceil(w / step) * Math.ceil(h / step);
    return { x: x0, y: y0, w: x1 - x0 + step, h: y1 - y0 + step, cover: px / total };
  }

  isShown(l) {
    return this.solo >= 0 ? l.index === this.solo : l.visible;
  }

  paint() {
    const [W, H] = this.size;
    if (!W || !this.layers.length) return;
    const box = this.stage.getBoundingClientRect();
    const scale = Math.min((box.width - 16) / W, (box.height - 16) / H, 1);
    const cw = Math.max(1, Math.round(W * scale)), ch = Math.max(1, Math.round(H * scale));
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = cw * dpr; this.canvas.height = ch * dpr;
    this.canvas.style.width = cw + "px"; this.canvas.style.height = ch + "px";
    this.scale = scale;
    const ctx = this.canvas.getContext("2d");
    ctx.setTransform(dpr * scale, 0, 0, dpr * scale, 0, 0);
    ctx.clearRect(0, 0, W, H);
    if (this.compare && this.composite) {
      ctx.drawImage(this.composite, 0, 0, W, H);
      return;
    }
    for (const idx of [...this.order].reverse()) {
      const l = this.layers[idx];
      if (!this.isShown(l)) continue;
      ctx.globalAlpha = l.opacity;
      ctx.globalCompositeOperation = l.blend === "normal" ? "source-over" : l.blend;
      ctx.drawImage(l.canvas, 0, 0, W, H);
    }
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = "source-over";
    const s = this.layers[this.sel];
    if (this.outline && s?.bbox) {
      const lw = 2 / scale;
      ctx.setLineDash([8 / scale, 5 / scale]);
      ctx.lineWidth = lw;
      ctx.strokeStyle = "#a78bfa";
      ctx.strokeRect(s.bbox.x, s.bbox.y, s.bbox.w, s.bbox.h);
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(139,92,246,.9)";
      const label = s.name.slice(0, 40);
      ctx.font = `${12 / scale}px Inter, sans-serif`;
      const tw = ctx.measureText(label).width + 10 / scale;
      const ly = Math.max(0, s.bbox.y - 18 / scale);
      ctx.fillRect(s.bbox.x, ly, tw, 18 / scale);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, s.bbox.x + 5 / scale, ly + 13 / scale);
    }
  }

  onStageClick(e) {
    if (!this.layers.length) return;
    e.stopPropagation();
    const rect = this.canvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / rect.width * this.size[0]);
    const y = Math.floor((e.clientY - rect.top) / rect.height * this.size[1]);
    if (x < 0 || y < 0 || x >= this.size[0] || y >= this.size[1]) return;
    for (const idx of this.order) {
      const l = this.layers[idx];
      if (!this.isShown(l) && !e.altKey) continue;
      const sx = Math.floor(x * l.canvas.width / this.size[0]), sy = Math.floor(y * l.canvas.height / this.size[1]);
      const a = l.ctx.getImageData(sx, sy, 1, 1).data[3];
      if (a > 24) {
        if (e.altKey) { l.visible = false; this.flash(`Hidden: ${l.name}`); this.commit(); return; }
        this.select(idx);
        this.flash(l.name);
        return;
      }
    }
  }

  select(idx) {
    this.sel = idx;
    const l = this.layers[idx];
    if (l) {
      this.opRange.value = Math.round(l.opacity * 100);
      this.opLabel.textContent = this.opRange.value;
      this.blendSel.value = l.blend;
    }
    this.renderList();
    this.paint();
  }

  renderList() {
    this.list.innerHTML = "";
    this.order.forEach((idx, pos) => {
      const l = this.layers[idx];
      const it = el("div", "ming-item" + (idx === this.sel ? " sel" : "") + (this.isShown(l) ? "" : " hidden"));
      it.draggable = true;
      const grip = el("span", "ming-grip", "⋮⋮");
      const th = el("div", "ming-thumb");
      const ti = el("img");
      ti.src = l.img.src;
      th.append(ti);
      const meta = el("div", "ming-meta");
      const nm = el("div", "ming-name");
      nm.textContent = l.name;
      nm.title = l.name + "  (double-click to rename)";
      const sub = el("div", "ming-sub", `#${pos + 1}${pos === 0 ? " · front" : pos === this.order.length - 1 ? " · back" : ""} · ${Math.round(l.opacity * 100)}% · ${l.blend}${l.bbox ? "" : " · empty"}`);
      meta.append(nm, sub);
      const eye = el("button", "ming-ico" + (l.visible ? "" : " off"), l.visible ? "👁" : "◌");
      eye.title = "Show / hide";
      eye.onclick = (e) => { e.stopPropagation(); l.visible = !l.visible; this.commit(); };
      const solo = el("button", "ming-ico" + (this.solo === idx ? " solo" : ""), "S");
      solo.title = "Solo (preview only this layer)";
      solo.onclick = (e) => { e.stopPropagation(); this.solo = this.solo === idx ? -1 : idx; this.renderList(); this.paint(); };
      it.append(grip, th, meta, eye, solo);
      it.onclick = () => this.select(idx);
      nm.ondblclick = (e) => {
        e.stopPropagation();
        const inp = el("input");
        inp.value = l.name;
        nm.innerHTML = "";
        nm.append(inp);
        inp.focus(); inp.select();
        const done = () => { l.name = inp.value.trim() || l.name; this.commit(); };
        inp.onkeydown = (ev) => { ev.stopPropagation(); if (ev.key === "Enter") inp.blur(); if (ev.key === "Escape") { inp.value = l.name; inp.blur(); } };
        inp.onblur = done;
      };
      it.ondragstart = (e) => { this.dragPos = pos; e.dataTransfer.effectAllowed = "move"; };
      it.ondragover = (e) => { e.preventDefault(); it.classList.add("drop"); };
      it.ondragleave = () => it.classList.remove("drop");
      it.ondrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        it.classList.remove("drop");
        if (this.dragPos === undefined || this.dragPos === pos) return;
        const [moved] = this.order.splice(this.dragPos, 1);
        this.order.splice(pos, 0, moved);
        this.dragPos = undefined;
        this.commit();
      };
      this.list.append(it);
    });
    const l = this.layers[this.sel];
    this.props.style.opacity = l ? 1 : 0.4;
  }

  stateObj() {
    return { layers: this.order.map((idx) => {
      const l = this.layers[idx];
      return { index: idx, name: l.name, visible: l.visible, opacity: +l.opacity.toFixed(3), blend: l.blend };
    }) };
  }

  commit() {
    const w = this.node.widgets?.find((x) => x.name === "studio_state");
    const st = this.stateObj();
    if (w) w.value = JSON.stringify(st);
    this.node.properties.ming_last = { ...(this.node.properties.ming_last || {}), state: st };
    this.renderList();
    this.paint();
    this.node.setDirtyCanvas?.(true, true);
  }

  resetState() {
    this.order = this.layers.map((_, i) => i);
    this.layers.forEach((l, i) => { l.visible = true; l.opacity = 1; l.blend = "normal"; });
    this.solo = -1;
    this.commit();
    this.flash("Layer state reset");
  }

  async exportAs(fmt, btn) {
    if (!this.layers.length) return this.flash("Nothing to export yet");
    let ids = this.order;
    if (fmt === "selected") {
      if (this.sel < 0) return this.flash("Select a layer first");
      ids = [this.sel];
    }
    const payload = {
      format: fmt === "selected" ? "png" : fmt,
      prefix: fmt === "selected" ? this.layers[this.sel].name : "ming_layers",
      crop: true,
      layers: ids.map((idx) => {
        const l = this.layers[idx];
        return { file: l.file, name: l.name, visible: fmt === "selected" ? true : l.visible, opacity: l.opacity, blend: l.blend };
      }),
    };
    const old = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = "⏳ Exporting…";
    try {
      const r = await api.fetchApi("/ming_studio/export", { method: "POST", body: JSON.stringify(payload), headers: { "Content-Type": "application/json" } });
      const j = await r.json();
      if (j.error) throw new Error(j.error);
      download(j);
      this.flash(`Saved output/ming_studio/${j.filename}`);
    } catch (err) {
      this.flash("Export failed: " + err.message);
    } finally {
      btn.disabled = false;
      btn.innerHTML = old;
    }
  }
}

// ------------------------------------------------------------------ Website card
function renderSite(container, s) {
  container.innerHTML = "";
  const card = el("div", "ming-card");
  card.append(el("h4", "", `🌐 ${s.site || s.title || "Website"} ${s.dark ? '<span class="ming-chip">dark</span>' : ""}`));
  if (s.title) card.append(el("div", "ming-sub", s.title));
  const sw = el("div", "ming-swatches");
  (s.palette || []).forEach((c) => {
    const d = el("div", "ming-sw", c);
    d.style.background = c;
    const rgb = parseInt(c.slice(1), 16);
    const lum = (0.2126 * ((rgb >> 16) & 255) + 0.7152 * ((rgb >> 8) & 255) + 0.0722 * (rgb & 255)) / 255;
    d.style.color = lum > 0.55 ? "#111" : "#fff";
    d.title = "Copy " + c;
    d.onclick = (e) => { e.stopPropagation(); navigator.clipboard?.writeText(c); };
    sw.append(d);
  });
  card.append(sw);
  const tags = (label, arr) => {
    if (!arr?.length) return;
    const t = el("div", "ming-tags");
    t.append(el("span", "ming-sub", label));
    arr.forEach((x) => t.append(el("span", "ming-chip", x)));
    card.append(t);
  };
  tags("Fonts", s.fonts);
  tags("Nav", s.nav);
  tags("CTAs", s.buttons);
  (s.notes || []).forEach((n) => card.append(el("div", "ming-warn", "⚠ " + n)));
  container.append(card);
}

// ------------------------------------------------------------------ extension
app.registerExtension({
  name: "ming.studio",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!MING_NODES.includes(nodeData.name)) return;
    injectCss();
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      this.color = "#2e2350";
      this.bgcolor = "#1b1528";

      if (nodeData.name === "MingLayerStudio") {
        hideWidget(this, "studio_state");
        this.studio = new LayerStudio(this);
        this.addDOMWidget("ming_studio_ui", "ming_studio", this.studio.root, { serialize: false, getMinHeight: () => 520 });
        this.setSize([Math.max(this.size[0], 980), Math.max(this.size[1], 720)]);
      }
      if (nodeData.name === "MingWebsiteDesign") {
        this.siteBox = el("div", "ming-root");
        this.siteBox.style.minHeight = "220px";
        this.siteBox.append(el("div", "ming-card", '<span class="ming-sub">Run to preview the extracted palette, fonts and structure.</span>'));
        this.addDOMWidget("ming_site_ui", "ming_site", this.siteBox, { serialize: false, getMinHeight: () => 240 });
        this.setSize([Math.max(this.size[0], 420), this.size[1]]);
      }
      if (nodeData.name === "MingSavePSD") {
        this.psdBox = el("div", "ming-root");
        const inner = el("div", "ming-dl", '<span class="ming-sub">PSD appears here after the run.</span>');
        this.psdBox.append(inner);
        this.psdInner = inner;
        this.addDOMWidget("ming_psd_ui", "ming_psd", this.psdBox, { serialize: false, getMinHeight: () => 70 });
      }
      return r;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      if (nodeData.name === "MingLayerStudio" && message?.ming_layers) {
        const last = { files: message.ming_layers, state: message.ming_state?.[0], size: message.ming_size, composite: message.ming_composite?.[0] };
        this.properties.ming_last = last;
        const w = this.widgets?.find((x) => x.name === "studio_state");
        if (w && last.state) w.value = JSON.stringify(last.state);
        this.studio.load(last.files, last.state, last.size, last.composite);
      }
      if (nodeData.name === "MingWebsiteDesign" && message?.ming_site) {
        this.properties.ming_site = message.ming_site[0];
        renderSite(this.siteBox, message.ming_site[0]);
      }
      if (nodeData.name === "MingSavePSD" && message?.ming_psd) {
        const f = message.ming_psd[0];
        this.psdInner.innerHTML = "";
        const b = el("button", "ming-btn primary", `⬇ Download ${f.filename}`);
        b.style.justifyContent = "center";
        b.onclick = (e) => { e.stopPropagation(); download(f); };
        this.psdInner.append(b, el("span", "ming-sub", `Saved in output/${f.subfolder ? f.subfolder + "/" : ""}`));
      }
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      if (nodeData.name === "MingLayerStudio" && this.properties?.ming_last?.files) {
        const l = this.properties.ming_last;
        setTimeout(() => this.studio?.load(l.files, l.state, l.size, l.composite), 50);
      }
      if (nodeData.name === "MingWebsiteDesign" && this.properties?.ming_site) {
        renderSite(this.siteBox, this.properties.ming_site);
      }
      return r;
    };
  },
});
