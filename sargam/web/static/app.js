/* Sargam UI: library, upload, progress, notation with synced playback, reference comparison. No framework. */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtTime = (s) => { s = Math.max(0, s || 0); const m = Math.floor(s / 60); return `${m}:${String(Math.floor(s % 60)).padStart(2, "0")}`; };
const pct = (x) => (x == null ? "–" : `${Math.round(x * 100)}%`);

const store = {
  get(k, d) { try { const v = localStorage.getItem("sargam." + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("sargam." + k, JSON.stringify(v)); } catch { /* private mode */ } },
};

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  let body = null;
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("json")) body = await res.json();
  if (!res.ok) {
    const msg = (body && (body.error || body.detail)) || `${res.status} ${res.statusText}`;
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return body;
}
const postJSON = (path, data) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });

let toastTimer = null;
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), 2200);
}

/* ------------------------------------------------------------------ state */

const S = {
  schema: [], stages: [], exts: [],
  items: [], busy: false,
  view: null, viewToken: 0,
  item: null, render: null,
  style: store.get("style", "nw"), detail: store.get("detail", 0),
  pendingFile: null,
};
const field = (key) => S.schema.find((f) => f.key === key);
const analysisFields = () => S.schema.filter((f) => f.affects === "analysis");
const defaults = () => Object.fromEntries(S.schema.map((f) => [f.key, f.default]));

/* ------------------------------------------------------------------ library sidebar */

let libTimer = null;
async function refreshLibrary() {
  try {
    const r = await api("/api/jobs");
    S.items = r.items;
    S.busy = r.busy || r.items.some((i) => i.state === "queued" || i.state === "running");
  } catch { return; }
  drawLibrary();
  clearTimeout(libTimer);
  libTimer = setTimeout(refreshLibrary, S.busy ? 2000 : 20000);
}

function relDate(ts) {
  const d = (Date.now() / 1000 - ts);
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.floor(d / 60)} min ago`;
  if (d < 86400) return `${Math.floor(d / 3600)} h ago`;
  return new Date(ts * 1000).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function stateChip(it) {
  if (it.state === "queued") return `<span class="chip run">queued</span>`;
  if (it.state === "running") return `<span class="chip run">analysing…</span>`;
  if (it.state === "error") return `<span class="chip bad">error</span>`;
  return "";
}

function drawLibrary() {
  const ul = $("#libList");
  const cur = S.item && S.item.id;
  ul.innerHTML = S.items.map((it) => `
    <li><a href="#/item/${esc(it.id)}" class="${it.id === cur ? "active" : ""}">
      <span class="lib-title">${esc(it.title)}</span>
      <span class="lib-sub">${it.sa ? `<b>${esc(it.sa)}</b>` : ""}${it.duration_s ? `<span>${fmtTime(it.duration_s)}</span>` : ""}
        ${stateChip(it)}${it.f1 != null ? `<span class="chip ${it.f1 >= 0.8 ? "ok" : "warn"}" title="F1 against the saved reference">F1 ${it.f1.toFixed(2)}</span>` : ""}
        <span>${relDate(it.created)}</span></span></a>
      <button class="lib-del" data-del="${esc(it.id)}" title="Delete" aria-label="Delete ${esc(it.title)}">&#x2715;</button></li>`).join("");
  $("#libEmpty").hidden = S.items.length > 0;
  const running = S.items.filter((i) => i.state === "running" || i.state === "queued");
  const b = $("#busy");
  b.hidden = running.length === 0;
  if (running.length) { b.textContent = running.length === 1 ? `Analysing ${running[0].title}` : `${running.length} songs in progress`; b.onclick = () => (location.hash = `#/item/${running[0].id}`); }
}

$("#libList").addEventListener("click", async (e) => {
  const del = e.target.closest("[data-del]");
  if (!del) { closeDrawer(); return; }
  e.preventDefault();
  const it = S.items.find((i) => i.id === del.dataset.del);
  if (!confirm(`Delete “${it ? it.title : "this song"}” from the library? Its audio copy, notation and saved reference are removed.`)) return;
  try {
    await api(`/api/jobs/${del.dataset.del}`, { method: "DELETE" });
    toast("Deleted");
    if (S.item && S.item.id === del.dataset.del) location.hash = "#/";
    refreshLibrary();
  } catch (err) { toast(err.message); }
});

function closeDrawer() { $("#library").classList.remove("open"); $("#libToggle").setAttribute("aria-expanded", "false"); }
$("#libToggle").addEventListener("click", () => {
  const open = $("#library").classList.toggle("open");
  $("#libToggle").setAttribute("aria-expanded", String(open));
});

/* ------------------------------------------------------------------ option forms */

function fieldHTML(f, value, idPrefix) {
  const id = `${idPrefix}-${f.key}`;
  const help = f.help ? `<div class="help">${esc(f.help)}</div>` : "";
  if (f.type === "bool") {
    return `<div class="field check"><label><input type="checkbox" id="${id}" data-key="${f.key}" ${value ? "checked" : ""}> ${esc(f.label)}</label>${help}</div>`;
  }
  let input;
  if (f.type === "select") {
    input = `<select id="${id}" data-key="${f.key}">${f.choices.map((c) => {
      const label = (f.labels && (f.labels[c] ?? f.labels[String(c)])) ?? c;
      return `<option value="${esc(c)}" ${String(c) === String(value) ? "selected" : ""}>${esc(label)}</option>`;
    }).join("")}</select>`;
  } else if (f.type === "number") {
    input = `<input type="number" id="${id}" data-key="${f.key}" value="${esc(value)}" min="${f.min ?? ""}" max="${f.max ?? ""}" step="${f.step ?? "any"}">`;
  } else {
    input = `<input type="text" id="${id}" data-key="${f.key}" value="${esc(value)}" placeholder="${esc(f.placeholder || "")}">`;
  }
  return `<div class="field"><label for="${id}">${esc(f.label)}</label>${input}${help}</div>`;
}

function readFields(root) {
  const out = {};
  $$("[data-key]", root).forEach((el) => {
    const f = field(el.dataset.key);
    if (!f) return;
    if (f.type === "bool") out[f.key] = el.checked;
    else if (f.type === "number") out[f.key] = Number(el.value || 0);
    else if (f.type === "select" && typeof f.choices[0] === "number") out[f.key] = Number(el.value);
    else out[f.key] = el.value;
  });
  return out;
}

/* ------------------------------------------------------------------ router */

function route() {
  closeDrawer();
  const h = location.hash || "#/";
  const m = h.match(/^#\/item\/([\w-]+)/);
  S.viewToken++;
  if (m) showItem(m[1]);
  else showUpload();
  drawLibrary();
}
window.addEventListener("hashchange", route);

/* ------------------------------------------------------------------ upload view */

function showUpload() {
  player.detach();
  S.view = "upload"; S.item = null; S.render = null; S.lastComparison = null;
  const opts = { ...defaults(), ...store.get("opts", {}) };
  const accept = S.exts.join(",");
  $("#main").innerHTML = `
    <section class="upload">
      <h1>Turn a song into bansuri sargam</h1>
      <p class="lead">Drop a song. The vocal line is separated from the music, its melody tracked, and written out in sargam
        for your flute, with the scale to play it in.</p>
      <label class="drop" id="drop">
        <input type="file" id="file" accept="${esc(accept)},audio/*,video/mp4" hidden>
        <div class="big-icon">&#9835;</div>
        <div class="drop-main">Drop a song here, or <u>choose a file</u></div>
        <div class="muted small">mp3, wav, m4a, flac or mp4</div>
        <div class="file-name" id="fileName"></div>
      </label>
      <details class="opts" ${store.get("optsOpen", false) ? "open" : ""}>
        <summary>Options</summary>
        <div class="form-grid" id="optForm">${S.schema.map((f) => fieldHTML(f, opts[f.key], "up")).join("")}</div>
      </details>
      <div class="go-row">
        <button class="btn primary big" id="go" disabled>Transcribe</button>
        <span class="muted small" id="goHint">Separation takes about a minute for a 4-minute song; repeat runs are instant.</span>
      </div>
    </section>`;
  const fileInput = $("#file");
  const setFile = (f) => {
    if (!f) return;
    S.pendingFile = f;
    $("#fileName").textContent = `${f.name} · ${(f.size / 1e6).toFixed(1)} MB`;
    $("#go").disabled = false;
    $("#go").focus();
  };
  fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
  const drop = $("#drop");
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove("over")));
  $(".opts").addEventListener("toggle", (e) => store.set("optsOpen", e.target.open));
  $("#go").addEventListener("click", () => submitUpload());
  if (S.pendingFile) setFile(S.pendingFile);
  upload.setFile = setFile;
}
const upload = { setFile: null };

async function submitUpload() {
  const f = S.pendingFile;
  if (!f) return;
  const opts = readFields($("#optForm"));
  store.set("opts", opts);
  if (opts.style) { S.style = opts.style; store.set("style", S.style); }
  if (opts.detail != null) { S.detail = opts.detail; store.set("detail", S.detail); }
  const fd = new FormData();
  fd.append("file", f, f.name);
  fd.append("options", JSON.stringify(opts));
  const go = $("#go");
  go.disabled = true; go.textContent = "Uploading…";
  try {
    const meta = await api("/api/jobs", { method: "POST", body: fd });
    S.pendingFile = null;
    location.hash = `#/item/${meta.id}`;
    refreshLibrary();
  } catch (err) {
    go.disabled = false; go.textContent = "Transcribe";
    toast(err.message);
  }
}

/* global drag & drop: dropping a file anywhere starts a new upload */
let dragDepth = 0;
window.addEventListener("dragenter", (e) => { if (e.dataTransfer && Array.from(e.dataTransfer.types).includes("Files")) { dragDepth++; if (S.view !== "upload") $("#dropHint").hidden = false; } });
window.addEventListener("dragleave", () => { dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) $("#dropHint").hidden = true; });
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0; $("#dropHint").hidden = true;
  const f = e.dataTransfer && e.dataTransfer.files[0];
  if (!f) return;
  S.pendingFile = f;
  if (S.view === "upload" && upload.setFile) upload.setFile(f);
  else location.hash = "#/";
});

/* ------------------------------------------------------------------ item view */

async function showItem(id) {
  const token = S.viewToken;
  let meta;
  try { meta = await api(`/api/jobs/${id}`); } catch (err) {
    $("#main").innerHTML = `<div class="card"><h2>Not found</h2><p class="muted">${esc(err.message)}</p><a class="btn" href="#/">Back</a></div>`;
    return;
  }
  if (token !== S.viewToken) return;
  if (!S.item || S.item.id !== id) { player.detach(); S.lastComparison = null; }
  S.item = meta;
  drawLibrary();
  if (meta.state === "queued" || meta.state === "running") { showProgress(meta, token); return; }
  if (meta.state === "error") { showError(meta); return; }
  await showResult(meta, token);
}

function showProgress(meta, token) {
  S.view = "progress";
  player.detach();
  const keys = S.stages.map((s) => s.key);
  const draw = (m) => {
    const cur = m.stage_key ? keys.indexOf(m.stage_key) : -1;
    const queued = m.state === "queued";
    $("#main").innerHTML = `
      <section class="card progress-card">
        <h2>${queued ? "Waiting to start" : "Transcribing"} “${esc(m.title)}”</h2>
        <p class="muted small">${queued ? (m.queue_position > 0 ? `${m.queue_position} ahead in the queue` : "starting…") : esc(m.stage || "")}
          ${m.elapsed != null ? ` · ${fmtTime(m.elapsed)} elapsed` : ""}</p>
        <ol class="stages">${S.stages.map((s, i) => `<li class="${queued ? "" : i < cur ? "done" : i === cur ? "now" : ""}">${esc(s.label)}</li>`).join("")}</ol>
        <div class="bar"><div></div></div>
        <p class="muted small" style="margin-top:12px">You can upload more songs meanwhile; they run one after another.</p>
      </section>`;
  };
  draw(meta);
  const poll = async () => {
    if (token !== S.viewToken) return;
    let m;
    try { m = await api(`/api/jobs/${meta.id}`); } catch { setTimeout(poll, 1500); return; }
    if (token !== S.viewToken) return;
    if (m.state === "queued" || m.state === "running") { draw(m); setTimeout(poll, 800); return; }
    refreshLibrary();
    showItem(meta.id);
  };
  setTimeout(poll, 600);
}

function showError(meta) {
  S.view = "error";
  player.detach();
  $("#main").innerHTML = `
    <section class="card error-box" style="max-width:720px">
      <h2>Couldn't transcribe “${esc(meta.title)}”</h2>
      <div class="error-msg">${esc(meta.error || "Unknown error")}</div>
      <p class="muted small">Adjust the options if needed and run it again. Details are in <code>error.log</code> in the song's library folder.</p>
      <div class="form-grid" id="errForm">${analysisFields().map((f) => fieldHTML(f, meta.options[f.key], "err")).join("")}</div>
      <div class="go-row" style="margin-top:16px"><button class="btn primary" id="retry">Run again</button></div>
    </section>`;
  $("#retry").addEventListener("click", () => rerun(meta.id, readFields($("#errForm"))));
}

async function rerun(id, options) {
  try {
    await postJSON(`/api/jobs/${id}/rerun`, { options });
    refreshLibrary();
    S.viewToken++;
    showItem(id);
  } catch (err) { toast(err.message); }
}

const LEGENDS = {
  nw: "<code>S R G m P D N</code> middle octave · <code>p d n</code> lowercase = lower octave · <code>S'</code> upper octave · " +
      "<code>R(k) G(k) D(k) N(k)</code> komal · <code>M</code> tivra Ma · each dot after a note = one unit of length",
  dots: "<code>S R G m P D N</code> shuddh · <code>r g d n</code> komal · <code>M</code> tivra Ma · <code>.P</code> lower octave · " +
        "<code>S'</code> upper octave · <code>{RS}</code> murki · <code>(R)</code> kann · <code>S~G</code> meend · dots = length",
  dashes: "<code>S R G m P D N</code> shuddh · <code>r g d n</code> komal · <code>M</code> tivra Ma · <code>.P</code> lower octave · " +
          "<code>S'</code> upper octave · <code>{RS}</code> murki · <code>(R)</code> kann · <code>S~G</code> meend · <code>-</code> hold one more beat",
};

async function showResult(meta, token) {
  S.view = "result";
  const s = meta.summary || {};
  const confClass = { high: "ok", medium: "warn", low: "bad" }[s.tonic_label] || "";
  const warnings = [];
  if (s.tonic_label === "low" && !s.overridden) {
    warnings.push(`<div class="callout warn"><span><b>Unsure about Sa.</b> ${esc(s.ambiguity || "")} If it sounds off, try Sa = ${esc(s.alternative)}.</span>
      <button class="btn small" data-sa="${esc(s.alternative)}">Use Sa = ${esc(s.alternative)}</button></div>`);
  }
  (s.key_changes || []).forEach((k) => warnings.push(`<div class="callout warn"><span><b>Possible key change</b> to ${esc(k.tonic)} around ${fmtTime(k.start)}–${fmtTime(k.end)}.</span>
      <button class="btn small" data-seek="${k.start}">Listen</button></div>`));
  const created = new Date(meta.created * 1000).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
  $("#main").innerHTML = `
    <section class="result">
      <div class="title-row"><h1 id="title" title="Click to rename">${esc(meta.title)}</h1>
        <span class="muted small">${esc(meta.filename)} · ${fmtTime(s.duration_s)} · ${esc(created)}${meta.revision > 1 ? ` · revision ${meta.revision}` : ""}</span></div>
      <div class="card scale-card">
        <div class="scale-main"><div class="scale-label">Scale of the flute</div><div class="scale-big">${esc(s.sa)}</div>
          <div class="scale-hz">Sa = ${esc(s.sa_hz)} Hz</div></div>
        <div class="facts">
          <div class="row"><span class="k">Sa</span><span class="chip ${confClass}">${s.overridden ? "set by you" : `${esc(s.tonic_label)} confidence`}</span>
            <span class="muted small">margin ${esc(s.tonic_margin)}</span></div>
          <div class="row"><span class="k">Scale</span><span>${esc(s.scale)}${s.mode ? ` · ${esc(s.sa)} ${esc(s.mode)}` : ""}</span></div>
          ${s.tempo_bpm ? `<div class="row"><span class="k">Tempo</span><span>${esc(s.tempo_bpm)} bpm</span></div>` : ""}
          <div class="row"><span class="k">Range</span><span>${esc(s.range)}</span>${s.fits ? "" : `<span class="chip warn">beyond the flute range</span>`}</div>
          <div class="row"><span class="k">Flute</span><span>${esc(s.flute_advice)}</span></div>
          ${warnings.join("")}
        </div>
      </div>

      <div class="toolbar">
        <label>Style ${selectHTML("style", S.style)}</label>
        <label>Detail ${selectHTML("detail", S.detail)}</label>
        <span class="grow"></span>
        <button class="btn" id="copyBtn" title="Copy the notation text">Copy</button>
        <div class="menu"><button class="btn" id="dlBtn" aria-haspopup="true">Download &#9662;</button>
          <div class="menu-list" id="dlMenu" hidden>
            <a id="dlTxt" download>Notation (.txt)</a>
            <a href="/api/jobs/${esc(meta.id)}/download/mid" download>Melody (.mid)</a>
            <a href="/api/jobs/${esc(meta.id)}/download/json" download>Timed notes (.json)</a>
          </div></div>
        <button class="btn" id="rerunBtn" aria-expanded="false">Re-analyse…</button>
      </div>
      <div class="card panel" id="rerunPanel" hidden>
        <h3>Re-analyse with different settings</h3>
        <p class="muted small">Stems and pitch tracks are cached, so this takes seconds unless you switch the pitch tracker.</p>
        <div class="form-grid" id="rerunForm">${analysisFields().map((f) => fieldHTML(f, meta.options[f.key], "rr")).join("")}</div>
        <div class="go-row" style="margin-top:14px"><button class="btn primary" id="rerunGo">Re-run</button>
          <button class="btn ghost" id="rerunCancel">Cancel</button></div>
      </div>
      <div class="legend" id="legend"></div>
      <div class="notation" id="notation" aria-label="Sargam notation"><p class="muted">Loading…</p></div>
      <p class="muted small" style="margin-top:8px">Click a note to jump there · click a phrase number to loop it ·
        <kbd>space</kbd> play/pause · <kbd>&larr;</kbd><kbd>&rarr;</kbd> previous/next phrase · <kbd>L</kbd> loop phrase · <kbd>Esc</kbd> stop looping</p>

      <section class="card ref" id="refPanel">
        <h3>Compare with a reference notation</h3>
        <p class="muted small">Paste notation from notationsworld, or its page URL. The transcription is scored against it, and the
          pair is saved with this song so it can be used for evaluation and training.</p>
        <div id="savedRef"></div>
        <div class="row"><input type="url" id="refUrl" placeholder="https://www.notationsworld.com/....html">
          <button class="btn" id="refFetch">Fetch &amp; compare</button></div>
        <textarea id="refText" rows="5" placeholder="SCALE OF THE SONG IS F MAJOR&#10;G..m.. R..G..R.S.R.. G..m.. P.G.P.G.R.S.."></textarea>
        <div class="row"><label class="small"><input type="checkbox" id="refSave" checked> Save as this song's reference</label>
          <span class="grow" style="flex:1"></span><button class="btn" id="refCompare">Compare pasted notation</button></div>
        <div id="refResult"></div>
      </section>

      <details class="diag"><summary>Analysis details</summary>
        <pre>${esc([
          `pitch tracker: ${s.pitch_backend}   separation: ${s.separation || "none"}   tuning: ${s.tuning_cents} cents`,
          `notes: ${s.n_notes}   phrases: ${s.n_phrases}   analysis time: ${meta.elapsed != null ? fmtTime(meta.elapsed) : "–"}`,
          `options: ${JSON.stringify(meta.options)}`,
          ...(s.log || []),
        ].join("\n"))}</pre></details>
    </section>`;

  $("#title").addEventListener("click", async () => {
    const t = prompt("Rename this song", meta.title);
    if (!t || t === meta.title) return;
    try { await api(`/api/jobs/${meta.id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title: t }) });
      meta.title = t; $("#title").textContent = t; refreshLibrary(); } catch (err) { toast(err.message); }
  });
  $$("[data-sa]").forEach((b) => b.addEventListener("click", () => rerun(meta.id, { ...meta.options, sa: b.dataset.sa })));
  $$(".facts [data-seek]").forEach((b) => b.addEventListener("click", () => { player.seek(Number(b.dataset.seek)); player.play(); }));
  $("[data-render=style]").addEventListener("change", (e) => { S.style = e.target.value; store.set("style", S.style); loadRender(meta.id); });
  $("[data-render=detail]").addEventListener("change", (e) => { S.detail = Number(e.target.value); store.set("detail", S.detail); loadRender(meta.id); });
  $("#copyBtn").addEventListener("click", async () => {
    if (!S.render) return;
    try { await navigator.clipboard.writeText(S.render.text); toast("Notation copied"); } catch { toast("Couldn't access the clipboard"); }
  });
  $("#dlBtn").addEventListener("click", (e) => { e.stopPropagation(); $("#dlMenu").hidden = !$("#dlMenu").hidden; });
  const toggleRerun = (show) => { $("#rerunPanel").hidden = !show; $("#rerunBtn").setAttribute("aria-expanded", String(show)); };
  $("#rerunBtn").addEventListener("click", () => toggleRerun($("#rerunPanel").hidden));
  $("#rerunCancel").addEventListener("click", () => toggleRerun(false));
  $("#rerunGo").addEventListener("click", () => rerun(meta.id, readFields($("#rerunForm"))));
  $("#refFetch").addEventListener("click", () => submitReference({ url: $("#refUrl").value }));
  $("#refUrl").addEventListener("keydown", (e) => { if (e.key === "Enter") submitReference({ url: $("#refUrl").value }); });
  $("#refCompare").addEventListener("click", () => submitReference({ text: $("#refText").value }));

  player.attach(meta);
  await loadRender(meta.id, token);
  if (meta.truth) loadSavedReference(meta.id);
}

function selectHTML(key, value) {
  const f = field(key);
  return `<select data-render="${key}">${f.choices.map((c) => {
    const label = (f.labels && (f.labels[c] ?? f.labels[String(c)])) ?? c;
    return `<option value="${esc(c)}" ${String(c) === String(value) ? "selected" : ""}>${esc(label)}</option>`;
  }).join("")}</select>`;
}

async function loadRender(id, token = S.viewToken) {
  let r;
  try { r = await api(`/api/jobs/${id}/render?style=${encodeURIComponent(S.style)}&detail=${S.detail}`); } catch (err) {
    $("#notation").innerHTML = `<p class="error-msg">${esc(err.message)}</p>`;
    return;
  }
  if (token !== S.viewToken) return;
  S.render = r;
  $("#legend").innerHTML = LEGENDS[r.style] || "";
  $("#dlTxt").href = `/api/jobs/${id}/download/txt?style=${encodeURIComponent(r.style)}&detail=${r.detail}`;
  drawNotation(r);
  player.setNotes(r);
  if (S.lastComparison) drawComparison(S.lastComparison);
}

function drawNotation(r) {
  const per = r.phrases_per_block || 4;
  const blocks = [];
  let k = 0;
  r.phrases.forEach((p, i) => {
    if (i % per === 0) blocks.push([]);
    const notes = p.segs.map((sg) => {
      if (sg.s == null) return esc(sg.t);
      const cls = sg.c < 0.5 && r.detail >= 2 ? "n unsure" : "n";
      return `<span class="${cls}" data-k="${k++}" data-s="${sg.s}" data-e="${sg.e}">${esc(sg.t)}</span>`;
    }).join("");
    blocks[blocks.length - 1].push(`<div class="line" data-i="${i}" data-start="${p.start}" data-end="${p.end}">
      <button class="pnum" data-loop="${i}" title="Loop phrase ${i + 1}">${i + 1}</button>
      <span class="ptime">${fmtTime(p.start)}</span><div class="notes">${notes}</div></div>`);
  });
  $("#notation").innerHTML = blocks.length ? blocks.map((b) => `<div class="block">${b.join("")}</div>`).join("")
    : `<p class="muted">No notes were found. If the input is a solo instrument, re-analyse with “skip vocal separation”.</p>`;
}

/* ------------------------------------------------------------------ reference comparison */

async function submitReference(body) {
  const id = S.item && S.item.id;
  if (!id) return;
  if (!body.url && !(body.text || "").trim()) { toast(body.url !== undefined ? "Enter a notation page URL" : "Paste some notation first"); return; }
  body.save = $("#refSave").checked;
  const out = $("#refResult");
  out.innerHTML = `<p class="muted">Comparing…</p>`;
  try {
    const r = await postJSON(`/api/jobs/${id}/reference`, body);
    S.lastComparison = r.comparison;
    drawSavedRef(r.saved);
    drawComparison(r.comparison);
    if (r.saved) { refreshLibrary(); toast("Reference saved with this song"); }
  } catch (err) { out.innerHTML = `<p class="error-msg">${esc(err.message)}</p>`; }
}

async function loadSavedReference(id) {
  try {
    const r = await api(`/api/jobs/${id}/reference`);
    if (!r.saved || !S.item || S.item.id !== id) return;
    S.lastComparison = r.comparison;
    drawSavedRef(r.saved);
    drawComparison(r.comparison);
  } catch (err) { $("#refResult").innerHTML = `<p class="error-msg">${esc(err.message)}</p>`; }
}

function drawSavedRef(saved) {
  const el = $("#savedRef");
  if (!el) return;
  if (!saved) { el.innerHTML = ""; return; }
  const src = saved.source && saved.source.startsWith("http") ? `<a href="${esc(saved.source)}" target="_blank" rel="noopener">${esc(saved.source)}</a>` : esc(saved.source);
  el.innerHTML = `<div class="saved-ref"><span class="chip ok">reference saved</span><span class="muted small">${src} · ${esc(saved.n_lines)} lines, Sa ${esc(saved.sa || "?")}</span>
    <button class="btn small ghost" id="refRemove">Remove</button></div>`;
  $("#refRemove").addEventListener("click", async () => {
    if (!confirm("Remove the saved reference for this song?")) return;
    try { await api(`/api/jobs/${S.item.id}/reference`, { method: "DELETE" }); el.innerHTML = ""; $("#refResult").innerHTML = ""; S.lastComparison = null; refreshLibrary(); }
    catch (err) { toast(err.message); }
  });
}

function drawComparison(c) {
  const out = $("#refResult");
  if (!out || !c) return;
  const nw = S.style === "nw";
  const sym = (o, which) => esc(nw ? o[which + "_nw"] : o[which]);
  const t = c.tonic;
  const tonic = t.truth ? `${esc(t.pred)} vs ${esc(t.truth)} ${t.ok ? "✓" : "✗"}` : `${esc(t.pred)} (reference has no scale line)`;
  const chip = (o) => {
    const at = o.t != null ? ` data-t="${o.t}"` : "";
    if (o.op === "match") return `<span class="d match"${at}>${sym(o, "truth")}</span>`;
    if (o.op === "sub") return `<span class="d sub"${at} title="transcribed ${sym(o, "pred")}, reference ${sym(o, "truth")}"><s>${sym(o, "pred")}</s>${sym(o, "truth")}</span>`;
    if (o.op === "octave") return `<span class="d oct"${at} title="octave differs: transcribed ${sym(o, "pred")}, reference ${sym(o, "truth")}"><s>${sym(o, "pred")}</s>${sym(o, "truth")}</span>`;
    if (o.op === "del") return `<span class="d miss" title="missing from the transcription">${sym(o, "truth")}</span>`;
    return `<span class="d extra"${at} title="extra note in the transcription">${sym(o, "pred")}</span>`;
  };
  out.innerHTML = `
    <div class="scores">
      <div class="score"><div class="v">${tonic}</div><div class="k">Sa (tonic)</div></div>
      <div class="score"><div class="v">${c.f1.toFixed(2)}</div><div class="k">F1 (aligned)</div></div>
      <div class="score"><div class="v">${pct(c.precision)}</div><div class="k">Precision</div></div>
      <div class="score"><div class="v">${pct(c.recall)}</div><div class="k">Recall</div></div>
      <div class="score"><div class="v">${c.ser.toFixed(2)}</div><div class="k">Sargam error rate</div></div>
    </div>
    <div class="diff-legend"><span><span class="d match">G</span> match</span><span><span class="d sub"><s>R</s>G</span> wrong note</span>
      <span><span class="d oct"><s>n</s>N</span> wrong octave</span><span><span class="d miss">G</span> missed</span><span><span class="d extra">R</span> extra</span>
      <span>· ${c.n_pred} transcribed vs ${c.n_truth} reference notes · click a note to hear that spot</span></div>
    <div class="diff">${c.lines.map((l) => `<div class="dline"><span class="num">${l.i + 1}</span><div>${l.ops.map(chip).join("")}</div></div>`).join("")}</div>`;
}

document.addEventListener("click", () => { const m = $("#dlMenu"); if (m) m.hidden = true; });
$("#main").addEventListener("click", (e) => {
  const d = e.target.closest(".d[data-t]");
  if (d) { player.seek(Number(d.dataset.t) - 0.3); player.play(); return; }
  const n = e.target.closest(".n");
  if (n) { player.seek(Number(n.dataset.s) - 0.02); if (player.paused()) player.preview(Number(n.dataset.k)); return; }
  const pn = e.target.closest(".pnum");
  if (pn) { player.toggleLoop(Number(pn.dataset.loop)); }
});

/* ------------------------------------------------------------------ synth (WebAudio) */

const synth = {
  ctx: null, master: null, noise: null, live: new Set(), volume: 0.5,
  ensure() {
    if (!this.ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      this.ctx = new AC();
      const comp = this.ctx.createDynamicsCompressor();
      this.master = this.ctx.createGain();
      this.master.gain.value = this.volume;
      this.master.connect(comp).connect(this.ctx.destination);
      const len = this.ctx.sampleRate * 2;
      this.noise = this.ctx.createBuffer(1, len, this.ctx.sampleRate);
      const d = this.noise.getChannelData(0);
      for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
    }
    if (this.ctx.state === "suspended") this.ctx.resume();
    return this.ctx;
  },
  setVolume(v) { this.volume = v; if (this.master) this.master.gain.setTargetAtTime(v, this.ctx.currentTime, 0.03); },
  freq(midi, cents) { return 440 * Math.pow(2, (midi - 69 + (cents || 0) / 100) / 12); },
  /* a soft bansuri-like voice: sine + a little 2nd/3rd harmonic, breathy band-passed noise, gentle vibrato */
  voice(f, when, dur) {
    const ctx = this.ctx;
    const attack = 0.035, release = 0.08, peak = 0.28;
    const end = when + Math.max(dur, attack + 0.03);
    const env = ctx.createGain();
    env.gain.setValueAtTime(0, when);
    env.gain.linearRampToValueAtTime(peak, when + attack);
    env.gain.setTargetAtTime(peak * 0.82, when + attack, 0.1);
    env.gain.setTargetAtTime(0, end, release / 3);
    env.connect(this.master);
    const nodes = [];
    [[1, 1], [2, 0.12], [3, 0.045]].forEach(([h, g]) => {
      const o = ctx.createOscillator();
      o.type = "sine";
      o.frequency.value = f * h;
      const gn = ctx.createGain();
      gn.gain.value = g;
      o.connect(gn).connect(env);
      nodes.push(o);
    });
    if (dur > 0.35) { // vibrato on held notes, after the attack
      const lfo = ctx.createOscillator(), depth = ctx.createGain();
      lfo.frequency.value = 5.2;
      depth.gain.setValueAtTime(0, when);
      depth.gain.linearRampToValueAtTime(f * 0.004, when + 0.35);
      lfo.connect(depth);
      nodes.slice(0, 1).forEach((o) => depth.connect(o.frequency));
      nodes.push(lfo);
    }
    const n = ctx.createBufferSource();
    n.buffer = this.noise; n.loop = true;
    const bp = ctx.createBiquadFilter();
    bp.type = "bandpass"; bp.frequency.value = Math.min(8000, f * 2.5); bp.Q.value = 1.2;
    const ng = ctx.createGain();
    ng.gain.setValueAtTime(0.09, when);
    ng.gain.setTargetAtTime(0.035, when + attack, 0.06);
    n.connect(bp).connect(ng).connect(env);
    nodes.push(n);
    const stopAt = end + release * 3 + 0.05;
    nodes.forEach((x) => { x.start(when); x.stop(stopAt); });
    const v = { env, nodes };
    this.live.add(v);
    nodes[0].onended = () => this.live.delete(v);
    return v;
  },
  silence() {
    if (!this.ctx) return;
    const t = this.ctx.currentTime;
    this.live.forEach((v) => {
      v.env.gain.cancelScheduledValues(t);
      v.env.gain.setTargetAtTime(0, t, 0.012);
      v.nodes.forEach((x) => { try { x.stop(t + 0.07); } catch { /* not started yet */ } });
    });
    this.live.clear();
  },
};

/* ------------------------------------------------------------------ player */

const audio = $("#audio");
const player = {
  meta: null, src: store.get("src", "song"), synthOn: store.get("synthOn", false), balance: store.get("balance", 50),
  notes: [], noteEls: [], noteTimes: [], lines: [], next: 0, curNote: -1, curLine: -1, loop: null,
  timer: null, raf: null, seeking: false, lastUserScroll: 0, pendingSeek: null,

  attach(meta) {
    this.meta = meta;
    document.body.classList.add("has-player");
    $("#player").hidden = false;
    const hasVocals = !!(meta.playback && meta.playback.vocals);
    $$(".seg button").forEach((b) => { b.disabled = b.dataset.src === "vocals" && !hasVocals; });
    if (this.src === "vocals" && !hasVocals) this.src = "song";
    this.loop = null;
    this.drawControls();
    this.load(0, false);
  },
  detach() {
    this.stop();
    this.meta = null;
    audio.removeAttribute("src");
    audio.load();
    $("#player").hidden = true;
    document.body.classList.remove("has-player");
  },
  load(t, play) {
    if (!this.meta) return;
    const which = this.src === "vocals" ? "vocals" : "song";
    const url = `/api/jobs/${this.meta.id}/audio/${which}`;
    if (audio.getAttribute("src") !== url) {
      this.pendingSeek = { t, play };
      audio.src = url;
      audio.load();
    } else {
      audio.currentTime = t;
      if (play) this.play();
    }
    this.applyMix();
  },
  setNotes(r) {
    this.notes = r.notes;
    this.tuning = r.tuning_cents;
    this.noteEls = $$("#notation .n");
    this.noteTimes = this.noteEls.map((el) => [Number(el.dataset.s), Number(el.dataset.e)]);
    this.lines = $$("#notation .line").map((el) => ({ el, start: Number(el.dataset.start), end: Number(el.dataset.end), i: Number(el.dataset.i) }));
    this.curNote = -1; this.curLine = -1;
    if (this.loop != null) this.markLoop();
    this.resync();
    this.highlight();
  },
  paused() { return audio.paused; },
  async play() {
    if (!this.meta) return;
    if (this.synthOn) synth.ensure();
    try { await audio.play(); } catch (err) { toast("Can't play: " + err.message); }
  },
  pause() { audio.pause(); },
  toggle() { audio.paused ? this.play() : this.pause(); },
  stop() { audio.pause(); synth.silence(); clearInterval(this.timer); this.timer = null; cancelAnimationFrame(this.raf); },
  seek(t) {
    if (!this.meta) return;
    audio.currentTime = Math.max(0, Math.min(t, audio.duration || t));
    this.highlight();
  },
  applyMix() {
    const b = this.balance / 100;
    const song = this.src === "none" ? 0 : Math.min(1, 2 * (1 - b));
    audio.volume = this.synthOn ? song : (this.src === "none" ? 0 : 1);
    synth.setVolume(this.synthOn ? Math.min(1, 2 * b) * 0.6 : 0);
  },
  drawControls() {
    $$(".seg button").forEach((b) => b.classList.toggle("on", b.dataset.src === this.src));
    $("#synthOn").checked = this.synthOn;
    $("#balance").value = this.balance;
    $(".balance").style.opacity = this.synthOn && this.src !== "none" ? 1 : 0.4;
    $("#speed").value = String(audio.playbackRate || 1);
    const lc = $("#loopChip");
    lc.hidden = this.loop == null;
    if (this.loop != null) lc.textContent = `Looping phrase ${this.loop + 1}  ✕`;
  },
  /* schedule transcription notes a little ahead of the media clock */
  schedule() {
    if (!this.synthOn || audio.paused || !synth.ctx) return;
    const t = audio.currentTime, rate = audio.playbackRate || 1, look = 0.35 * rate;
    const loopEnd = this.loop != null ? this.lines[this.loop].end + 0.15 : Infinity;
    while (this.next < this.notes.length && this.notes[this.next][0] < t + look) {
      const [s, e, m] = this.notes[this.next];
      if (e > t && s < loopEnd) {
        const from = Math.max(s, t);
        const when = synth.ctx.currentTime + (from - t) / rate;
        synth.voice(synth.freq(m, this.tuning), when, (Math.min(e, loopEnd) - from) / rate);
      }
      this.next++;
    }
  },
  resync() {
    synth.silence();
    const t = audio.currentTime || 0;
    let lo = 0, hi = this.notes.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (this.notes[mid][1] <= t) lo = mid + 1; else hi = mid; }
    this.next = lo;
    this.schedule();
  },
  preview(k) {
    const nt = this.noteTimes[k];
    if (!nt) return;
    synth.ensure();
    synth.setVolume(0.6);
    const midi = this.midiAt((nt[0] + nt[1]) / 2);
    if (midi != null) synth.voice(synth.freq(midi, this.tuning), synth.ctx.currentTime + 0.01, Math.min(0.9, Math.max(0.25, nt[1] - nt[0])));
    setTimeout(() => this.applyMix(), 1200);
    this.setCurrentNote(k);
  },
  midiAt(t) {
    const n = this.notes.find((x) => x[0] <= t && t < x[1]) || this.notes.find((x) => x[0] >= t);
    return n ? n[2] : null;
  },
  findNote(t) {
    let lo = 0, hi = this.noteTimes.length - 1, ans = -1;
    while (lo <= hi) { const mid = (lo + hi) >> 1; if (this.noteTimes[mid][0] <= t) { ans = mid; lo = mid + 1; } else hi = mid - 1; }
    if (ans >= 0 && t < this.noteTimes[ans][1] + 0.05) return ans;
    return -1;
  },
  findLine(t) {
    let lo = 0, hi = this.lines.length - 1, ans = -1;
    while (lo <= hi) { const mid = (lo + hi) >> 1; if (this.lines[mid].start <= t + 0.05) { ans = mid; lo = mid + 1; } else hi = mid - 1; }
    return ans;
  },
  setCurrentNote(k) {
    if (k === this.curNote) return;
    if (this.curNote >= 0 && this.noteEls[this.curNote]) this.noteEls[this.curNote].classList.remove("now");
    this.curNote = k;
    if (k >= 0 && this.noteEls[k]) this.noteEls[k].classList.add("now");
  },
  highlight() {
    const t = audio.currentTime || 0;
    $("#tCur").textContent = fmtTime(t);
    if (!this.seeking && audio.duration) $("#seek").value = String(Math.round((t / audio.duration) * 1000));
    this.setCurrentNote(this.findNote(t));
    const li = this.findLine(t);
    if (li !== this.curLine) {
      if (this.curLine >= 0 && this.lines[this.curLine]) this.lines[this.curLine].el.classList.remove("current");
      this.curLine = li;
      if (li >= 0) {
        const el = this.lines[li].el;
        el.classList.add("current");
        if (!audio.paused && Date.now() - this.lastUserScroll > 2500) {
          const r = el.getBoundingClientRect(), bottom = window.innerHeight - 110;
          if (r.top < 70 || r.bottom > bottom) el.scrollIntoView({ block: "center", behavior: "smooth" });
        }
      }
    }
  },
  tick() {
    if (this.loop != null && this.lines[this.loop]) {
      const L = this.lines[this.loop];
      if (audio.currentTime >= L.end + 0.15 || audio.currentTime < L.start - 0.5) { audio.currentTime = Math.max(0, L.start - 0.1); }
    }
    this.highlight();
    if (!audio.paused) this.raf = requestAnimationFrame(() => this.tick());
  },
  toggleLoop(i) {
    if (this.loop === i) { this.clearLoop(); return; }
    this.loop = i;
    this.markLoop();
    this.drawControls();
    this.seek(Math.max(0, this.lines[i].start - 0.1));
    this.play();
  },
  markLoop() { this.lines.forEach((l) => l.el.classList.toggle("looping", l.i === this.loop)); },
  clearLoop() { this.loop = null; this.markLoop(); this.drawControls(); this.resync(); },
  jumpPhrase(dir) {
    if (!this.lines.length) return;
    const t = audio.currentTime;
    let li = this.findLine(t);
    if (dir > 0) li = Math.min(this.lines.length - 1, li + 1);
    else if (li >= 0 && t - this.lines[li].start < 1.0) li = Math.max(0, li - 1);
    li = Math.max(0, li);
    if (this.loop != null) { this.loop = li; this.markLoop(); this.drawControls(); }
    this.seek(Math.max(0, this.lines[li].start - 0.1));
  },
};

audio.addEventListener("loadedmetadata", () => {
  $("#tDur").textContent = fmtTime(audio.duration);
  if (player.pendingSeek) {
    const { t, play } = player.pendingSeek;
    player.pendingSeek = null;
    audio.currentTime = t;
    if (play) player.play();
  }
});
audio.addEventListener("play", () => {
  $("#playBtn").innerHTML = "&#10074;&#10074;";
  $("#playBtn").setAttribute("aria-label", "Pause (space)");
  player.resync();
  clearInterval(player.timer);
  player.timer = setInterval(() => player.schedule(), 40);
  player.tick();
});
audio.addEventListener("pause", () => {
  $("#playBtn").innerHTML = "&#9654;";
  $("#playBtn").setAttribute("aria-label", "Play (space)");
  clearInterval(player.timer); player.timer = null;
  synth.silence();
  player.highlight();
});
audio.addEventListener("seeked", () => { player.resync(); player.highlight(); });
audio.addEventListener("ratechange", () => player.resync());
audio.addEventListener("ended", () => synth.silence());
audio.addEventListener("error", () => { if (player.meta && audio.getAttribute("src")) toast("This audio can't be played in the browser"); });

$("#playBtn").addEventListener("click", (e) => { e.currentTarget.blur(); player.toggle(); });
$("#seek").addEventListener("input", (e) => {
  player.seeking = true;
  if (audio.duration) { audio.currentTime = (Number(e.target.value) / 1000) * audio.duration; $("#tCur").textContent = fmtTime(audio.currentTime); }
});
$("#seek").addEventListener("change", () => { player.seeking = false; });
$$(".seg button").forEach((b) => b.addEventListener("click", () => {
  b.blur();
  player.src = b.dataset.src;
  store.set("src", player.src);
  if (player.src === "none" && !player.synthOn) { player.synthOn = true; store.set("synthOn", true); synth.ensure(); player.resync(); }
  player.drawControls();
  player.load(audio.currentTime || 0, !audio.paused);
}));
$("#synthOn").addEventListener("change", (e) => {
  player.synthOn = e.target.checked;
  store.set("synthOn", player.synthOn);
  if (player.synthOn) synth.ensure();
  player.applyMix(); player.drawControls(); player.resync();
});
$("#balance").addEventListener("input", (e) => { player.balance = Number(e.target.value); store.set("balance", player.balance); player.applyMix(); });
$("#speed").addEventListener("change", (e) => {
  audio.defaultPlaybackRate = audio.playbackRate = Number(e.target.value); // default survives switching song/vocals
  e.target.blur();
});
$("#loopChip").addEventListener("click", () => player.clearLoop());
["wheel", "touchmove"].forEach((ev) => window.addEventListener(ev, () => { player.lastUserScroll = Date.now(); }, { passive: true }));

document.addEventListener("keydown", (e) => {
  if (S.view !== "result" || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.target.closest("input, textarea, select, [contenteditable]")) return;
  if (e.code === "Space") { e.preventDefault(); player.toggle(); }
  else if (e.key === "ArrowRight") { e.preventDefault(); player.jumpPhrase(1); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); player.jumpPhrase(-1); }
  else if (e.key === "l" || e.key === "L") { const li = player.findLine(audio.currentTime); if (li >= 0) player.toggleLoop(li); }
  else if (e.key === "Escape" && player.loop != null) player.clearLoop();
  else if (e.key === "PageDown" || e.key === "PageUp") player.lastUserScroll = Date.now();
});
document.addEventListener("keyup", (e) => { if (e.code === "Space" && e.target.tagName === "BUTTON" && S.view === "result") e.preventDefault(); });

/* ------------------------------------------------------------------ boot */

(async function boot() {
  try {
    const o = await api("/api/options");
    S.schema = o.schema; S.stages = o.stages; S.exts = o.audio_exts;
  } catch (err) {
    $("#main").innerHTML = `<div class="card"><h2>Can't reach the sargam server</h2><p class="error-msg">${esc(err.message)}</p></div>`;
    return;
  }
  if (!field("style").choices.includes(S.style)) S.style = field("style").default;
  await refreshLibrary();
  route();
})();
