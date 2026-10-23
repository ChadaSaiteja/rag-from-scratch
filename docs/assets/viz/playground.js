// Playground: type any sentence and watch it become tokens, vocabulary IDs, a
// TF-IDF vector, a point in the same 3D space as the stored chunks, and a ranked
// list of nearest chunks. Plain math in the browser, no model involved.
import { createStage, hasWebGL, makeGlowPoint, makeLine, setPointOpacity, THREE } from "./core.js";
import { DATA, buildPoints, escapeHtml } from "./space-common.js";

const SP = DATA.space;
const root = document.querySelector(".pg");
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
const STOP = new Set(SP.stopwords);
const VOCAB_INDEX = new Map(SP.vocab.map((w, i) => [w, i]));
// mean · component, so projecting a sparse vector only needs its non-zero entries
const MEAN_DOT = SP.comps.map((comp) => comp.reduce((s, c, i) => s + c * SP.mean[i], 0));

const $ = (sel) => root.querySelector(sel);
const out = (stage) => root.querySelector(`.pg-step[data-stage=${stage}] .pg-out`);
const steps = [...root.querySelectorAll(".pg-step")];

// ------------------------------------------------------------------ the math

function analyse(text) {
  const raw = text.match(/[A-Za-z0-9]+|[^\sA-Za-z0-9]/g) || [];
  const tokens = raw.map((piece) => {
    const lower = piece.toLowerCase();
    let status = "kept";
    if (!/[a-z0-9]/i.test(piece)) status = "punct";
    else if (STOP.has(lower)) status = "stop";
    else if (lower.length < 2) status = "short";
    const id = status === "kept" ? VOCAB_INDEX.get(lower) : undefined;
    if (status === "kept" && id === undefined) status = "oov";
    return { piece, lower, status, id };
  });

  const counts = new Map();
  tokens.filter((t) => t.status === "kept").forEach((t) => counts.set(t.id, (counts.get(t.id) || 0) + 1));
  let entries = [...counts].map(([id, n]) => [id, (1 + Math.log(n)) * SP.idf[id]]);
  const norm = Math.hypot(...entries.map(([, v]) => v));
  entries = norm ? entries.map(([id, v]) => [id, v / norm]) : [];
  entries.sort((a, b) => b[1] - a[1]);

  const pos = entries.length
    ? SP.comps.map((comp, k) => (entries.reduce((s, [id, v]) => s + v * comp[id], 0) - MEAN_DOT[k]) * SP.scale)
    : null;

  const weights = new Map(entries);
  const nearest = SP.chunkVecs
    .map((vec, cid) => [cid, vec.reduce((s, [id, v]) => s + v * (weights.get(id) || 0), 0)])
    .filter(([, s]) => s > 0)
    .sort((a, b) => b[1] - a[1])
    .slice(0, DATA.settings.k);

  return { tokens, entries, pos, nearest };
}

// ------------------------------------------------------------------ 3D view

let view = null;
function setupView() {
  const stageEl = root.querySelector(".pg-stage");
  if (!hasWebGL()) {
    stageEl.querySelector(".viz-fallback").hidden = false;
    return;
  }
  const stage = createStage(stageEl, { cameraPos: [15, 9, 25], autoRotate: true });
  const group = new THREE.Group();
  stage.scene.add(group);
  const points = buildPoints(stage, group);
  const query = makeGlowPoint(0xffffff, 1.8);
  setPointOpacity(query, 0);
  group.add(query);
  const lines = [0, 1, 2].map(() => {
    const l = makeLine(new THREE.Vector3(), new THREE.Vector3(1, 0, 0), 0xffffff, 0);
    l.visible = false;
    group.add(l);
    return l;
  });
  stage.onFrame((t) => {
    if (query.visible) query.userData.halo.scale.setScalar(2.9 + 0.35 * Math.sin(t * 3));
  });
  view = { stage, points, query, lines };
}

function showInSpace(result) {
  if (!view) return;
  const { stage, points, query, lines } = view;
  stage.cancelTweens();
  lines.forEach((l) => { l.visible = false; });
  const nearIds = new Set(result.nearest.map(([cid]) => cid));
  points.forEach((p, i) => { setPointOpacity(p, result.pos && !nearIds.has(i) ? 0.35 : 1); p.scale.setScalar(nearIds.has(i) ? 1.5 : 1); });
  if (!result.pos) { setPointOpacity(query, 0); return; }
  const from = new THREE.Vector3(0, 16, 6);
  const to = new THREE.Vector3(...result.pos);
  setPointOpacity(query, 1);
  stage.animate(1300, (e) => query.position.lerpVectors(from, to, e)).then(() => {
    result.nearest.forEach(([cid], k) => {
      const l = lines[k];
      const p = l.geometry.attributes.position;
      const target = DATA.chunks[cid].pos;
      p.setXYZ(0, to.x, to.y, to.z);
      p.setXYZ(1, target[0], target[1], target[2]);
      p.needsUpdate = true;
      l.geometry.computeBoundingSphere();
      l.visible = true;
      stage.animate(500, (e) => { l.material.opacity = 0.9 * e; });
    });
  });
}

// ------------------------------------------------------------------ rendering

const cells = [];
function buildGrid() {
  const grid = root.querySelector(".pg-grid");
  const frag = document.createDocumentFragment();
  SP.vocab.forEach((w, i) => {
    const cell = document.createElement("span");
    cell.className = "pg-cell";
    cell.title = `#${i} ${w}`;
    frag.appendChild(cell);
    cells.push(cell);
  });
  grid.appendChild(frag);
}

const STATUS_TEXT = {
  kept: "kept", stop: "common word, dropped", punct: "punctuation, dropped",
  short: "too short, dropped", oov: "not in vocabulary",
};

function render(text, result) {
  out("sentence").innerHTML = `<p class="pg-sentence">“${escapeHtml(text)}”</p>` +
    `<p class="pg-meta">${text.length} characters</p>`;

  out("tokens").innerHTML = `<div class="pg-chips">${result.tokens.map((t, i) =>
    `<span class="pg-chip ${t.status}" style="--i:${i}" title="${STATUS_TEXT[t.status]}">${escapeHtml(t.piece)}</span>`).join("")}</div>` +
    `<p class="pg-meta">${result.tokens.length} pieces · ${result.tokens.filter((t) => t.status === "kept" || t.status === "oov").length} meaningful words</p>`;

  const words = result.tokens.filter((t) => t.status === "kept" || t.status === "oov");
  out("ids").innerHTML = words.length
    ? `<div class="pg-ids">${words.map((t, i) => t.status === "kept"
        ? `<span class="pg-id" style="--i:${i}"><b>${escapeHtml(t.lower)}</b><i>→</i><code>#${t.id}</code></span>`
        : `<span class="pg-id oov" style="--i:${i}"><b>${escapeHtml(t.lower)}</b><i>→</i><code>unknown</code></span>`).join("")}</div>` +
      `<p class="pg-meta">IDs are positions in a ${SP.vocab.length}-word vocabulary built from the ${DATA.chunks.length} chunks.</p>`
    : `<p class="pg-empty">No meaningful words to look up. Try a sentence about RAG.</p>`;

  cells.forEach((c) => { c.style.removeProperty("--w"); c.classList.remove("on"); c.style.removeProperty("--i"); });
  result.entries.forEach(([id, v], k) => {
    const c = cells[id];
    c.style.setProperty("--w", v.toFixed(3));
    c.style.setProperty("--i", k);
    c.classList.add("on");
    c.title = `#${id} ${SP.vocab[id]} · weight ${v.toFixed(3)}`;
  });
  out("vector").querySelector(".pg-vector-meta").innerHTML = result.entries.length
    ? `<b>${result.entries.length}</b> of ${SP.vocab.length} numbers are non-zero:` +
      `<span class="pg-weights">${result.entries.map(([id, v]) => `<span>${escapeHtml(SP.vocab[id])} <code>${v.toFixed(2)}</code></span>`).join("")}</span>`
    : `All ${SP.vocab.length} numbers are zero: none of these words appear in the chunks.`;

  out("space").querySelector(".pg-space-meta").innerHTML = result.pos
    ? `Position: <code>(${result.pos.map((x) => x.toFixed(2)).join(", ")})</code>. The white point is your sentence; coloured points are the stored chunks.`
    : `Nothing to place: an all-zero vector has no direction.`;

  const maxS = result.nearest[0]?.[1] || 1;
  out("nearest").innerHTML = result.nearest.length
    ? `<ol class="pg-results">${result.nearest.map(([cid, s], k) => {
        const c = DATA.chunks[cid];
        const lesson = DATA.lessons[c.lesson];
        return `<li style="--i:${k}"><div class="pg-res-head"><i class="pg-dot" style="background:${lesson.color}"></i><span>${escapeHtml(c.filename)}</span> · chunk ${c.index + 1}` +
          `<code>${s.toFixed(3)}</code></div><div class="pg-bar"><span style="width:${(100 * s / maxS).toFixed(1)}%;background:${lesson.color}"></span></div>` +
          `<p>${escapeHtml(c.text.slice(0, 180))}…</p></li>`;
      }).join("")}</ol>`
    : `<p class="pg-empty">No chunk shares a word with this sentence, so every score is 0. Keyword representations can only match shared words. A real embedding model compares <em>meaning</em>: it would still match a reworded question about RAG, and would place an off-topic question like this far from every chunk.</p>`;
}

let runId = 0;
function run(text) {
  text = text.trim().slice(0, 200) || "How does chunk overlap help retrieval?";
  const id = ++runId;
  const result = analyse(text);
  render(text, result);
  steps.forEach((s) => s.classList.remove("shown"));
  if (REDUCED) {
    steps.forEach((s) => s.classList.add("shown"));
    showInSpace(result);
    return;
  }
  steps.forEach((s, i) => setTimeout(() => {
    if (id !== runId) return;
    s.classList.add("shown");
    if (s.dataset.stage === "space") showInSpace(result);
  }, i * 650));
}

// ------------------------------------------------------------------ wiring

buildGrid();
setupView();
const input = $(".pg-text");
$(".pg-input").addEventListener("submit", (e) => { e.preventDefault(); run(input.value); });
root.querySelectorAll("[data-example]").forEach((b) => b.addEventListener("click", () => {
  input.value = b.dataset.example;
  run(input.value);
}));
run(input.value);
