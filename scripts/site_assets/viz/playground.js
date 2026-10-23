// RAG Pipeline Playground: the whole pipeline on one page, with no model involved.
//   Part A (indexing, once):   load -> clean -> chunk -> vectorize -> store, for a chosen lesson
//   Part B (answering, per question): sentence -> tokens -> IDs -> vector -> space ->
//   vector search -> BM25 -> fusion -> prompt -> grounding preview -> checks
// All numbers come from this repo's corpus; the math runs in the browser and matches
// scripts/viz_data.py (TF-IDF + PCA, BM25, weighted Reciprocal Rank Fusion).
import { createStage, hasWebGL, makeGlowPoint, makeLine, setPointOpacity, THREE } from "./core.js";
import { DATA, buildPoints, escapeHtml } from "./space-common.js";

const SP = DATA.space;
const BM = SP.bm25;
const K = DATA.settings.k;
const [W_BM25, W_VEC] = DATA.settings.weights;
const RRF_C = DATA.settings.rrfC;
const RELEVANCE_FLOOR = 0.1;
const root = document.querySelector(".pg");
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
const STOP = new Set(SP.stopwords);
const VOCAB_INDEX = new Map(SP.vocab.map((w, i) => [w, i]));
// mean · component, so projecting a sparse vector only needs its non-zero entries
const MEAN_DOT = SP.comps.map((comp) => comp.reduce((s, c, i) => s + c * SP.mean[i], 0));
const CHUNK_TINTS = ["rgba(94,234,212,0.18)", "rgba(147,197,253,0.2)", "rgba(244,114,182,0.18)", "rgba(167,139,250,0.2)"];

const out = (stage) => root.querySelector(`.pg-step[data-stage=${stage}] .pg-out`);
const short = (c) => `${c.filename.replace(".txt", "")} #${c.index + 1}`;
const dot = (c) => `<i class="pg-dot" style="background:${DATA.lessons[c.lesson].color}"></i>`;

// ================================================================== Part A: indexing

function cleanText(text) {
  const rules = [];
  let t = text.replace(/\r\n?/g, "\n");
  const count = (re, s) => (s.match(re) || []).length;
  const apply = (label, re, rep) => {
    const n = count(re, t);
    t = t.replace(re, rep);
    rules.push([label, n]);
  };
  apply("invisible / control characters removed", /[\u0000-\u0008\u000B\u000C\u000E-\u001F​-‍﻿]/g, "");
  apply("trailing spaces trimmed", /[ \t]+$/gm, "");
  apply("repeated spaces collapsed", / {2,}/g, " ");
  apply("runs of blank lines collapsed", /\n{3,}/g, "\n\n");
  const before = t.length;
  t = t.trim();
  rules.push(["leading / trailing whitespace characters trimmed", before - t.length]);
  return { text: t, rules };
}

function renderIndexing(lessonId) {
  const lesson = DATA.lessons.find((l) => l.id === Number(lessonId));
  const chunks = lesson.chunkIds.map((id) => DATA.chunks[id]);

  // A1 Load
  const lines = lesson.text.split("\n").length;
  out("load").innerHTML =
    `<div class="pg-kv"><span>File</span><b>data/raw/${escapeHtml(lesson.filename)}</b><span>Size</span><b>${lesson.chars.toLocaleString()} characters · ${lines} lines</b></div>` +
    `<pre class="pg-pre">${escapeHtml(lesson.text.slice(0, 420))}${lesson.text.length > 420 ? "\n…" : ""}</pre>`;

  // A2 Clean
  const cleaned = cleanText(lesson.text);
  const removed = lesson.chars - cleaned.text.length;
  out("clean").innerHTML =
    `<ul class="pg-rules">${cleaned.rules.map(([label, n]) => `<li class="${n ? "hit" : ""}"><b>${n}</b> ${label}</li>`).join("")}</ul>` +
    `<p class="pg-meta">${lesson.chars.toLocaleString()} → ${cleaned.text.length.toLocaleString()} characters (${removed} removed). ` +
    `These lessons are already clean text. Real documents (PDFs, web pages) need much more here: headers, footers, page numbers, broken lines.</p>`;

  // A3 Chunk: colour each character by the chunk(s) that contain it
  const marks = new Set([0, lesson.text.length]);
  chunks.forEach((c) => { marks.add(c.start); marks.add(c.end); });
  const cuts = [...marks].sort((a, b) => a - b);
  let html = "";
  for (let i = 0; i < cuts.length - 1; i++) {
    const [a, b] = [cuts[i], cuts[i + 1]];
    const inside = chunks.map((c, k) => (c.start <= a && c.end >= b ? k : -1)).filter((k) => k >= 0);
    const seg = escapeHtml(lesson.text.slice(a, b));
    if (inside.length > 1) html += `<mark class="pg-overlap" title="In chunks ${inside.map((k) => k + 1).join(" and ")}">${seg}</mark>`;
    else if (inside.length === 1) html += `<span style="background:${CHUNK_TINTS[inside[0] % CHUNK_TINTS.length]}" title="Chunk ${inside[0] + 1}">${seg}</span>`;
    else html += `<span class="pg-gap" title="Dropped whitespace between chunks">${seg}</span>`;
  }
  out("chunk").innerHTML =
    `<div class="pg-legend">${chunks.map((c, k) => `<span><i style="background:${CHUNK_TINTS[k % CHUNK_TINTS.length]}"></i>Chunk ${k + 1} · ${c.chars} chars ≈ ${c.tokens} tokens</span>`).join("")}` +
    (chunks.some((c) => c.overlapPrev) ? `<span><i class="pg-overlap-key"></i>overlap (in two chunks)</span>` : "") + `</div>` +
    `<pre class="pg-pre pg-chunked">${html}</pre>` +
    `<p class="pg-meta">RecursiveCharacterTextSplitter(chunk_size=${DATA.settings.chunkSize}, chunk_overlap=${DATA.settings.chunkOverlap}): split at paragraph breaks first, then lines, then spaces.</p>`;

  // A4 Vectorize
  out("vectorize").innerHTML = chunks.map((c, k) => {
    const max = c.topTerms[0]?.[1] || 1;
    return `<div class="pg-vecrow"><div class="pg-vecname">Chunk ${k + 1}<small>${SP.chunkVecs[c.id].length} of ${SP.vocab.length} non-zero</small></div>` +
      `<div class="pg-terms">${c.topTerms.slice(0, 6).map(([w, v]) => `<span><b style="width:${(100 * v / max).toFixed(0)}%"></b><em>${escapeHtml(w)}</em><code>${v.toFixed(2)}</code></span>`).join("")}</div></div>`;
  }).join("") + `<p class="pg-meta">Strongest dimensions of each chunk's vector. The query in Part B is turned into a vector the same way.</p>`;

  // A5 Store
  out("store").innerHTML =
    `<div class="pg-table-wrap"><table class="pg-table"><tr><th>id</th><th>metadata</th><th>text</th><th>vector</th></tr>` +
    chunks.map((c) => `<tr><td><code>${c.id}</code></td><td><code>{"filename": "${escapeHtml(c.filename)}"}</code></td>` +
      `<td>${escapeHtml(c.text.slice(0, 70))}…</td><td>${SP.chunkVecs[c.id].length} non-zero / ${SP.vocab.length}</td></tr>`).join("") +
    `</table></div><p class="pg-meta">${chunks.length} of ${DATA.chunks.length} records in the collection come from this lesson. ` +
    `The real pipeline stores them in Chroma (<code>gemini_collection</code> in <code>./chroma_gemini_db</code>).</p>`;
}

// ================================================================== Part B: answering

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

  // TF-IDF vector (sublinear tf, L2-normalised), as in viz_data.py
  const counts = new Map();
  tokens.filter((t) => t.status === "kept").forEach((t) => counts.set(t.id, (counts.get(t.id) || 0) + 1));
  let entries = [...counts].map(([id, n]) => [id, (1 + Math.log(n)) * SP.idf[id]]);
  const norm = Math.hypot(...entries.map(([, v]) => v));
  entries = norm ? entries.map(([id, v]) => [id, v / norm]) : [];
  entries.sort((a, b) => b[1] - a[1]);
  const pos = entries.length
    ? SP.comps.map((comp, k) => (entries.reduce((s, [id, v]) => s + v * comp[id], 0) - MEAN_DOT[k]) * SP.scale)
    : null;

  // Vector search: cosine against every stored chunk
  const weights = new Map(entries);
  const cosine = SP.chunkVecs.map((vec) => vec.reduce((s, [id, v]) => s + v * (weights.get(id) || 0), 0));
  const vector = cosine.map((s, cid) => [cid, s]).filter(([, s]) => s > 0).sort((a, b) => b[1] - a[1]).slice(0, K);

  // Keyword search: BM25 over the query's known words
  const qIds = [...counts.keys()];
  const bm25Scores = BM.counts.map((doc, cid) => {
    const tf = new Map(doc);
    const len = BM.lengths[cid];
    return qIds.reduce((s, id) => {
      const f = tf.get(id);
      return f ? s + BM.idf[id] * f * (BM.k1 + 1) / (f + BM.k1 * (1 - BM.b + BM.b * len / BM.avgdl)) : s;
    }, 0);
  });
  const bm25 = bm25Scores.map((s, cid) => [cid, s]).filter(([, s]) => s > 0).sort((a, b) => b[1] - a[1]).slice(0, K);
  const matched = (cid) => qIds.filter((id) => BM.counts[cid].some(([i]) => i === id)).map((id) => SP.vocab[id]);

  // Weighted Reciprocal Rank Fusion
  const fusedMap = new Map();
  [[W_BM25, bm25], [W_VEC, vector]].forEach(([w, list]) => list.forEach(([cid], r) => {
    fusedMap.set(cid, (fusedMap.get(cid) || 0) + w / (r + 1 + RRF_C));
  }));
  const fused = [...fusedMap].sort((a, b) => b[1] - a[1]);

  return { tokens, entries, pos, vector, bm25, fused, cosine, matched, qIds };
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
  const nearIds = new Set(result.vector.map(([cid]) => cid));
  points.forEach((p, i) => { setPointOpacity(p, result.pos && !nearIds.has(i) ? 0.35 : 1); p.scale.setScalar(nearIds.has(i) ? 1.5 : 1); });
  if (!result.pos) { setPointOpacity(query, 0); return; }
  const from = new THREE.Vector3(0, 16, 6);
  const to = new THREE.Vector3(...result.pos);
  setPointOpacity(query, 1);
  stage.animate(1300, (e) => query.position.lerpVectors(from, to, e)).then(() => {
    result.vector.forEach(([cid], k) => {
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

function rankedList(list, fmt, extra = () => "") {
  if (!list.length) return `<p class="pg-empty">No chunk scored above zero.</p>`;
  const max = list[0][1] || 1;
  return `<ol class="pg-results">${list.map(([cid, s], k) => {
    const c = DATA.chunks[cid];
    const color = DATA.lessons[c.lesson].color;
    return `<li style="--i:${k}"><div class="pg-res-head">${dot(c)}<span>${escapeHtml(c.filename)}</span> · chunk ${c.index + 1}<code>${fmt(s)}</code></div>` +
      `<div class="pg-bar"><span style="width:${(100 * s / max).toFixed(1)}%;background:${color}"></span></div>${extra(cid)}</li>`;
  }).join("")}</ol>`;
}

function splitSentences(text) {
  return text.split(/(?<=[.?!])\s+|\n+/).map((s) => s.trim()).filter((s) => s.length > 20);
}

function render(text, r) {
  out("sentence").innerHTML = `<p class="pg-sentence">“${escapeHtml(text)}”</p><p class="pg-meta">${text.length} characters</p>`;

  out("tokens").innerHTML = `<div class="pg-chips">${r.tokens.map((t, i) =>
    `<span class="pg-chip ${t.status}" style="--i:${i}" title="${STATUS_TEXT[t.status]}">${escapeHtml(t.piece)}</span>`).join("")}</div>` +
    `<p class="pg-meta">${r.tokens.length} pieces · ${r.tokens.filter((t) => t.status === "kept" || t.status === "oov").length} meaningful words</p>`;

  const words = r.tokens.filter((t) => t.status === "kept" || t.status === "oov");
  out("ids").innerHTML = words.length
    ? `<div class="pg-ids">${words.map((t, i) => t.status === "kept"
        ? `<span class="pg-id" style="--i:${i}"><b>${escapeHtml(t.lower)}</b><i>→</i><code>#${t.id}</code></span>`
        : `<span class="pg-id oov" style="--i:${i}"><b>${escapeHtml(t.lower)}</b><i>→</i><code>unknown</code></span>`).join("")}</div>` +
      `<p class="pg-meta">IDs are positions in the ${SP.vocab.length}-word vocabulary built from the ${DATA.chunks.length} chunks in Part A.</p>`
    : `<p class="pg-empty">No meaningful words to look up. Try a sentence about RAG.</p>`;

  cells.forEach((c) => { c.style.removeProperty("--w"); c.classList.remove("on"); c.style.removeProperty("--i"); });
  r.entries.forEach(([id, v], k) => {
    const c = cells[id];
    c.style.setProperty("--w", v.toFixed(3));
    c.style.setProperty("--i", k);
    c.classList.add("on");
    c.title = `#${id} ${SP.vocab[id]} · weight ${v.toFixed(3)}`;
  });
  out("vector").querySelector(".pg-vector-meta").innerHTML = r.entries.length
    ? `<b>${r.entries.length}</b> of ${SP.vocab.length} numbers are non-zero:` +
      `<span class="pg-weights">${r.entries.map(([id, v]) => `<span>${escapeHtml(SP.vocab[id])} <code>${v.toFixed(2)}</code></span>`).join("")}</span>`
    : `All ${SP.vocab.length} numbers are zero: none of these words appear in the chunks.`;

  out("space").querySelector(".pg-space-meta").innerHTML = r.pos
    ? `Position: <code>(${r.pos.map((x) => x.toFixed(2)).join(", ")})</code>. The white point is your sentence; coloured points are the stored chunks.`
    : `Nothing to place: an all-zero vector has no direction.`;

  out("vsearch").innerHTML = rankedList(r.vector, (s) => `cos ${s.toFixed(3)}`) +
    (r.vector.length ? `<p class="pg-meta">The ${K} chunks pointing in the most similar direction (cosine similarity).</p>` : "");

  out("bm25").innerHTML = rankedList(r.bm25, (s) => s.toFixed(2),
    (cid) => `<p class="pg-match">matched: ${r.matched(cid).map((w) => `<code>${escapeHtml(w)}</code>`).join(" ")}</p>`) +
    (r.bm25.length ? `<p class="pg-meta">BM25 rewards exact shared words, rare words more, with diminishing returns for repeats.</p>` : "");

  const rankOf = (list, cid) => { const i = list.findIndex(([c]) => c === cid); return i < 0 ? "–" : i + 1; };
  out("fusion").innerHTML = r.fused.length
    ? `<div class="pg-table-wrap"><table class="pg-table"><tr><th>chunk</th><th>BM25 rank</th><th>vector rank</th><th>fused score</th></tr>` +
      r.fused.map(([cid, s]) => `<tr><td>${dot(DATA.chunks[cid])}${escapeHtml(short(DATA.chunks[cid]))}</td><td>${rankOf(r.bm25, cid)}</td><td>${rankOf(r.vector, cid)}</td><td><code>${s.toFixed(4)}</code></td></tr>`).join("") +
      `</table></div><p class="pg-meta">score = ${W_BM25}/(${RRF_C} + BM25 rank) + ${W_VEC}/(${RRF_C} + vector rank). Ranks, not raw scores, so the two scales never need comparing. ` +
      `The real pipeline also asks Gemini for 3 rephrasings of the question before vector search.</p>`
    : `<p class="pg-empty">Neither retriever found anything, so there is nothing to fuse.</p>`;

  // Prompt, exactly as generation.py builds it
  const context = r.fused.map(([cid]) => `[Source: ${DATA.chunks[cid].filename}]\n${DATA.chunks[cid].text}`).join("\n\n");
  const prompt = `${DATA.promptHead}Context:\n${context}\n\nQuestion: ${text}`;
  out("prompt").innerHTML =
    `<div class="pg-prompt-bar"><span>${prompt.length.toLocaleString()} characters ≈ ${Math.round(prompt.length / 4).toLocaleString()} tokens · ${r.fused.length} context blocks</span>` +
    `<button type="button" class="pg-copy">Copy prompt</button></div>` +
    `<pre class="pg-pre pg-prompt">${escapeHtml(prompt).replace(/\[Source: ([^\]]+)\]/g, '<mark class="pg-src">[Source: $1]</mark>')}</pre>`;
  out("prompt").querySelector(".pg-copy").addEventListener("click", (e) => {
    navigator.clipboard?.writeText(prompt).then(() => { e.target.textContent = "Copied ✓"; }, () => {});
  });

  // Grounding preview: the context sentences sharing the most (idf-weighted) query words
  // Overlap repeats text across neighbouring chunks, so de-duplicate; and skip
  // questions, since evidence should be statements.
  const qSet = new Set(r.qIds);
  const seen = new Set();
  const scored = [];
  r.fused.forEach(([cid]) => splitSentences(DATA.chunks[cid].text).forEach((s) => {
    const key = s.toLowerCase();
    const words = s.split(/\s+/).length;
    const statement = /[.!]["”']?$/.test(s) ? words >= 4 : words >= 8 && !/:$/.test(s); // skip headings and list labels
    if (seen.has(key) || /\?["”']?$/.test(s) || !statement) return;
    seen.add(key);
    const ids = new Set((s.toLowerCase().match(/[a-z0-9]+/g) || []).map((w) => VOCAB_INDEX.get(w)).filter((id) => qSet.has(id)));
    const score = [...ids].reduce((sum, id) => sum + SP.idf[id], 0);
    if (score > 0) scored.push({ s, score, file: DATA.chunks[cid].filename, ids });
  }));
  scored.sort((a, b) => b.score - a.score);
  const highlight = (s, ids) => escapeHtml(s).replace(/[A-Za-z0-9]+/g, (w) => (ids.has(VOCAB_INDEX.get(w.toLowerCase())) ? `<b>${w}</b>` : w));
  out("ground").innerHTML = scored.length
    ? `<ul class="pg-ground">${scored.slice(0, 3).map((x, k) => `<li style="--i:${k}">${highlight(x.s, x.ids)} <span class="pg-cite">(Source: ${escapeHtml(x.file)})</span></li>`).join("")}</ul>`
    : `<p class="pg-empty">No sentence in the retrieved context shares a word with the question. The prompt tells the model to say it doesn't know, and that's the right answer here.</p>`;

  // Checks
  const topCos = r.fused.length ? Math.max(...r.fused.map(([cid]) => r.cosine[cid])) : 0;
  const passFloor = topCos >= RELEVANCE_FLOOR;
  const evalCase = DATA.queries.find((q) => q.question.toLowerCase() === text.toLowerCase().trim());
  const checks = [
    [passFloor, `Relevance: best context similarity is ${topCos.toFixed(3)} (floor ${RELEVANCE_FLOOR}).`,
      passFloor ? "Enough evidence to answer." : "Too weak: the right response is \"I don't know\". This repo has no such floor yet (README known issue #4), so it would still send these chunks."],
    [r.fused.length > 0, `Citations: ${r.fused.length} context block${r.fused.length === 1 ? "" : "s"}, each tagged with its filename.`,
      r.fused.length ? "Every claim in the answer can point to a source file." : "Nothing to cite."],
  ];
  if (evalCase) {
    const hit = r.fused.some(([cid]) => DATA.chunks[cid].filename === evalCase.expected);
    checks.push([hit, `Evaluation question: expected source <code>${escapeHtml(evalCase.expected)}</code>.`,
      hit ? "Retrieved: the retrieval check in evaluation.py would pass." : "Missed: the retrieval check in evaluation.py would fail."]);
  }
  out("check").innerHTML = `<ul class="pg-checks">${checks.map(([ok, what, why], k) =>
    `<li class="${ok ? "ok" : "bad"}" style="--i:${k}"><span>${ok ? "✓" : "✗"}</span><div><b>${what}</b><br>${why}</div></li>`).join("")}</ul>` +
    (evalCase ? "" : `<p class="pg-meta">Pick one of the evaluation questions under “Try” to see the scored retrieval check from <code>evaluation.py</code>.</p>`);
}

// ------------------------------------------------------------------ sequencing

let runId = 0;
function reveal(stepEls, gap, onShown) {
  const id = ++runId;
  stepEls.forEach((s) => s.classList.remove("shown"));
  if (REDUCED) {
    stepEls.forEach((s) => { s.classList.add("shown"); onShown?.(s); });
    return;
  }
  stepEls.forEach((s, i) => setTimeout(() => {
    if (id !== runId) return;
    s.classList.add("shown");
    onShown?.(s);
  }, i * gap));
}

function runQuery(text) {
  text = text.trim().slice(0, 200) || DATA.queries[0].question;
  const result = analyse(text);
  render(text, result);
  const stepsB = [...root.querySelectorAll(".pg-part-b .pg-step")];
  reveal(stepsB, 450, (s) => { if (s.dataset.stage === "space") showInSpace(result); });
}

function runIndexing(lessonId) {
  renderIndexing(lessonId);
  const stepsA = [...root.querySelectorAll(".pg-part-a .pg-step")];
  stepsA.forEach((s) => s.classList.add("shown"));
}

// ------------------------------------------------------------------ wiring

buildGrid();
setupView();
const lessonPick = root.querySelector(".pg-lesson");
lessonPick.addEventListener("change", () => runIndexing(lessonPick.value));
runIndexing(lessonPick.value);

const input = root.querySelector(".pg-text");
root.querySelector(".pg-input").addEventListener("submit", (e) => { e.preventDefault(); runQuery(input.value); });
root.querySelectorAll("[data-example]").forEach((b) => b.addEventListener("click", () => {
  input.value = b.dataset.example;
  runQuery(input.value);
}));
runQuery(input.value);
