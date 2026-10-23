// "How RAG works": one scroll-driven story. A sticky stage on the left keeps the
// same objects alive across steps (documents -> chunks -> vectors -> stored dots ->
// retrieved sources -> prompt -> answer) while the narrative scrolls on the right.
// Pure SVG + HTML, no WebGL: the vector space is a 3D point cloud projected by hand.
const DATA = window.RAG_VIZ;
const root = document.querySelector(".story");
const svg = root.querySelector(".story-svg");
const stage = root.querySelector(".story-stage");
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
const NS = "http://www.w3.org/2000/svg";
const W = 1000, H = 620;
const FOCUS = DATA.lessons.find((l) => l.filename.startsWith("03_")) || DATA.lessons[0];
const QUERY = DATA.queries[0];
const TINT = ["#5eead4", "#93c5fd", "#f9a8d4"];
const AMBER = "#fbbf24";

const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const el = (tag, attrs = {}, parent = svg) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  parent.appendChild(n);
  return n;
};
const lerp = (a, b, t) => a + (b - a) * t;

// ------------------------------------------------------------------ smoothing
// Every animated thing has `cur` (what is drawn) and `to` (where it should be).
// Each frame cur moves toward to with exponential smoothing, so steps can be
// changed at any moment without bookkeeping.
const animated = [];
function track(obj, keys) {
  obj.cur = {}; obj.to = {};
  keys.forEach((k) => { obj.cur[k] = 0; obj.to[k] = 0; });
  animated.push(obj);
  return obj;
}
function settle(dt) {
  const k = REDUCED ? 1 : 1 - Math.exp(-dt * 9);
  for (const o of animated) for (const key in o.to) o.cur[key] = lerp(o.cur[key], o.to[key], k);
}

// ------------------------------------------------------------------ build the scene
const gDocs = el("g"), gLinks = el("g"), gPieces = el("g"), gCloud = el("g"), gDots = el("g"), gQuery = el("g"), gLabels = el("g");

// 10 document cards
const DOC_W = 150, DOC_H = 92;
const docs = DATA.lessons.map((l, i) => {
  const g = el("g", { class: "st-doc" }, gDocs);
  const rect = el("rect", { rx: 8, fill: "#f8fafc", stroke: l.color, "stroke-width": 2 }, g);
  const title = el("text", { class: "st-doc-title", x: 10, y: 20 }, g);
  title.textContent = l.filename.replace(".txt", "");
  const lines = Array.from({ length: 26 }, () => el("rect", { rx: 2, fill: "#cbd5e1" }, g));
  const bands = [0, 1, 2].map((k) => el("rect", { fill: TINT[k], opacity: 0 }, g));
  const badge = el("g", {}, g);
  const badgeRect = el("rect", { rx: 5, fill: l.color }, badge);
  const badgeText = el("text", { class: "st-badge" }, badge);
  badgeText.textContent = `filename: ${l.filename}`;
  const size = el("text", { class: "st-size" }, g);
  size.textContent = `${l.chars.toLocaleString()} chars`;
  return track({ l, g, rect, title, lines, bands, badge, badgeRect, badgeText, size, i }, ["x", "y", "w", "h", "o", "badge", "lines"]);
});

// 21 chunk pieces (cards that later show their vector bars)
const PIECE_W = 120, PIECE_H = 64;
const pieces = DATA.chunks.map((c) => {
  const lesson = DATA.lessons[c.lesson];
  const g = el("g", { class: "st-piece" }, gPieces);
  const rect = el("rect", { rx: 6, fill: "#f8fafc", stroke: lesson.color, "stroke-width": 1.5 }, g);
  const overlap = el("rect", { fill: AMBER, opacity: 0.55 }, g);
  const label = el("text", { class: "st-piece-label" }, g);
  label.textContent = `${lesson.filename.replace(".txt", "").slice(0, 2)} · chunk ${c.index + 1}`;
  const lines = [0, 1, 2].map(() => el("rect", { rx: 1.5, fill: "#cbd5e1" }, g));
  const max = c.topTerms[0]?.[1] || 1;
  const bars = c.topTerms.slice(0, 6).map((t) => { const b = el("rect", { rx: 1.5, fill: lesson.color }, g); b.dataset.h = (t[1] / max); return b; });
  return track({ c, lesson, g, rect, overlap, label, lines, bars }, ["x", "y", "w", "h", "o", "bars", "text", "ov"]);
});

// 21 dots: the stored vectors
const dots = DATA.chunks.map((c) => {
  const lesson = DATA.lessons[c.lesson];
  const g = el("g", { class: "st-dot" }, gDots);
  const halo = el("circle", { fill: lesson.color, opacity: 0.25 }, g);
  const core = el("circle", { fill: lesson.color }, g);
  const ring = el("circle", { fill: "none", stroke: AMBER, "stroke-width": 2.5, opacity: 0 }, g);
  return track({ c, lesson, g, halo, core, ring, gx: 0, gy: 0 }, ["x", "y", "r", "o", "cloud", "ring", "lit"]);
});

// cloud frame + 3D axes hint
const cloudFrame = el("g", { class: "st-cloud" }, gCloud);
const cloudRect = el("rect", { rx: 18, fill: "none", stroke: "#334155", "stroke-dasharray": "6 6" }, cloudFrame);
const cloudLabel = el("text", { class: "st-cloud-label" }, cloudFrame);
cloudLabel.textContent = "Chroma collection · 21 vectors";
const cloud = track({ cx: 500, cy: 330, s: 17, yaw: 0, rect: cloudRect, label: cloudLabel }, ["cx", "cy", "s", "o", "spin"]);

// query dot + search lines + labels
const qg = el("g", { class: "st-query" }, gQuery);
const qHalo = el("circle", { fill: "#fff", opacity: 0.25 }, qg);
const qCore = el("circle", { fill: "#fff", stroke: "#0f172a", "stroke-width": 1.5 }, qg);
const qdot = track({ g: qg, halo: qHalo, core: qCore }, ["x", "y", "r", "o", "cloud"]);
const links = [0, 1, 2].map(() => track({ line: el("line", { stroke: "#fff", "stroke-width": 2, "stroke-linecap": "round" }, gLinks) }, ["o"]));
const lineLabels = [0, 1, 2].map(() => {
  const g = el("g", {}, gLabels);
  const bg = el("rect", { rx: 5, fill: "#0f172a", opacity: 0.9 }, g);
  const t = el("text", { class: "st-line-label" }, g);
  return track({ g, bg, t }, ["x", "y", "o"]);
});
const connectors = [0, 1, 2].map(() => track({ line: el("line", { stroke: "#64748b", "stroke-width": 1.5, "stroke-dasharray": "4 4" }, gLinks) }, ["o"]));

// ------------------------------------------------------------------ 3D projection
function project(p, yaw, cx, cy, s) {
  const [x, y, z] = p;
  const X = x * Math.cos(yaw) - z * Math.sin(yaw);
  const Z = x * Math.sin(yaw) + z * Math.cos(yaw);
  const depth = (Z + 10) / 20;              // 0 far .. 1 near
  return { x: cx + X * s, y: cy - y * s * 0.9 + Z * s * 0.18, depth };
}

// ------------------------------------------------------------------ layouts per step
const DOC_GRID = (i) => ({ x: 60 + (i % 5) * 190, y: 150 + Math.floor(i / 5) * 170 });
const PIECE_GRID = (k) => ({ x: 60 + (k % 7) * 132, y: 110 + Math.floor(k / 7) * 150 });
const CLOUD_HOME = { cx: 500, cy: 335, s: 17 };
const CLOUD_RIGHT = { cx: 650, cy: 320, s: 15 };
const Q_START = { x: 335, y: 190 };

function hideAll() {
  docs.forEach((d) => { d.to.o = 0; d.to.badge = 0; d.to.lines = 1; d.bands.forEach((b) => b.setAttribute("opacity", 0)); });
  pieces.forEach((p) => { p.to.o = 0; p.to.bars = 0; p.to.text = 1; p.to.ov = 0; });
  dots.forEach((d) => { d.to.o = 0; d.to.cloud = 0; d.to.ring = 0; d.to.lit = 0; d.to.r = 6; });
  cloud.to.o = 0; cloud.to.spin = 0;
  qdot.to.o = 0; qdot.to.cloud = 0;
  links.forEach((l) => { l.to.o = 0; });
  lineLabels.forEach((l) => { l.to.o = 0; });
  connectors.forEach((l) => { l.to.o = 0; });
}

function layoutDocs(show = true, badges = false) {
  docs.forEach((d, i) => {
    const p = DOC_GRID(i);
    Object.assign(d.to, { x: p.x, y: p.y, w: DOC_W, h: DOC_H, o: show ? 1 : 0, badge: badges ? 1 : 0, lines: 1 });
  });
}

function layoutFocusChunking() {
  const focusPieces = FOCUS.chunkIds.map((id) => pieces[id]);
  const docX = 40, docY = 60, docW = 280, docH = 480;
  docs.forEach((d) => {
    if (d.l.id === FOCUS.id) Object.assign(d.to, { x: docX, y: docY, w: docW, h: docH, o: 1, badge: 1, lines: 1 });
    else Object.assign(d.to, { o: 0 });
  });
  const focusDoc = docs[FOCUS.id];
  const total = focusPieces.reduce((s, p) => s + p.c.chars, 0);
  const scale = (docH - 40 - 18 * (focusPieces.length - 1)) / total;
  let y = docY + 20;
  focusPieces.forEach((p, k) => {
    const h = Math.max(60, p.c.chars * scale);
    Object.assign(p.to, { x: 440, y, w: 330, h, o: 1, bars: 0, text: 1, ov: p.c.overlapPrev ? 1 : 0 });
    p.ovH = (p.c.overlapPrev / p.c.chars) * h;
    // band on the document showing where this piece comes from
    const b = focusDoc.bands[k];
    const top = docY + 30 + (p.c.start / FOCUS.chars) * (docH - 40);
    const bottom = docY + 30 + (p.c.end / FOCUS.chars) * (docH - 40);
    b.setAttribute("x", docX + 6); b.setAttribute("width", docW - 12);
    b.setAttribute("y", top); b.setAttribute("height", Math.max(4, bottom - top));
    b.setAttribute("opacity", 0.35);
    const c = connectors[k];
    c.line.setAttribute("x1", docX + docW); c.line.setAttribute("y1", (top + bottom) / 2);
    c.line.setAttribute("x2", 440); c.line.setAttribute("y2", y + h / 2);
    c.to.o = 1;
    y += h + 18;
  });
  pieces.forEach((p) => { if (!FOCUS.chunkIds.includes(p.c.id)) p.to.o = 0; });
}

function layoutPieces({ bars = 0, text = 1 } = {}) {
  pieces.forEach((p, k) => {
    const g = PIECE_GRID(k);
    Object.assign(p.to, { x: g.x, y: g.y, w: PIECE_W, h: PIECE_H + (bars ? 46 : 0), o: 1, bars, text, ov: 0 });
    dots[k].gx = g.x + PIECE_W / 2; dots[k].gy = g.y + PIECE_H / 2 + (bars ? 46 : 0);
  });
}

function layoutCloud(where, { pieceFade = true } = {}) {
  Object.assign(cloud.to, where, { o: 1, spin: 1 });
  dots.forEach((d) => { d.to.o = 1; d.to.cloud = 1; d.to.r = 6; });
  if (pieceFade) pieces.forEach((p) => { p.to.o = 0; });
}

function markVector(on) {
  QUERY.vector.forEach(([cid, score], k) => {
    dots[cid].to.lit = on ? 1 : 0;
    dots[cid].to.r = on ? 8 : 6;
    links[k].to.o = on ? 0.9 : 0;
    lineLabels[k].to.o = on ? 1 : 0;
    lineLabels[k].t.textContent = `#${k + 1} vector · cos ${score.toFixed(2)}`;
  });
}
function markBM25(on) {
  QUERY.bm25.forEach(([cid]) => { dots[cid].to.ring = on ? 1 : 0; });
}

const STEPS = {
  overview() { hideAll(); },
  documents() { hideAll(); layoutDocs(true, false); },
  load() { hideAll(); layoutDocs(true, true); },
  chunk() { hideAll(); layoutFocusChunking(); },
  embed() { hideAll(); layoutPieces({ bars: 1, text: 0.35 }); },
  store() { hideAll(); layoutPieces({ bars: 1, text: 0 }); layoutCloud(CLOUD_HOME); },
  question() { hideAll(); layoutCloud(CLOUD_RIGHT); Object.assign(qdot.to, { x: Q_START.x, y: Q_START.y, r: 7, o: 0, cloud: 0 }); },
  embedq() { hideAll(); layoutCloud(CLOUD_RIGHT); Object.assign(qdot.to, { x: Q_START.x, y: Q_START.y, r: 8, o: 1, cloud: 1 }); dots.forEach((d) => { d.to.o = 0.55; }); },
  vsearch() { STEPS.embedq(); markVector(true); },
  bm25() { STEPS.embedq(); markVector(false); markBM25(true); dots.forEach((d) => { d.to.o = 0.45; }); QUERY.bm25.forEach(([cid]) => { dots[cid].to.o = 1; dots[cid].to.r = 8; }); },
  fuse() { STEPS.embedq(); dots.forEach((d) => { d.to.o = 0.25; }); QUERY.fused.forEach(([cid]) => { dots[cid].to.o = 1; dots[cid].to.lit = 1; dots[cid].to.r = 8; }); cloud.to.o = 0.5; },
  prompt() { hideAll(); },
  generate() { hideAll(); },
  evaluate() { hideAll(); },
  done() { hideAll(); },
};
const ORDER = Object.keys(STEPS);

// ------------------------------------------------------------------ draw
function draw(now, dt) {
  settle(dt);
  if (!REDUCED) cloud.yaw += dt * 0.25 * cloud.cur.spin;
  const yaw = cloud.yaw;

  docs.forEach((d) => {
    const { x, y, w, h, o, badge, lines } = d.cur;
    d.g.setAttribute("transform", `translate(${x} ${y})`);
    d.g.setAttribute("opacity", o.toFixed(3));
    d.g.style.display = o < 0.01 ? "none" : "";
    d.rect.setAttribute("width", Math.max(0, w)); d.rect.setAttribute("height", Math.max(0, h));
    const big = h > 200;
    const gap = big ? 17 : 12;
    d.lines.forEach((ln, k) => {
      const visible = big ? k < 26 : k < 4;
      const lw = Math.max(0, w - 20 - ((k * 7) % 4) * 16);
      ln.setAttribute("x", 10); ln.setAttribute("width", lw);
      ln.setAttribute("y", 32 + k * gap); ln.setAttribute("height", big ? 6 : 5);
      ln.setAttribute("opacity", visible && 32 + k * gap + 6 < h - 28 ? lines * 0.9 : 0);
    });
    d.title.setAttribute("font-size", big ? 15 : 12);
    d.badge.setAttribute("opacity", badge.toFixed(3));
    d.badgeRect.setAttribute("x", 8); d.badgeRect.setAttribute("y", h - 24); d.badgeRect.setAttribute("width", Math.max(0, w - 16)); d.badgeRect.setAttribute("height", 18);
    d.badgeText.setAttribute("x", 14); d.badgeText.setAttribute("y", h - 11);
    d.badgeText.setAttribute("font-size", big ? 12 : 9.5);
    d.size.setAttribute("x", w - 10); d.size.setAttribute("y", 20);
    d.size.setAttribute("opacity", big ? 1 : 0);
  });

  pieces.forEach((p) => {
    const { x, y, w, h, o, bars, text, ov } = p.cur;
    p.g.setAttribute("transform", `translate(${x} ${y})`);
    p.g.setAttribute("opacity", o.toFixed(3));
    p.g.style.display = o < 0.01 ? "none" : "";
    p.rect.setAttribute("width", Math.max(0, w)); p.rect.setAttribute("height", Math.max(0, h));
    p.label.setAttribute("x", 8); p.label.setAttribute("y", 16);
    p.label.setAttribute("opacity", Math.max(text, 0.6));
    p.lines.forEach((ln, k) => {
      ln.setAttribute("x", 8); ln.setAttribute("y", 26 + k * 10); ln.setAttribute("height", 4);
      ln.setAttribute("width", Math.max(8, (w - 16) * (1 - k * 0.18)));
      ln.setAttribute("opacity", text);
    });
    const ovH = (p.ovH || 0) * ov;
    p.overlap.setAttribute("x", 1); p.overlap.setAttribute("y", 1); p.overlap.setAttribute("width", Math.max(0, w - 2)); p.overlap.setAttribute("height", Math.max(0, ovH));
    p.overlap.setAttribute("opacity", 0.5 * ov);
    const barArea = 40;
    p.bars.forEach((b, k) => {
      const bh = barArea * Number(b.dataset.h) * bars;
      const bw = Math.max(0, (w - 16) / p.bars.length - 4);
      b.setAttribute("x", 8 + k * (bw + 4)); b.setAttribute("width", bw);
      b.setAttribute("y", h - 8 - bh); b.setAttribute("height", Math.max(0, bh));
      b.setAttribute("opacity", bars);
    });
  });

  // cloud frame
  const cs = cloud.cur;
  cloudFrame.setAttribute("opacity", cs.o.toFixed(3));
  cloudFrame.style.display = cs.o < 0.01 ? "none" : "";
  const half = cs.s * 11;
  cloudRect.setAttribute("x", cs.cx - half); cloudRect.setAttribute("y", cs.cy - half * 0.95);
  cloudRect.setAttribute("width", Math.max(0, half * 2)); cloudRect.setAttribute("height", Math.max(0, half * 1.9));
  cloudLabel.setAttribute("x", cs.cx); cloudLabel.setAttribute("y", cs.cy - half * 0.95 - 10);

  // dots: blend between their grid position and the projected 3D position
  const order = [];
  dots.forEach((d) => {
    const pr = project(d.c.pos, yaw, cs.cx, cs.cy, cs.s);
    const m = d.cur.cloud;
    const x = lerp(d.gx, pr.x, m), y = lerp(d.gy, pr.y, m);
    d.px = x; d.py = y;
    const r = Math.max(0, d.cur.r * lerp(1, 0.6 + 0.6 * pr.depth, m));
    d.g.setAttribute("transform", `translate(${x} ${y})`);
    d.g.setAttribute("opacity", d.cur.o.toFixed(3));
    d.g.style.display = d.cur.o < 0.01 ? "none" : "";
    d.core.setAttribute("r", r);
    d.halo.setAttribute("r", r * (2 + d.cur.lit));
    d.halo.setAttribute("opacity", 0.2 + 0.35 * d.cur.lit);
    d.ring.setAttribute("r", r + 5); d.ring.setAttribute("opacity", d.cur.ring);
    order.push([pr.depth, d.g]);
  });
  // draw far dots first
  order.sort((a, b) => a[0] - b[0]).forEach(([, g]) => gDots.appendChild(g));

  const qp = project(QUERY.pos, yaw, cs.cx, cs.cy, cs.s);
  const qx = lerp(qdot.cur.x, qp.x, qdot.cur.cloud), qy = lerp(qdot.cur.y, qp.y, qdot.cur.cloud);
  qg.setAttribute("transform", `translate(${qx} ${qy})`);
  qg.setAttribute("opacity", qdot.cur.o.toFixed(3));
  qg.style.display = qdot.cur.o < 0.01 ? "none" : "";
  qCore.setAttribute("r", Math.max(0, qdot.cur.r)); qHalo.setAttribute("r", Math.max(0, qdot.cur.r * 2.6 + 2 * Math.sin(now / 300)));

  QUERY.vector.forEach(([cid], k) => {
    const d = dots[cid], l = links[k];
    l.line.setAttribute("x1", qx); l.line.setAttribute("y1", qy);
    l.line.setAttribute("x2", d.px); l.line.setAttribute("y2", d.py);
    l.line.setAttribute("opacity", l.cur.o.toFixed(3));
    const lab = lineLabels[k];
    const lx = (qx + d.px) / 2, ly = (qy + d.py) / 2 - 12;
    lab.g.setAttribute("transform", `translate(${lx} ${ly})`);
    lab.g.setAttribute("opacity", lab.cur.o.toFixed(3));
    const tw = lab.t.getComputedTextLength ? lab.t.getComputedTextLength() : 110;
    lab.bg.setAttribute("x", -tw / 2 - 6); lab.bg.setAttribute("y", -11); lab.bg.setAttribute("width", Math.max(0, tw + 12)); lab.bg.setAttribute("height", 20);
  });
  connectors.forEach((c) => c.line.setAttribute("opacity", c.cur.o.toFixed(3)));
}

// ------------------------------------------------------------------ overlays + narrative
const overlays = [...root.querySelectorAll(".story-overlay [data-panel]")];
const stepEls = [...root.querySelectorAll(".story-step")];
const railItems = [...root.querySelectorAll(".story-rail button")];
const counter = root.querySelector(".story-counter");
let current = -1;

function typeInto(elm, text, speed = 14) {
  elm.textContent = "";
  if (REDUCED) { elm.innerHTML = text; return; }
  let i = 0;
  const plain = text;
  clearInterval(elm._typer);
  elm._typer = setInterval(() => {
    i += 2;
    elm.innerHTML = plain.slice(0, i);
    if (i >= plain.length) { clearInterval(elm._typer); elm.innerHTML = plain; }
  }, speed);
}

function setStep(index) {
  index = Math.max(0, Math.min(ORDER.length - 1, index));
  if (index === current) return;
  current = index;
  const key = ORDER[index];
  root.dataset.current = key;
  STEPS[key]();
  overlays.forEach((o) => o.classList.toggle("on", o.dataset.panel.split(" ").includes(key)));
  stepEls.forEach((s, i) => s.classList.toggle("active", i === index));
  railItems.forEach((b, i) => { b.classList.toggle("active", i === index); b.classList.toggle("done", i < index); });
  if (counter) counter.textContent = `${index + 1} / ${ORDER.length}`;
  if (key === "generate") {
    const ans = root.querySelector("[data-answer]");
    typeInto(ans, ans.dataset.answer);
  }
}

// Scroll-driven: the step whose panel crosses the middle of the viewport is active.
// Scroll-driven: the step whose card crosses the activation line is active.
// Geometry on scroll (not IntersectionObserver) so it works the same in every
// browser and when the page is restored at a scroll position. On phones the
// stage is pinned to the top ~46% of the screen, so the line sits below it.
const NARROW = matchMedia("(max-width: 960px)").matches;
function pickActive() {
  const line = innerHeight * (NARROW ? 0.64 : 0.5);
  let best = 0, bestDist = Infinity;
  stepEls.forEach((s, i) => {
    const r = s.getBoundingClientRect();
    const dist = r.top <= line && r.bottom >= line ? 0 : Math.min(Math.abs(r.top - line), Math.abs(r.bottom - line));
    if (dist < bestDist) { bestDist = dist; best = i; }
  });
  setStep(best);
}
// Time-throttled (not requestAnimationFrame) so it keeps working when the tab is
// in the background, where animation frames are paused.
let lastPick = 0, pickTimer = null;
function onScroll() {
  const now = performance.now();
  clearTimeout(pickTimer);
  if (now - lastPick > 60) { lastPick = now; pickActive(); }
  else pickTimer = setTimeout(() => { lastPick = performance.now(); pickActive(); }, 70);
}
addEventListener("scroll", onScroll, { passive: true });
addEventListener("resize", onScroll);
// Belt and braces: some embedded browsers deliver no scroll events to a hidden
// tab but still run observers and timers, so those trigger the same geometry check.
const stepObserver = new IntersectionObserver(onScroll, { threshold: [0, 0.5, 1] });
stepEls.forEach((s) => stepObserver.observe(s));
let lastScrollY = -1;
setInterval(() => { if (scrollY !== lastScrollY) { lastScrollY = scrollY; pickActive(); } }, 250);

function scrollToStep(i) {
  i = Math.max(0, Math.min(ORDER.length - 1, i));
  // No explicit behavior: the page's CSS scroll-behavior decides (smooth, unless reduced motion).
  stepEls[i].scrollIntoView({ block: NARROW ? "start" : "center" });
}
railItems.forEach((b, i) => b.addEventListener("click", () => scrollToStep(i)));
root.querySelector("[data-nav=prev]").addEventListener("click", () => scrollToStep(current - 1));
root.querySelector("[data-nav=next]").addEventListener("click", () => scrollToStep(current + 1));
document.addEventListener("keydown", (e) => {
  if (e.target instanceof Element && e.target.closest("input, select, textarea")) return;
  if (e.key === "ArrowRight" || e.key === "ArrowDown" && e.altKey) { e.preventDefault(); scrollToStep(current + 1); }
  if (e.key === "ArrowLeft" || e.key === "ArrowUp" && e.altKey) { e.preventDefault(); scrollToStep(current - 1); }
});

// Autoplay: advance every few seconds until the end.
const playBtn = root.querySelector("[data-nav=play]");
let timer = null;
function setPlaying(on) {
  clearInterval(timer);
  timer = on ? setInterval(() => { if (current >= ORDER.length - 1) return setPlaying(false); scrollToStep(current + 1); }, 5200) : null;
  playBtn.textContent = on ? "❚❚ Pause" : "▶ Play";
  playBtn.setAttribute("aria-pressed", String(on));
}
playBtn.addEventListener("click", () => setPlaying(!timer));
window.addEventListener("wheel", () => { if (timer) setPlaying(false); }, { passive: true });
window.addEventListener("touchstart", () => { if (timer) setPlaying(false); }, { passive: true });

// ------------------------------------------------------------------ run
hideAll();
animated.forEach((o) => { for (const k in o.to) o.cur[k] = o.to[k]; });
docs.forEach((d, i) => { const p = DOC_GRID(i); Object.assign(d.cur, { x: p.x, y: p.y, w: DOC_W, h: DOC_H }); });
pieces.forEach((p, k) => { const g = PIECE_GRID(k); Object.assign(p.cur, { x: g.x, y: g.y, w: PIECE_W, h: PIECE_H }); });
dots.forEach((d, k) => { const g = PIECE_GRID(k); d.gx = g.x + PIECE_W / 2; d.gy = g.y + PIECE_H / 2; d.cur.r = 6; });
Object.assign(cloud.cur, CLOUD_HOME);
Object.assign(qdot.cur, Q_START, { r: 7 });
setStep(0);
pickActive();

let last = performance.now();
let visible = true;
new IntersectionObserver((e) => { visible = e[0].isIntersecting; }).observe(stage);
(function loop(now) {
  requestAnimationFrame(loop);
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  if (visible) draw(now, dt);
})(last);
