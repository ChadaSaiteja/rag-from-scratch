// Embedding & storage: real chunks turn into vectors, the vectors become points in
// 3D, neighbours link up into a search graph, and everything is stored with metadata.
// Positions are TF-IDF vectors reduced to 3D (an offline stand-in for embeddings).
import { boot, createStage, createPlayer, makeTextPlane, makeLabel, makeLine, makeGlowPoint, setPointOpacity, THREE } from "./core.js";
import { DATA, buildPoints, buildLessonLabels, legendHtml, chunkTip } from "./space-common.js";

const root = document.querySelector(".viz[data-scene=space]");
const CAM_HOME = [0, 3, 30];

boot(root, (stageEl) => {
  const stage = createStage(stageEl, { cameraPos: CAM_HOME });
  const hud = stageEl.querySelector(".viz-hud");
  const group = new THREE.Group();
  stage.scene.add(group);
  const chunks = DATA.chunks;

  // --- Cards laid out in a grid (the text before embedding)
  const COLS = Math.ceil(Math.sqrt(chunks.length * 1.6));
  const cardW = 3.2;
  const cards = chunks.map((c, i) => {
    const lesson = DATA.lessons[c.lesson];
    const card = makeTextPlane(c.text.slice(0, 150) + "…", {
      width: cardW, cols: 28, canvasWidth: 512, border: lesson.color,
    });
    const row = Math.floor(i / COLS);
    const col = i % COLS;
    const rows = Math.ceil(chunks.length / COLS);
    card.home = new THREE.Vector3((col - (COLS - 1) / 2) * (cardW + 0.6), ((rows - 1) / 2 - row) * 3.3, 0);
    card.mesh.position.copy(card.home);
    group.add(card.mesh);
    stage.addPickable(card.mesh, () => chunkTip(c));

    // The vector: bars for this chunk's strongest TF-IDF dimensions
    const bars = new THREE.Group();
    const maxW = c.topTerms[0]?.[1] || 1;
    c.topTerms.forEach((t, k) => {
      const h = 0.25 + 2.2 * (t[1] / maxW);
      const bar = new THREE.Mesh(
        new THREE.BoxGeometry(0.26, h, 0.26),
        new THREE.MeshBasicMaterial({ color: lesson.color, transparent: true, opacity: 0 }),
      );
      bar.position.set((k - (c.topTerms.length - 1) / 2) * 0.36, h / 2 - 1, 0);
      bars.add(bar);
    });
    bars.position.copy(card.home);
    group.add(bars);
    return { card, bars };
  });

  // --- Points, lesson labels
  const points = buildPoints(stage, group);
  const labels = buildLessonLabels(group);

  // --- Neighbour graph (k=3 by cosine, undirected)
  const edgeKey = new Set();
  const edges = [];
  const adj = chunks.map(() => new Set());
  chunks.forEach((c) => c.neighbors.forEach(([j]) => {
    adj[c.id].add(j);
    adj[j].add(c.id);
    const key = [Math.min(c.id, j), Math.max(c.id, j)].join("-");
    if (edgeKey.has(key)) return;
    edgeKey.add(key);
    const line = makeLine(new THREE.Vector3(...c.pos), new THREE.Vector3(...chunks[j].pos), 0x7dd3fc, 0.35);
    group.add(line);
    edges.push({ line, a: c.id, b: j });
  }));

  // Focus chunk for "nearest neighbours": the first chunk of the vector-database lesson
  const focusId = DATA.lessons.find((l) => l.filename.startsWith("05_"))?.chunkIds[0] ?? 0;
  const focus = chunks[focusId];
  const focusLines = focus.neighbors.map(([j, sim]) => {
    const line = makeLine(new THREE.Vector3(...focus.pos), new THREE.Vector3(...chunks[j].pos), 0xffffff, 0.9);
    group.add(line);
    const label = makeLabel(`cos ${sim.toFixed(2)}`, { color: "#f8fafc", size: 0.42 });
    const mid = new THREE.Vector3(...focus.pos).add(new THREE.Vector3(...chunks[j].pos)).multiplyScalar(0.5);
    label.position.copy(mid).add(new THREE.Vector3(0, 0.4, 0));
    group.add(label);
    return { line, label };
  });
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.55, 0.62, 48),
    new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false }),
  );
  ring.position.set(...focus.pos);
  group.add(ring);

  // Greedy hop path through the graph: start far away, always step to the neighbour
  // closest to the target (how graph indexes like HNSW avoid comparing against everything).
  const target = focus;
  const dist = (a, b) => Math.hypot(a.pos[0] - b.pos[0], a.pos[1] - b.pos[1], a.pos[2] - b.pos[2]);
  const entry = chunks.reduce((far, c) => (dist(c, target) > dist(far, target) ? c : far), chunks[0]);
  const path = [entry.id];
  for (let guard = 0; guard < 20; guard++) {
    const here = chunks[path[path.length - 1]];
    if (here.id === target.id) break;
    let best = here;
    adj[here.id].forEach((j) => { if (dist(chunks[j], target) < dist(best, target)) best = chunks[j]; });
    if (best.id === here.id) break;
    path.push(best.id);
  }
  const traveller = makeGlowPoint(0xffffff, 1.6);
  group.add(traveller);
  const pathLines = path.slice(1).map((id, k) => {
    const line = makeLine(new THREE.Vector3(...chunks[path[k]].pos), new THREE.Vector3(...chunks[id].pos), 0xfde047, 1);
    group.add(line);
    return line;
  });

  // --- The stored collection boundary
  const shell = new THREE.Mesh(
    new THREE.IcosahedronGeometry(12.5, 2),
    new THREE.MeshBasicMaterial({ color: 0x38bdf8, wireframe: true, transparent: true, opacity: 0, depthWrite: false }),
  );
  group.add(shell);
  const shellLabel = makeLabel("Chroma collection → ./chroma_gemini_db", { color: "#7dd3fc", size: 0.7 });
  shellLabel.position.set(0, 13.6, 0);
  group.add(shellLabel);

  const lineOpacity = (l, o) => { l.material.opacity = o; l.visible = o > 0.001; };

  function reset() {
    stage.camera.position.set(...CAM_HOME);
    stage.controls.target.set(0, 0, 0);
    stage.controls.autoRotate = false;
    cards.forEach(({ card, bars }) => {
      card.mesh.position.copy(card.home);
      card.mesh.scale.setScalar(1);
      card.mesh.material.opacity = 0;
      card.mesh.visible = false;
      bars.position.copy(card.home);
      bars.scale.set(1, 0.001, 1);
      bars.children.forEach((b) => { b.material.opacity = 0; });
      bars.visible = false;
    });
    points.forEach((p, i) => {
      p.position.copy(cards[i].card.home);
      p.scale.setScalar(1);
      setPointOpacity(p, 0);
    });
    labels.forEach((l) => { l.material.opacity = 0; });
    edges.forEach((e) => lineOpacity(e.line, 0));
    focusLines.forEach((f) => { lineOpacity(f.line, 0); f.label.material.opacity = 0; });
    ring.material.opacity = 0;
    pathLines.forEach((l) => lineOpacity(l, 0));
    setPointOpacity(traveller, 0);
    shell.material.opacity = 0;
    shellLabel.material.opacity = 0;
    hud.innerHTML = `<div class="hud-title">${chunks.length} chunks from ${DATA.lessons.length} lessons</div>` +
      `<div class="hud-note">Hover any card or point to read it.</div>`;
  }

  const sample = chunks[focusId];
  const steps = [
    // 1. The chunks, still text
    async (ctx) => {
      await ctx.anim(1000, (e) => cards.forEach(({ card }, i) => {
        const local = Math.min(1, Math.max(0, e * 1.6 - i / chunks.length * 0.6));
        card.mesh.visible = true;
        card.mesh.material.opacity = local;
        card.mesh.scale.setScalar(0.85 + 0.15 * local);
      }));
    },
    // 2. Text becomes numbers
    async (ctx) => {
      ctx.set(() => {
        hud.innerHTML = `<div class="hud-title">One chunk → ${DATA.settings.vocab} numbers</div>` +
          sample.topTerms.slice(0, 6).map((t) => `<div class="hud-row"><span>${t[0]}</span><b>${t[1].toFixed(2)}</b></div>`).join("") +
          `<div class="hud-note">Bars show each chunk's strongest dimensions.<br>Stand-in: TF-IDF keyword weights, not Gemini.</div>`;
      });
      await ctx.anim(1200, (e) => cards.forEach(({ card, bars }) => {
        card.mesh.material.opacity = 1 - 0.75 * e;
        card.mesh.scale.setScalar(1 - 0.35 * e);
        bars.visible = true;
        bars.position.set(card.home.x, card.home.y, 0.6 * e);
        bars.scale.set(1, Math.max(0.001, e), 1);
        bars.children.forEach((b) => { b.material.opacity = 0.95 * e; });
      }));
    },
    // 3. Numbers become a position
    async (ctx) => {
      await ctx.anim(700, (e) => cards.forEach(({ card, bars }, i) => {
        card.mesh.material.opacity = 0.25 * (1 - e);
        bars.scale.set(1 - 0.9 * e, 1 - 0.95 * e, 1);
        setPointOpacity(points[i], e);
      }));
      ctx.set(() => cards.forEach(({ card, bars }) => { card.mesh.visible = false; bars.visible = false; }));
      await Promise.all([
        ctx.anim(2200, (e) => points.forEach((p, i) => {
          p.position.lerpVectors(cards[i].card.home, p.userData.home, e);
        })),
        ctx.camera([16, 9, 28], [0, -0.5, 0], 2200),
      ]);
    },
    // 4. Similar topics land close together
    async (ctx) => {
      ctx.set(() => { hud.innerHTML = legendHtml(); stage.controls.autoRotate = !stage.reduced; });
      await ctx.anim(900, (e) => labels.forEach((l) => { l.material.opacity = e; }));
    },
    // 5. Nearest neighbours of one chunk
    async (ctx) => {
      ctx.set(() => {
        stage.controls.autoRotate = false;
        const lesson = DATA.lessons[focus.lesson];
        hud.innerHTML = `<div class="hud-title">Nearest to ${lesson.filename} · chunk ${focus.index + 1}</div>` +
          focus.neighbors.map(([j, s], k) => `<div class="hud-row"><span>#${k + 1} ${chunks[j].filename.replace(".txt", "")}</span><b>${s.toFixed(2)}</b></div>`).join("") +
          `<div class="hud-note">Cosine similarity, higher = closer in meaning</div>`;
      });
      await Promise.all([
        ctx.camera([focus.pos[0] + 7, focus.pos[1] + 4, focus.pos[2] + 11], focus.pos, 1600),
        ctx.anim(1200, (e) => {
          labels.forEach((l) => { l.material.opacity = 1 - 0.7 * e; });
          points.forEach((p, i) => {
            const near = i === focus.id || focus.neighbors.some(([j]) => j === i);
            setPointOpacity(p, near ? 1 : 1 - 0.75 * e);
          });
          ring.material.opacity = e;
          focusLines.forEach((f) => { lineOpacity(f.line, 0.9 * e); f.label.material.opacity = e; });
        }),
      ]);
    },
    // 6. The index: a graph you can hop through
    async (ctx) => {
      ctx.set(() => {
        hud.innerHTML = `<div class="hud-title">Index: links between neighbours</div>` +
          `<div class="hud-row"><span>Links</span><b>${edges.length}</b></div>` +
          `<div class="hud-row"><span>Hops to target</span><b>${path.length - 1}</b></div>` +
          `<div class="hud-note">Graph indexes like HNSW hop toward the query instead of comparing against every vector. With 21 chunks it barely matters; with 21 million it's everything.</div>`;
      });
      await Promise.all([
        ctx.camera([16, 10, 22], [0, 0, 0], 1500),
        ctx.anim(1200, (e) => {
          ring.material.opacity = 1 - e;
          focusLines.forEach((f) => { lineOpacity(f.line, 0.9 * (1 - e)); f.label.material.opacity = 1 - e; });
          points.forEach((p) => setPointOpacity(p, Math.max(p.userData.core.material.opacity, 0.85 * e)));
          edges.forEach((ed) => lineOpacity(ed.line, 0.35 * e));
        }),
      ]);
      ctx.set(() => { setPointOpacity(traveller, 1); traveller.position.set(...chunks[path[0]].pos); });
      for (let k = 1; k < path.length; k++) {
        const a = new THREE.Vector3(...chunks[path[k - 1]].pos);
        const b = new THREE.Vector3(...chunks[path[k]].pos);
        await ctx.anim(750, (e) => {
          traveller.position.lerpVectors(a, b, e);
          lineOpacity(pathLines[k - 1], e);
        });
      }
      ctx.set(() => { ring.material.opacity = 1; });
    },
    // 7. Stored, with metadata, in a collection on disk
    async (ctx) => {
      ctx.set(() => {
        hud.innerHTML = `<div class="hud-title">Stored in Chroma</div>` +
          `<div class="hud-row"><span>Vectors</span><b>${chunks.length}</b></div>` +
          `<div class="hud-row"><span>Metadata</span><b>source, filename</b></div>` +
          `<div class="hud-row"><span>Collection</span><b>gemini_collection</b></div>` +
          `<div class="hud-note">Each point keeps its text and filename, which is what makes citations possible later. Hover a point to see it.</div>`;
        stage.controls.autoRotate = !stage.reduced;
      });
      await Promise.all([
        ctx.camera([0, 10, 36], [0, 0, 0], 1800),
        ctx.anim(1500, (e) => {
          setPointOpacity(traveller, 1 - e);
          ring.material.opacity = 1 - e;
          pathLines.forEach((l) => lineOpacity(l, 1 - e));
          edges.forEach((ed) => lineOpacity(ed.line, 0.35 - 0.2 * e));
          points.forEach((p) => setPointOpacity(p, 1));
          labels.forEach((l) => { l.material.opacity = 0.3 + 0.5 * e; });
          shell.material.opacity = 0.14 * e;
          shellLabel.material.opacity = e;
        }),
      ]);
    },
  ];

  // Gentle breathing so the space feels alive
  stage.onFrame((t) => {
    points.forEach((p, i) => p.userData.halo.scale.setScalar(1.6 + 0.12 * Math.sin(t * 1.6 + i)));
    ring.lookAt(stage.camera.position);
  });

  reset();
  createPlayer(root, stage, { steps, reset }).restart();
});
