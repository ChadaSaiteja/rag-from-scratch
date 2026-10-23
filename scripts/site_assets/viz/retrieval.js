// Retrieval: one of the pipeline's evaluation questions becomes a point, vector
// search and BM25 each pick their top 3, Reciprocal Rank Fusion merges them, and
// the winners are handed to the prompt with their source filenames.
import { boot, createStage, createPlayer, makeTextPlane, makeLabel, makeLine, makeGlowPoint, setPointOpacity, THREE } from "./core.js";
import { DATA, buildPoints, escapeHtml } from "./space-common.js";

const root = document.querySelector(".viz[data-scene=retrieval]");
const CAM_HOME = [6, 6, 28];
const VEC = 0x67e8f9;
const KEY = 0xfbbf24;

boot(root, (stageEl) => {
  const stage = createStage(stageEl, { cameraPos: CAM_HOME });
  const hud = stageEl.querySelector(".viz-hud");
  const picker = root.querySelector(".viz-picker");
  const base = new THREE.Group();
  stage.scene.add(base);
  const chunks = DATA.chunks;
  const points = buildPoints(stage, base);
  let world = null;

  const lineOpacity = (l, o) => { l.material.opacity = o; l.visible = o > 0.001; };
  const short = (c) => `${c.filename.replace(".txt", "")} #${c.index + 1}`;

  function build(q) {
    if (world) {
      stage.scene.remove(world.group);
      world.group.traverse((o) => { o.geometry?.dispose(); o.material?.map?.dispose(); o.material?.dispose(); });
    }
    const group = new THREE.Group();
    stage.scene.add(group);
    const qPos = new THREE.Vector3(...q.pos);

    // The question as a card, then as a point
    const card = makeTextPlane(q.question, { width: 6, cols: 34, canvasWidth: 768, border: "#f8fafc", color: "#f8fafc" });
    card.home = new THREE.Vector3(0, 0, 10);
    group.add(card.mesh);
    const qPoint = makeGlowPoint(0xffffff, 1.7);
    group.add(qPoint);
    const pulse = new THREE.Mesh(
      new THREE.RingGeometry(0.9, 1.0, 64),
      new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false }),
    );
    pulse.position.copy(qPos);
    group.add(pulse);

    // Vector leg: lines + rank labels to the top-3 by cosine
    const vec = q.vector.map(([cid, score], r) => {
      const to = new THREE.Vector3(...chunks[cid].pos);
      const line = makeLine(qPos, to, VEC, 0.95);
      const label = makeLabel(`vector #${r + 1} · ${score.toFixed(2)}`, { color: "#a5f3fc", size: 0.42 });
      label.position.copy(to).add(new THREE.Vector3(0, 0.75, 0));
      group.add(line, label);
      return { cid, line, label };
    });

    // Keyword leg: a panel of the question's terms, beams to the BM25 top-3
    const termsLabel = makeLabel(`BM25 terms: ${q.terms.join(" · ")}`, { color: "#fde68a", size: 0.5, bg: "rgba(41,30,4,0.85)" });
    const termsPos = qPos.clone().add(new THREE.Vector3(0, 3.2, 0));
    termsLabel.position.copy(termsPos);
    group.add(termsLabel);
    const bm = q.bm25.map(([cid, score], r) => {
      const to = new THREE.Vector3(...chunks[cid].pos);
      const line = makeLine(termsPos, to, KEY, 0.95);
      const label = makeLabel(`BM25 #${r + 1} · ${score.toFixed(1)}`, { color: "#fde68a", size: 0.42 });
      label.position.copy(to).add(new THREE.Vector3(0, -0.75, 0));
      group.add(line, label);
      return { cid, line, label };
    });

    // Winners: cards that fly to the front as the prompt's context
    const fused = q.fused.slice(0, 4);
    const winners = fused.map(([cid], k) => {
      const c = chunks[cid];
      const lesson = DATA.lessons[c.lesson];
      const w = makeTextPlane(`[Source: ${c.filename}]\n${c.text.slice(0, 170)}…`, {
        width: 5, cols: 34, canvasWidth: 768, border: lesson.color,
      });
      w.from = new THREE.Vector3(...c.pos);
      w.to = new THREE.Vector3((k - (fused.length - 1) / 2) * 5.6, -1, 9);
      group.add(w.mesh);
      return w;
    });

    world = { group, q, qPos, card, qPoint, pulse, vec, bm, termsLabel, winners, fused };
  }

  function reset() {
    const w = world;
    stage.camera.position.set(...CAM_HOME);
    stage.controls.target.set(0, 0, 0);
    stage.controls.autoRotate = false;
    points.forEach((p) => { setPointOpacity(p, 0.9); p.scale.setScalar(1); });
    w.card.mesh.position.copy(w.card.home);
    w.card.mesh.scale.setScalar(1);
    w.card.mesh.material.opacity = 0;
    w.card.mesh.visible = false;
    setPointOpacity(w.qPoint, 0);
    w.qPoint.position.copy(w.card.home);
    w.pulse.material.opacity = 0;
    [...w.vec, ...w.bm].forEach((x) => { lineOpacity(x.line, 0); x.label.material.opacity = 0; });
    w.termsLabel.material.opacity = 0;
    w.winners.forEach((m) => { m.mesh.position.copy(m.from); m.mesh.scale.setScalar(0.05); m.mesh.material.opacity = 0; m.mesh.visible = false; });
    hud.innerHTML = `<div class="hud-title">${chunks.length} stored chunks</div><div class="hud-note">Pick a question above, or hover a point to read it.</div>`;
  }

  function rankTable() {
    const q = world.q;
    const rankOf = (list, cid) => { const i = list.findIndex(([c]) => c === cid); return i < 0 ? "–" : i + 1; };
    const rows = q.fused.map(([cid, s]) => {
      const c = chunks[cid];
      const hit = c.filename === q.expected;
      return `<tr${hit ? ' class="hit"' : ""}><td>${short(c)}</td><td>${rankOf(q.bm25, cid)}</td><td>${rankOf(q.vector, cid)}</td><td>${s.toFixed(4)}</td></tr>`;
    }).join("");
    return `<div class="hud-title">Reciprocal Rank Fusion</div><table class="hud-table"><tr><th>chunk</th><th>BM25</th><th>vec</th><th>score</th></tr>${rows}</table>` +
      `<div class="hud-note">score = 0.5/(60+rank<sub>BM25</sub>) + 0.5/(60+rank<sub>vec</sub>)</div>`;
  }

  const steps = [
    // 1. The stored space
    async (ctx) => {
      await ctx.camera([6, 6, 28], [0, 0, 0], 1000);
    },
    // 2. The question becomes a point
    async (ctx) => {
      ctx.set(() => {
        hud.innerHTML = `<div class="hud-title">The question</div><div class="hud-note">${escapeHtml(world.q.question)}</div>` +
          `<div class="hud-note">Embedded with the same model as the chunks, so it lands in the same space.</div>`;
        world.card.mesh.visible = true;
      });
      await ctx.anim(700, (e) => { world.card.mesh.material.opacity = e; });
      await ctx.wait(900);
      await Promise.all([
        ctx.anim(1500, (e) => {
          world.card.mesh.scale.setScalar(1 - 0.95 * e);
          world.card.mesh.position.lerpVectors(world.card.home, world.qPos, e);
          world.card.mesh.material.opacity = 1 - e;
          setPointOpacity(world.qPoint, e);
          world.qPoint.position.lerpVectors(world.card.home, world.qPos, e);
        }),
        ctx.camera([world.qPos.x + 12, world.qPos.y + 9, world.qPos.z + 22], [world.qPos.x * 0.5, world.qPos.y * 0.5 + 1, world.qPos.z * 0.5], 1500),
      ]);
      ctx.set(() => { world.card.mesh.visible = false; });
    },
    // 3. Vector search
    async (ctx) => {
      ctx.set(() => {
        hud.innerHTML = `<div class="hud-title">Vector search · top ${DATA.settings.k}</div>` +
          world.q.vector.map(([cid, s], r) => `<div class="hud-row"><span>#${r + 1} ${short(chunks[cid])}</span><b>${s.toFixed(2)}</b></div>`).join("") +
          `<div class="hud-note">Ranked by cosine similarity to the question.</div>`;
      });
      for (const [r, v] of world.vec.entries()) {
        await ctx.anim(650, (e) => { lineOpacity(v.line, 0.95 * e); v.label.material.opacity = e; points[v.cid].scale.setScalar(1 + 0.6 * e); });
        if (r < world.vec.length - 1) await ctx.wait(150);
      }
    },
    // 4. Keyword search (BM25)
    async (ctx) => {
      ctx.set(() => {
        hud.innerHTML = `<div class="hud-title">BM25 keyword search · top ${DATA.settings.k}</div>` +
          (world.q.bm25.length ? world.q.bm25.map(([cid, s], r) => `<div class="hud-row"><span>#${r + 1} ${short(chunks[cid])}</span><b>${s.toFixed(1)}</b></div>`).join("") : `<div class="hud-note">No chunk shares a keyword.</div>`) +
          `<div class="hud-note">Scores exact shared words, weighting rare words more.</div>`;
      });
      await ctx.anim(600, (e) => { world.termsLabel.material.opacity = e; });
      for (const b of world.bm) {
        await ctx.anim(600, (e) => { lineOpacity(b.line, 0.95 * e); b.label.material.opacity = e; points[b.cid].scale.setScalar(Math.max(points[b.cid].scale.x, 1 + 0.6 * e)); });
      }
    },
    // 5. Fusion
    async (ctx) => {
      ctx.set(() => { hud.innerHTML = rankTable(); });
      const fusedIds = new Set(world.q.fused.map(([c]) => c));
      const maxS = world.q.fused[0][1];
      await ctx.anim(1300, (e) => {
        points.forEach((p, i) => {
          const s = world.q.fused.find(([c]) => c === i)?.[1];
          setPointOpacity(p, fusedIds.has(i) ? 1 : 1 - 0.8 * e);
          if (s) p.scale.setScalar(1 + (0.4 + 1.2 * (s / maxS)) * e);
        });
        [...world.vec, ...world.bm].forEach((x) => { lineOpacity(x.line, 0.95 - 0.6 * e); x.label.material.opacity = 1 - e; });
        world.termsLabel.material.opacity = 1 - 0.6 * e;
      });
    },
    // 6. Hand-off to the prompt
    async (ctx) => {
      const q = world.q;
      ctx.set(() => {
        hud.innerHTML = rankTable() + `<div class="hud-verdict ${q.hit ? "ok" : "miss"}">Expected source: ${q.expected}<br>${q.hit ? "✓ retrieved" : "✗ missed"}</div>`;
        world.winners.forEach((m) => { m.mesh.visible = true; });
      });
      await Promise.all([
        ctx.camera([0, 1, 26], [0, -1, 6], 1600),
        ctx.anim(1700, (e) => world.winners.forEach((m, k) => {
          const local = Math.min(1, Math.max(0, e * 1.3 - k * 0.1));
          m.mesh.position.lerpVectors(m.from, m.to, local);
          m.mesh.scale.setScalar(0.05 + 0.95 * local);
          m.mesh.material.opacity = local;
        })),
      ]);
    },
  ];

  stage.onFrame((t) => {
    if (!world) return;
    world.pulse.lookAt(stage.camera.position);
    if (world.qPoint.visible) {
      const s = (t * 0.8) % 1;
      world.pulse.scale.setScalar(1 + s * 3);
      world.pulse.material.opacity = 0.6 * (1 - s) * (world.qPoint.userData.core.material.opacity);
    }
  });

  let player = null;
  function start(index) {
    build(DATA.queries[Number(index)]);
    if (!player) player = createPlayer(root, stage, { steps, reset });
    player.restart();
  }
  picker?.addEventListener("change", () => start(picker.value));
  start(picker ? picker.value : 0);
});
