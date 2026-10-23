// Chunking: one real lesson is split by the same RecursiveCharacterTextSplitter
// settings the pipeline uses (1,000 characters, 200 overlap).
import { boot, createStage, createPlayer, makeTextPlane, makeLabel, makeLine, wrapText, THREE } from "./core.js";

const DATA = window.RAG_VIZ;
const root = document.querySelector(".viz[data-scene=chunking]");
const COLS = 58;
const WIDTH = 7;
const CUT = 0x22d3ee;
const BREAK = 0x64748b;
const OVERLAP = "rgba(251,191,36,0.32)";

boot(root, (stageEl) => {
  const stage = createStage(stageEl, { cameraPos: [0, 0, 17] });
  const hud = stageEl.querySelector(".viz-hud");
  const picker = root.querySelector(".viz-picker");
  let world = null;
  let player = null;

  function lessonById(id) { return DATA.lessons.find((l) => l.id === Number(id)); }

  function build(lesson) {
    if (world) {
      stage.scene.remove(world.group);
      world.group.traverse((o) => { o.geometry?.dispose(); o.material?.map?.dispose(); o.material?.dispose(); });
      stage.clearPickables();
    }
    const group = new THREE.Group();
    stage.scene.add(group);
    const chunks = lesson.chunkIds.map((id) => DATA.chunks[id]);

    // Whole document plane
    const doc = makeTextPlane(lesson.text, { width: WIDTH, cols: COLS });
    const docTop = doc.height / 2;
    group.add(doc.mesh);
    const lineY = (lineIdx) => docTop - doc.padTop - (lineIdx + 0.5) * doc.lineHeight;
    const lineOf = (charPos) => {
      let idx = 0;
      doc.lines.forEach((ln, i) => { if (ln.start <= charPos) idx = i; });
      return idx;
    };

    // Paragraph breaks ("\n\n") the splitter tries first
    const breaks = [];
    doc.lines.forEach((ln, i) => {
      if (ln.text === "" && i > 0) {
        const y = lineY(i);
        const l = makeLine(new THREE.Vector3(-WIDTH / 2 - 0.3, y, 0.02), new THREE.Vector3(WIDTH / 2 + 0.3, y, 0.02), BREAK, 0.9);
        breaks.push(l);
        group.add(l);
      }
    });

    // Cut lines at the real chunk boundaries, and a scanning bar
    const cuts = chunks.slice(0, -1).map((c) => {
      const y = lineY(lineOf(c.end)) - doc.lineHeight * 0.5;
      const l = makeLine(new THREE.Vector3(-WIDTH / 2 - 0.6, y, 0.04), new THREE.Vector3(WIDTH / 2 + 0.6, y, 0.04), CUT, 1);
      l.userData.y = y;
      group.add(l);
      return l;
    });
    const scan = makeLine(new THREE.Vector3(-WIDTH / 2 - 0.6, 0, 0.05), new THREE.Vector3(WIDTH / 2 + 0.6, 0, 0.05), 0xffffff, 0.9);
    group.add(scan);

    // Chunk planes, each placed where its text sits in the document
    const gap = 1.3;
    const total = chunks.length * WIDTH + (chunks.length - 1) * gap;
    const pieces = chunks.map((c, i) => {
      const plane = makeTextPlane(c.text, { width: WIDTH, cols: COLS, border: lesson.color });
      const startY = lineY(lineOf(c.start)) + doc.lineHeight * 0.5 + plane.padTop - plane.height / 2;
      const home = new THREE.Vector3(0, startY, 0.1);
      const out = new THREE.Vector3(-total / 2 + WIDTH / 2 + i * (WIDTH + gap), 0, 0);
      group.add(plane.mesh);

      // Overlap overlays: top of this chunk repeats the end of the previous one
      const overlays = [];
      const addOverlay = (fromChar, toChar) => {
        const lines = wrapText(c.text, COLS);
        let a = 0;
        let b = lines.length - 1;
        lines.forEach((ln, li) => {
          if (ln.start <= fromChar) a = li;
          if (ln.start <= toChar) b = li;
        });
        const h = (b - a + 1) * plane.lineHeight;
        const mesh = new THREE.Mesh(
          new THREE.PlaneGeometry(WIDTH * 0.97, h),
          new THREE.MeshBasicMaterial({ color: 0xfbbf24, transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending }),
        );
        mesh.position.set(0, plane.height / 2 - plane.padTop - a * plane.lineHeight - h / 2, 0.01);
        plane.mesh.add(mesh);
        overlays.push(mesh);
      };
      if (c.overlapPrev > 0) addOverlay(0, c.overlapPrev - 1);
      const next = chunks[i + 1];
      if (next && next.overlapPrev > 0) addOverlay(c.chars - next.overlapPrev, c.chars - 1);

      const label = makeLabel(`Chunk ${i + 1} · ${c.chars} chars`, { color: lesson.color, size: 0.5 });
      label.position.set(0, plane.height / 2 + 0.5, 0);
      plane.mesh.add(label);
      const tokenLabel = makeLabel(`≈ ${c.tokens} tokens`, { color: "#fde68a", size: 0.45 });
      tokenLabel.position.set(0, plane.height / 2 + 1.1, 0);
      plane.mesh.add(tokenLabel);
      const meta = makeLabel(`{ filename: "${c.filename}", chunk: ${i} }`, { color: "#0f172a", size: 0.42, bg: lesson.color, weight: 700 });
      meta.position.set(0, -plane.height / 2 - 0.55, 0.05);
      plane.mesh.add(meta);

      stage.addPickable(plane.mesh, () =>
        `<strong>Chunk ${i + 1} of ${chunks.length}</strong> · ${c.filename}<br>` +
        `${c.chars} characters ≈ ${c.tokens} tokens · chars ${c.start}–${c.end}` +
        (c.overlapPrev ? `<br><em>First ${c.overlapPrev} chars repeat the previous chunk</em>` : "") +
        `<span class="viz-tip-text">${escapeHtml(c.text.slice(0, 220))}…</span>`);
      return { c, plane, home, out, overlays, label, tokenLabel, meta };
    });

    // Ribbons linking duplicated overlap text between neighbours
    const ribbons = pieces.slice(1).map(() => {
      const l = makeLine(new THREE.Vector3(), new THREE.Vector3(), 0xfbbf24, 0.8);
      group.add(l);
      return l;
    });

    world = { group, doc, breaks, cuts, scan, pieces, ribbons, chunks, lesson, docTop, total };
    return world;
  }

  function setOpacity(mesh, o) {
    mesh.material.opacity = o;
    mesh.visible = o > 0.001;
  }

  function reset() {
    const w = world;
    stage.camera.position.set(0, 0, 17);
    stage.controls.target.set(0, 0, 0);
    setOpacity(w.doc.mesh, 0);
    w.doc.mesh.position.set(0, 0, 0);
    w.breaks.forEach((l) => setOpacity(l, 0));
    w.cuts.forEach((l) => setOpacity(l, 0));
    setOpacity(w.scan, 0);
    w.pieces.forEach((p) => {
      p.plane.mesh.position.copy(p.home);
      p.plane.mesh.rotation.set(0, 0, 0);
      p.plane.mesh.scale.setScalar(1);
      setOpacity(p.plane.mesh, 0);
      p.overlays.forEach((o) => { o.material.opacity = 0; });
      [p.label, p.tokenLabel, p.meta].forEach((s) => { s.material.opacity = 0; });
    });
    w.ribbons.forEach((r) => setOpacity(r, 0));
    hud.innerHTML = hudHtml(w, "doc");
  }

  function hudHtml(w, mode) {
    const l = w.lesson;
    if (mode === "doc") {
      return `<div class="hud-title">${l.filename}</div><div class="hud-row"><span>Length</span><b>${l.chars.toLocaleString()} chars</b></div>` +
        `<div class="hud-row"><span>≈ Tokens</span><b>${Math.round(l.chars / 4).toLocaleString()}</b></div>`;
    }
    const rows = w.chunks.map((c, i) => `<div class="hud-row"><span>Chunk ${i + 1}</span><b>${mode === "tokens" ? `≈ ${c.tokens} tok` : `${c.chars} ch`}</b></div>`).join("");
    const settings = `<div class="hud-note">chunk_size=${DATA.settings.chunkSize} · overlap=${DATA.settings.chunkOverlap}<br>${mode === "tokens" ? "measured in characters, not tokens" : "RecursiveCharacterTextSplitter"}</div>`;
    return `<div class="hud-title">${l.filename} → ${w.chunks.length} chunks</div>${rows}${settings}`;
  }

  // Ribbon endpoints follow the moving planes
  stage.onFrame(() => {
    if (!world) return;
    world.ribbons.forEach((r, i) => {
      const a = world.pieces[i];
      const b = world.pieces[i + 1];
      const ao = a.overlays[a.overlays.length - 1];
      const bo = b.overlays[0];
      if (!ao || !bo || !r.visible) return;
      const pa = ao.getWorldPosition(new THREE.Vector3());
      const pb = bo.getWorldPosition(new THREE.Vector3());
      const p = r.geometry.attributes.position;
      p.setXYZ(0, pa.x + WIDTH * 0.45, pa.y, pa.z);
      p.setXYZ(1, pb.x - WIDTH * 0.45, pb.y, pb.z);
      p.needsUpdate = true;
    });
  });

  const steps = [
    // 1. One document
    async (ctx) => {
      ctx.set(() => { hud.innerHTML = hudHtml(world, "doc"); });
      const fitZ = Math.max(15, world.doc.height * 1.45);
      await Promise.all([
        ctx.anim(900, (e) => setOpacity(world.doc.mesh, e)),
        ctx.camera([0, 0, fitZ], [0, 0, 0], 1200),
      ]);
    },
    // 2. Natural breaks
    async (ctx) => {
      for (const [i, l] of world.breaks.entries()) {
        ctx.anim(500, (e) => setOpacity(l, e * 0.9));
        if (!ctx.instant) await ctx.wait(70 + i * 5);
      }
      await ctx.wait(300);
    },
    // 3. Scan and cut at the real boundaries
    async (ctx) => {
      const top = world.docTop;
      const bottom = -world.docTop;
      ctx.set(() => setOpacity(world.scan, 0.9));
      await ctx.anim(2600, (e) => {
        const y = top - (top - bottom) * e;
        const p = world.scan.geometry.attributes.position;
        p.setY(0, y);
        p.setY(1, y);
        p.needsUpdate = true;
        world.cuts.forEach((c) => setOpacity(c, y <= c.userData.y ? 1 : 0));
      }, (t) => t);
      ctx.set(() => setOpacity(world.scan, 0));
    },
    // 4. Chunks separate
    async (ctx) => {
      ctx.set(() => { hud.innerHTML = hudHtml(world, "chars"); });
      await ctx.anim(500, (e) => {
        setOpacity(world.doc.mesh, 1 - e);
        world.breaks.forEach((l) => setOpacity(l, 0.9 * (1 - e)));
        world.cuts.forEach((l) => setOpacity(l, 1 - e));
        world.pieces.forEach((p) => setOpacity(p.plane.mesh, e));
      });
      const fitZ = Math.max(18, world.total * 0.95);
      await Promise.all([
        ctx.anim(1600, (e) => world.pieces.forEach((p, i) => {
          p.plane.mesh.position.lerpVectors(p.home, p.out, e);
          p.plane.mesh.rotation.y = (i - (world.pieces.length - 1) / 2) * -0.12 * e;
          p.label.material.opacity = e;
        })),
        ctx.camera([0, 2, fitZ], [0, 0, 0], 1600),
      ]);
    },
    // 5. Overlap
    async (ctx) => {
      await ctx.anim(900, (e) => {
        world.pieces.forEach((p) => p.overlays.forEach((o) => { o.material.opacity = 0.38 * e; }));
        world.ribbons.forEach((r) => setOpacity(r, 0.85 * e));
      });
    },
    // 6. Characters vs tokens
    async (ctx) => {
      ctx.set(() => { hud.innerHTML = hudHtml(world, "tokens"); });
      await ctx.anim(700, (e) => world.pieces.forEach((p) => { p.tokenLabel.material.opacity = e; }));
    },
    // 7. Metadata stamp, then hand-off
    async (ctx) => {
      await ctx.anim(700, (e) => world.pieces.forEach((p) => { p.meta.material.opacity = e; }));
      await ctx.wait(1200);
      await Promise.all([
        ctx.anim(1500, (e) => world.pieces.forEach((p, i) => {
          p.plane.mesh.position.z = p.out.z - 3 * e;
          p.plane.mesh.position.y = p.out.y + Math.sin(e * Math.PI) * 0.6 * (i % 2 ? 1 : -1);
          p.plane.mesh.rotation.x = -0.25 * e;
        })),
        ctx.camera([6, 5, Math.max(18, world.total * 0.9)], [0, -0.5, -2], 1500),
      ]);
    },
  ];

  function start(lessonId) {
    build(lessonById(lessonId));
    if (!player) player = createPlayer(root, stage, { steps, reset });
    player.restart();
  }

  picker?.addEventListener("change", () => start(picker.value));
  start(picker ? picker.value : 2);
});

function escapeHtml(s) {
  return s.replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]));
}
