// Shared engine for the RAG 3D visualizations: scene, camera, step player,
// tweening, glowing points, text planes, labels and hover tooltips.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export { THREE };

const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
const MOBILE = matchMedia("(max-width: 700px)").matches;

export const ease = {
  inOut: (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2),
  out: (t) => 1 - Math.pow(1 - t, 3),
  linear: (t) => t,
};

export function hasWebGL() {
  try {
    const c = document.createElement("canvas");
    return !!(window.WebGLRenderingContext && (c.getContext("webgl2") || c.getContext("webgl")));
  } catch (e) {
    return false;
  }
}

// ------------------------------------------------------------------ textures

let glowTexture = null;
function getGlowTexture() {
  if (glowTexture) return glowTexture;
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d");
  const grad = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.18, "rgba(255,255,255,0.85)");
  grad.addColorStop(0.45, "rgba(255,255,255,0.22)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, 128, 128);
  glowTexture = new THREE.CanvasTexture(c);
  glowTexture.colorSpace = THREE.SRGBColorSpace;
  return glowTexture;
}

/** A glowing point: additive halo sprite + small solid core (the core is what hover picks). */
export function makeGlowPoint(color, size = 1) {
  const group = new THREE.Group();
  const halo = new THREE.Sprite(new THREE.SpriteMaterial({
    map: getGlowTexture(), color, transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending, opacity: 0.9,
  }));
  halo.scale.setScalar(1.6 * size);
  const core = new THREE.Mesh(
    new THREE.SphereGeometry(0.16 * size, 16, 16),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 1 }),
  );
  group.add(halo, core);
  group.userData = { halo, core, baseSize: size };
  return group;
}

export function setPointOpacity(point, o) {
  point.userData.halo.material.opacity = 0.9 * o;
  point.userData.core.material.opacity = o;
  point.visible = o > 0.001;
}

/** Text label sprite that always faces the camera. */
export function makeLabel(text, { color = "#e2e8f0", size = 0.55, weight = 600, bg = null } = {}) {
  const fontPx = 48;
  const c = document.createElement("canvas");
  const g = c.getContext("2d");
  g.font = `${weight} ${fontPx}px system-ui, -apple-system, Segoe UI, sans-serif`;
  const w = Math.ceil(g.measureText(text).width) + 32;
  c.width = w;
  c.height = fontPx + 28;
  g.font = `${weight} ${fontPx}px system-ui, -apple-system, Segoe UI, sans-serif`;
  if (bg) {
    g.fillStyle = bg;
    roundRect(g, 0, 0, c.width, c.height, 18);
    g.fill();
  }
  g.fillStyle = color;
  g.textBaseline = "middle";
  g.fillText(text, 16, c.height / 2 + 2);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
  sprite.scale.set((size * c.width) / c.height, size, 1);
  sprite.renderOrder = 10;
  return sprite;
}

function roundRect(g, x, y, w, h, r) {
  g.beginPath();
  g.moveTo(x + r, y);
  g.arcTo(x + w, y, x + w, y + h, r);
  g.arcTo(x + w, y + h, x, y + h, r);
  g.arcTo(x, y + h, x, y, r);
  g.arcTo(x, y, x + w, y, r);
  g.closePath();
}

/** Wrap text into lines of at most `cols` characters, remembering each line's start offset. */
export function wrapText(text, cols) {
  const lines = [];
  let offset = 0;
  for (const para of text.split("\n")) {
    if (para.length === 0) {
      lines.push({ text: "", start: offset });
    } else {
      let i = 0;
      while (i < para.length) {
        let end = Math.min(i + cols, para.length);
        if (end < para.length) {
          const space = para.lastIndexOf(" ", end);
          if (space > i) end = space + 1;
        }
        lines.push({ text: para.slice(i, end), start: offset + i });
        i = end;
      }
    }
    offset += para.length + 1;
  }
  return lines;
}

/**
 * A plane showing monospaced text, with optional highlighted character ranges.
 * Returns { mesh, lines, lineHeight } where lineHeight is in world units.
 */
export function makeTextPlane(text, {
  width = 6, cols = 58, color = "#cbd5e1", bg = "rgba(15,23,42,0.88)", border = "#334155",
  highlights = [], baseOffset = 0, fontPx = null, canvasWidth = 1024,
} = {}) {
  const lines = wrapText(text, cols);
  const pad = Math.round(canvasWidth * 0.026);
  // Size the font so `cols` monospace characters fill the canvas width.
  fontPx = fontPx || Math.floor((canvasWidth - pad * 2) / (cols * 0.6));
  const lineH = Math.round(fontPx * 1.45);
  const c = document.createElement("canvas");
  c.width = canvasWidth;
  c.height = Math.max(64, Math.ceil(pad * 2 + lines.length * lineH));
  const g = c.getContext("2d");
  g.fillStyle = bg;
  roundRect(g, 2, 2, c.width - 4, c.height - 4, 22);
  g.fill();
  g.strokeStyle = border;
  g.lineWidth = 3;
  g.stroke();
  g.font = `${fontPx}px ui-monospace, Cascadia Code, Consolas, monospace`;
  const charW = g.measureText("M").width;
  const scaleX = Math.min(1, (c.width - pad * 2) / (cols * charW));
  lines.forEach((ln, li) => {
    const y = pad + li * lineH;
    for (const h of highlights) {
      const s = Math.max(h.start - baseOffset, ln.start);
      const e = Math.min(h.end - baseOffset, ln.start + ln.text.length);
      if (e > s) {
        g.fillStyle = h.color;
        g.fillRect(pad + (s - ln.start) * charW * scaleX, y + 2, (e - s) * charW * scaleX, lineH - 4);
      }
    }
    g.fillStyle = color;
    g.save();
    g.translate(pad, y + lineH * 0.72);
    g.scale(scaleX, 1);
    g.fillText(ln.text, 0, 0);
    g.restore();
  });
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  const height = (width * c.height) / c.width;
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(width, height),
    new THREE.MeshBasicMaterial({ map: tex, transparent: true, side: THREE.DoubleSide, depthWrite: false }),
  );
  return { mesh, lines, height, lineHeight: (lineH / c.height) * height, padTop: (pad / c.height) * height };
}

export function makeLine(a, b, color, opacity = 0.6) {
  const geom = new THREE.BufferGeometry().setFromPoints([a.clone(), b.clone()]);
  const line = new THREE.Line(geom, new THREE.LineBasicMaterial({
    color, transparent: true, opacity, blending: THREE.AdditiveBlending, depthWrite: false,
  }));
  line.userData.baseOpacity = opacity;
  return line;
}

export function setLineEnds(line, a, b) {
  const p = line.geometry.attributes.position;
  p.setXYZ(0, a.x, a.y, a.z);
  p.setXYZ(1, b.x, b.y, b.z);
  p.needsUpdate = true;
  line.geometry.computeBoundingSphere();
}

// ------------------------------------------------------------------ stage

export function createStage(container, { cameraPos = [0, 0, 22], target = [0, 0, 0], autoRotate = false } = {}) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, MOBILE ? 1.5 : 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.setClearColor(0x04060c, 1);
  container.prepend(renderer.domElement);

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x04060c, 0.012);
  const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 400);
  camera.position.set(...cameraPos);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(...target);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 4;
  controls.maxDistance = 70;
  controls.autoRotate = autoRotate && !REDUCED;
  controls.autoRotateSpeed = 0.35;
  controls.enablePan = !MOBILE;

  // Background star field (decoration only, not data).
  const starCount = MOBILE ? 700 : 1600;
  const starPos = new Float32Array(starCount * 3);
  for (let i = 0; i < starCount; i++) {
    const r = 45 + Math.random() * 60;
    const th = Math.random() * Math.PI * 2;
    const ph = Math.acos(2 * Math.random() - 1);
    starPos.set([r * Math.sin(ph) * Math.cos(th), r * Math.sin(ph) * Math.sin(th), r * Math.cos(ph)], i * 3);
  }
  const starGeom = new THREE.BufferGeometry();
  starGeom.setAttribute("position", new THREE.BufferAttribute(starPos, 3));
  const stars = new THREE.Points(starGeom, new THREE.PointsMaterial({
    color: 0x8aa0c8, size: 0.18, transparent: true, opacity: 0.55, depthWrite: false,
  }));
  scene.add(stars);

  // ---- tweens
  const tweens = new Set();
  function animate(duration, fn, easing = ease.inOut) {
    return new Promise((resolve) => {
      if (REDUCED || duration <= 0) {
        fn(1);
        resolve();
        return;
      }
      tweens.add({ start: performance.now(), duration, fn, easing, resolve });
    });
  }
  function cancelTweens() {
    for (const t of tweens) t.resolve();
    tweens.clear();
  }

  // ---- render loop (paused while off screen)
  const frameHooks = new Set();
  let visible = true;
  const io = new IntersectionObserver((entries) => { visible = entries[0].isIntersecting; });
  io.observe(container);
  const clock = new THREE.Clock();
  function loop() {
    requestAnimationFrame(loop);
    if (!visible) return;
    const now = performance.now();
    for (const t of [...tweens]) {
      const p = Math.min(1, (now - t.start) / t.duration);
      t.fn(t.easing(p));
      if (p >= 1) {
        tweens.delete(t);
        t.resolve();
      }
    }
    const dt = clock.getDelta();
    const time = clock.elapsedTime;
    stars.rotation.y += dt * 0.004;
    for (const hook of frameHooks) hook(time, dt);
    controls.update();
    renderer.render(scene, camera);
  }

  function resize() {
    const w = container.clientWidth;
    const h = container.clientHeight;
    renderer.setSize(w, h, false);
    renderer.domElement.style.width = "100%";
    renderer.domElement.style.height = "100%";
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(container);
  resize();
  loop();

  // ---- camera moves
  function cameraTo(pos, tgt, duration = 1400) {
    const p0 = camera.position.clone();
    const t0 = controls.target.clone();
    const p1 = new THREE.Vector3(...pos);
    const t1 = new THREE.Vector3(...tgt);
    return animate(duration, (e) => {
      camera.position.lerpVectors(p0, p1, e);
      controls.target.lerpVectors(t0, t1, e);
    });
  }

  // ---- hover / tap tooltips
  const tooltip = container.querySelector(".viz-tooltip");
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  const pickables = new Map(); // mesh -> () => html
  function addPickable(mesh, htmlFn) { pickables.set(mesh, htmlFn); }
  function clearPickables() { pickables.clear(); hideTip(); }
  function hideTip() { if (tooltip) tooltip.hidden = true; container.style.cursor = ""; }
  function pick(clientX, clientY) {
    if (!tooltip || pickables.size === 0) return;
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const meshes = [...pickables.keys()].filter((m) => m.visible && worldVisible(m));
    const hit = raycaster.intersectObjects(meshes, false)[0];
    if (!hit) return hideTip();
    tooltip.innerHTML = pickables.get(hit.object)();
    tooltip.hidden = false;
    const x = Math.min(clientX - rect.left + 14, rect.width - tooltip.offsetWidth - 8);
    const y = Math.min(clientY - rect.top + 14, rect.height - tooltip.offsetHeight - 8);
    tooltip.style.transform = `translate(${Math.max(8, x)}px, ${Math.max(8, y)}px)`;
    container.style.cursor = "pointer";
  }
  function worldVisible(obj) {
    for (let o = obj; o; o = o.parent) if (!o.visible) return false;
    return true;
  }
  renderer.domElement.addEventListener("pointermove", (e) => { if (e.pointerType === "mouse") pick(e.clientX, e.clientY); });
  renderer.domElement.addEventListener("pointerleave", hideTip);
  renderer.domElement.addEventListener("pointerup", (e) => { if (e.pointerType !== "mouse") pick(e.clientX, e.clientY); });

  return {
    THREE, scene, camera, renderer, controls, animate, cancelTweens, cameraTo,
    onFrame: (fn) => frameHooks.add(fn), addPickable, clearPickables, reduced: REDUCED, mobile: MOBILE,
  };
}

// ------------------------------------------------------------------ step player

/**
 * Drives a list of async steps. Each step receives `ctx` with `instant` set when
 * it is being fast-forwarded (jumping to a later step), so it can apply its end
 * state without animating. `reset()` restores the scene to its initial state.
 */
export function createPlayer(root, stage, { steps, reset, dwell = 4200 }) {
  const items = [...root.querySelectorAll(".viz-steps > li")];
  const caption = root.querySelector(".viz-caption");
  const dotsBox = root.querySelector(".viz-dots");
  const playBtn = root.querySelector("[data-action=play]");
  const prevBtn = root.querySelector("[data-action=prev]");
  const nextBtn = root.querySelector("[data-action=next]");
  if (items.length !== steps.length) console.warn("viz: caption count", items.length, "!= steps", steps.length);

  let current = -1;
  let generation = 0;
  let playing = !stage.reduced;
  let timer = null;

  const dots = steps.map((_, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "viz-dot";
    b.setAttribute("aria-label", `Step ${i + 1}: ${items[i]?.querySelector("h3")?.textContent || ""}`);
    b.addEventListener("click", () => { setPlaying(false); goTo(i); });
    dotsBox?.appendChild(b);
    return b;
  });
  items.forEach((li, i) => li.addEventListener("click", () => { setPlaying(false); goTo(i); }));

  function ctxFor(gen, instant) {
    const alive = () => gen === generation;
    // Steps must only touch the scene through anim()/set(), so a step that was
    // interrupted by a jump can never change the scene after the jump.
    return {
      instant,
      alive,
      set: (fn) => { if (alive()) fn(); },
      anim: (ms, fn, easing) => {
        if (!alive()) return Promise.resolve();
        if (instant) { fn(1); return Promise.resolve(); }
        return stage.animate(ms, (e) => { if (alive()) fn(e); }, easing);
      },
      camera: (pos, tgt, ms = 1400) => {
        if (!alive()) return Promise.resolve();
        if (instant) {
          stage.camera.position.set(...pos);
          stage.controls.target.set(...tgt);
          return Promise.resolve();
        }
        return stage.cameraTo(pos, tgt, ms);
      },
      wait: (ms) => (instant || stage.reduced ? Promise.resolve() : new Promise((r) => setTimeout(r, ms))),
    };
  }

  async function goTo(index) {
    index = Math.max(0, Math.min(steps.length - 1, index));
    const gen = ++generation;
    clearTimeout(timer);
    stage.cancelTweens();
    // Always rebuild: reset, fast-forward earlier steps to their end state, then
    // animate the target step. Scenes create objects once and steps only change
    // properties, so this is cheap and makes interruptions safe.
    reset();
    for (let i = 0; i < index; i++) {
      await steps[i](ctxFor(gen, true));
      if (gen !== generation) return;
    }
    current = index;
    render();
    await steps[index](ctxFor(gen, false));
    if (gen !== generation) return;
    if (playing && index < steps.length - 1) timer = setTimeout(() => goTo(index + 1), dwell);
    if (index === steps.length - 1) { setPlaying(false, true); }
  }

  function render() {
    items.forEach((li, i) => li.classList.toggle("active", i === current));
    dots.forEach((d, i) => {
      d.classList.toggle("active", i === current);
      d.classList.toggle("done", i < current);
    });
    const li = items[current];
    if (caption && li) {
      caption.innerHTML = `<span class="viz-caption-num">Step ${current + 1} / ${steps.length}</span>` +
        `<strong>${li.querySelector("h3").textContent}</strong><span>${li.querySelector("p").innerHTML}</span>`;
    }
    prevBtn.disabled = current <= 0;
    nextBtn.disabled = current >= steps.length - 1;
  }

  function setPlaying(on, finished = false) {
    playing = on;
    clearTimeout(timer);
    playBtn.textContent = on ? "❚❚ Pause" : finished ? "↻ Replay" : "▶ Play";
    playBtn.setAttribute("aria-pressed", String(on));
  }

  playBtn.addEventListener("click", () => {
    if (playing) return setPlaying(false);
    setPlaying(true);
    goTo(current >= steps.length - 1 ? 0 : current + 1);
  });
  prevBtn.addEventListener("click", () => { setPlaying(false); goTo(current - 1); });
  nextBtn.addEventListener("click", () => { setPlaying(false); goTo(current + 1); });
  root.addEventListener("keydown", (e) => {
    if (e.target.closest("select, input")) return;
    if (e.key === "ArrowRight") { setPlaying(false); goTo(current + 1); }
    if (e.key === "ArrowLeft") { setPlaying(false); goTo(current - 1); }
  });

  setPlaying(playing);
  return {
    goTo,
    restart: () => { current = -1; goTo(0); },
    get current() { return current; },
  };
}

/** Shared bootstrap: checks WebGL, builds the stage, and reports failures in the stage area. */
export function boot(root, build) {
  const stageEl = root.querySelector(".viz-stage");
  if (!hasWebGL()) {
    stageEl.classList.add("viz-nogl");
    stageEl.querySelector(".viz-fallback").hidden = false;
    return;
  }
  try {
    build(stageEl);
  } catch (err) {
    console.error(err);
    stageEl.classList.add("viz-nogl");
    const fb = stageEl.querySelector(".viz-fallback");
    fb.hidden = false;
    fb.textContent = "The 3D scene could not start in this browser. The steps below describe what it shows.";
  }
}
