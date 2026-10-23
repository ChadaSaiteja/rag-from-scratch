// Shared pieces for scenes that show the stored vector space.
import { makeGlowPoint, makeLabel, THREE } from "./core.js";

export const DATA = window.RAG_VIZ;

export function escapeHtml(s) {
  return s.replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]));
}

export function chunkTip(c) {
  const lesson = DATA.lessons[c.lesson];
  return `<strong style="color:${lesson.color}">${c.filename}</strong> · chunk ${c.index + 1}/${lesson.chunkIds.length}<br>` +
    `${c.chars} chars ≈ ${c.tokens} tokens<br>` +
    `<em>top terms:</em> ${c.topTerms.slice(0, 5).map((t) => t[0]).join(", ")}` +
    `<span class="viz-tip-text">${escapeHtml(c.text.slice(0, 200))}…</span>`;
}

/** One glowing, hoverable point per real chunk, placed at its 3D position. */
export function buildPoints(stage, group) {
  return DATA.chunks.map((c) => {
    const color = DATA.lessons[c.lesson].color;
    const point = makeGlowPoint(color, 1);
    point.userData.home = new THREE.Vector3(...c.pos);
    point.position.copy(point.userData.home);
    group.add(point);
    stage.addPickable(point.userData.core, () => chunkTip(c));
    return point;
  });
}

/** A floating label at the centre of each lesson's chunks. */
export function buildLessonLabels(group) {
  return DATA.lessons.map((l) => {
    const pts = l.chunkIds.map((id) => DATA.chunks[id].pos);
    const centre = pts.reduce((a, p) => a.map((v, i) => v + p[i] / pts.length), [0, 0, 0]);
    const label = makeLabel(l.title, { color: l.color, size: 0.55 });
    label.position.set(centre[0], centre[1] + 1.1, centre[2]);
    label.material.opacity = 0;
    group.add(label);
    return label;
  });
}

export function legendHtml() {
  return `<div class="hud-title">Colour = source lesson</div>` + DATA.lessons.map((l) =>
    `<div class="hud-legend"><i style="background:${l.color}"></i>${l.filename.replace(".txt", "")}</div>`).join("");
}
