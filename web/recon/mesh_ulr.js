// C2: real frames blended on the fitted plate (VDTM / unstructured lumigraph rendering).
// The views are the capture's frames rectified onto the filmed face (WebP, alpha = seen, feathered
// towards where the frame stopped seeing the face). Each frame 12 views are bound: up to 4 kept for
// coverage for the whole recording (the manifest's ulr.cover_views, chosen by cover_views in
// flawmap/recon/candidates/c2_ulr.py), then the nearest to the eye's direction (select_views there)
// among the views at least 3 degrees apart (ulr.active_views, distinct_views there).
// Per pixel every bound view is weighted, w = ramp x alpha x (1/a)(1 - a/cap), a the angle between
// the eye's ray and the view's ray to that point; the ramp (rank_ramp there) is 1 for the coverage
// views and falls to 0 for a view about to leave the set or just joining it, so the picture changes
// continuously as the eye turns. The blend's share is (1 - fade) x cover x held x nearCap: the
// weighted alpha, the weight against one full view's at 15 degrees, and a fall to 0 before the cap; past
// 12-25 degrees from the captured rays it fades to the inspection texture, as does everything off
// the filmed face. This mirrors C2ULR.blend_views in c2_ulr.py.
//
// The view binding and the blend (ulrGLSL, ulrViews) are shared with C7 (mesh_hybrid.js), which
// blends residuals against C1 instead of colours.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { firstMesh, loadTex } from './common.js';

export const N = 12;

export const vert = /* glsl */`
varying vec2 vUv;
varying vec3 vPos;
void main() {
  vUv = uv;
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vPos = wp.xyz;
  gl_Position = projectionMatrix * viewMatrix * wp;
}`;

// ulrBlend(t, d, ex): the stored views blended at face position t (0-1 over the views' extent) for the
// eye direction d; .rgb the blend, .a its share (1 - fade) x cover (0 where no bound view saw it).
// With ULR_EXTRA defined, the including shader defines vec3 ulrExtra(int k), a value per bound view,
// and ex returns it blended with the same weights (C7: C1 seen from each view); else ex is 0.
export const ulrGLSL = /* glsl */`
#ifdef ULR_EXTRA
vec3 ulrExtra(int k);
#endif
uniform sampler2D v0; uniform sampler2D v1; uniform sampler2D v2; uniform sampler2D v3;
uniform sampler2D v4; uniform sampler2D v5; uniform sampler2D v6; uniform sampler2D v7;
uniform sampler2D v8; uniform sampler2D v9; uniform sampler2D v10; uniform sampler2D v11;
uniform vec3 vc[${N}];
uniform float vsel[${N}];   // the ramp of each bound view (0 while a view is not loaded)
uniform vec2 viewSizeMM;    // the stored views' extent (cols x rows x res)
uniform vec2 fadeDeg;
uniform float angleCap;
uniform float capRamp;
uniform float heldRef;      // the share is full while the weight is at least that of one full view at this angle
uniform float ulrOn;
vec4 sv(int k, vec2 t) {
  if (k == 0) return texture2D(v0, t); if (k == 1) return texture2D(v1, t);
  if (k == 2) return texture2D(v2, t); if (k == 3) return texture2D(v3, t);
  if (k == 4) return texture2D(v4, t); if (k == 5) return texture2D(v5, t);
  if (k == 6) return texture2D(v6, t); if (k == 7) return texture2D(v7, t);
  if (k == 8) return texture2D(v8, t); if (k == 9) return texture2D(v9, t);
  if (k == 10) return texture2D(v10, t); return texture2D(v11, t);
}
vec4 ulrBlend(vec2 t, vec3 d, out vec3 exBlend) {
  vec3 acc = vec3(0.0); vec3 accEx = vec3(0.0);
  float ws = 0.0; float asum = 0.0; float ts = 0.0;
  for (int k = 0; k < ${N}; k++) {
    vec4 s = sv(k, t);
    float th = max(degrees(acos(clamp(dot(normalize(vc[k] - vPos), d), -1.0, 1.0))), 0.05);
    float w0 = (1.0 / th) * max(1.0 - th / angleCap, 0.0) * s.a;
    float w = vsel[k] * w0;
    acc += w * s.rgb;
#ifdef ULR_EXTRA
    if (w > 0.0) accEx += w * ulrExtra(k);
#endif
    ws += w; asum += w * s.a; ts += w * th;
  }
  exBlend = vec3(0.0);
  if (ws <= 1e-9) return vec4(0.0);
  exBlend = accEx / ws;
  float tEff = ts / ws;
  float fade = clamp((tEff - fadeDeg.x) / (fadeDeg.y - fadeDeg.x), 0.0, 1.0);
  float nearCap = clamp((angleCap - tEff) / capRamp, 0.0, 1.0);
  float held = clamp(ws / ((1.0 - heldRef / angleCap) / heldRef), 0.0, 1.0);
  return vec4(acc / ws, (1.0 - fade) * (asum / ws) * held * nearCap);
}`;

const frag = /* glsl */`
uniform sampler2D baseMap;
uniform vec4 faceRect;      // atlas pixels: x0, y0, width, height of the filmed face's chart
uniform float faceRes;      // mm per atlas pixel on that chart
uniform vec2 atlasSize;
varying vec2 vUv;
varying vec3 vPos;
${ulrGLSL}
void main() {
  vec3 col = texture2D(baseMap, vUv).rgb;
  vec2 px = vUv * atlasSize;
  if (ulrOn > 0.5 && px.x >= faceRect.x && px.y >= faceRect.y && px.x < faceRect.x + faceRect.z && px.y < faceRect.y + faceRect.w) {
    vec3 ex;
    vec4 b = ulrBlend((px - faceRect.xy) * faceRes / viewSizeMM, normalize(cameraPosition - vPos), ex);
    col = mix(col, b.rgb, b.a);
  }
  gl_FragColor = linearToOutputTexel(vec4(col, 1.0));
}`;

// the uniforms of ulrGLSL for the manifest's ulr entry
export function ulrUniforms(u) {
  const blank = new THREE.DataTexture(new Uint8Array([0, 0, 0, 0]), 1, 1);
  blank.needsUpdate = true;
  const uniforms = {
    vc: { value: Array.from({ length: N }, () => new THREE.Vector3()) }, vsel: { value: new Array(N).fill(0) },
    viewSizeMM: { value: new THREE.Vector2(u.shape[1] * u.res_mm, u.shape[0] * u.res_mm) },
    fadeDeg: { value: new THREE.Vector2(u.fade_deg[0], u.fade_deg[1]) }, ulrOn: { value: 1 },
    angleCap: { value: u.angle_cap ?? 30 }, capRamp: { value: u.cap_ramp_deg ?? 5 }, heldRef: { value: u.held_ref_deg ?? 15 },
  };
  for (let k = 0; k < N; k++) uniforms[`v${k}`] = { value: blank };
  return { uniforms, blank };
}

// the views: directions from the device centre, the 12 bound each frame with their ramps, a small
// texture cache filled on demand
export function ulrViews(u, base, uniforms, blank, { srgb = true, onLoad = null } = {}) {   // onLoad: a view arrived (draw again)
  const views = u.views.map((v) => {
    const c = new THREE.Vector3(...v.center_mm).multiplyScalar(1e-3);
    return { file: v.file, url: v.url, c, dir: c.clone().normalize(), tex: null, loading: false };   // url: a ready address (a packed view)
  });
  const fixed = (u.cover_views || []).slice(0, N).filter((k) => k >= 0 && k < views.length);
  // the views that may be bound (distinct_views in c2_ulr.py: at least 3 degrees apart); all if absent
  const active = u.active_views ? u.active_views.filter((k) => k >= 0 && k < views.length) : views.map((_, k) => k);
  const isActive = new Uint8Array(views.length);
  active.forEach((k) => { isActive[k] = 1; });
  const rampFrac = u.ramp_frac ?? 0, rampMin = u.ramp_min_deg ?? 5;
  // select_views and rank_ramp in c2_ulr.py: the coverage views, then the nearest in direction; the
  // ramp falls to 0 at the angle of the nearest view left out
  function select(e) {
    const dots = views.map((v) => v.dir.dot(e));
    const order = active.slice().sort((p, q) => dots[q] - dots[p] || p - q);
    const chosen = fixed.slice();
    for (const k of order) { if (chosen.length >= N) break; if (!chosen.includes(k)) chosen.push(k); }
    const sel = chosen.map(() => 1);
    const ang = dots.map((x) => (Math.acos(Math.min(1, Math.max(-1, x))) * 180) / Math.PI);
    let cut = Infinity;
    for (const k of active) if (!chosen.includes(k)) cut = Math.min(cut, ang[k]);
    const dirK = chosen.filter((k) => !fixed.includes(k));
    if (rampFrac > 0 && Number.isFinite(cut) && dirK.length) {
      const first = Math.min(...dirK.map((k) => ang[k]));
      const width = Math.max(rampFrac * (cut - first), rampMin);
      chosen.forEach((k, j) => { if (!fixed.includes(k)) sel[j] = Math.min(1, Math.max(0, (cut - ang[k]) / width)); });
    }
    return chosen.map((k, j) => ({ v: views[k], s: sel[j] }));
  }
  const cache = [];
  let bound = [];
  function ensure(v) {
    if (v.tex || v.loading) return;
    v.loading = true;
    loadTex(v.url || base + v.file, { aniso: 4, srgb }).then((t) => {
      v.tex = t; v.loading = false; cache.push(v); onLoad?.();
      while (cache.length > 32) { const old = cache.shift(); if (!bound.includes(old)) { old.tex.dispose(); old.tex = null; } else cache.push(old); }
    }).catch(() => { v.loading = false; });
  }
  const e = new THREE.Vector3();
  return {
    update(camera) {
      e.copy(camera.position).normalize();
      const picked = select(e);
      bound = picked.map((p) => p.v);
      picked.forEach(({ v, s }, k) => {
        ensure(v);
        uniforms[`v${k}`].value = v.tex || blank;
        uniforms.vsel.value[k] = v.tex ? s : 0;
        uniforms.vc.value[k].copy(v.c);
      });
    },
    dispose() { views.forEach((v) => v.tex?.dispose()); },
  };
}

export async function create({ base, manifest, scene }) {
  const u = manifest.ulr;
  const [gltf, baseMap] = await Promise.all([new GLTFLoader().loadAsync(base + manifest.files.model), loadTex(base + manifest.files.texture)]);
  const mesh = firstMesh(gltf.scene);
  const [x0, y0, res] = u.atlas.layout[u.chart];
  const { uniforms: ulrU, blank } = ulrUniforms(u);
  const uniforms = {
    ...ulrU, baseMap: { value: baseMap },
    faceRect: { value: new THREE.Vector4(x0, y0, u.chart_size_mm[0] / res, u.chart_size_mm[1] / res) },
    faceRes: { value: res }, atlasSize: { value: new THREE.Vector2(u.atlas.w, u.atlas.h) },
  };
  mesh.material = new THREE.ShaderMaterial({ uniforms, vertexShader: vert, fragmentShader: frag, side: THREE.DoubleSide });
  scene.add(mesh);
  const views = ulrViews(u, base, uniforms, blank);
  return {
    update(camera) { views.update(camera); },
    dispose() { mesh.geometry.dispose(); mesh.material.dispose(); baseMap.dispose(); views.dispose(); },
  };
}
