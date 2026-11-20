// flawmap 모델링 비교: one example capture, three models of it on three mats that turn together.
//   기존 (C0): the service's model before: the inspection texture on the fitted plate, a synthetic room's light
//   안 된 방법 (C5a): plain 3D Gaussian splatting of the whole frame (cut to a box around the device)
//   최종 (C7 5라운드): the measured reflection map plus the real frames' residuals (web/recon/mesh_hybrid.js)
// Every model is in the device frame (metres; +z out of the screen), so one camera serves all three.
// three.js comes from the CDN's module builds (no import map: the page may be wrapped after other scripts).
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.180.0/+esm';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/controls/OrbitControls.js/+esm';
import { GLTFLoader } from 'https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/loaders/GLTFLoader.js/+esm';
import { RoomEnvironment } from 'https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/environments/RoomEnvironment.js/+esm';
import { loadSplat } from './gs_lite.js';

// The artifact host serves web types only, so binary files (.glb .f16 .bin .splat) are published as
// base64 text beside their names (<name>.b64.txt, scripts/team_page/build.py). This wrapper answers a
// fetch of the binary name with the decoded bytes, so three.js's loaders and the C7 modules read them as usual.
const BINARY = /\.(glb|f16|bin|splat)$/;
const rawFetch = window.fetch.bind(window);
window.fetch = async (input, init) => {
  const url = typeof input === 'string' ? input : input.url;
  if (!BINARY.test(new URL(url, location.href).pathname)) return rawFetch(input, init);
  const r = await rawFetch(`${url}.b64.txt`, init?.signal ? { signal: init.signal } : undefined);
  if (!r.ok) return r;
  const text = await r.text();
  const bin = atob(text);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Response(bytes, { status: 200, headers: { 'Content-Type': 'application/octet-stream' } });
};

const $ = (id) => document.getElementById(id);
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
const DATA = './data/';
const FACE_KO = { back: '뒷면', screen: '화면' };

// ------------------------------------------------------------------ the shared camera
const bench = $('bench');
const camera = new THREE.PerspectiveCamera(30, 4 / 3, 0.005, 20);
const controls = new OrbitControls(camera, bench);
controls.enableDamping = !reduce;
controls.dampingFactor = 0.1;
controls.enablePan = false;
controls.rotateSpeed = 0.8;
// on a phone the panels stack: a vertical swipe scrolls the page, a sideways drag turns the models
const narrow = matchMedia('(max-width: 900px)');
const setTouch = () => { bench.style.touchAction = narrow.matches ? 'pan-y' : 'none'; };
setTouch();
narrow.addEventListener('change', setTouch);
let dirtyUntil = 0;
const markDirty = (ms = 150) => { dirtyUntil = Math.max(dirtyUntil, performance.now() + ms); };
controls.addEventListener('change', () => markDirty(250));

// ------------------------------------------------------------------ panels
function makePanel(key) {
  const el = $(`panel-${key}`);
  const canvas = el.querySelector('canvas');
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  const p = { key, el, canvas, renderer, state: el.querySelector('.state') };
  p.load = newLoad(p);
  return p;
}
const panels = { old: makePanel('old'), fail: makePanel('fail'), final: makePanel('final') };

// One load of one panel: its own scene, parts and blob addresses. A load that an example switch has
// overtaken keeps adding to its own scene, never to the new one, and is thrown away when it ends.
function newLoad(p) {
  return { scene: new THREE.Scene(), parts: [], urls: [], renderer: p.renderer };
}
function dropLoad(l) {
  for (const x of l.parts) { try { x.dispose?.(); } catch { /* already gone */ } }
  for (const u of l.urls) URL.revokeObjectURL(u);
  l.parts = []; l.urls = [];
}

const gltfCache = new Map();
function loadGltf(url) {
  if (!gltfCache.has(url)) gltfCache.set(url, new GLTFLoader().loadAsync(url));
  return gltfCache.get(url).then((g) => g.scene.clone(true));
}
function firstMesh(root) {
  let mesh = null;
  root.traverse((o) => { if (o.isMesh && !mesh) mesh = o; });
  if (mesh && !mesh.geometry.attributes.normal) mesh.geometry.computeVertexNormals();
  return mesh;
}
function loadTex(url, srgb = true) {
  return new Promise((res, rej) => new THREE.TextureLoader().load(url, (t) => {
    t.flipY = false; t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace; t.anisotropy = 8; res(t);
  }, undefined, rej));
}

// 기존: the result viewer's material and lights before the realistic view (web/viewer.js)
async function loadOld(l, ex, base) {
  const pmrem = new THREE.PMREMGenerator(l.renderer);
  const env = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  l.parts.push({ dispose() { env.dispose(); pmrem.dispose(); } });
  l.scene.environment = env;
  l.scene.add(new THREE.HemisphereLight(0xffffff, 0x9aa0a8, 0.7));
  const head = new THREE.DirectionalLight(0xffffff, 0.9);
  l.scene.add(head);
  const [root, tex, rough] = await Promise.all([loadGltf(base + ex.files.model), loadTex(base + ex.files.texture),
    ex.files.roughness ? loadTex(base + ex.files.roughness, false).catch(() => null) : Promise.resolve(null)]);
  const mesh = firstMesh(root);
  mesh.material = new THREE.MeshStandardMaterial({ map: tex, roughness: rough ? 1.0 : 0.55, roughnessMap: rough, metalness: 0,
    envMapIntensity: 0.9, side: THREE.DoubleSide });
  l.scene.add(mesh);
  l.parts.push({ update(cam) { head.position.copy(cam.position); },
    dispose() { mesh.material.dispose(); tex.dispose(); rough?.dispose(); } });
}

// 안 된 방법: one splat model per recording (gs_lite.js). A recording of the back has the floor where the
// screen is (and the other way round), so with two recordings each is drawn from its own face's side.
const FACE_N = { screen: 1, back: -1 };   // the face's outward normal along the device z
async function loadFail(l, ex, base) {
  const faces = ex.faces.filter((f) => f.gs);
  if (!faces.length) throw new Error('이 예시의 가우시안 결과가 없습니다');
  const only = faces.length > 1;
  await Promise.all(faces.map(async (f) => {
    const s = await loadSplat(base + f.gs.file, { shUrl: f.gs.sh && base + f.gs.sh, shRest: f.gs.sh_rest || 0, shScale: f.gs.sh_scale || 1 });
    l.scene.add(s.mesh);
    const side = FACE_N[f.face] ?? 1;
    l.parts.push({
      update(cam, renderer) {
        s.mesh.visible = !only || cam.position.z * side >= 0;
        if (s.mesh.visible) s.update(cam, renderer);
      },
      dispose() { s.dispose(); },
    });
  }));
}

// 최종: one C7 layer per filmed face; a second face's layer draws its own face only, over the first
let hybrid = null;
async function loadFinal(l, ex, base) {
  hybrid = hybrid || await import('./recon/mesh_hybrid.js');
  for (const [k, f] of ex.faces.entries()) {
    const dir = base + f.c7.replace(/[^/]*$/, '');
    const man = await (await fetch(base + f.c7)).json();
    if (man.ulr?.pack) {
      const blob = await (await fetch(dir + man.ulr.pack)).blob();
      for (const v of man.ulr.views) {
        const [off, len] = v.range;
        v.url = URL.createObjectURL(blob.slice(off, off + len, 'image/webp'));
        l.urls.push(v.url);
      }
    }
    l.parts.push(await hybrid.create({ base: dir, manifest: man, scene: l.scene, onlyFace: k > 0, depthBias: k > 0 ? 1 : 0,
      onLoad: () => markDirty(250) }));
  }
}

const LOADERS = { old: loadOld, fail: loadFail, final: loadFinal };
let token = 0;
let current = null;

async function showExample(ex) {
  const my = ++token;
  current = ex;
  renderNotes(ex);
  aimCamera(ex, 'oblique');
  const base = `${DATA}${ex.id}/`;
  await Promise.all(Object.values(panels).map(async (p) => {
    dropLoad(p.load);
    const l = newLoad(p);
    p.load = l;
    p.state.hidden = false;
    p.state.textContent = '불러오는 중…';
    try {
      await LOADERS[p.key](l, ex, base);
      if (p.load === l) p.state.hidden = true;
    } catch (e) {
      if (p.load === l) p.state.textContent = `불러오지 못했습니다: ${e?.message || e}`;
    }
    if (p.load !== l) dropLoad(l);       // overtaken by another example while loading
    markDirty(600);
  }));
}

// ------------------------------------------------------------------ views
// For checks before publishing: look through one of the recording's cameras (T: device-frame pose in
// mm with OpenCV axes, row-major 4×4; K: its intrinsics). Dragging again leaves this view.
let lookMode = false;
window.__flawmapLook = (T, K, h) => {
  const m = new THREE.Matrix4().set(...T.flat()).multiply(new THREE.Matrix4().makeScale(1, -1, -1));
  camera.position.setFromMatrixPosition(m).multiplyScalar(1e-3);
  camera.quaternion.setFromRotationMatrix(m);
  camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(h / 2 / K[1][1]));
  camera.updateProjectionMatrix();
  lookMode = true;
  markDirty(1500);
};
controls.addEventListener('start', () => {
  if (!lookMode) return;
  lookMode = false;
  camera.fov = 30;
  camera.updateProjectionMatrix();
  const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
  controls.target.copy(camera.position).addScaledVector(fwd, camera.position.length());
});
let upSign = 1;
function aimCamera(ex, how) {
  lookMode = false;
  camera.fov = 30;
  camera.updateProjectionMatrix();
  const main = ex.faces[0]?.face || 'screen';
  upSign = main === 'back' ? -1 : 1;
  camera.up.set(0, 0, upSign);
  const L = ex.size_mm[0] * 1e-3;
  const d = (L / 2) / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.25;
  const el = { top: 89.5, oblique: 38, side: 6, flip: -55 }[how] ?? 38;
  const az = THREE.MathUtils.degToRad(-60);
  const e = THREE.MathUtils.degToRad(el);
  const dir = new THREE.Vector3(Math.cos(e) * Math.cos(az), Math.cos(e) * Math.sin(az), Math.sin(e) * upSign);
  camera.position.copy(dir.multiplyScalar(d));
  controls.target.set(0, 0, 0);
  controls.minDistance = d * 0.25;
  controls.maxDistance = d * 2.5;
  camera.lookAt(0, 0, 0);
  controls.update();
  markDirty(400);
}
for (const b of document.querySelectorAll('[data-view]')) {
  b.addEventListener('click', () => { if (current) aimCamera(current, b.dataset.view); });
}
const spinBtn = $('spin');
function setSpin(on) {
  controls.autoRotate = on;
  controls.autoRotateSpeed = 2.0;
  spinBtn.setAttribute('aria-pressed', String(on));
  spinBtn.textContent = on ? '돌리기 멈춤' : '천천히 돌리기';
  markDirty();
}
spinBtn.addEventListener('click', () => setSpin(!controls.autoRotate));

// ------------------------------------------------------------------ notes beside the bench
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
function renderNotes(ex) {
  $('exTitle').textContent = ex.title;
  $('exKind').textContent = `${ex.kind} · ${ex.device}`;
  $('exLook').textContent = ex.look;
  const rows = [
    ['놓인 상태', ex.place],
    ['잰 크기', `${ex.size_mm[0]} × ${ex.size_mm[1]} × ${ex.size_mm[2]} mm`],
    ['찾은 하자', `${ex.defects}곳 (검수 결과, 세 모델과 무관)`],
  ];
  $('exFacts').innerHTML = rows.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join('');
  $('exFaces').innerHTML = ex.faces.map((f) => {
    const c = f.choice || {};
    const sep = c.min_view_sep_deg ? `거의 같은 방향의 사진을 3° 간격으로 줄임 (모든 사진으로는 회전 튐 ${c.turn_jump_every_view}배)` : `사진 ${f.active || f.views}장 모두 사용 (회전 튐 ${c.turn_jump_every_view}배)`;
    const val = f.frames?.val ? `검증 프레임 ${f.frames.val}장으로 고름` : '녹화가 짧아 검증 구간이 없어 기본값';
    return `<li><b>${esc(FACE_KO[f.face] || f.face)}</b>: 학습 프레임 ${f.frames?.train ?? '?'}장, ${sep}.
      설정 fade ${(c.fade_deg || []).join('–')}°, β ${c.beta} (${val}).${f.gs ? ` 가우시안 ${f.gs.splats.toLocaleString()}개 (학습된 ${f.gs.splats_trained.toLocaleString()}개 중 기기 주변).` : ''}</li>`;
  }).join('');
}

// ------------------------------------------------------------------ example picker
function renderPicker(list) {
  const nav = $('picker');
  nav.innerHTML = list.map((ex, i) => `<button type="button" class="pick" data-i="${i}" aria-pressed="false">
      <span class="pick-t">${esc(ex.title)}</span><span class="pick-k">${esc(ex.kind)}</span></button>`).join('');
  for (const b of nav.querySelectorAll('.pick')) {
    b.addEventListener('click', () => {
      for (const o of nav.querySelectorAll('.pick')) o.setAttribute('aria-pressed', String(o === b));
      showExample(list[+b.dataset.i]);
    });
  }
}

// ------------------------------------------------------------------ size and drawing
function resize() {
  let aspect = 4 / 3;
  for (const p of Object.values(panels)) {
    const r = p.canvas.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) continue;
    p.renderer.setSize(r.width, r.height, false);
    aspect = r.width / r.height;
  }
  camera.aspect = aspect;
  camera.updateProjectionMatrix();
  markDirty(200);
}
new ResizeObserver(resize).observe(bench);

let visible = true;
new IntersectionObserver((es) => { visible = es.some((e) => e.isIntersecting); if (visible) markDirty(300); }).observe(bench);
function frame(now) {
  requestAnimationFrame(frame);
  if (!visible || (!controls.autoRotate && now > dirtyUntil)) return;
  if (!lookMode) controls.update();
  camera.updateMatrixWorld();
  for (const p of Object.values(panels)) {
    for (const x of p.load.parts) x.update?.(camera, p.renderer);
    p.renderer.render(p.load.scene, camera);
  }
}
requestAnimationFrame(frame);

// ------------------------------------------------------------------ start
try {
  const list = await (await fetch(`${DATA}examples.json`)).json();
  renderPicker(list);
  resize();
  document.querySelector('.pick')?.click();
} catch (e) {
  for (const p of Object.values(panels)) { p.state.hidden = false; p.state.textContent = `예시 목록을 읽지 못했습니다: ${e?.message || e}`; }
}
