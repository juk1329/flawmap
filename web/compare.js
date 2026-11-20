// Candidate comparison: every reconstruction candidate of one recording on its own mat, all turned by
// one camera. Reads runs/compare/<rec>/index.json (written by `flawmap compare`).
//   ?rec=<recording>        which recording
//   &cands=c0,c1,c7         only these candidates, in this order (a set to rate together)
//   &blind=1                start blind: the set shuffled over the panels under letters, the scores
//                           hidden until the ratings are saved (then each letter's candidate is shown)
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const RENDERERS = {
  'mesh-standard': () => import('./recon/mesh_standard.js'),
  'mesh-ulr': () => import('./recon/mesh_ulr.js'),
  'mesh-vcolor': () => import('./recon/mesh_vcolor.js'),
  splat: () => import('./recon/splat.js'),
  'splat-mesh': () => import('./recon/splat_mesh.js'),
  'mesh-srm': () => import('./recon/plate_vd.js'),
  'mesh-shtex': () => import('./recon/plate_vd.js'),
  'mesh-hybrid': () => import('./recon/mesh_hybrid.js'),
  'splat-srm': () => import('./recon/splat_srm.js'),
};
const status = (t) => { $('status').textContent = t; };

// ------------------------------------------------------------------ which recording
const params = new URLSearchParams(location.search);
let recs = [];
try { recs = await (await fetch('/runs/compare/recs.json', { cache: 'no-store' })).json(); } catch { /* none yet */ }
const rec = params.get('rec') || recs[0]?.rec;
for (const r of recs) $('rec').append(new Option(`${r.rec} (${r.face === 'back' ? '뒷면' : '화면'}, 후보 ${r.n})`, r.rec));
if (rec && !recs.some((r) => r.rec === rec)) $('rec').append(new Option(rec, rec));
$('rec').value = rec || '';
// switching recordings keeps the other options (the candidate set, blind)
$('rec').addEventListener('change', () => { params.set('rec', $('rec').value); location.search = params.toString(); });
if (!rec) { status('비교 결과가 아직 없습니다. flawmap compare를 먼저 실행하세요.'); throw new Error('no recording'); }
const ROOT = `/runs/compare/${rec}/`;
const index = await (await fetch(ROOT + 'index.json', { cache: 'no-store' })).json();
const allCands = index.candidates || [];
const only = (params.get('cands') || '').split(',').map((s) => s.trim().toLowerCase()).filter(Boolean);
const picked = only.map((id) => allCands.find((c) => c.id === id)).filter(Boolean);
const cands = picked.length ? picked : allCands;

// ------------------------------------------------------------------ one camera for every panel
const dev = index.device;
const L = dev.length_mm * 1e-3, W = dev.width_mm * 1e-3;
const N = new THREE.Vector3(0, 0, index.face === 'back' ? -1 : 1);        // the filmed face's normal: "up"
const X = new THREE.Vector3(1, 0, 0), Y = new THREE.Vector3(0, 1, 0);
const camera = new THREE.PerspectiveCamera(32, 4 / 3, 0.005, 5);
camera.up.copy(N);
const controls = new OrbitControls(camera, $('stage'));
controls.enableDamping = !matchMedia('(prefers-reduced-motion: reduce)').matches;
controls.dampingFactor = 0.12;
controls.minDistance = 0.04;
controls.maxDistance = 1.5;
const fitDist = (0.5 * Math.hypot(L, W)) / Math.tan(THREE.MathUtils.degToRad(32) / 2) * 1.05;
function place(elDeg, azDeg, d = fitDist, target = new THREE.Vector3()) {
  leaveTestMode();
  const e = THREE.MathUtils.degToRad(elDeg), a = THREE.MathUtils.degToRad(azDeg);
  const dir = X.clone().multiplyScalar(Math.cos(e) * Math.cos(a)).addScaledVector(Y, Math.cos(e) * Math.sin(a)).addScaledVector(N, Math.sin(e));
  controls.target.copy(target);
  camera.position.copy(target).addScaledVector(dir, d);
  camera.lookAt(target);
  window.__flawmapCompareDirty?.();
}

// ------------------------------------------------------------------ test-frame cameras (OpenCV K and pose)
let testMode = null;
function setTestView(tv) {
  const T = tv.T_cad_cam, K = tv.K, w = tv.W, h = tv.H, n = 0.005, f = 5;
  const m = new THREE.Matrix4().set(
    T[0][0], -T[0][1], -T[0][2], T[0][3] * 1e-3,
    T[1][0], -T[1][1], -T[1][2], T[1][3] * 1e-3,
    T[2][0], -T[2][1], -T[2][2], T[2][3] * 1e-3,
    0, 0, 0, 1);
  camera.position.setFromMatrixPosition(m);
  camera.quaternion.setFromRotationMatrix(m);
  camera.updateMatrixWorld(true);
  const P = camera.projectionMatrix;
  P.set(2 * K[0][0] / w, 0, 1 - 2 * (K[0][2] + 0.5) / w, 0,
    0, 2 * K[1][1] / h, 2 * (K[1][2] + 0.5) / h - 1, 0,
    0, 0, -(f + n) / (f - n), -2 * f * n / (f - n),
    0, 0, -1, 0);
  camera.projectionMatrixInverse.copy(P).invert();
  // keep the orbit centre in front of the camera so the next drag turns about the device
  const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
  controls.target.copy(camera.position).addScaledVector(fwd, camera.position.length());
  testMode = tv;
  for (const p of panels) p.real.src = ROOT + tv.image;
  updateReal();
  window.__flawmapCompareDirty?.();
}
function leaveTestMode() {
  if (!testMode) return;
  testMode = null;
  camera.updateProjectionMatrix();
  $('testframe').value = '';
  updateReal();
}
controls.addEventListener('start', () => { if (testMode) { const p = camera.position.clone(); leaveTestMode(); camera.position.copy(p); } });

// ------------------------------------------------------------------ panels
const panels = [];
let blind = false;
let revealed = false;   // blind ratings saved: each letter's candidate and the scores are shown
let assignment = [];
const LETTERS = 'ABCDEF';

function pinObjects() {
  const g = new THREE.Group();
  const mat = new THREE.MeshBasicMaterial({ color: 0xe5251e });
  for (const d of index.defects || []) {
    if (!d.centroid_mm) continue;
    const s = new THREE.Mesh(new THREE.SphereGeometry(0.0014, 16, 12), mat);
    const nn = d.normal ? new THREE.Vector3(...d.normal) : N.clone();
    s.position.set(...d.centroid_mm).multiplyScalar(1e-3).addScaledVector(nn, 0.0014);
    g.add(s);
  }
  return g;
}

function makePanel(k) {
  const el = document.createElement('div');
  el.className = 'panel';
  el.innerHTML = `<div class="head"><span class="tag"></span><select aria-label="패널 ${k + 1}의 후보"></select></div>
    <canvas></canvas><img class="real" alt=""><div class="state"></div>`;
  $('stage').append(el);
  const canvas = el.querySelector('canvas');
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  const sel = el.querySelector('select');
  for (const c of cands) sel.append(new Option(`${c.id}: ${c.label}`, c.id));
  const p = { k, el, canvas, renderer, sel, tag: el.querySelector('.tag'), real: el.querySelector('.real'),
    state: el.querySelector('.state'), scene: null, inst: null, cid: null };
  sel.addEventListener('change', () => { load(p, sel.value); renderTables(); });
  return p;
}

async function load(p, cid) {
  p.inst?.dispose();
  p.inst = null;
  p.cid = cid;
  p.scene = new THREE.Scene();
  p.scene.add(pinObjects());
  const c = cands.find((x) => x.id === cid);
  labelPanel(p);
  if (!c) return;
  p.state.textContent = '불러오는 중…';
  try {
    const manifest = await (await fetch(ROOT + c.manifest, { cache: 'no-store' })).json();
    const mod = await RENDERERS[manifest.renderer]();
    const scene = p.scene;
    const inst = await mod.create({ base: ROOT + c.manifest.replace(/manifest\.json$/, ''), manifest, renderer: p.renderer, scene });
    if (p.scene === scene) { p.inst = inst; p.state.textContent = ''; window.__flawmapCompareDirty?.(); }
    else inst.dispose();
  } catch (e) {
    p.state.textContent = `불러오지 못함: ${e.message}`;
  }
}

function labelPanel(p) {
  const c = cands.find((x) => x.id === p.cid);
  p.tag.textContent = blind ? (revealed && c ? `${LETTERS[p.k]} = ${c.id.toUpperCase()}` : LETTERS[p.k]) : (c ? c.id.toUpperCase() : '');
  p.sel.hidden = blind;
  p.sel.value = p.cid || '';
}

function layout(nWanted) {
  // never more panels than candidates (a repeated candidate would be rated twice)
  const n = Math.max(1, Math.min(nWanted, cands.length, LETTERS.length));
  $('stage').className = `grid n${n <= 2 ? 2 : n <= 4 ? 4 : 6}`;
  while (panels.length < n) panels.push(makePanel(panels.length));
  panels.forEach((p, i) => { p.el.hidden = i >= n; });
  const want = blind ? assignment : cands.map((c) => c.id);
  panels.slice(0, n).forEach((p, i) => { if (!p.cid || p.cid !== want[i]) load(p, want[i]); });
  resize();
  renderTables();
}

function resize() {
  for (const p of panels) {
    if (p.el.hidden) continue;
    const w = p.canvas.clientWidth, h = Math.round(w * 3 / 4);
    p.renderer.setSize(w, h, false);
  }
}
addEventListener('resize', resize);

$('npanel').addEventListener('change', () => layout(+$('npanel').value));
function setBlind(on) {
  blind = on;
  revealed = false;
  $('blind').classList.toggle('on', blind);
  $('blind').setAttribute('aria-pressed', String(blind));
  if (blind) {
    const ids = cands.map((c) => c.id);
    for (let i = ids.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [ids[i], ids[j]] = [ids[j], ids[i]]; }
    assignment = ids;
    panels.forEach((p) => { p.cid = null; });
  }
  layout(+$('npanel').value);
}
$('blind').addEventListener('click', () => setBlind(!blind));

// ------------------------------------------------------------------ presets
let spinning = false;
document.querySelectorAll('[data-preset]').forEach((b) => b.addEventListener('click', () => {
  const k = b.dataset.preset;
  if (k === 'home') place(55, -60);
  if (k === 'graze') place(5, -60);
  if (k === 'under') place(-45, -60);
  if (k === 'spin') {
    spinning = !spinning;
    b.classList.toggle('on', spinning);
    b.setAttribute('aria-pressed', String(spinning));
    if (spinning && testMode) leaveTestMode();
    controls.autoRotate = spinning;
    controls.autoRotateSpeed = 2.5;
  }
}));
for (const d of index.defects || []) $('defect').append(new Option(`${d.id}. ${d.label || ''} ${d.where || ''}`.trim(), d.id));
$('defect').addEventListener('change', () => {
  const d = (index.defects || []).find((x) => String(x.id) === $('defect').value);
  if (d?.centroid_mm) place(65, -60, 0.08, new THREE.Vector3(...d.centroid_mm).multiplyScalar(1e-3));
});
(index.test_views || []).forEach((tv, i) => $('testframe').append(new Option(`${i + 1}. 프레임 ${tv.frame}`, i)));
$('testframe').addEventListener('change', () => {
  const i = $('testframe').value;
  if (i === '') leaveTestMode();
  else setTestView(index.test_views[+i]);
});
let spaceDown = false;
function updateReal() {
  const show = testMode && ($('showReal').checked || spaceDown);
  for (const p of panels) p.real.style.opacity = show ? '1' : '0';
}
$('showReal').addEventListener('change', updateReal);
addEventListener('keydown', (e) => { if (e.code === 'Space' && e.target === document.body) { e.preventDefault(); spaceDown = true; updateReal(); } });
addEventListener('keyup', (e) => { if (e.code === 'Space') { spaceDown = false; updateReal(); } });

// ------------------------------------------------------------------ tables: scores and ratings
const COLS = [
  ['psnr_g', '시험 PSNR', 'hi', 2], ['hf_ncc', '세부 상관', 'hi', 3], ['lpips_g', 'LPIPS', 'lo', 3], ['coverage', '덮은 비율', 'hi', 2],
  ['turn30_float', '떠다니는 점', 'lo', 3], ['under_hole', '안 찍힌 면 빈 곳', 'lo', 2], ['flicker', '깜빡임', 'lo', 4],
  ['turn_jump', '회전 튐', 'lo', 2],
  ['seconds', '학습 초', 'lo', 0], ['web_mb', 'MB', 'lo', 1],
];
function shown() {
  const seen = new Set();
  return panels.filter((p) => !p.el.hidden).map((p, i) => ({ letter: LETTERS[i], c: cands.find((x) => x.id === p.cid) }))
    .filter((x) => x.c && !seen.has(x.c.id) && seen.add(x.c.id));
}
// who a row is: the candidate, or in blind mode its letter (and, once the ratings are saved, the candidate)
const who = (letter, c) => (!blind ? `${c.id.toUpperCase()} ${c.label}` : revealed ? `${letter} = ${c.id.toUpperCase()} ${c.label}` : letter);
function renderMetrics() {
  if (blind && !revealed) {
    // the scores would tell the letters apart, so they wait until the ratings are saved
    $('metrics').innerHTML = '<p class="note">블라인드 평가 중에는 지표를 가립니다. 평가를 저장하면 글자마다 어느 후보였는지와 지표가 나옵니다.</p>';
    return;
  }
  const rows = blind ? shown() : cands.map((c) => ({ letter: null, c }));
  const best = {};
  for (const [k, , dir] of COLS) {
    const v = rows.map((r) => r.c.summary?.[k]).filter((x) => x != null);
    if (v.length) best[k] = dir === 'hi' ? Math.max(...v) : Math.min(...v);
  }
  $('metrics').innerHTML = `<table><thead><tr><th>후보</th>${COLS.map((c) => `<th>${c[1]}</th>`).join('')}</tr></thead><tbody>${
    rows.map((r) => `<tr><td>${who(r.letter, r.c)}</td>${COLS.map(([k, , , dp]) => {
      const v = r.c.summary?.[k];
      return `<td class="${v != null && v === best[k] ? 'best' : ''}">${v == null ? '—' : (+v).toFixed(dp)}</td>`;
    }).join('')}</tr>`).join('')}</tbody></table>`;
}
function renderTables() {
  renderMetrics();
  $('ratings').innerHTML = shown().map(({ letter, c }) => `<div class="rate" data-cid="${c.id}" data-letter="${letter}">
    <div class="who">${who(letter, c)}</div>
    ${[['realism', '사실감'], ['stable', '돌려도 깨지지 않음'], ['defects', '하자가 잘 보임']].map(([k, t]) => `<div class="row"><span>${t}</span>${
      [1, 2, 3, 4, 5].map((v) => `<label><input type="radio" name="${letter}-${k}" value="${v}"> ${v}</label>`).join(' ')}</div>`).join('')}
    <textarea placeholder="메모 (예: 360° 돌릴 때 옆면이 깨짐)"></textarea></div>`).join('');
}
$('saveRatings').addEventListener('click', async () => {
  const items = [...document.querySelectorAll('.rate')].map((el) => {
    const pick = (k) => +(el.querySelector(`input[name="${el.dataset.letter}-${k}"]:checked`)?.value || 0) || null;
    return { panel: el.dataset.letter, candidate: el.dataset.cid, realism: pick('realism'), stable: pick('stable'), defects: pick('defects'),
      note: el.querySelector('textarea').value.trim() };
  });
  const payload = { rec, blind, set: cands.map((c) => c.id), time: new Date().toISOString(), ratings: items };
  try {
    const r = await fetch('/api/compare/ratings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    $('savedMsg').textContent = blind ? '저장했습니다. 글자마다 어느 후보였는지 아래와 패널에 나옵니다.' : '저장했습니다.';
    $('ratingsOut').hidden = true;
  } catch {
    try {
      const all = JSON.parse(localStorage.getItem('flawmap-compare-ratings') || '[]');
      all.push(payload);
      localStorage.setItem('flawmap-compare-ratings', JSON.stringify(all));
    } catch { /* storage blocked */ }
    $('savedMsg').textContent = '서버에 저장하지 못해 이 브라우저에 보관했습니다. 아래 내용을 복사해 전달해 주세요.';
    $('ratingsOut').hidden = false;
    $('ratingsOut').textContent = JSON.stringify(payload, null, 1);
  }
  // the ratings are kept either way, so a blind round now shows who was who (the form keeps its answers)
  if (blind && !revealed) {
    revealed = true;
    panels.forEach((p) => { if (!p.el.hidden) labelPanel(p); });
    renderMetrics();
    document.querySelectorAll('.rate').forEach((el) => {
      const c = cands.find((x) => x.id === el.dataset.cid);
      if (c) el.querySelector('.who').textContent = who(el.dataset.letter, c);
    });
  }
});

// ------------------------------------------------------------------ start
// a chosen set opens with enough panels for all of it (at most six)
if (picked.length) $('npanel').value = String(picked.length <= 2 ? 2 : picked.length <= 4 ? 4 : 6);
if (params.get('blind') === '1') setBlind(true);
else layout(+$('npanel').value);
place(55, -60);
status(`${index.rec}: ${index.face === 'back' ? '뒷면' : '화면'} 촬영, 후보 ${cands.length}개${picked.length ? `(전체 ${allCands.length}개 중)` : ''}${blind ? ', 블라인드' : ''}, 시험 프레임 ${index.test_views?.length || 0}장`);
// draw only when something changed (four WebGL canvases at 60 fps are heavy on a laptop): the camera
// moved, a model or texture arrived, the window resized, or the turntable is on; plus a slow refresh
// so textures that finish loading show up
let dirty = 4, lastDraw = 0;
const markDirty = () => { dirty = Math.max(dirty, 2); };
controls.addEventListener('change', markDirty);
addEventListener('resize', markDirty);
window.__flawmapCompareDirty = markDirty;
renderer_loop();
function renderer_loop(now = 0) {
  requestAnimationFrame(renderer_loop);
  if (!testMode) controls.update();
  if (!(dirty > 0 || spinning || now - lastDraw > 500)) return;
  dirty = Math.max(0, dirty - 1);
  lastDraw = now;
  for (const p of panels) {
    if (p.el.hidden || !p.scene) continue;
    p.inst?.update(camera);
    p.renderer.render(p.scene, camera);
  }
}
