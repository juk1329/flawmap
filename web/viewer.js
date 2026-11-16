// flawmap viewer: the reference model with the capture's appearance, numbered defect pins, the
// evidence frames for each defect, a 360 view made of the real keyframes, and coverage.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const $ = (id) => document.getElementById(id);
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const PART_KO = { screen: '액정', back: '후면', frame: '프레임', camera: '카메라', lens: '카메라 렌즈', button: '버튼' };
const CONF_KO = { high: '높음', medium: '보통', low: '낮음' };

function fmtTime(s) {
  const m = Math.floor(s / 60);
  const r = Math.floor(s % 60);
  return `${m}:${String(r).padStart(2, '0')}`;
}

// Served by the inspection service, the page carries its configuration in window.FLAWMAP:
// {bundle, role: owner|viewer, back, inspection, share_url, published, listed}.
const CFG = window.FLAWMAP || null;

async function pickBundle() {
  if (CFG?.bundle) return CFG.bundle;
  const q = new URLSearchParams(location.search);
  if (q.get('bundle')) return q.get('bundle').replace(/\/?$/, '/');
  const res = await fetch('/api/bundles');
  const list = await res.json();
  if (q.get('run')) {
    const hit = list.find((b) => b.name === q.get('run'));
    if (hit) return hit.url;
  }
  return list.length ? list[0].url : null;
}

const base = await pickBundle();
if (!base) {
  $('status3d').textContent = '처리된 결과가 없습니다. flawmap run으로 촬영본을 먼저 처리하세요.';
  throw new Error('no bundle');
}
const data = await (await fetch(base + 'defects.json')).json();
const defects = data.defects;
let selected = null;

// ------------------------------------------------------------------ header, summary, coverage
$('title').textContent = `${data.device.name} 상태 확인`;
document.title = `${data.device.name} 상태 확인 | flawmap`;
const cov = data.coverage;
const unseenParts = Object.entries(cov.by_part).filter(([, v]) => v < 0.5).map(([k]) => PART_KO[k] || k);
$('facts').innerHTML =
  `<span class="${defects.length ? 'bad' : ''}">하자 <b>${defects.length}곳</b></span>` +
  `<span>촬영된 면 <b>${Math.round(cov.overall * 100)}%</b></span>`;
$('summary').innerHTML = defects.length
  ? `<p class="verdict">하자 ${defects.length}곳을 찾았습니다.</p><p>빨간 번호를 누르면 그 하자가 찍힌 실제 장면을 볼 수 있습니다.</p>`
  : `<p class="verdict">찾은 하자가 없습니다.</p><p>촬영된 면(${Math.round(cov.overall * 100)}%) 안에서 확인한 결과입니다.</p>`;
if (unseenParts.length) {
  $('summary').innerHTML += `<p>촬영되지 않아 확인하지 못한 곳: ${unseenParts.join(', ')}</p>`;
}
if (data.processing?.anchor === 'template') {
  $('summary').innerHTML += `<p class="note">원본 도면 없이 촬영본에서 잰 판 모양 모델로 확인한 결과입니다. 카메라 돌출부와 버튼 위치를 몰라 그 근처는 덜 정확합니다.${data.processing.frame_checked === false ? ' 테두리(프레임)의 찍힘·까짐은 이 모델로 확인하지 않습니다.' : ''}${data.processing.scale ? ' 크기는 기기 폭을 71.5 mm로 가정한 추정값이고, 위아래 방향은 정하지 못합니다.' : ''}</p>`;
}

// ------------------------------------------------------------------ inspection record (service only)
const MODE_KO = {
  board: '촬영 보드로 위치 측정', boardless: '보드 없이 기기 외곽으로 위치 측정', mixed: '보드와 기기 외곽으로 위치 측정', sfm: '공개 데이터에 딸린 SfM 카메라 (크기는 가정)',
  cad: '기종 도면 모델', 'cad-estimated': '치수로 추정한 기종의 도면 모델', template: '촬영본에서 맞춘 판형 모델 (추정)',
};
const escHTML = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
if (CFG?.inspection) {
  const i = CFG.inspection;
  const pr = i.summary?.processing || {};
  const t = i.finished ? new Date(i.finished * 1000) : null;
  const p2 = (x) => String(x).padStart(2, '0');
  const rows = [
    ['검수 번호', i.id],
    ...(t ? [['처리 일시', `${t.getFullYear()}-${p2(t.getMonth() + 1)}-${p2(t.getDate())} ${p2(t.getHours())}:${p2(t.getMinutes())}`]] : []),
    ...(i.requester ? [['요청자', i.requester]] : []),
    ...i.passes.map((p) => [p.face_up === 'back' ? '뒷면 위 촬영' : '화면 위 촬영',
      `${p.public ? `공개 영상 uCO3D ${p.public} (${p.license || 'CC BY 4.0'}), ` : ''}${p.sha256 ? `영상 SHA-256 ${p.sha256}…` : '올린 영상'}`]),
    ...(pr.pose_mode ? [['위치 측정', MODE_KO[pr.pose_mode] || pr.pose_mode]] : []),
    ...(pr.anchor ? [['기기 모델', MODE_KO[pr.anchor] || pr.anchor]] : []),
    ...(pr.device_estimate ? [['기종 추정', `잰 크기 ${pr.device_estimate.size_mm.map((x) => x.toFixed(1)).join(' × ')} mm, ${pr.device_estimate.matched_name ? `${pr.device_estimate.matched_name} 도면과 ${pr.device_estimate.tolerance_mm} mm 안에서 일치` : '맞는 도면 없음'}`]] : []),
    ...(pr.version ? [['프로그램 버전', pr.version]] : []),
  ];
  const who = CFG.role === 'owner'
    ? `검수를 요청한 사람에게만 보이는 화면입니다. ${CFG.published ? (CFG.listed ? '링크로 공유하고 공개 목록에도 올렸습니다.' : '링크로 공유 중입니다.') : '아직 공유하지 않았습니다.'}`
    : '요청자가 올린 촬영 영상을 검수 서버가 처리한 결과입니다. 요청자는 이 결과를 고칠 수 없습니다.';
  $('svc').hidden = false;
  $('svc').innerHTML =
    (i.superseded_by ? '<p class="svc-warn">요청자가 다시 찍어 올려 새 검수로 대체된 결과입니다.</p>' : '') +
    `<p class="svc-who">${who}</p>` +
    (i.title ? `<p class="svc-title">${escHTML(i.title)}</p>` : '') +
    `<dl class="facts-grid">${rows.map(([k, v]) => `<dt>${escHTML(k)}</dt><dd>${escHTML(v)}</dd>`).join('')}</dl>` +
    (CFG.back ? `<p class="svc-back"><a href="${escHTML(CFG.back)}">${CFG.role === 'owner' ? '내 검수 화면으로 돌아가기' : (CFG.back === '/gallery' ? '공개된 결과 목록으로' : 'flawmap 처음 화면으로')}</a></p>` : '');
}

const covOrder = ['screen', 'back', 'frame', 'camera', 'lens', 'button'];
$('coverage').innerHTML = covOrder.filter((k) => k in cov.by_part).map((k) => {
  const v = Math.round(cov.by_part[k] * 100);
  return `<div class="cov-row"><span>${PART_KO[k]}</span><div class="cov-bar"><i style="width:${v}%"></i></div><span>${v}%</span></div>`;
}).join('') + `<p class="cov-note">두 장 이상의 장면에서 정면에 가깝게(입사각 60° 이내) 보인 면적의 비율입니다. 빗금은 확인하지 못한 곳입니다.</p>`;

// ------------------------------------------------------------------ defect list and detail
const list = $('defectList');
if (!defects.length) list.innerHTML = '<li><p class="empty">표시할 하자가 없습니다.</p></li>';
for (const d of defects) {
  const li = document.createElement('li');
  li.innerHTML = `<button data-id="${d.id}"><span class="num">${d.id}</span><span><span class="dname">${d.label_ko}</span><span class="dwhere">${d.location_ko}</span></span></button>`;
  li.querySelector('button').addEventListener('click', () => select(d.id, true));
  list.appendChild(li);
}

function sizeText(d) {
  const L = d.size.length_mm;
  if (d.type === 'chip') return `지름 약 ${Math.max(L, Math.sqrt(d.size.area_mm2)).toFixed(1)} mm`;
  if (d.type === 'dent') {
    const o = d.channels?.outline_mm;
    return `테두리를 따라 약 ${L.toFixed(0)} mm` + (o ? `, 윤곽이 약 ${o.toFixed(1)} mm 안쪽으로 들어감` : '');
  }
  return `길이 약 ${L.toFixed(0)} mm`;
}

function evidenceText(d) {
  const c = d.channels || {};
  const parts = [];
  if (c.outline_mm) parts.push('기기 윤곽의 들어간 모양');
  if (c.appearance) parts.push(d.type === 'crack' || d.type === 'scratch' ? '유리 위의 가는 선' : '도장이 벗겨진 색');
  return parts.join(', ');
}

function showDetail(d) {
  $('detail').hidden = false;
  $('dTitle').innerHTML = `<span class="num" style="display:inline-grid;margin-right:8px;vertical-align:-6px">${d.id}</span>${d.label_ko}`;
  $('dFacts').innerHTML =
    `<dt>위치</dt><dd>${d.location_ko}</dd>` +
    `<dt>부위</dt><dd>${d.part_ko}</dd>` +
    `<dt>크기</dt><dd>${sizeText(d)}</dd>` +
    (evidenceText(d) ? `<dt>찾은 근거</dt><dd>${evidenceText(d)}</dd>` : '') +
    `<dt>확실한 정도</dt><dd>${CONF_KO[d.confidence] || d.confidence} (본 장면 ${d.n_views}개 중 ${Math.round(d.persistence * 100)}%에서 보임)</dd>`;
  const ev = $('dEvidence');
  ev.innerHTML = '';
  d.evidence.forEach((e, k) => {
    const fig = document.createElement('figure');
    const passKo = e.pass === 'back' ? '뒷면 위 촬영' : '화면 위 촬영';
    fig.innerHTML =
      `<button class="shot" aria-label="장면 ${k + 1} 크게 보기"><img src="${base + e.crop}" alt="${d.label_ko}이 보이는 장면 ${k + 1}"></button>` +
      `<figcaption><span>${passKo}, ${fmtTime(e.time_s)}</span><button class="play">영상 ${fmtTime(e.time_s)}부터 보기</button></figcaption>`;
    fig.querySelector('.shot').addEventListener('click', () => openModal(d, e));
    fig.querySelector('.play').addEventListener('click', () => playAt(e.pass, e.time_s));
    ev.appendChild(fig);
  });
}

function select(id, moveCamera) {
  selected = id;
  for (const b of list.querySelectorAll('button[data-id]')) b.setAttribute('aria-current', String(Number(b.dataset.id) === id));
  for (const p of pinEls) p.el.classList.toggle('selected', p.d.id === id);
  const d = defects.find((x) => x.id === id);
  if (!d) return;
  showDetail(d);
  if (moveCamera && mode === 'model') flyTo(d);
  if (mode === 'spin') jumpSpinTo(d);
  drawSpinOverlay();
}

// ------------------------------------------------------------------ modal and video
let modalEv = null;
function openModal(d, e) {
  modalEv = e;
  $('modalImg').src = base + e.full;
  $('modalImg').alt = `${d.label_ko}이 보이는 전체 장면`;
  $('modalCap').textContent = `${d.id}번 ${d.label_ko}, ${e.pass === 'back' ? '뒷면 위 촬영' : '화면 위 촬영'} ${fmtTime(e.time_s)}. 빨간 사각형 안에 하자가 있습니다.`;
  $('modal').hidden = false;
  $('modalClose').focus();
}
function closeModal() { $('modal').hidden = true; }
$('modalClose').addEventListener('click', closeModal);
$('modalPlay').addEventListener('click', () => { closeModal(); if (modalEv) playAt(modalEv.pass, modalEv.time_s); });
$('modal').addEventListener('click', (ev) => { if (ev.target === $('modal')) closeModal(); });

function playAt(pass, t) {
  const name = data.capture.videos?.[pass];
  if (!name) return;
  const v = $('video');
  const src = base + name;
  if (!v.src.endsWith(src)) v.src = src;
  $('player').hidden = false;
  const go = () => { v.currentTime = Math.max(0, t - 0.5); v.play().catch(() => {}); };
  if (v.readyState >= 1) go(); else v.addEventListener('loadedmetadata', go, { once: true });
}
$('playerClose').addEventListener('click', () => { $('video').pause(); $('player').hidden = true; });
document.addEventListener('keydown', (ev) => {
  if (ev.key === 'Escape') { closeModal(); }
  if (mode === 'spin' && (ev.key === 'ArrowLeft' || ev.key === 'ArrowRight')) stepSpin(ev.key === 'ArrowRight' ? 1 : -1);
});

// ------------------------------------------------------------------ 3D view
const canvas = $('c3d');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(26, 1, 0.005, 5);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = !reduceMotion;
controls.dampingFactor = 0.12;
controls.minDistance = 0.08;
controls.maxDistance = 0.9;
scene.add(new THREE.HemisphereLight(0xffffff, 0x9aa0a8, 1.25));
const headlight = new THREE.DirectionalLight(0xffffff, 1.4);
scene.add(headlight);

const passes = data.capture.passes.map((p) => p.face_up);
const startFace = passes.includes('back') && !passes.includes('screen') ? 'back' : 'screen';
setView(startFace, false);

const tl = new THREE.TextureLoader();
function loadTex(url) {
  return new Promise((res, rej) => tl.load(url, (t) => { t.flipY = false; t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = renderer.capabilities.getMaxAnisotropy(); res(t); }, undefined, rej));
}
const [gltf, tex, ovl] = await Promise.all([
  new GLTFLoader().loadAsync(base + data.files.model), loadTex(base + data.files.texture), loadTex(base + data.files.overlay),
]);
let mesh = null;
gltf.scene.traverse((o) => { if (o.isMesh && !mesh) mesh = o; });
// the exported glTF carries positions and UVs only; lighting needs normals
if (!mesh.geometry.attributes.normal) mesh.geometry.computeVertexNormals();
mesh.material = new THREE.MeshStandardMaterial({ map: tex, roughness: 0.55, metalness: 0.05, side: THREE.DoubleSide });
scene.add(mesh);
const overlay = new THREE.Mesh(mesh.geometry, new THREE.MeshBasicMaterial({
  map: ovl, transparent: true, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2, side: THREE.DoubleSide,
}));
overlay.position.copy(mesh.position); overlay.quaternion.copy(mesh.quaternion); overlay.scale.copy(mesh.scale);
scene.add(overlay);
$('showDefects').addEventListener('change', (e) => { overlay.visible = e.target.checked; });
$('status3d').textContent = '';

// defect-id lookup in the atlas (R channel = id)
const idImg = new Image();
idImg.src = base + data.files.idmap;
await idImg.decode();
const idCanvas = new OffscreenCanvas(idImg.width, idImg.height);
const idCtx = idCanvas.getContext('2d', { willReadFrequently: true });
idCtx.drawImage(idImg, 0, 0);
function idAt(uv) {
  const x = Math.min(idImg.width - 1, Math.max(0, Math.floor(uv.x * idImg.width)));
  const y = Math.min(idImg.height - 1, Math.max(0, Math.floor(uv.y * idImg.height)));
  return idCtx.getImageData(x, y, 1, 1).data[0];
}

// pins
const pinEls = defects.map((d) => {
  const el = document.createElement('button');
  el.className = 'pin';
  el.textContent = d.id;
  el.setAttribute('aria-label', `${d.id}번 ${d.label_ko}, ${d.location_ko}`);
  el.addEventListener('click', () => select(d.id, true));
  $('pins').appendChild(el);
  const n = new THREE.Vector3(...d.normal).normalize();
  const p = new THREE.Vector3(...d.centroid_mm).multiplyScalar(0.001).addScaledVector(n, 0.0012);
  return { d, el, p, n };
});

// click on the surface picks the defect painted there
const ray = new THREE.Raycaster();
let downAt = null;
canvas.addEventListener('pointerdown', (e) => { downAt = [e.clientX, e.clientY]; });
canvas.addEventListener('pointerup', (e) => {
  if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 4) return;
  const r = canvas.getBoundingClientRect();
  ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
  const hit = ray.intersectObject(mesh, false)[0];
  if (hit && hit.uv) {
    const id = idAt(hit.uv);
    if (id > 0) select(id, false);
  }
});

function setView(face, animate = true) {
  const dirs = { screen: [0.0, -0.18, 1], back: [0.0, -0.18, -1], side: [1, -0.25, 0.35] };
  const dir = new THREE.Vector3(...dirs[face]).normalize();
  tween(new THREE.Vector3(0, 0, 0), dir.multiplyScalar(0.36), animate);
}
$('viewFront').addEventListener('click', () => setView('screen'));
$('viewBack').addEventListener('click', () => setView('back'));
$('viewSide').addEventListener('click', () => setView('side'));

function flyTo(d) {
  const p = new THREE.Vector3(...d.centroid_mm).multiplyScalar(0.001);
  const n = new THREE.Vector3(...d.normal).normalize();
  // look at the defect from its normal, tilted a little so the frame band keeps some context
  const off = n.clone().multiplyScalar(0.16).add(new THREE.Vector3(0, -0.03, Math.sign(n.z || 1) * 0.03));
  tween(p, p.clone().add(off), true);
}

let anim = null;
function tween(target, pos, animate) {
  if (!animate || reduceMotion) {
    controls.target.copy(target); camera.position.copy(pos); camera.lookAt(target); controls.update(); return;
  }
  anim = { t0: performance.now(), dur: 650, fromT: controls.target.clone(), toT: target.clone(), fromP: camera.position.clone(), toP: pos.clone() };
}

const tmp = new THREE.Vector3();
function resize() {
  const r = canvas.getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / Math.max(r.height, 1);
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

function frame(now) {
  if (anim) {
    const k = Math.min(1, (now - anim.t0) / anim.dur);
    const e = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
    controls.target.lerpVectors(anim.fromT, anim.toT, e);
    camera.position.lerpVectors(anim.fromP, anim.toP, e);
    if (k >= 1) anim = null;
  }
  controls.update();
  headlight.position.copy(camera.position);
  renderer.render(scene, camera);
  const r = canvas.getBoundingClientRect();
  for (const pin of pinEls) {
    tmp.copy(pin.p).project(camera);
    const facing = pin.n.dot(camera.position.clone().sub(pin.p)) > 0;
    pin.el.style.left = `${(tmp.x * 0.5 + 0.5) * r.width}px`;
    pin.el.style.top = `${(-tmp.y * 0.5 + 0.5) * r.height}px`;
    pin.el.classList.toggle('behind', !facing);
    pin.el.style.display = tmp.z < 1 ? '' : 'none';
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// ------------------------------------------------------------------ 360 view of real keyframes
let mode = 'model';
const spin = data.spin?.passes || [];
let sp = { pass: 0, band: 0, idx: 0 };
const spinImg = $('spinimg');
const spinOvl = $('spinovl');

function spinBand() { return spin[sp.pass]?.bands[sp.band]; }
const passLabel = (n) => (n === 'back' ? '뒷면 위 촬영' : '화면 위 촬영');
if (spin.length > 1) {
  spin.forEach((p, pi) => {
    const b = document.createElement('button');
    b.textContent = passLabel(p.name);
    b.setAttribute('aria-pressed', String(pi === 0));
    b.addEventListener('click', () => { sp = { pass: pi, band: Math.min(1, p.bands.length - 1), idx: 0 }; showSpin(); });
    $('spinPasses').appendChild(b);
  });
}
function showSpin() {
  const band = spinBand();
  if (!band) { $('spinmeta').textContent = '실제 영상 장면이 없습니다.'; return; }
  sp.idx = (sp.idx + band.frames.length) % band.frames.length;
  const f = band.frames[sp.idx];
  [...$('spinPasses').children].forEach((b, pi) => b.setAttribute('aria-pressed', String(pi === sp.pass)));
  spinImg.src = base + f.img;
  spinImg.alt = `실제 촬영 장면, ${spin[sp.pass].name === 'back' ? '뒷면 위' : '화면 위'}, 둘레 ${Math.round(f.az)}°, 높이 ${Math.round(f.el)}°`;
  $('spinmeta').textContent = `${spin[sp.pass].name === 'back' ? '뒷면 위 촬영' : '화면 위 촬영'}, 둘레 ${Math.round(f.az)}°, 높이 약 ${Math.round(f.el)}°, 영상 ${fmtTime(f.time_s)}`;
}
spinImg.addEventListener('load', drawSpinOverlay);
function drawSpinOverlay() {
  const band = spinBand();
  if (!band || mode !== 'spin') return;
  const f = band.frames[sp.idx];
  const w = spinImg.clientWidth, h = spinImg.clientHeight;
  spinOvl.width = w * devicePixelRatio; spinOvl.height = h * devicePixelRatio;
  spinOvl.style.width = `${w}px`; spinOvl.style.height = `${h}px`;
  spinOvl.style.left = `${spinImg.offsetLeft}px`; spinOvl.style.top = `${spinImg.offsetTop}px`;
  const ctx = spinOvl.getContext('2d');
  ctx.setTransform(devicePixelRatio * w / spinImg.naturalWidth, 0, 0, devicePixelRatio * h / spinImg.naturalHeight, 0, 0);
  ctx.clearRect(0, 0, spinImg.naturalWidth, spinImg.naturalHeight);
  for (const o of f.outlines) {
    const sel = o.id === selected;
    ctx.lineWidth = sel ? 4 : 2.5;
    ctx.strokeStyle = '#e5251e';
    ctx.fillStyle = sel ? 'rgba(229,37,30,0.18)' : 'rgba(229,37,30,0.0)';
    for (const poly of o.polys) {
      ctx.beginPath();
      poly.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.closePath(); ctx.fill(); ctx.stroke();
    }
    const [x0, y0] = o.polys[0][0];
    ctx.fillStyle = '#e5251e'; ctx.beginPath(); ctx.arc(x0 + 14, y0 - 14, 12, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#fff'; ctx.font = '600 14px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(String(o.id), x0 + 14, y0 - 13);
  }
}
function stepSpin(k) { sp.idx += k; showSpin(); }
function jumpSpinTo(d) {
  let best = null;
  spin.forEach((p, pi) => p.bands.forEach((b, bi) => b.frames.forEach((f, fi) => {
    const o = f.outlines.find((x) => x.id === d.id);
    if (!o) return;
    const area = o.polys.reduce((a, poly) => a + poly.length, 0);
    if (!best || area > best.area) best = { pi, bi, fi, area };
  })));
  if (best) { sp = { pass: best.pi, band: best.bi, idx: best.fi }; showSpin(); }
}
let drag = null;
const sf = document.querySelector('.spinframe');
sf.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY, moved: false }; sf.setPointerCapture(e.pointerId); });
sf.addEventListener('pointermove', (e) => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  if (Math.abs(dx) > 14) { stepSpin(dx > 0 ? -1 : 1); drag.x = e.clientX; drag.moved = true; }
  if (Math.abs(dy) > 50) {
    const bands = spin[sp.pass]?.bands || [];
    const nb = Math.max(0, Math.min(bands.length - 1, sp.band + (dy > 0 ? -1 : 1)));
    if (nb !== sp.band) { const az = spinBand().frames[sp.idx].az; sp.band = nb; sp.idx = nearestAz(az); showSpin(); }
    drag.y = e.clientY; drag.moved = true;
  }
});
sf.addEventListener('pointerup', (e) => {
  if (drag && !drag.moved) pickSpin(e);
  drag = null;
});
function nearestAz(az) {
  const fr = spinBand().frames;
  let k = 0, bd = 1e9;
  fr.forEach((f, i) => { const d = Math.abs(((f.az - az + 540) % 360) - 180); if (d < bd) { bd = d; k = i; } });
  return k;
}
function pickSpin(e) {
  const band = spinBand();
  if (!band) return;
  const r = spinImg.getBoundingClientRect();
  const x = (e.clientX - r.left) * spinImg.naturalWidth / r.width;
  const y = (e.clientY - r.top) * spinImg.naturalHeight / r.height;
  const ctx = spinOvl.getContext('2d');
  for (const o of band.frames[sp.idx].outlines) {
    for (const poly of o.polys) {
      const path = new Path2D();
      poly.forEach(([px, py], k) => (k ? path.lineTo(px, py) : path.moveTo(px, py)));
      path.closePath();
      ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0);
      const inside = ctx.isPointInPath(path, x, y) || ctx.isPointInStroke(path, x, y);
      ctx.restore();
      if (inside) { select(o.id, false); return; }
    }
  }
}

for (const b of document.querySelectorAll('.modes button')) {
  b.addEventListener('click', () => {
    mode = b.dataset.mode;
    for (const x of document.querySelectorAll('.modes button')) x.setAttribute('aria-selected', String(x === b));
    $('view3d').hidden = mode !== 'model';
    $('viewspin').hidden = mode !== 'spin';
    document.querySelector('.tools .seg').style.visibility = mode === 'model' ? '' : 'hidden';
    if (mode === 'spin') { if (selected) jumpSpinTo(defects.find((d) => d.id === selected)); showSpin(); }
    else resize();
  });
}
window.addEventListener('resize', drawSpinOverlay);

// URL options for demos and screenshots: ?select=<id>&mode=spin
{
  const q = new URLSearchParams(location.search);
  if (q.get('mode') === 'spin') document.querySelector('.modes button[data-mode="spin"]').click();
  if (q.get('select')) select(Number(q.get('select')), true);
}
