// Shared helpers for the service pages: fetching, formatting and the inspection tag.
export const $ = (id) => document.getElementById(id);

export async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw Object.assign(new Error(`HTTP ${r.status}`), { status: r.status });
  return r.json();
}

export const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

export function when(ts) {
  if (!ts) return '';
  const d = new Date(ts * 1000);
  const p = (x) => String(x).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

export function duration(s) {
  if (s == null) return '';
  s = Math.max(0, Math.round(s));
  if (s < 60) return `${s}초`;
  const m = Math.floor(s / 60);
  return s % 60 ? `${m}분 ${s % 60}초` : `${m}분`;
}

export const TYPE_KO = { chip: '프레임 까짐', dent: '프레임 찍힘', crack: '유리 깨짐', scratch: '흠집', spot: '점 자국', design: '설계·인쇄·얼룩(추정)' };
export const PART_KO = { screen: '액정', back: '후면', frame: '프레임', camera: '카메라', lens: '카메라 렌즈', button: '버튼' };
export const FACE_KO = { screen: '화면이 위로 온 촬영', back: '뒷면이 위로 온 촬영' };
export const MODE_KO = {
  board: '촬영 보드로 위치 측정', boardless: '보드 없이 기기 외곽으로 위치 측정', mixed: '보드와 기기 외곽으로 위치 측정', sfm: '공개 데이터에 딸린 SfM 카메라 (크기는 가정)',
  cad: '기종 도면 모델', 'cad-estimated': '치수로 추정한 기종의 도면 모델', template: '촬영본에서 맞춘 판형 모델 (추정)',
};

// The inspection tag: defect count (red only when there are defects), coverage with the unseen
// part hatched, and the inspection number.
export function tagHTML(insp, { href = null, big = false } = {}) {
  const s = insp.summary || {};
  const n = s.n_defects ?? 0;
  const cov = Math.round((s.coverage ?? 0) * 100);
  const dn = s.device_name || insp.device_name || '';
  const head = `<span>flawmap 3D 검수</span><span>${esc(dn.startsWith('추정 판형') ? '휴대폰 (도면 없음)' : dn)}</span>`;
  const count = n ? `하자 ${n}곳` : '찾은 하자 없음';
  const body =
    `<div class="t-head">${head}</div>` +
    `<div class="t-count${n ? ' bad' : ''}">${count}</div>` +
    `<div class="t-cov"><div class="covbar" role="img" aria-label="촬영된 면 ${cov}%"><i style="width:${cov}%"></i></div><span>촬영된 면 ${cov}%</span></div>` +
    `<div class="t-id"><span>검수 번호 ${esc(insp.id)}</span>${insp.finished ? `<span>${when(insp.finished)}</span>` : ''}</div>`;
  const cls = `tag${big ? ' big' : ''}`;
  return href ? `<a class="${cls}" href="${esc(href)}" aria-label="검수 결과 보기: ${count}, 촬영된 면 ${cov}%">${body}</a>` : `<div class="${cls}">${body}</div>`;
}

export function topbar(current) {
  const links = [['/new', '검수 요청'], ['/gallery', '검수 결과']];
  return `<div class="wrap"><a class="brand" href="/"><span class="mark" aria-hidden="true"></span>flawmap</a>` +
    `<nav>${links.map(([h, t]) => `<a href="${h}"${h === current ? ' aria-current="page"' : ''}>${t}</a>`).join('')}</nav></div>`;
}
