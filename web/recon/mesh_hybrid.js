// C7: C1's measured reflection everywhere, plus the real frames' residuals against C1 near the
// directions they were taken from (flawmap/recon/candidates/c7_hybrid.py):
//   out = C1(eye) + share (r_blend - (1 - beta) (C1(eye) - C1_blend)),   share = (1 - fade) cover,
// with the residual r_k = (stored - 0.5) / 0.5 of the views bound and blended as C2 does
// (mesh_ulr.js: ulrViews, ulrBlend: every bound view weighted by angle, continuous as the eye turns)
// and C1_blend C1 as seen from each blended view (ULR_EXTRA);
// beta was chosen on the validation frames. The views hold signed residuals, so they are read as
// stored (no sRGB decoding).
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { firstMesh, loadTex } from './common.js';
import { common, envLookup, halfTex } from './plate_vd.js';
import { ulrGLSL, ulrUniforms, ulrViews, vert } from './mesh_ulr.js';

const frag = /* glsl */`
#define ULR_EXTRA
${common}
${envLookup}
uniform sampler2D sTex; uniform sampler2D envTex;
uniform float rOff; uniform float rScale; uniform float beta;
uniform float onlyFace;     // 1: draw this face only (a second face's layer over the first, web/viewer.js)
vec3 gBase; float gS;
${ulrGLSL}
// C1 at this point as seen from bound view k
vec3 ulrExtra(int k) {
  if (beta >= 1.0) return vec3(0.0);
  vec3 dk = normalize(vc[k] - vPos);
  vec3 wk = 2.0 * dot(dk, faceN) * faceN - dk;
  return toCode(gBase + gS * texture2D(envTex, envUV(vPos * 1000.0, wk)).rgb);
}
void main() {
  vec3 lin = toLin(texture2D(atlas, vUv).rgb);
  vec2 t;
  bool face = onFace(t);
  if (onlyFace > 0.5 && !face) discard;
  vec3 d = normalize(cameraPosition - vPos);
  gBase = lin; gS = 0.0;
  if (vdOn > 0.5 && face) {
    gS = texture2D(sTex, t).r;
    vec3 w = 2.0 * dot(d, faceN) * faceN - d;
    lin += gS * texture2D(envTex, envUV(vPos * 1000.0, w)).rgb;
  }
  vec3 col = toCode(lin);
  if (ulrOn > 0.5 && face) {
    vec3 c1v;
    vec4 b = ulrBlend(t * faceSize / viewSizeMM, d, c1v);
    vec3 corr = beta < 1.0 ? (1.0 - beta) * (col - c1v) : vec3(0.0);
    col = clamp(col + b.a * ((b.rgb - rOff) / rScale - corr), 0.0, 1.0);
  }
  gl_FragColor = vec4(col, 1.0);
}`;

// onlyFace: draw the filmed face only; depthBias > 0 draws it in front of a coplanar mesh (the
// viewer lays one face's layer over another's, or over the inspection model).
export async function create({ base, manifest, scene, onlyFace = false, depthBias = 0, onLoad = null }) {
  const f = manifest.files, face = manifest.face, s = manifest.srm, u = manifest.ulr;
  const [gltf, atlas, sT, eT] = await Promise.all([new GLTFLoader().loadAsync(base + f.model), loadTex(base + f.texture, { srgb: false }),
    halfTex(base + s.s, s.s_w, s.s_h), halfTex(base + s.env, s.env_w, s.env_h)]);
  eT.wrapS = THREE.RepeatWrapping;
  const mesh = firstMesh(gltf.scene);
  const { uniforms: ulrU, blank } = ulrUniforms(u);
  const r = u.residual || { offset: 0.5, scale: 0.5 };
  const uniforms = {
    ...ulrU,
    atlas: { value: atlas }, faceRect: { value: new THREE.Vector4(...face.rect) }, faceRes: { value: face.res_mm },
    faceSize: { value: new THREE.Vector2(...face.size_mm) }, atlasSize: { value: new THREE.Vector2(...face.atlas) },
    faceN: { value: new THREE.Vector3(...face.normal) }, vdOn: { value: 1 },
    sTex: { value: sT }, envTex: { value: eT }, envR: { value: s.room_radius_mm || 0 },
    rOff: { value: r.offset }, rScale: { value: r.scale }, beta: { value: u.beta ?? 1.0 }, onlyFace: { value: onlyFace ? 1 : 0 },
  };
  mesh.material = new THREE.ShaderMaterial({ uniforms, vertexShader: vert, fragmentShader: frag, side: THREE.DoubleSide,
    polygonOffset: depthBias > 0, polygonOffsetFactor: -depthBias, polygonOffsetUnits: -depthBias });
  scene.add(mesh);
  const views = ulrViews(u, base, uniforms, blank, { srgb: false, onLoad });
  return {
    mesh,
    update(camera) { views.update(camera); },
    dispose() { mesh.geometry.dispose(); mesh.material.dispose(); atlas.dispose(); sT.dispose(); eT.dispose(); views.dispose(); },
  };
}
