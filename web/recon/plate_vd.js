// C1 and C3 in the browser: the plate mesh with the fitted atlas, and on the filmed face a
// view-dependent term added in linear light (mirrors flawmap/recon/candidates/plate_fit.py):
//   C3 (mesh-shtex): per-texel SH (degree 2) of the view direction;
//   C1 (mesh-srm):   s(u) x L(mirror direction), L a 64 x 32 equirectangular map in the device frame.
// Code values are decoded and re-encoded with BT.709, like the Python renderer.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { firstMesh, loadTex } from './common.js';

export async function halfTex(url, w, h) {
  const buf = await (await fetch(url)).arrayBuffer();
  const t = new THREE.DataTexture(new Uint16Array(buf), w, h, THREE.RGBAFormat, THREE.HalfFloatType);
  t.minFilter = THREE.LinearFilter; t.magFilter = THREE.LinearFilter; t.generateMipmaps = false;
  t.wrapS = THREE.ClampToEdgeWrapping; t.wrapT = THREE.ClampToEdgeWrapping;
  t.needsUpdate = true;
  return t;
}

const vert = /* glsl */`
varying vec2 vUv; varying vec3 vPos;
void main() { vUv = uv; vec4 wp = modelMatrix * vec4(position, 1.0); vPos = wp.xyz; gl_Position = projectionMatrix * viewMatrix * wp; }`;

export const common = /* glsl */`
uniform sampler2D atlas;
uniform vec4 faceRect; uniform float faceRes; uniform vec2 faceSize; uniform vec2 atlasSize; uniform vec3 faceN;
uniform float vdOn;
varying vec2 vUv; varying vec3 vPos;
vec3 toLin(vec3 c) { return mix(c / 4.5, pow((c + 0.099) / 1.099, vec3(1.0 / 0.45)), step(0.081, c)); }
vec3 toCode(vec3 l) { l = max(l, 0.0); return clamp(mix(4.5 * l, 1.099 * pow(l, vec3(0.45)) - 0.099, step(0.018, l)), 0.0, 1.0); }
bool onFace(out vec2 t) {
  vec2 px = vUv * atlasSize;
  t = (px - faceRect.xy) * faceRes / faceSize;
  return px.x >= faceRect.x && px.y >= faceRect.y && px.x < faceRect.x + faceRect.z && px.y < faceRect.y + faceRect.w;
}`;

const fragSH = /* glsl */`
${common}
uniform sampler2D shTex;
vec4 tile(int j, vec2 t) { return texture2D(shTex, vec2((float(j) + clamp(t.x, 0.001, 0.999)) / 7.0, t.y)); }
void main() {
  vec3 base = toLin(texture2D(atlas, vUv).rgb);
  vec2 t;
  if (vdOn > 0.5 && onFace(t)) {
    vec3 d = normalize(cameraPosition - vPos);
    float x = d.x, y = d.y, z = d.z;
    float Y[9];
    Y[0] = 0.282095; Y[1] = 0.488603 * y; Y[2] = 0.488603 * z; Y[3] = 0.488603 * x; Y[4] = 1.092548 * x * y;
    Y[5] = 1.092548 * y * z; Y[6] = 0.315392 * (3.0 * z * z - 1.0); Y[7] = 1.092548 * x * z; Y[8] = 0.546274 * (x * x - y * y);
    float c[28];
    for (int j = 0; j < 7; j++) { vec4 v = tile(j, t); c[4 * j] = v.x; c[4 * j + 1] = v.y; c[4 * j + 2] = v.z; c[4 * j + 3] = v.w; }
    vec3 r = vec3(0.0);
    for (int k = 0; k < 9; k++) r += Y[k] * vec3(c[3 * k], c[3 * k + 1], c[3 * k + 2]);
    base += r;
  }
  gl_FragColor = vec4(toCode(base), 1.0);
}`;

// the environment is read where the mirror ray meets a sphere of envR mm about the device centre
// (a finite room; envR 0: the mirror direction itself), as envmap.proxy_dirs does
export const envLookup = /* glsl */`
uniform float envR;
vec2 envUV(vec3 pmm, vec3 w) {
  if (envR > 0.0) {
    float pw = dot(pmm, w);
    float t = -pw + sqrt(max(pw * pw - dot(pmm, pmm) + envR * envR, 1e-6));
    w = normalize(pmm + t * w);
  }
  return vec2(atan(w.y, w.x) / 6.2831853 + 0.5, acos(clamp(w.z, -1.0, 1.0)) / 3.1415927);
}`;

const fragSRM = /* glsl */`
${common}
${envLookup}
uniform sampler2D sTex; uniform sampler2D envTex;
void main() {
  vec3 base = toLin(texture2D(atlas, vUv).rgb);
  vec2 t;
  if (vdOn > 0.5 && onFace(t)) {
    vec3 d = normalize(cameraPosition - vPos);
    vec3 w = 2.0 * dot(d, faceN) * faceN - d;
    base += texture2D(sTex, t).r * texture2D(envTex, envUV(vPos * 1000.0, w)).rgb;
  }
  gl_FragColor = vec4(toCode(base), 1.0);
}`;

export async function create({ base, manifest, scene }) {
  const f = manifest.files, face = manifest.face;
  const [gltf, atlas] = await Promise.all([new GLTFLoader().loadAsync(base + f.model), loadTex(base + f.texture, { srgb: false })]);
  const mesh = firstMesh(gltf.scene);
  const uniforms = {
    atlas: { value: atlas }, faceRect: { value: new THREE.Vector4(...face.rect) }, faceRes: { value: face.res_mm },
    faceSize: { value: new THREE.Vector2(...face.size_mm) }, atlasSize: { value: new THREE.Vector2(...face.atlas) },
    faceN: { value: new THREE.Vector3(...face.normal) }, vdOn: { value: 1 },
  };
  let frag;
  if (manifest.renderer === 'mesh-shtex') {
    uniforms.shTex = { value: await halfTex(base + manifest.sh.file, manifest.sh.w, manifest.sh.h) };
    frag = fragSH;
  } else {
    const s = manifest.srm;
    const [sT, eT] = await Promise.all([halfTex(base + s.s, s.s_w, s.s_h), halfTex(base + s.env, s.env_w, s.env_h)]);
    eT.wrapS = THREE.RepeatWrapping;
    uniforms.sTex = { value: sT }; uniforms.envTex = { value: eT }; uniforms.envR = { value: s.room_radius_mm || 0 };
    frag = fragSRM;
  }
  mesh.material = new THREE.ShaderMaterial({ uniforms, vertexShader: vert, fragmentShader: frag, side: THREE.DoubleSide });
  scene.add(mesh);
  return { update() {}, dispose() { mesh.geometry.dispose(); mesh.material.dispose(); atlas.dispose(); } };
}
