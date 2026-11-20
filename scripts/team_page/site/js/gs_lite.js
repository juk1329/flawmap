// A small Gaussian-splat renderer for the comparison page: the 3DGS image model (each splat's 3D
// covariance projected to an ellipse on screen, Kerbl et al. 2023) drawn back to front, with no
// workers and no WebAssembly so it runs inside a page with a strict content policy.
// Input: the 32-byte ".splat" records (position f32×3, scale f32×3, RGBA u8, rotation u8×4 as w x y z) and,
// optionally, the view-dependent colour: spherical harmonics above the base colour, one byte a coefficient
// ((v / shScale) * 127 + 128), evaluated per view direction as 3DGS and gsplat do.
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.180.0/+esm';

const ROW = 512;    // splats per texture row (3 texels each: 1536 wide, inside every WebGL2 limit)
const SH_ROW = 128; // splats per row of the harmonics texture (12 RGBA texels each)

const vertexShader = (shRest) => /* glsl */`
#define SH_REST ${shRest}
precision highp float;
precision highp int;
uniform highp sampler2D uData;     // 3 texels a splat: (x y z s00) (s01 s02 s11 s12) (s22 . . .)
uniform highp sampler2D uColor;    // 1 texel a splat: RGBA
uniform vec2 uViewport;            // drawing buffer, px
uniform vec2 uFocal;               // px
uniform highp sampler2D uSH;       // 12 texels a splat: up to 15 coefficients × RGB, one byte each
uniform float uSHScale;
uniform vec3 uCamPos;              // the camera in the splats' frame
in float splatIndex;
out vec4 vColor;
out vec2 vPos;
void main() {
  int i = int(splatIndex + 0.5);
  ivec2 b = ivec2((i % ${ROW}) * 3, i / ${ROW});
  vec4 t0 = texelFetch(uData, b, 0);
  vec4 t1 = texelFetch(uData, b + ivec2(1, 0), 0);
  vec4 t2 = texelFetch(uData, b + ivec2(2, 0), 0);
  vec4 cam = modelViewMatrix * vec4(t0.xyz, 1.0);
  vec4 clip = projectionMatrix * cam;
  float bound = 1.2 * clip.w;
  if (cam.z > -0.01 || clip.x < -bound || clip.x > bound || clip.y < -bound || clip.y > bound) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    return;
  }
  mat3 S = mat3(t0.w, t1.x, t1.y,  t1.x, t1.z, t1.w,  t1.y, t1.w, t2.x);
  mat3 W = mat3(modelViewMatrix);
  mat3 Sv = W * S * transpose(W);
  // pixel coordinates (right, up) of a view-space point: u = -f.x x / z, v = -f.y y / z
  float z = cam.z;
  vec2 lim = 1.3 * 0.5 * uViewport / uFocal;                   // as 3DGS: the slope clamped to 1.3× the view
  vec2 t = clamp(cam.xy / -z, -lim, lim) * -z;
  mat3 J = mat3(-uFocal.x / z, 0.0, 0.0,
                0.0, -uFocal.y / z, 0.0,
                uFocal.x * t.x / (z * z), uFocal.y * t.y / (z * z), 0.0);
  mat3 C = J * Sv * transpose(J);
  float a = C[0][0] + 0.3, d = C[1][1] + 0.3, o = C[0][1];      // + 0.3 px²: the 3DGS low-pass
  float mid = 0.5 * (a + d), rad = length(vec2(0.5 * (a - d), o));
  float l1 = mid + rad, l2 = mid - rad;
  if (l2 <= 0.0) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
  vec2 e1 = abs(o) > 1e-12 ? normalize(vec2(o, l1 - a)) : (a >= d ? vec2(1.0, 0.0) : vec2(0.0, 1.0));
  vec2 e2 = vec2(-e1.y, e1.x);
  vec2 off = position.x * min(3.0 * sqrt(l1), 2048.0) * e1 + position.y * min(3.0 * sqrt(l2), 2048.0) * e2;
  gl_Position = vec4(clip.xy / clip.w + 2.0 * off / uViewport, clip.z / clip.w, 1.0);
  vColor = texelFetch(uColor, ivec2(i % ${ROW}, i / ${ROW}), 0);
#if SH_REST > 0
  float c[48];
  ivec2 sb = ivec2((i % ${SH_ROW}) * 12, i / ${SH_ROW});
  for (int k = 0; k < 12; k++) {
    vec4 v = texelFetch(uSH, sb + ivec2(k, 0), 0);
    c[4 * k] = v.x; c[4 * k + 1] = v.y; c[4 * k + 2] = v.z; c[4 * k + 3] = v.w;
  }
  vec3 sh[15];
  for (int k = 0; k < SH_REST; k++) sh[k] = (vec3(c[3 * k], c[3 * k + 1], c[3 * k + 2]) * 255.0 - 128.0) / 127.0 * uSHScale;
  vec3 dir = normalize(t0.xyz - uCamPos);
  float x = dir.x, y = dir.y, zz = dir.z;
  vec3 rgb = -0.4886025119029199 * y * sh[0] + 0.4886025119029199 * zz * sh[1] - 0.4886025119029199 * x * sh[2];
#if SH_REST > 3
  float xx = x * x, yy = y * y, z2 = zz * zz, xy = x * y, yz = y * zz, xz = x * zz;
  rgb += 1.0925484305920792 * xy * sh[3] - 1.0925484305920792 * yz * sh[4] + 0.31539156525252005 * (2.0 * z2 - xx - yy) * sh[5]
       - 1.0925484305920792 * xz * sh[6] + 0.5462742152960396 * (xx - yy) * sh[7];
#if SH_REST > 8
  rgb += -0.5900435899266435 * y * (3.0 * xx - yy) * sh[8] + 2.890611442640554 * xy * zz * sh[9]
       - 0.4570457994644658 * y * (4.0 * z2 - xx - yy) * sh[10] + 0.3731763325901154 * zz * (2.0 * z2 - 3.0 * xx - 3.0 * yy) * sh[11]
       - 0.4570457994644658 * x * (4.0 * z2 - xx - yy) * sh[12] + 1.445305721320277 * zz * (xx - yy) * sh[13]
       - 0.5900435899266435 * x * (xx - 3.0 * yy) * sh[14];
#endif
#endif
  vColor.rgb = clamp(vColor.rgb + rgb, 0.0, 1.0);
#endif
  vPos = 3.0 * position.xy;
}`;

const fragmentShader = /* glsl */`
precision highp float;
in vec4 vColor;
in vec2 vPos;
out vec4 fragColor;
void main() {
  float r2 = dot(vPos, vPos);
  if (r2 > 9.0) discard;
  float alpha = min(0.999, vColor.a * exp(-0.5 * r2));   // as gsplat, which trained these splats
  if (alpha < 1.0 / 255.0) discard;
  fragColor = vec4(vColor.rgb * alpha, alpha);
}`;

function parse(buf) {
  const n = Math.floor(buf.byteLength / 32);
  const f = new Float32Array(buf, 0, n * 8);
  const u = new Uint8Array(buf);
  const rows = Math.ceil(n / ROW);
  const data = new Float32Array(ROW * 3 * rows * 4);
  const color = new Uint8Array(ROW * rows * 4);
  const centers = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const x = f[8 * i], y = f[8 * i + 1], zz = f[8 * i + 2];
    const sx = f[8 * i + 3], sy = f[8 * i + 4], sz = f[8 * i + 5];
    const o = 32 * i + 28;
    let w = (u[o] - 128) / 128, qx = (u[o + 1] - 128) / 128, qy = (u[o + 2] - 128) / 128, qz = (u[o + 3] - 128) / 128;
    const ql = Math.hypot(w, qx, qy, qz) || 1;
    w /= ql; qx /= ql; qy /= ql; qz /= ql;
    // R (row-major) from the unit quaternion, then M = R diag(s): Σ = M Mᵀ
    const r00 = 1 - 2 * (qy * qy + qz * qz), r01 = 2 * (qx * qy - w * qz), r02 = 2 * (qx * qz + w * qy);
    const r10 = 2 * (qx * qy + w * qz), r11 = 1 - 2 * (qx * qx + qz * qz), r12 = 2 * (qy * qz - w * qx);
    const r20 = 2 * (qx * qz - w * qy), r21 = 2 * (qy * qz + w * qx), r22 = 1 - 2 * (qx * qx + qy * qy);
    const m00 = r00 * sx, m01 = r01 * sy, m02 = r02 * sz;
    const m10 = r10 * sx, m11 = r11 * sy, m12 = r12 * sz;
    const m20 = r20 * sx, m21 = r21 * sy, m22 = r22 * sz;
    const b = ((Math.floor(i / ROW) * ROW * 3) + (i % ROW) * 3) * 4;
    data[b] = x; data[b + 1] = y; data[b + 2] = zz;
    data[b + 3] = m00 * m00 + m01 * m01 + m02 * m02;
    data[b + 4] = m00 * m10 + m01 * m11 + m02 * m12;
    data[b + 5] = m00 * m20 + m01 * m21 + m02 * m22;
    data[b + 6] = m10 * m10 + m11 * m11 + m12 * m12;
    data[b + 7] = m10 * m20 + m11 * m21 + m12 * m22;
    data[b + 8] = m20 * m20 + m21 * m21 + m22 * m22;
    color.set(u.subarray(32 * i + 24, 32 * i + 28), 4 * i);
    centers[3 * i] = x; centers[3 * i + 1] = y; centers[3 * i + 2] = zz;
  }
  return { n, rows, data, color, centers };
}

async function bytes(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.arrayBuffer();
}

// shUrl, shRest (coefficients a colour: 15 for degree 3), shScale: the view-dependent colour (optional)
export async function loadSplat(url, { shUrl = null, shRest = 0, shScale = 1 } = {}) {
  const [buf, shBuf] = await Promise.all([bytes(url), shUrl && shRest ? bytes(shUrl) : null]);
  const { n, rows, data, color, centers } = parse(buf);
  const dataTex = new THREE.DataTexture(data, ROW * 3, rows, THREE.RGBAFormat, THREE.FloatType);
  const colorTex = new THREE.DataTexture(color, ROW, rows, THREE.RGBAFormat, THREE.UnsignedByteType);
  let shTex = null;
  if (shBuf) {
    const shRows = Math.ceil(n / SH_ROW);
    const shData = new Uint8Array(SH_ROW * 12 * shRows * 4);
    const src = new Uint8Array(shBuf);
    const per = shRest * 3;
    for (let i = 0; i < n; i++) {
      const o = ((Math.floor(i / SH_ROW) * SH_ROW * 12) + (i % SH_ROW) * 12) * 4;
      shData.set(src.subarray(i * per, (i + 1) * per), o);
    }
    shTex = new THREE.DataTexture(shData, SH_ROW * 12, shRows, THREE.RGBAFormat, THREE.UnsignedByteType);
  } else {
    shRest = 0;
  }
  for (const t of [dataTex, colorTex, shTex]) {
    if (!t) continue;
    t.minFilter = t.magFilter = THREE.NearestFilter; t.generateMipmaps = false; t.needsUpdate = true;
  }

  const geo = new THREE.InstancedBufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]), 3));
  geo.setIndex([0, 1, 2, 0, 2, 3]);
  const order = new Float32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  const idx = new THREE.InstancedBufferAttribute(order, 1);
  idx.setUsage(THREE.DynamicDrawUsage);
  geo.setAttribute('splatIndex', idx);
  geo.instanceCount = n;

  const uniforms = { uData: { value: dataTex }, uColor: { value: colorTex }, uViewport: { value: new THREE.Vector2(1, 1) },
    uFocal: { value: new THREE.Vector2(1, 1) }, uSH: { value: shTex || colorTex }, uSHScale: { value: shScale }, uCamPos: { value: new THREE.Vector3() } };
  const mat = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3, vertexShader: vertexShader(shRest), fragmentShader, uniforms, transparent: true, depthTest: false, depthWrite: false,
    blending: THREE.CustomBlending, blendEquation: THREE.AddEquation,
    blendSrc: THREE.OneFactor, blendDst: THREE.OneMinusSrcAlphaFactor, blendSrcAlpha: THREE.OneFactor, blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.frustumCulled = false;

  // back-to-front order by view depth: a 16-bit counting sort, redone when the view changes
  const keys = new Uint16Array(n);
  const depth = new Float32Array(n);
  const counts = new Uint32Array(65536);
  const last = new THREE.Matrix4().set(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  const size = new THREE.Vector2();
  function sort(camera) {
    const m = camera.matrixWorldInverse.elements;
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < n; i++) {
      const d = -(m[2] * centers[3 * i] + m[6] * centers[3 * i + 1] + m[10] * centers[3 * i + 2] + m[14]);
      depth[i] = d;
      if (d < lo) lo = d;
      if (d > hi) hi = d;
    }
    const s = 65535 / Math.max(hi - lo, 1e-9);
    counts.fill(0);
    for (let i = 0; i < n; i++) { const k = ((depth[i] - lo) * s) | 0; keys[i] = k; counts[k]++; }
    let acc = 0;
    for (let k = 65535; k >= 0; k--) { const c = counts[k]; counts[k] = acc; acc += c; }   // far first
    for (let i = 0; i < n; i++) order[counts[keys[i]]++] = i;
    idx.needsUpdate = true;
  }
  return {
    mesh, count: n,
    update(camera, renderer) {
      renderer.getDrawingBufferSize(size);
      uniforms.uViewport.value.copy(size);
      const P = camera.projectionMatrix.elements;
      uniforms.uFocal.value.set(P[0] * size.x / 2, P[5] * size.y / 2);
      uniforms.uCamPos.value.setFromMatrixPosition(camera.matrixWorld);
      if (!last.equals(camera.matrixWorldInverse)) { sort(camera); last.copy(camera.matrixWorldInverse); }
    },
    dispose() { geo.dispose(); mat.dispose(); dataTex.dispose(); colorTex.dispose(); shTex?.dispose(); },
  };
}
