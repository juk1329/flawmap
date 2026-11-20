// C4b: splats bound to the fitted plate's filmed face over the plate mesh (as C4), then the shared
// reflection s(u) L(lookup) added on the face by a second pass of the mesh with additive blending
// (flawmap/recon/candidates/c4b_plate_gs_reflect.py adds it to the splats' code values the same way).
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark';
import { firstMesh, loadTex } from './common.js';
import { common, envLookup, halfTex } from './plate_vd.js';
import { vert } from './mesh_ulr.js';

const fragSpec = /* glsl */`
${common}
${envLookup}
uniform sampler2D sTex; uniform sampler2D envTex;
void main() {
  vec2 t;
  vec3 d = normalize(cameraPosition - vPos);
  if (vdOn < 0.5 || !onFace(t) || dot(d, faceN) <= 0.0) discard;
  vec3 w = 2.0 * dot(d, faceN) * faceN - d;
  gl_FragColor = vec4(texture2D(sTex, t).r * texture2D(envTex, envUV(vPos * 1000.0, w)).rgb, 1.0);
}`;

export async function create({ base, manifest, renderer, scene }) {
  const f = manifest.files, face = manifest.face, s = manifest.srm;
  const [gltf, tex, sT, eT] = await Promise.all([new GLTFLoader().loadAsync(base + f.model), loadTex(base + f.texture),
    halfTex(base + s.s, s.s_w, s.s_h), halfTex(base + s.env, s.env_w, s.env_h)]);
  eT.wrapS = THREE.RepeatWrapping;
  const mesh = firstMesh(gltf.scene);
  mesh.material = new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 });
  scene.add(mesh);
  const spark = new SparkRenderer({ renderer });
  scene.add(spark);
  const splat = new SplatMesh({ url: base + f.splat });
  await splat.initialized;
  scene.add(splat);
  const uniforms = {
    atlas: { value: tex }, faceRect: { value: new THREE.Vector4(...face.rect) }, faceRes: { value: face.res_mm },
    faceSize: { value: new THREE.Vector2(...face.size_mm) }, atlasSize: { value: new THREE.Vector2(...face.atlas) },
    faceN: { value: new THREE.Vector3(...face.normal) }, vdOn: { value: 1 },
    sTex: { value: sT }, envTex: { value: eT }, envR: { value: s.room_radius_mm || 0 },
  };
  const overlay = new THREE.Mesh(mesh.geometry, new THREE.ShaderMaterial({
    uniforms, vertexShader: vert, fragmentShader: fragSpec, side: THREE.DoubleSide, transparent: true,
    blending: THREE.AdditiveBlending, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1,
  }));
  overlay.renderOrder = 10;
  scene.add(overlay);
  return {
    update() {},
    dispose() {
      mesh.geometry.dispose(); mesh.material.dispose(); overlay.material.dispose(); tex.dispose(); sT.dispose(); eT.dispose();
      splat.dispose?.(); spark.dispose?.();
    },
  };
}
