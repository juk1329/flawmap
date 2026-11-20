// C4: splats bound to the fitted plate's filmed face, over the plate mesh with the inspection texture
// (unlit, as in training). The mesh hides splats seen from the other side (depth test).
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark';
import { firstMesh, loadTex } from './common.js';

export async function create({ base, manifest, renderer, scene }) {
  const f = manifest.files;
  const [gltf, tex] = await Promise.all([new GLTFLoader().loadAsync(base + f.model), loadTex(base + f.texture)]);
  const mesh = firstMesh(gltf.scene);
  mesh.material = new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 });
  scene.add(mesh);
  const spark = new SparkRenderer({ renderer });
  scene.add(spark);
  const splat = new SplatMesh({ url: base + f.splat });
  await splat.initialized;
  scene.add(splat);
  return {
    update() {},
    dispose() { mesh.geometry.dispose(); mesh.material.dispose(); tex.dispose(); splat.dispose?.(); spark.dispose?.(); },
  };
}
