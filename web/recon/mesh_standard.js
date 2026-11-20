// C0: today's look, the same material and lights as the result viewer (viewer.js): the inspection
// texture on the fitted plate, a synthetic room for reflections, a fill light and a headlight.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { firstMesh, loadTex } from './common.js';

export async function create({ base, manifest, renderer, scene }) {
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x9aa0a8, 0.7));
  const head = new THREE.DirectionalLight(0xffffff, 0.9);
  scene.add(head);
  const f = manifest.files;
  const [gltf, tex, rough] = await Promise.all([
    new GLTFLoader().loadAsync(base + f.model), loadTex(base + f.texture),
    f.roughness ? loadTex(base + f.roughness, { srgb: false }).catch(() => null) : Promise.resolve(null),
  ]);
  const mesh = firstMesh(gltf.scene);
  mesh.material = new THREE.MeshStandardMaterial({
    map: tex, roughness: rough ? 1.0 : 0.55, roughnessMap: rough || null, metalness: 0.0, envMapIntensity: 0.9, side: THREE.DoubleSide,
  });
  scene.add(mesh);
  return {
    update(camera) { head.position.copy(camera.position); head.target.position.set(0, 0, 0); head.target.updateMatrixWorld(); },
    dispose() { mesh.geometry.dispose(); mesh.material.dispose(); tex.dispose(); rough?.dispose(); },
  };
}
