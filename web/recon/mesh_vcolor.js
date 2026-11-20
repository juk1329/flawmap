// Meshes that carry their colour per vertex or in their own texture (C6, the RGB-D scan), unlit.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

export async function create({ base, manifest, scene }) {
  const gltf = await new GLTFLoader().loadAsync(base + manifest.files.model);
  const meshes = [];
  gltf.scene.traverse((o) => {
    if (!o.isMesh) return;
    const hasColor = !!o.geometry.attributes.color;
    const map = o.material?.map || null;
    o.material = new THREE.MeshBasicMaterial({ vertexColors: hasColor, map, side: THREE.DoubleSide });
    meshes.push(o);
  });
  scene.add(gltf.scene);
  return { update() {}, dispose() { meshes.forEach((m) => { m.geometry.dispose(); m.material.dispose(); }); } };
}
