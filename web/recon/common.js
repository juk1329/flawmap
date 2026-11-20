// Shared pieces of the candidate renderers on the comparison page: texture loading and the
// device-frame conventions (metres; x = width, y = length, +z = screen; GLB UVs top-left).
import * as THREE from 'three';

const loader = new THREE.TextureLoader();

export function loadTex(url, { srgb = true, aniso = 8 } = {}) {
  return new Promise((res, rej) => loader.load(url, (t) => {
    t.flipY = false;
    t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    t.anisotropy = aniso;
    res(t);
  }, undefined, rej));
}

export async function loadJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}

export function firstMesh(root) {
  let mesh = null;
  root.traverse((o) => { if (o.isMesh && !mesh) mesh = o; });
  if (mesh && !mesh.geometry.attributes.normal) mesh.geometry.computeVertexNormals();
  return mesh;
}
