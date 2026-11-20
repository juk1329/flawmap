// Gaussian-splat candidates (C4, C5): drawn by Spark (MIT, World Labs) inside the panel's three.js
// scene. The splat files are written in the device frame (metres), like the meshes.
import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark';

export async function create({ base, manifest, renderer, scene }) {
  const spark = new SparkRenderer({ renderer });
  scene.add(spark);
  const splat = new SplatMesh({ url: base + manifest.files.splat });
  await splat.initialized;
  scene.add(splat);
  return {
    update() {},
    dispose() { splat.dispose?.(); spark.dispose?.(); },
  };
}
