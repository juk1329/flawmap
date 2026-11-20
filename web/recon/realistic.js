// The realistic view of a result bundle (bundle version 2, flawmap/recon/service.py): one C7 layer per
// filmed face (mesh_hybrid.js, the comparison page's renderer of C7). The first face's layer draws the
// whole model; a second face's layer draws its own face only, over the first. Used by the result
// viewer (web/viewer.js) and the home page's example (web/service/hero3d.js). Throws when the files
// are not there (a shared link without the requester's leave gets 404 for them).
export async function realisticLayers(base, data, scene) {
  if (!data.realistic?.index) return [];
  const get = async (rel) => {
    const r = await fetch(base + rel);
    if (!r.ok) throw new Error(`${rel}: HTTP ${r.status}`);
    return r.json();
  };
  const idx = await get(data.realistic.index);
  const { create } = await import('./mesh_hybrid.js');
  const layers = [];
  try {
    for (const [k, f] of (idx.faces || []).entries()) {
      layers.push(await create({ base: base + f.manifest.replace(/[^/]*$/, ''), manifest: await get(f.manifest), scene,
        onlyFace: k > 0, depthBias: k > 0 ? 1 : 0 }));
    }
  } catch (e) {
    for (const l of layers) { scene.remove(l.mesh); l.dispose(); }
    throw e;
  }
  return layers;
}
