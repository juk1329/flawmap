// The home page's live example: a real result turning slowly on the grey mat, its defects outlined
// in red. It is the same bundle the result page shows (model, surface texture, outline overlay), in
// its realistic view when the bundle has one and the link may show it (web/recon/realistic.js).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

export async function heroModel(canvas, base) {
  const data = await (await fetch(base + 'defects.json')).json();
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  const scene = new THREE.Scene();
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x9aa0a8, 0.7));
  const head = new THREE.DirectionalLight(0xffffff, 0.9);
  scene.add(head);
  const L = data.device.length_mm || 146.7;
  const scale = Math.max(1, L / 146.7);
  const camera = new THREE.PerspectiveCamera(26, 1, 0.005, 5 * scale);
  const controls = new OrbitControls(camera, canvas);
  controls.enableZoom = false;
  controls.enablePan = false;
  controls.enableDamping = !reduce;
  controls.autoRotate = !reduce;
  controls.autoRotateSpeed = 0.9;
  // open on the filmed face that carries more of the defects (the screen when it is a tie)
  const faces = data.capture.passes.map((p) => p.face_up);
  const onFace = (c) => data.defects.filter((d) => d.chart === c).length;
  const z = faces.includes('back') && (!faces.includes('screen') || onFace('back') > onFace('screen')) ? -1 : 1;
  camera.position.set(0, -0.2 * 0.36 * scale, z * 0.36 * scale * 1.05);
  controls.target.set(0, 0, 0);

  const tl = new THREE.TextureLoader();
  const tex = (u) => new Promise((res, rej) => tl.load(base + u, (t) => { t.flipY = false; t.colorSpace = THREE.SRGBColorSpace; res(t); }, undefined, rej));
  const opt = (k) => (data.files[k] ? tex(data.files[k]).catch(() => null) : Promise.resolve(null));
  const [gltf, map, ovl, rough] = await Promise.all([new GLTFLoader().loadAsync(base + data.files.model), tex(data.files.texture),
    tex(data.files.overlay), opt('roughness')]);
  let mesh = null;
  gltf.scene.traverse((o) => { if (o.isMesh && !mesh) mesh = o; });
  if (!mesh.geometry.attributes.normal) mesh.geometry.computeVertexNormals();
  if (rough) rough.colorSpace = THREE.NoColorSpace;
  mesh.material = new THREE.MeshStandardMaterial({ map, roughness: rough ? 1 : 0.55, roughnessMap: rough, metalness: 0, envMapIntensity: 0.9,
    side: THREE.DoubleSide });
  scene.add(mesh);
  const o = new THREE.Mesh(mesh.geometry, new THREE.MeshBasicMaterial({ map: ovl, transparent: true, depthWrite: false, polygonOffset: true,
    polygonOffsetFactor: -2, polygonOffsetUnits: -2, side: THREE.DoubleSide }));
  scene.add(o);
  // the realistic view replaces the surface model once it has loaded; the outlines stay on top
  let layers = [];
  import('/recon/realistic.js').then(({ realisticLayers }) => realisticLayers(base, data, scene))
    .then((ls) => { layers = ls; if (layers.length) mesh.visible = false; })
    .catch(() => { layers = []; });

  function resize() {
    const r = canvas.getBoundingClientRect();
    renderer.setSize(r.width, r.height, false);
    camera.aspect = r.width / Math.max(r.height, 1);
    camera.updateProjectionMatrix();
  }
  window.addEventListener('resize', resize);
  resize();
  renderer.setAnimationLoop(() => {
    controls.update();
    head.position.copy(camera.position);
    for (const l of layers) l.update(camera);
    renderer.render(scene, camera);
  });
  return data;
}
