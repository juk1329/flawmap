"""C6, an RGB-D scan mesh: what phone LiDAR scanning apps do. The LiDAR depth of the training frames
is fused into a truncated signed distance volume (Curless & Levoy 1996; KinectFusion, Newcombe et al.
2011) with the frames' cameras, the zero level set is meshed, cropped to the device box, and coloured
by Zhou & Koltun's colour map optimisation (2014; Open3D, MIT) from 960 x 720 keyframes.

Geometry is whatever the 256 x 192 LiDAR saw: on a 5 mm slab of glass and aluminium it is a lumpy
sheet (the point of this baseline). Colour is per vertex (1.5 mm voxels), so it is softer than the
texture-based candidates; scanning apps bake a texture instead, which this baseline does not.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ...utils.logging import get_logger
from ..context import View
from .base import Candidate

log = get_logger("flawmap.recon")
VOXEL = 0.0015
TRUNC = 0.006


class C6TSDF(Candidate):
    id = "c6"
    label_ko = "RGB-D 스캔 메시 (LiDAR TSDF + 색 지도 최적화)"
    family = "free"
    renderer = "mesh-vcolor"
    research = ["KinectFusion TSDF (Newcombe 2011)", "색 지도 최적화 (Zhou·Koltun 2014, Open3D)"]

    def fit(self) -> dict:
        import open3d as o3d

        ctx = self.ctx
        frames = ctx.train_frames[::2]
        # a cube round the device (ScalableTSDFVolume returned nothing in Open3D 0.20 here, even for a
        # synthetic plane; the uniform volume works)
        side = 0.35
        res = int(np.ceil(side / VOXEL))
        vol = o3d.pipelines.integration.UniformTSDFVolume(length=side, resolution=res, sdf_trunc=TRUNC,
                                                          color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
                                                          origin=np.array([-side / 2] * 3))
        used = 0
        for i in frames:
            d, conf = ctx.depth(i)
            if d is None:
                continue
            h, w = d.shape
            img = ctx.image(i)
            col = np.ascontiguousarray(_resize(img, (w, h)))
            occ = _resize(ctx.occluded(i).astype(np.uint8), (w, h), nearest=True) > 0      # a stand or a hand (context.occluded)
            dep = np.where((conf >= 2) & ~occ, d, 0).astype(np.uint16)
            cam = ctx.cap.camera(i)
            s = w / cam.width
            intr = o3d.camera.PinholeCameraIntrinsic(w, h, cam.fx * s, cam.fy * s, (cam.cx + 0.5) * s - 0.5, (cam.cy + 0.5) * s - 0.5)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(col), o3d.geometry.Image(dep), depth_scale=1000.0,
                                                                      depth_trunc=1.0, convert_rgb_to_intensity=False)
            vol.integrate(rgbd, intr, _extrinsic(ctx.T_cad_cam[i]))
            used += 1
        mesh = vol.extract_triangle_mesh()
        s = ctx.model.spec
        h = np.array([s.width / 2, s.length / 2, max(abs(ctx.model.z_front), abs(ctx.model.z_back))]) * 1e-3 + 0.01
        mesh = mesh.crop(o3d.geometry.AxisAlignedBoundingBox(-h, h))
        mesh.remove_unreferenced_vertices()
        n_before = len(mesh.triangles)
        # colour map optimisation from about 30 keyframes at the working scale
        kf = ctx.train_frames[:: max(1, len(ctx.train_frames) // 30)]
        rgbds, traj = [], o3d.camera.PinholeCameraTrajectory()
        params = []
        for i in kf:
            v = ctx.view(i)
            img = np.ascontiguousarray(ctx.image(i))
            d, conf = ctx.depth(i)
            if d is None:
                continue
            dep = _resize(np.where(conf >= 2, d, 0).astype(np.float32), (v.W, v.H), nearest=True)
            dep = np.where(ctx.occluded(i), 0, dep).astype(np.uint16)
            rgbds.append(o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(img), o3d.geometry.Image(dep), depth_scale=1000.0,
                                                                            depth_trunc=1.0, convert_rgb_to_intensity=False))
            p = o3d.camera.PinholeCameraParameters()
            p.intrinsic = o3d.camera.PinholeCameraIntrinsic(v.W, v.H, v.K[0, 0], v.K[1, 1], v.K[0, 2], v.K[1, 2])
            p.extrinsic = _extrinsic(v.T_cad_cam)
            params.append(p)
        traj.parameters = params
        try:
            opt = o3d.pipelines.color_map.RigidOptimizerOption(maximum_iteration=60)
            mesh, _ = o3d.pipelines.color_map.run_rigid_optimizer(mesh, rgbds, traj, opt)
            cmo = True
        except Exception as e:     # pragma: no cover - keep the TSDF colours
            log.warning(f"colour map optimisation failed: {e}")
            cmo = False
        o3d.io.write_triangle_mesh(str(self.art / "mesh.ply"), mesh)
        _write_glb(self.art / "model.glb", np.asarray(mesh.vertices), np.asarray(mesh.triangles), np.asarray(mesh.vertex_colors))
        return {"frames_fused": used, "triangles": len(mesh.triangles), "triangles_before_crop": n_before, "color_map_optimised": cmo}

    def load(self) -> None:
        import open3d as o3d

        m = o3d.io.read_triangle_mesh(str(self.art / "mesh.ply"))
        self.V = np.asarray(m.vertices) * 1e3                     # mm, like the views
        self.F = np.asarray(m.triangles)
        self.Cv = np.asarray(m.vertex_colors).astype(np.float32)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(self.V.astype(np.float32)), o3d.core.Tensor(self.F.astype(np.uint32)))

    def render(self, view: View):
        import open3d as o3d

        cam = view.cam
        us, vs = cam.pixel_grid(1)
        d = cam.ray_dirs_cam(us.ravel(), vs.ravel())
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
        T = np.asarray(view.T_cad_cam, float)
        dw = d @ T[:3, :3].T
        o = np.broadcast_to(T[:3, 3], dw.shape)
        rays = o3d.core.Tensor(np.concatenate([o, dw], 1).astype(np.float32))
        r = self.scene.cast_rays(rays)
        prim = r["primitive_ids"].numpy().astype(np.int64)
        uv = r["primitive_uvs"].numpy()
        hit = prim != o3d.t.geometry.RaycastingScene.INVALID_ID
        f = self.F[np.where(hit, prim, 0)]
        b1, b2 = uv[:, 0], uv[:, 1]
        col = (1 - b1 - b2)[:, None] * self.Cv[f[:, 0]] + b1[:, None] * self.Cv[f[:, 1]] + b2[:, None] * self.Cv[f[:, 2]]
        col[~hit] = 0
        return col.reshape(view.H, view.W, 3).astype(np.float32), hit.reshape(view.H, view.W).astype(np.float32)

    def export_web(self, web: Path) -> dict:
        import shutil

        web.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.art / "model.glb", web / "model.glb")
        return self.write_manifest(web, {"files": {"model": "model.glb"}, "triangles": self.fit_stats().get("triangles")})


def _extrinsic(T_cad_cam_mm: np.ndarray) -> np.ndarray:
    T = np.asarray(T_cad_cam_mm, float).copy()
    T[:3, 3] *= 1e-3
    return np.linalg.inv(T)


def _resize(img: np.ndarray, size: tuple[int, int], nearest: bool = False) -> np.ndarray:
    import cv2

    return cv2.resize(img, size, interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)


def _write_glb(path: Path, V: np.ndarray, F: np.ndarray, C: np.ndarray) -> None:
    import trimesh

    # glTF vertex colours are linear; the frames' colours are sRGB
    c = np.clip(C, 0, 1)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = trimesh.Trimesh(vertices=V, faces=F, vertex_colors=(lin * 255).round().astype(np.uint8), process=False)
    m.export(str(path))
