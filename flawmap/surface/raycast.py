"""Ray casting of the reference model into a camera (adapted from ISCO ``render/raycast_backend.py``).

Every registered frame is turned into per-pixel surface correspondences: which triangle of
the reference model a pixel sees, where on it (barycentric coordinates), how far away, and
at what viewing angle. Everything downstream (masks, coverage, unwrapped appearance maps,
evidence frames) is built from this.

Changes from ISCO: a full intrinsic matrix (``geometry.camera.Camera``) instead of a
field-of-view camera, triangle ids and barycentric coordinates are returned, and the
Embree scene is built once and reused for all frames (ISCO rebuilt it per call). Measured
in ISCO: 1.10 s for a 1920x1440 frame on a 348k-face mesh including scene build; the
quarter-resolution casts used for coverage are ~16x cheaper.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.camera import Camera
from ..geometry.mesh_io import LabeledMesh
from ..geometry.transforms import apply_T

_MAX_RAYS_PER_CALL = 8_000_000


@dataclass
class RayHits:
    """Dense per-pixel results on a (rows, cols) grid of sampled pixels.

    ``face`` is -1 where the ray misses the model. ``u``/``v`` are the pixel coordinates of
    the grid in the full camera, so a quarter-resolution cast can still be compared with
    full-resolution images.
    """
    u: np.ndarray
    v: np.ndarray
    face: np.ndarray
    depth: np.ndarray
    bary: np.ndarray          # (rows, cols, 2) barycentric (b1, b2) of the hit point
    cos: np.ndarray           # |cos| between the ray and the face normal
    point: np.ndarray         # (rows, cols, 3) hit point in the model (object) frame

    @property
    def hit(self) -> np.ndarray:
        return self.face >= 0


class RaycastScene:
    """An Embree scene of one mesh placed at ``T_world_obj``; cast any number of cameras."""

    def __init__(self, mesh: LabeledMesh, T_world_obj: np.ndarray | None = None, n_threads: int = 0, backend: str = "auto"):
        import open3d as o3d

        self.mesh = mesh
        self.T_world_obj = np.eye(4) if T_world_obj is None else np.asarray(T_world_obj, float)
        self.n_threads = n_threads
        self._scene = o3d.t.geometry.RaycastingScene(nthreads=n_threads)
        v_world = apply_T(self.T_world_obj, mesh.vertices).astype(np.float32)
        self._gid = self._scene.add_triangles(o3d.core.Tensor(v_world), o3d.core.Tensor(mesh.faces.astype(np.uint32)))
        tri = mesh.vertices[mesh.faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        self.face_normals_obj = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        self._warp = None
        if backend in ("auto", "warp"):
            try:
                import warp as wp

                wp.init()
                if wp.is_cuda_available():
                    from .raycast_warp import WarpCaster

                    self._warp = WarpCaster(v_world, mesh.faces)
            except Exception:  # pragma: no cover - no GPU / no warp
                if backend == "warp":
                    raise
        self.backend = "warp" if self._warp is not None else "embree"

    def _cast(self, origins: np.ndarray, dirs: np.ndarray):
        if self._warp is not None:
            t, prim, puv, hit = self._warp.cast(origins, dirs)
            return t, prim.astype(np.int64), puv, hit
        import open3d as o3d

        rays = np.concatenate([origins, dirs], axis=1).astype(np.float32)
        t = np.empty(len(rays), np.float32)
        gid = np.empty(len(rays), np.int64)
        prim = np.empty(len(rays), np.int64)
        puv = np.empty((len(rays), 2), np.float32)
        for s in range(0, len(rays), _MAX_RAYS_PER_CALL):
            e = min(s + _MAX_RAYS_PER_CALL, len(rays))
            ans = self._scene.cast_rays(o3d.core.Tensor(rays[s:e]), nthreads=self.n_threads)
            t[s:e] = ans["t_hit"].numpy()
            gid[s:e] = ans["geometry_ids"].numpy().astype(np.int64)
            prim[s:e] = ans["primitive_ids"].numpy().astype(np.int64)
            puv[s:e] = ans["primitive_uvs"].numpy()
        hit = (gid == self._gid) & np.isfinite(t)
        return t, prim, puv, hit

    def cast(self, cam: Camera, T_world_cam: np.ndarray, step: int = 1) -> RayHits:
        """Cast every ``step``-th pixel of ``cam`` placed at ``T_world_cam``."""
        us, vs = cam.pixel_grid(step)
        shape = us.shape
        d_cam = cam.ray_dirs_cam(us.ravel(), vs.ravel())
        norm = np.linalg.norm(d_cam, axis=1)
        d_w = (d_cam / norm[:, None]) @ np.asarray(T_world_cam)[:3, :3].T
        o_w = np.broadcast_to(np.asarray(T_world_cam)[:3, 3], d_w.shape)
        t, prim, puv, hit = self._cast(o_w, d_w)
        face = np.where(hit, prim, -1)
        depth = np.where(hit, t / norm, 0.0)
        # normals in world frame
        n_w = self.face_normals_obj[np.where(hit, prim, 0)] @ self.T_world_obj[:3, :3].T
        cosv = np.where(hit, np.abs(np.einsum("ij,ij->i", n_w, d_w)), 0.0)
        p_w = o_w + d_w * np.where(hit, t, 0.0)[:, None]
        R, tt = self.T_world_obj[:3, :3], self.T_world_obj[:3, 3]
        p_obj = (p_w - tt) @ R
        return RayHits(us, vs, face.reshape(shape), depth.reshape(shape).astype(np.float32),
                       puv.reshape(shape + (2,)), cosv.reshape(shape).astype(np.float32),
                       p_obj.reshape(shape + (3,)))

    def first_hit_distance(self, origins: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        """Ray length to the first hit (inf on a miss); ``dirs`` need not be unit length."""
        dirs = np.asarray(dirs, float)
        nrm = np.linalg.norm(dirs, axis=1)
        t, _, _, hit = self._cast(np.asarray(origins, float), dirs / nrm[:, None])
        return np.where(hit, t, np.inf)

    def closest_points(self, pts_world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Closest points on the mesh (world frame), their distances and face ids."""
        import open3d as o3d

        q = o3d.core.Tensor(np.asarray(pts_world, np.float32))
        ans = self._scene.compute_closest_points(q, nthreads=self.n_threads)
        cp = ans["points"].numpy().astype(float)
        fid = ans["primitive_ids"].numpy().astype(np.int64)
        return cp, np.linalg.norm(cp - np.asarray(pts_world, float), axis=1), fid
