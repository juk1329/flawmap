"""GPU ray casting with NVIDIA Warp (``warp-lang``, already in env_isaaclab).

The CPU Embree path (``raycast.RaycastScene``) takes ~1 s per 1920x1440 frame and the server's CPUs
are mostly busy with other jobs, so full-resolution casts (synthetic captures, appearance maps) go
through the GPU when CUDA is available. Results use the same conventions as the Embree path:
barycentric ``(b1, b2)`` with point = (1 - b1 - b2) v0 + b1 v1 + b2 v2.
"""
from __future__ import annotations

import numpy as np
import warp as wp

wp.config.quiet = True


@wp.kernel
def _cast_kernel(mesh_id: wp.uint64, origins: wp.array(dtype=wp.vec3), dirs: wp.array(dtype=wp.vec3),
                 t_out: wp.array(dtype=wp.float32), face_out: wp.array(dtype=wp.int32),
                 u_out: wp.array(dtype=wp.float32), v_out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    q = wp.mesh_query_ray(mesh_id, origins[i], dirs[i], 1.0e7)
    if q.result:
        t_out[i] = q.t
        face_out[i] = q.face
        u_out[i] = q.u
        v_out[i] = q.v
    else:
        t_out[i] = -1.0
        face_out[i] = -1
        u_out[i] = 0.0
        v_out[i] = 0.0


class WarpCaster:
    """Static triangle mesh on the GPU; ``cast`` returns hit distance, face and barycentrics."""

    def __init__(self, vertices_world: np.ndarray, faces: np.ndarray, device: str = "cuda"):
        wp.init()
        self.device = device
        self._v = wp.array(np.asarray(vertices_world, np.float32), dtype=wp.vec3, device=device)
        self._f = wp.array(np.asarray(faces, np.int32).ravel(), dtype=wp.int32, device=device)
        self.mesh = wp.Mesh(points=self._v, indices=self._f)

    def cast(self, origins: np.ndarray, dirs: np.ndarray, chunk: int = 4_000_000):
        n = len(dirs)
        t = np.empty(n, np.float32)
        face = np.empty(n, np.int32)
        b = np.empty((n, 2), np.float32)
        origins = np.broadcast_to(np.asarray(origins, np.float32), (n, 3))
        dirs = np.asarray(dirs, np.float32)
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            m = e - s
            o_d = wp.array(np.ascontiguousarray(origins[s:e]), dtype=wp.vec3, device=self.device)
            d_d = wp.array(np.ascontiguousarray(dirs[s:e]), dtype=wp.vec3, device=self.device)
            t_d = wp.empty(m, dtype=wp.float32, device=self.device)
            f_d = wp.empty(m, dtype=wp.int32, device=self.device)
            u_d = wp.empty(m, dtype=wp.float32, device=self.device)
            v_d = wp.empty(m, dtype=wp.float32, device=self.device)
            wp.launch(_cast_kernel, dim=m, inputs=[self.mesh.id, o_d, d_d, t_d, f_d, u_d, v_d], device=self.device)
            t[s:e] = t_d.numpy()
            face[s:e] = f_d.numpy()
            # warp: p = u v0 + v v1 + (1 - u - v) v2  ->  Open3D-style (b1, b2) = weights of v1, v2
            u, v = u_d.numpy(), v_d.numpy()
            b[s:e, 0] = v
            b[s:e, 1] = 1.0 - u - v
        hit = face >= 0
        return np.where(hit, t, np.inf), face, b, hit
