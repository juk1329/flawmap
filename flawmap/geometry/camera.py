"""Pinhole camera with a full intrinsic matrix (OpenCV convention).

Pixel centres are at integer coordinates: pixel ``(u, v)`` (column, row) has its centre at
``(u, v)`` and the ray through it is ``((u - cx) / fx, (v - cy) / fy, 1)`` in camera
coordinates. Depth is the camera-frame z (distance to the image plane), not ray length.

ISCO's ``PinholeCamera`` derived fx = fy from a field of view and put the principal point
at the image centre; phone frames need the calibrated K (ARKit or ChArUco), so this class
takes K directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Camera:
    K: np.ndarray
    width: int
    height: int
    dist: tuple[float, ...] | None = None   # OpenCV distortion; frames are used undistorted when None

    def __post_init__(self) -> None:
        object.__setattr__(self, "K", np.asarray(self.K, dtype=float).reshape(3, 3))

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    def scaled(self, s: float) -> "Camera":
        """Camera for the image resized by ``s`` (e.g. 0.25 for quarter resolution).

        With integer pixel centres, a continuous coordinate maps as u' = (u + 0.5) s - 0.5.
        """
        K = self.K.copy()
        K[0, 0] *= s
        K[1, 1] *= s
        K[0, 2] = (K[0, 2] + 0.5) * s - 0.5
        K[1, 2] = (K[1, 2] + 0.5) * s - 0.5
        return Camera(K, int(round(self.width * s)), int(round(self.height * s)), self.dist)

    def project(self, pts_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Continuous pixel coordinates (u, v) and depth z of camera-frame points."""
        pts_cam = np.asarray(pts_cam, dtype=float)
        z = pts_cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.fx * pts_cam[:, 0] / z + self.cx
            v = self.fy * pts_cam[:, 1] / z + self.cy
        return u, v, z

    def in_image(self, u: np.ndarray, v: np.ndarray, z: np.ndarray, margin: float = 0.0) -> np.ndarray:
        return (z > 0) & (u >= -0.5 + margin) & (u < self.width - 0.5 - margin) & \
            (v >= -0.5 + margin) & (v < self.height - 0.5 - margin)

    def unproject(self, u: np.ndarray, v: np.ndarray, z: np.ndarray) -> np.ndarray:
        u, v, z = (np.asarray(a, dtype=float) for a in (u, v, z))
        x = (u - self.cx) / self.fx * z
        y = (v - self.cy) / self.fy * z
        return np.stack([x, y, z], axis=-1)

    def pixel_grid(self, step: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Integer pixel coordinates of every ``step``-th pixel, as (rows, cols) grids."""
        vs, us = np.mgrid[0:self.height:step, 0:self.width:step]
        return us, vs

    def ray_dirs_cam(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Camera-frame ray directions (z = 1) through pixel coordinates."""
        u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
        return np.stack([(u - self.cx) / self.fx, (v - self.cy) / self.fy, np.ones_like(u)], axis=-1)

    def to_dict(self) -> dict:
        return {"K": self.K.tolist(), "width": self.width, "height": self.height,
                "dist": list(self.dist) if self.dist is not None else None}

    @classmethod
    def from_dict(cls, d: dict) -> "Camera":
        dist = d.get("dist")
        return cls(np.asarray(d["K"], float), int(d["width"]), int(d["height"]), tuple(dist) if dist else None)


def project_points(cam: Camera, T_world_cam: np.ndarray, pts_world: np.ndarray):
    """Project world points: returns (u, v, z) with z the camera depth."""
    R, t = T_world_cam[:3, :3], T_world_cam[:3, 3]
    pc = (np.asarray(pts_world, float) - t) @ R          # world -> camera (R^T (p - t))
    return cam.project(pc)
