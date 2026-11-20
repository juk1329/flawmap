"""Read a uCO3D sequence (public crowd-sourced 360-degree object videos, CC BY 4.0) as posed frames.

Layout (``data/datasets/uco3d``): ``metadata.sqlite`` (tables ``sequence_annots``, ``frame_annots``)
and per sequence ``<super_category>/<category>/<name>/`` with ``rgb_video.mp4``, ``mask_video.mkv``,
``segmented_point_cloud.ply`` and ``gaussian_splats/``.

Cameras are VGGSfM reconstructions in PyTorch3D conventions, stored as float32 blobs per frame:

* ``_viewpoint_R`` (3x3), ``_viewpoint_T`` (3): ``X_cam = X_world @ R + T`` (row vectors), camera
  axes +x left, +y up, +z forward;
* ``_viewpoint_focal_length``, ``_viewpoint_principal_point`` in ``ndc_isotropic`` units: pixels
  are ``f * s`` and ``(W/2 - p_x * s, H/2 - p_y * s)`` with ``s = min(W, H) / 2``;
* ``_viewpoint_colmap_distortion_coeffs`` (12): only the first (radial k1) is non-zero; frames are
  undistorted with it so the pinhole model holds;
* ``_image_size`` is (H, W).

The scene scale is arbitrary (SfM units); there is no metric depth. Annotated frames carry a
timestamp, and the video frame is ``round(timestamp * fps)``.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..geometry.camera import Camera
from ..geometry.transforms import inv_T, make_T

P3D_TO_CV = np.diag([-1.0, -1.0, 1.0])


@dataclass
class Uco3dSequence:
    name: str
    root: Path
    category: str
    caption: str
    frame_numbers: list[int]
    timestamps: np.ndarray
    K: dict[int, np.ndarray]                  # frame_number -> 3x3 (pixels, undistorted frames)
    T_world_cam: dict[int, np.ndarray]        # OpenCV camera, world in SfM units
    k1: dict[int, float]
    width: int
    height: int
    fps: float
    quality: dict = field(default_factory=dict)

    @property
    def video(self) -> Path:
        return self.root / "rgb_video.mp4"

    def camera(self, fn: int) -> Camera:
        return Camera(self.K[fn], self.width, self.height)

    def video_index(self, fn: int) -> int:
        return int(round(float(self.timestamps[self.frame_numbers.index(fn)]) * self.fps))


def _f32(b, n=None):
    a = np.frombuffer(b, np.float32).astype(np.float64)
    return a if n is None else a[:n]


def find_root(data_root: Path, name: str) -> Path:
    hits = list(Path(data_root).glob(f"*/*/{name}"))
    if not hits:
        raise FileNotFoundError(f"uCO3D sequence {name} not under {data_root}")
    return hits[0]


def load_sequence(data_root: str | Path, name: str) -> Uco3dSequence:
    from .stray import _ffprobe_like

    data_root = Path(data_root)
    root = find_root(data_root, name)
    con = sqlite3.connect(f"file:{data_root / 'metadata.sqlite'}?mode=ro", uri=True)
    try:
        s = con.execute("SELECT category, _caption_text, _reconstruction_quality_viewpoint, "
                        "_reconstruction_quality_sfm_n_registered_cameras FROM sequence_annots WHERE sequence_name=?", (name,)).fetchone()
        rows = con.execute("SELECT frame_number, frame_timestamp, _image_size, _viewpoint_R, _viewpoint_T, _viewpoint_focal_length, "
                           "_viewpoint_principal_point, _viewpoint_intrinsics_format, _viewpoint_colmap_distortion_coeffs "
                           "FROM frame_annots WHERE sequence_name=? ORDER BY frame_number", (name,)).fetchall()
    finally:
        con.close()
    info = _ffprobe_like(root / "rgb_video.mp4")
    fps = 30.0
    import re

    m = re.search(r"(\d+(?:\.\d+)?) fps", info.get("ffmpeg_banner", ""))
    if m:
        fps = float(m.group(1))
    K, T, k1, fns, ts = {}, {}, {}, [], []
    W = H = None
    for fn, t, size, R, Tv, f, pp, fmt, dist in rows:
        if fmt != "ndc_isotropic":
            raise ValueError(f"{name}: unsupported intrinsics format {fmt}")
        H, W = (int(x) for x in np.frombuffer(size, np.int32)[:2])
        sc = min(W, H) / 2.0
        fx, fy = _f32(f, 2) * sc
        px, py = _f32(pp, 2)
        K[fn] = np.array([[fx, 0.0, W / 2.0 - px * sc], [0.0, fy, H / 2.0 - py * sc], [0.0, 0.0, 1.0]])
        Rm = _f32(R).reshape(3, 3)
        T_cam_world = make_T(P3D_TO_CV @ Rm.T, P3D_TO_CV @ _f32(Tv, 3))
        T[fn] = inv_T(T_cam_world)
        k1[fn] = float(_f32(dist)[0]) if dist is not None else 0.0
        fns.append(int(fn))
        ts.append(float(t))
    return Uco3dSequence(name, root, s[0] if s else "", (s[1] or "") if s else "", fns, np.asarray(ts), K, T, k1,
                         int(W), int(H), fps, {"viewpoint": s[2] if s else None, "n_registered": s[3] if s else None})


def undistort(img: np.ndarray, K: np.ndarray, k1: float) -> np.ndarray:
    """Remove the radial k1 distortion (COLMAP SIMPLE_RADIAL / RADIAL convention, normalised
    coordinates) keeping the same K."""
    import cv2

    if abs(k1) < 1e-9:
        return img
    return cv2.undistort(img, K, np.array([k1, 0.0, 0.0, 0.0, 0.0]), None, K)


def read_frames(seq: Uco3dSequence, frame_numbers: list[int], undistorted: bool = True, masks: bool = False) -> dict[int, np.ndarray]:
    """RGB frames (or mask frames, bool) for annotated frame numbers."""
    from .stray import iter_frames

    want = {seq.video_index(fn): fn for fn in frame_numbers}
    src = seq.root / ("mask_video.mkv" if masks else "rgb_video.mp4")
    out = {}
    for vi, img in iter_frames(src, sorted(want)):
        fn = want[vi]
        if masks:
            m = (img[..., 0] if img.ndim == 3 else img) > 127
            out[fn] = undistort(m.astype(np.uint8), seq.K[fn], seq.k1[fn]).astype(bool) if undistorted else m
        else:
            out[fn] = undistort(img, seq.K[fn], seq.k1[fn]) if undistorted else img
    return out


def read_points(seq: Uco3dSequence, segmented: bool = True) -> np.ndarray:
    from plyfile import PlyData

    p = PlyData.read(str(seq.root / ("segmented_point_cloud.ply" if segmented else "point_cloud.ply")))
    v = p.elements[0]
    return np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)


def read_splat_means(seq: Uco3dSequence) -> np.ndarray:
    """The positions of the dataset's own Gaussian-splat fit of the sequence (``gaussian_splats/``,
    the whole scene), in the SfM frame: two 8-bit PNGs (``means_u`` high byte, ``means_l`` low byte,
    RGB = xyz) give 16 bits a coordinate over ``meta.json``'s range, stored as sign(x) log(1 + |x|).
    The log decoding was checked on 2712-37354-8763: its segmented SfM points lie a median 0.009
    units from the foreground splats decoded this way, 0.39 units decoded linearly. Used where the
    full SfM point cloud was not downloaded (``point_cloud.ply``)."""
    import cv2

    d = seq.root / "gaussian_splats"
    meta = json.loads((d / "meta.json").read_text())["means"]
    lo = cv2.imread(str(d / "means_l.png"), cv2.IMREAD_UNCHANGED)[..., ::-1].reshape(-1, 3).astype(np.uint32)
    hi = cv2.imread(str(d / "means_u.png"), cv2.IMREAD_UNCHANGED)[..., ::-1].reshape(-1, 3).astype(np.uint32)
    mins, maxs = np.asarray(meta["mins"], np.float64), np.asarray(meta["maxs"], np.float64)
    m = mins + ((hi << 8) | lo) / 65535.0 * (maxs - mins)
    return (np.sign(m) * np.expm1(np.abs(m)))[: int(meta["shape"][0])]


def mask_agreement(seq: Uco3dSequence, frame_numbers: list[int], max_points: int = 5000) -> float:
    """Share of the segmented point cloud's projections that land inside the object mask (checks
    the camera conversion)."""
    pts = read_points(seq)
    if len(pts) > max_points:
        pts = pts[np.random.default_rng(0).choice(len(pts), max_points, replace=False)]
    ms = read_frames(seq, frame_numbers, masks=True)
    vals = []
    for fn, m in ms.items():
        Tc = inv_T(seq.T_world_cam[fn])
        pc = pts @ Tc[:3, :3].T + Tc[:3, 3]
        ok = pc[:, 2] > 0
        K = seq.K[fn]
        u = K[0, 0] * pc[ok, 0] / pc[ok, 2] + K[0, 2]
        v = K[1, 1] * pc[ok, 1] / pc[ok, 2] + K[1, 2]
        inside = (u >= 0) & (u < m.shape[1] - 1) & (v >= 0) & (v < m.shape[0] - 1)
        if inside.sum() < 50:
            continue
        vals.append(m[v[inside].astype(int), u[inside].astype(int)].mean())
    return float(np.median(vals)) if vals else 0.0
