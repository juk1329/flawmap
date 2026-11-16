"""Read a Stray Scanner recording (https://github.com/strayrobots/scanner/blob/main/docs/format.md).

``odometry.csv`` holds, per frame, the ARKit camera pose (camera-to-world, metres; ARKit camera
axes x right, y up, z backwards) and the per-frame intrinsics of ``rgb.mp4``. Poses are converted to
the OpenCV camera convention and millimetres here. The ARKit world frame is gravity-aligned (+Y up)
but its origin and heading are wherever the session started; ``poses.align`` maps it onto the board.

``rgb.mp4`` is decoded with the ffmpeg bundled in imageio-ffmpeg (HEVC capable) **without
auto-rotation**: ARKit intrinsics refer to the sensor (landscape) orientation, and an auto-rotated
frame would not match them. The decoded size is checked against the principal point.
"""
from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..geometry.camera import Camera
from ..geometry.transforms import make_T

ARKIT_TO_CV_CAM = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass
class StrayCapture:
    root: Path
    timestamps: np.ndarray            # (N,) seconds
    frame_ids: list[str]              # "000123"
    T_ark_cam: np.ndarray             # (N,4,4) ARKit world <- OpenCV camera, mm
    K: np.ndarray                     # (N,3,3) per-frame intrinsics
    width: int
    height: int
    has_depth: bool
    distortion_center: np.ndarray | None = None
    info: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frame_ids)

    def camera(self, i: int) -> Camera:
        return Camera(self.K[i], self.width, self.height)

    def depth(self, i: int) -> np.ndarray | None:
        import cv2

        p = self.root / "depth" / f"{self.frame_ids[i]}.png"
        return cv2.imread(str(p), cv2.IMREAD_UNCHANGED) if p.exists() else None

    def confidence(self, i: int) -> np.ndarray | None:
        import cv2

        p = self.root / "confidence" / f"{self.frame_ids[i]}.png"
        return cv2.imread(str(p), cv2.IMREAD_UNCHANGED) if p.exists() else None


def _ffprobe_like(video: Path) -> dict:
    """Width, height, frame count and rotation via the bundled ffmpeg (no ffprobe is bundled)."""
    import imageio_ffmpeg

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    r = subprocess.run([exe, "-hide_banner", "-i", str(video)], capture_output=True, text=True)
    txt = r.stderr
    import re

    m = re.search(r"Video: (\w+).*?, (\d{2,5})x(\d{2,5})", txt)
    rot = re.search(r"rotate\s*:\s*(-?\d+)|displaymatrix: rotation of (-?[\d.]+) degrees", txt)
    color = re.search(r"Video: [^\n]*?\(([^)]*?(?:bt2020|arib-std-b67|smpte2084)[^)]*)\)", txt)
    return {"codec": m.group(1) if m else None, "width": int(m.group(2)) if m else None,
            "height": int(m.group(3)) if m else None,
            "rotation": float(next(g for g in rot.groups() if g)) if rot else 0.0,
            "hdr_hint": color.group(1) if color else None, "ffmpeg_banner": txt[-2000:]}


def read_stray(root: str | Path) -> StrayCapture:
    root = Path(root)
    rows = []
    with open(root / "odometry.csv") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        for r in reader:
            if not r or not r[0].strip():
                continue
            rows.append({h: v.strip() for h, v in zip(header, r)})
    from scipy.spatial.transform import Rotation

    ts, ids, T, K = [], [], [], []
    dc = []
    for r in rows:
        ts.append(float(r["timestamp"]))
        ids.append(r["frame"].zfill(6))
        q = [float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"])]
        Ta = make_T(Rotation.from_quat(q).as_matrix(), 1000.0 * np.array([float(r["x"]), float(r["y"]), float(r["z"])]))
        T.append(Ta @ ARKIT_TO_CV_CAM)
        if "fx" in r and r["fx"]:
            K.append(np.array([[float(r["fx"]), 0, float(r["cx"])], [0, float(r["fy"]), float(r["cy"])], [0, 0, 1.0]]))
        dcx, dcy = r.get("distortion_center_x", ""), r.get("distortion_center_y", "")
        dc.append([float(dcx), float(dcy)] if dcx and dcy else [np.nan, np.nan])
    if not K:   # old recordings: one matrix for all frames
        K0 = np.loadtxt(root / "camera_matrix.csv", delimiter=",")
        K = [K0] * len(rows)
    info = _ffprobe_like(root / "rgb.mp4")
    w, h = info["width"], info["height"]
    Ks = np.asarray(K)
    # intrinsics refer to the video frame: the principal point must sit near the centre of the
    # decoded frame in the same orientation
    cx, cy = float(np.median(Ks[:, 0, 2])), float(np.median(Ks[:, 1, 2]))
    if w and h and not (abs(cx - w / 2) < 0.15 * w and abs(cy - h / 2) < 0.15 * h):
        info["warning"] = f"principal point ({cx:.0f},{cy:.0f}) does not fit the decoded {w}x{h} frame"
    dca = np.asarray(dc, float)
    return StrayCapture(root, np.asarray(ts), ids, np.asarray(T), Ks, int(w), int(h), (root / "depth").exists(),
                        None if np.all(np.isnan(dca)) else dca, info)


def iter_frames(video: str | Path, indices: list[int] | None = None, scale: float = 1.0):
    """Yield ``(index, RGB uint8)`` from a video, without auto-rotation. ``indices`` restricts the
    frames returned (decoding is sequential, so pass sorted indices)."""
    import imageio_ffmpeg

    want = None if indices is None else set(int(i) for i in indices)
    meta = _ffprobe_like(Path(video))
    w, h = meta["width"], meta["height"]
    if scale != 1.0:
        w, h = int(round(w * scale)), int(round(h * scale))
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [exe, "-hide_banner", "-loglevel", "error", "-noautorotate", "-i", str(video), "-vsync", "0"]
    if scale != 1.0:
        cmd += ["-vf", f"scale={w}:{h}:flags=area"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=w * h * 3 * 2)
    n = w * h * 3
    i = 0
    last = max(want) if want else None
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            if want is None or i in want:
                yield i, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            i += 1
            if last is not None and i > last:
                break
    finally:
        p.stdout.close()
        p.kill()
        p.wait()


def write_capture_summary(cap: StrayCapture, out: Path) -> None:
    from ..utils.io import write_json

    write_json(out, {"root": str(cap.root), "n_frames": len(cap), "width": cap.width, "height": cap.height,
                     "duration_s": float(cap.timestamps[-1] - cap.timestamps[0]) if len(cap) else 0.0,
                     "fx_median": float(np.median(cap.K[:, 0, 0])), "has_depth": cap.has_depth,
                     "video": {k: v for k, v in cap.info.items() if k != "ffmpeg_banner"}})


# ----------------------------------------------------------------------------- lens distortion
def distortion_lut(cap: StrayCapture, i: int) -> np.ndarray | None:
    """Per-frame radial lookup table (float32), if the recording has one."""
    p = cap.root / "distortion" / f"{cap.frame_ids[i]}.bin"
    if not p.exists():
        return None
    return np.fromfile(p, dtype="<f4")


def undistort_maps(lut: np.ndarray, center: tuple[float, float], width: int, height: int, inverse: bool = False):
    """``cv2.remap`` maps that rectify a frame with Apple's radial lookup table.

    Apple's ``lensDistortionLookupTable`` gives, for the radius r of an *undistorted* point (as a
    fraction of the largest radius in the image), the magnification m(r): the distorted position is
    c + (p - c)(1 + m). Stray Scanner does not say which of Apple's two tables it stores, so
    ``inverse=True`` reads it the other way; ``choose_undistortion`` picks by board reprojection
    error."""
    cx, cy = center
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    dx, dy = xs - cx, ys - cy
    r = np.sqrt(dx ** 2 + dy ** 2)
    r_max = float(np.max(np.hypot(np.array([0, width - 1]) - cx, np.array([[0], [height - 1]]) - cy)))
    t = np.clip(r / r_max, 0, 1) * (len(lut) - 1)
    m = np.interp(t.ravel(), np.arange(len(lut)), lut).reshape(r.shape).astype(np.float32)
    k = 1.0 / (1.0 + m) if inverse else (1.0 + m)
    return (cx + dx * k).astype(np.float32), (cy + dy * k).astype(np.float32)


def choose_undistortion(cap: StrayCapture, frames: dict[int, np.ndarray], board, n: int = 6, log=print) -> str:
    """'none', 'forward' or 'inverse': whichever gives the lowest median board reprojection error on
    a few frames (a reading that is wrong makes it clearly worse)."""
    import cv2

    from ..poses.board import BoardDetector

    if cap.distortion_center is None or not (cap.root / "distortion").exists():
        return "none"
    det = BoardDetector(board)
    ids = sorted(frames)[:: max(1, len(frames) // n)][:n]
    errs = {"none": [], "forward": [], "inverse": []}
    for i in ids:
        lut = distortion_lut(cap, i)
        if lut is None or not np.all(np.isfinite(cap.distortion_center[i])):
            continue
        c = cap.distortion_center[i]
        # the centre may be given in the camera's reference resolution; bring it to the frame
        if c[0] > cap.width or c[1] > cap.height:
            c = c * (cap.width / (2 * c[0]))
        for mode in errs:
            img = frames[i]
            if mode != "none":
                mx, my = undistort_maps(lut, (float(c[0]), float(c[1])), cap.width, cap.height, inverse=(mode == "inverse"))
                img = cv2.remap(img, mx, my, cv2.INTER_LINEAR)
            bp = det.pose(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cap.camera(i))
            if bp is not None:
                errs[mode].append(bp.reproj_px)
    med = {k: float(np.median(v)) for k, v in errs.items() if v}
    log(f"undistortion check (median board reprojection px): {med}")
    if not med or "none" not in med:
        return "none"
    best = min(med, key=med.get)
    return best if med[best] < 0.9 * med["none"] else "none"
