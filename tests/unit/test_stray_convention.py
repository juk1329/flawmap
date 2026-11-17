"""Camera-axis convention of Stray Scanner's odometry.csv, pinned with a real recording.

The camera looks at a flat floor in every frame. Fitting a plane to each LiDAR depth map gives the
floor normal in camera axes; mapped to the world with the odometry pose, it must be the same normal
(ARKit's up axis, +y) in every frame. A wrong camera-axis flip makes the normals scatter.
"""
import csv
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from flawmap.io.stray import ARKIT_TO_CV_CAM

FIX = Path(__file__).resolve().parents[1] / "data" / "stray_floor"


def _floor_normals_world():
    rows = list(csv.reader(open(FIX / "odometry.csv", encoding="utf-8")))
    head = [h.strip() for h in rows[0]]
    out = []
    for r in rows[1:]:
        d = dict(zip(head, (x.strip() for x in r)))
        depth = cv2.imread(str(FIX / "depth" / f"{d['frame'].zfill(6)}.png"), cv2.IMREAD_UNCHANGED).astype(np.float64)
        h, w = depth.shape
        s = 1920 / w                                   # intrinsics refer to the 1920x1440 video frame
        fx, fy = float(d["fx"]) / s, float(d["fy"]) / s
        cx, cy = (float(d["cx"]) + 0.5) / s - 0.5, (float(d["cy"]) + 0.5) / s - 0.5
        jj, ii = np.meshgrid(np.arange(w), np.arange(h))
        z = depth.ravel()
        x, y = (jj.ravel() - cx) / fx * z, (ii.ravel() - cy) / fy * z
        coef, *_ = np.linalg.lstsq(np.c_[x, y, np.ones_like(z)], z, rcond=None)
        n_cam = np.array([coef[0], coef[1], -1.0])        # OpenCV camera axes, towards the camera
        n_cam /= np.linalg.norm(n_cam)
        R = Rotation.from_quat([float(d[k]) for k in ("qx", "qy", "qz", "qw")]).as_matrix()
        out.append(R @ ARKIT_TO_CV_CAM[:3, :3] @ n_cam)
    return np.array(out)


def test_floor_normal_agrees_across_frames():
    n = _floor_normals_world()
    assert len(n) == 8
    mean = n.mean(0) / np.linalg.norm(n.mean(0))
    spread = np.degrees(np.arccos(np.clip(n @ mean, -1, 1)))
    assert spread.max() < 3.0, spread
    assert mean[1] > 0.99, mean                           # ARKit world: +y is up, the floor faces up
