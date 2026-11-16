"""Camera poses of a capture in the board frame.

Every selected frame gets a board pose from the ChArUco corners (metric, drift-free). The ARKit
trajectory is then fitted to those poses with one rigid transform (both are metric): frames where
the board is not detected - low passes where the device hides much of the board, motion blur - take
the aligned ARKit pose, and the fit residual tells how far ARKit drifted. ARKit drifts by
centimetres over a full circle, so the board pose wins wherever it exists.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..geometry.camera import Camera
from ..geometry.transforms import inv_T, make_T, pose_difference
from .board import BoardDetector, BoardSpec


@dataclass
class CapturePoses:
    frames: list[int]
    T_world_cam: dict[int, np.ndarray]
    source: dict[int, str]                       # "board" | "arkit"
    board_stats: dict[int, dict] = field(default_factory=dict)
    T_world_ark: np.ndarray | None = None
    arkit_residual: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"frames": self.frames,
                "poses": {str(i): {"T_world_cam": self.T_world_cam[i], "source": self.source[i], **self.board_stats.get(i, {})}
                          for i in self.frames},
                "T_world_ark": self.T_world_ark, "arkit_residual": self.arkit_residual}


def rigid_fit(A: np.ndarray, B: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Rigid T with B ~ T A (Kabsch, weighted)."""
    w = np.ones(len(A)) if w is None else w
    w = w / w.sum()
    ca, cb = (w[:, None] * A).sum(0), (w[:, None] * B).sum(0)
    H = ((A - ca) * w[:, None]).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return make_T(R, cb - R @ ca)


def align_arkit(T_world_cam_board: dict[int, np.ndarray], T_ark_cam: np.ndarray) -> tuple[np.ndarray, dict]:
    """``T_world_ark`` from frames that have both poses: camera centres plus points 100 mm ahead
    of each camera (so orientation counts), with one round of outlier rejection."""
    ids = sorted(T_world_cam_board)
    if len(ids) < 3:
        raise ValueError("need at least 3 board poses to align ARKit")

    def pts(T):
        return np.stack([T[:3, 3], T[:3, 3] + 100.0 * T[:3, 2]])

    A = np.concatenate([pts(T_ark_cam[i]) for i in ids])
    B = np.concatenate([pts(T_world_cam_board[i]) for i in ids])
    T = rigid_fit(A, B)
    for _ in range(2):
        r = np.linalg.norm((A @ T[:3, :3].T + T[:3, 3]) - B, axis=1)
        keep = r < max(3.0 * np.median(r), 5.0)
        T = rigid_fit(A[keep], B[keep])
    errs = [pose_difference(T @ T_ark_cam[i], T_world_cam_board[i]) for i in ids]
    rot = np.array([e[0] for e in errs])
    pos = np.array([e[1] for e in errs])
    return T, {"n": len(ids), "rot_deg_median": float(np.median(rot)), "rot_deg_max": float(rot.max()),
               "pos_mm_median": float(np.median(pos)), "pos_mm_max": float(pos.max())}


def estimate_poses(frames: dict[int, np.ndarray], cams: dict[int, Camera], board: BoardSpec,
                   T_ark_cam: np.ndarray | None = None, min_markers: int = 6, max_reproj_px: float = 1.5) -> CapturePoses:
    """Board poses for ``frames`` (index -> RGB image); ARKit fills the gaps when given."""
    det = BoardDetector(board)
    board_poses, stats = {}, {}
    for i, img in frames.items():
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        bp = det.pose(gray, cams[i], min_markers=min_markers)
        if bp is not None:
            stats[i] = {"n_markers": bp.n_markers, "n_corners": bp.n_corners, "reproj_px": bp.reproj_px}
            if bp.reproj_px <= max_reproj_px:
                board_poses[i] = bp.T_world_cam
    T_world, src = dict(board_poses), {i: "board" for i in board_poses}
    T_world_ark, resid = None, {}
    if T_ark_cam is not None and len(board_poses) >= 3:
        T_world_ark, resid = align_arkit(board_poses, T_ark_cam)
        for i in frames:
            if i not in T_world:
                T_world[i] = T_world_ark @ T_ark_cam[i]
                src[i] = "arkit"
    order = sorted(T_world)
    return CapturePoses(order, T_world, src, stats, T_world_ark, resid)
