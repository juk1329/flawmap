"""Rigid-transform helpers and the flawmap frame conventions (adapted from ISCO).

All lengths are **millimetres**.

* **world (board)** - the printed capture board: origin at the centre of the grey sheet
  where the device outline is printed, +X to the right of the sheet, +Y towards the
  printed "top" arrow, +Z up out of the paper.
* **cad (device)** - the reference model: origin at the centre of the device's bounding
  box, +X to the right when looking at the screen, +Y towards the top edge (earpiece),
  +Z out of the screen. A device lying screen-up with its top towards the arrow is
  therefore roughly ``T_world_cad = translate(0, 0, thickness / 2)``.
* **camera (OpenCV)** - x right, y down, z forward (optical axis). ``T_world_cam`` maps
  camera coordinates to world coordinates; its rotation columns are the camera axes in
  world coordinates. Pixel centres are at integer coordinates (OpenCV convention).

``T_a_b`` maps coordinates in frame b to frame a. Quaternions are scalar-first
``(w, x, y, z)`` unless a function says otherwise.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

WORLD_UP = np.array([0.0, 0.0, 1.0])


def make_T(R: np.ndarray | None = None, t: np.ndarray | None = None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=float)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def apply_T(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to (N,3) points."""
    pts = np.asarray(pts)
    return pts @ T[:3, :3].T + T[:3, 3]


def apply_R(T_or_R: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    R = T_or_R[:3, :3]
    return np.asarray(vecs) @ R.T


def rot_x(deg: float) -> np.ndarray:
    return Rotation.from_euler("x", deg, degrees=True).as_matrix()


def rot_y(deg: float) -> np.ndarray:
    return Rotation.from_euler("y", deg, degrees=True).as_matrix()


def rot_z(deg: float) -> np.ndarray:
    return Rotation.from_euler("z", deg, degrees=True).as_matrix()


def look_at_opencv(eye: np.ndarray, target: np.ndarray, up: np.ndarray = WORLD_UP) -> np.ndarray:
    """``T_world_cam`` (OpenCV axes) of a camera at ``eye`` looking at ``target``.

    Columns of the rotation are (right, down, forward). Image "up" follows ``up``; when the
    viewing direction is (anti)parallel to ``up`` the world +Y axis fixes the roll so that
    image up points towards +Y (the board's "top" arrow).
    """
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    f = target - eye
    n = np.linalg.norm(f)
    if n < 1e-12:
        raise ValueError("eye and target coincide")
    f /= n
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-8:  # looking straight up/down
        r = np.cross(f, np.array([0.0, 1.0, 0.0]))
        if np.linalg.norm(r) < 1e-8:
            r = np.array([1.0, 0.0, 0.0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f], axis=1)
    return make_T(R, eye)


def quat_wxyz_from_R(R: np.ndarray) -> np.ndarray:
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def R_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def pose_from_params(dx=0.0, dy=0.0, dz=0.0, yaw_deg=0.0, roll_deg=0.0, pitch_deg=0.0) -> np.ndarray:
    """Pose with rotation ``Rz(yaw) Ry(pitch) Rx(roll)`` about the origin, then translation."""
    R = rot_z(yaw_deg) @ rot_y(pitch_deg) @ rot_x(roll_deg)
    return make_T(R, [dx, dy, dz])


def rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(R).as_rotvec())))


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """4x4 transform from a 6-vector (rotation vector [rad], translation)."""
    xi = np.asarray(xi, dtype=float)
    return make_T(Rotation.from_rotvec(xi[:3]).as_matrix(), xi[3:])


def pose_difference(T_a: np.ndarray, T_b: np.ndarray) -> tuple[float, float]:
    """Rotation (deg) and translation (same unit as T) between two poses."""
    D = inv_T(T_a) @ T_b
    return rotation_angle_deg(D[:3, :3]), float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]))
