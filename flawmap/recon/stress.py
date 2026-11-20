"""Views a person rotates to when inspecting the model: where the user saw Gaussian splatting break.

All views look at the device frame (mm). Elevation is measured from the device plane towards the
filmed face's outward normal (the back: -z; the screen: +z), so 90 degrees is straight above the
filmed face and negative elevations look at the face that was not filmed.

Groups:
- ``turn30``: a full turn at 30 degrees, 1-degree steps (the turntable video and the flicker metric)
- ``ring10`` / ``ring60`` / ``ring85``: turns at 10 / 60 / 85 degrees, 10-degree steps
- ``graze``: 3 and 8 degrees (nearly edge-on), every 30 degrees
- ``under``: -40 and -75 degrees, every 45 degrees (the side the capture never saw)
- ``zoom``: 70 mm from the face centre and from each defect, 75 degrees
"""
from __future__ import annotations

import numpy as np

from .context import View


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """``T_cad_cam`` (OpenCV axes: x right, y down, z forward) of a camera at ``eye`` looking at ``target``."""
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    down = -np.asarray(up, float)
    down = down - f * (down @ f)
    if np.linalg.norm(down) < 1e-6:
        down = np.array([0.0, -1.0, 0.0]) - f * (np.array([0.0, -1.0, 0.0]) @ f)
    down /= np.linalg.norm(down)
    right = np.cross(down, f)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([right, down, f])
    T[:3, 3] = eye
    return T


def stress_K(W: int = 640, H: int = 480, hfov_deg: float = 66.0) -> np.ndarray:
    fx = (W / 2) / np.tan(np.radians(hfov_deg) / 2)
    return np.array([[fx, 0, (W - 1) / 2], [0, fx, (H - 1) / 2], [0, 0, 1.0]])


def stress_views(model, face: str, defects: list | None = None, W: int = 640, H: int = 480) -> list[View]:
    K = stress_K(W, H)
    n = np.array([0, 0, -1.0]) if face == "back" else np.array([0, 0, 1.0])
    x, y = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    spec = model.spec
    half = 0.5 * np.hypot(spec.length, spec.width)
    vfov = 2 * np.arctan((H / 2) / K[1, 1])
    dist = half / np.tan(vfov / 2) * 1.08
    c = np.zeros(3)
    out: list[View] = []

    def add(group: str, el: float, az: float, d: float = dist, target: np.ndarray = c, i: int | None = None):
        e, a = np.radians(el), np.radians(az)
        dirv = np.cos(e) * (np.cos(a) * x + np.sin(a) * y) + np.sin(e) * n
        eye = target + d * dirv
        up = n if abs(el) < 70 else y
        name = f"{group} el{el:g} az{az:g}" + (f" #{i}" if i is not None else "")
        out.append(View(K, look_at(eye, target, up), W, H, None, name))

    for az in range(0, 360, 1):
        add("turn30", 30, az)
    for el, g in ((10, "ring10"), (60, "ring60"), (85, "ring85")):
        for az in range(0, 360, 10):
            add(g, el, az)
    for el in (3, 8):
        for az in range(0, 360, 30):
            add("graze", el, az)
    for el in (-40, -75):
        for az in range(0, 360, 45):
            add("under", el, az)
    zface = model.z_back if face == "back" else model.z_front
    add("zoom", 75, 90, d=70.0, target=np.array([0, 0, zface]))
    for k, d in enumerate(defects or []):
        p = d.get("centroid_mm")
        if p is not None and d.get("chart") == face:
            add("zoom", 75, 90, d=70.0, target=np.asarray(p, float), i=d.get("id", k))
    return out
