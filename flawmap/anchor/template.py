"""A plate-shaped model fitted to the capture, for devices without a reference drawing.

The detectors never use the reference model's colours, only its shape as a coordinate system
(charts, mm, parts). For phones and tablets that shape is close to a rounded plate, so a plate whose
length, width, thickness and corner radius are fitted to the capture's outline can stand in for the
drawing-based model. What is lost: the camera bump, buttons and openings are unknown (no exclusion
zones, no part names for them), and without metric cameras the size is only assumed.

* ``template_spec``  - a ``DeviceSpec`` for a rounded plate with flat sides and small chamfers
* ``fit_dimensions`` - edge alignment (as ``register.register_device``) with three extra unknowns, the
  log-scales of width, length and thickness about the device centre; the model is rebuilt with
  the new size after every round, then the corner radius is chosen by a 1-D search
* ``identify``       - the drawing-based model whose length and width match the fitted plate
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from ..geometry.transforms import inv_T, se3_exp
from ..surface.raycast import RaycastScene
from .parametric import DeviceSpec, build_device_model
from .register import _grad, _project, match_edges, silhouette_points

GENERIC_PHONE = {"length": 146.7, "width": 71.5, "thickness": 7.6, "corner_radius": 10.0}


def template_spec(length: float, width: float, thickness: float, corner_radius: float = 10.0, name: str | None = None) -> DeviceSpec:
    r = float(np.clip(corner_radius, 1.0, min(width, length) / 2 - 1.0))
    th = np.radians(np.linspace(0.0, 90.0, 9))
    corner = [[float(r - r * np.cos(t)), float(r - r * np.sin(t))] for t in th]      # (inset from side, inset from top)
    h = thickness / 2.0
    c = min(0.6, 0.12 * thickness)
    profile = [[1.0, h, "screen"], [0.45, h - 0.05, "frame"], [0.12, h - c * 0.6, "frame"], [0.0, h - c, "frame"],
               [0.0, -(h - c), "frame"], [0.12, -(h - c * 0.6), "frame"], [0.45, -(h - 0.05), "back"], [1.0, -h, "back"]]
    nm = name or f"추정 판형 모델 {length:.1f} x {width:.1f} x {thickness:.1f} mm"
    return DeviceSpec(name=nm, slug="template", length=float(length), width=float(width), thickness=float(thickness),
                      corner=corner, side_profile=profile, camera={}, buttons=[], openings=[],
                      default_color={"frame": [60, 62, 66], "back": [70, 72, 76], "screen": [14, 14, 16], "camera": [40, 40, 44],
                                     "lens": [10, 10, 12], "button": [60, 62, 66]},
                      source="fitted to the capture (no reference drawing)",
                      raw={"template": True, "corner_radius": r})


def fit_dimensions(spec: DeviceSpec, frames: dict[int, np.ndarray], cams: dict, T_world_cam: dict[int, np.ndarray],
                   T_world_cad: np.ndarray, views: list[int], searches=(10.0, 6.0, 4.0, 3.0, 2.5), fit_thickness: bool = True,
                   max_points_per_view: int = 500, huber_px: float = 1.0, radii=(6.0, 8.0, 10.0, 12.0), log=print):
    """Pose and plate size from image edges with cameras fixed. Returns (spec, T_world_cad, info)."""
    L, W, T = spec.length, spec.width, spec.thickness
    r = spec.raw.get("corner_radius", GENERIC_PHONE["corner_radius"])
    grads = {i: _grad(frames[i]) for i in views}
    rng = np.random.default_rng(0)
    rounds = []
    Tw = T_world_cad.copy()

    def collect(spec_, Tw_, S):
        model = build_device_model(spec_, corner_step=0.5, edge_step=2.0, profile_step=0.3)
        scene = RaycastScene(model.mesh, None)
        data = []
        for i in views:
            X, n2 = silhouette_points(scene, cams[i], inv_T(Tw_) @ T_world_cam[i])
            if len(X) == 0:
                continue
            if len(X) > max_points_per_view:
                sel = rng.choice(len(X), max_points_per_view, replace=False)
                X, n2 = X[sel], n2[sel]
            p, _ = _project(cams[i], inv_T(T_world_cam[i]), X @ Tw_[:3, :3].T + Tw_[:3, 3])
            tgt, _ = match_edges(*grads[i], p, n2, S)
            ok = np.isfinite(tgt[:, 0])
            if ok.sum() >= 20:
                data.append((i, X[ok], n2[ok], tgt[ok], inv_T(T_world_cam[i])))
        return data

    def make_res(data, Tw_, with_scale):
        def res(x):
            xi = x[:6]
            sc = np.exp(np.r_[x[6:9], 0.0][:3]) if with_scale else np.ones(3)
            if with_scale and not fit_thickness:
                sc[2] = 1.0
            Tn = Tw_ @ se3_exp(xi)            # update in the device frame (rotation about the device centre)
            out = []
            for i, X, n2, tgt, Tcw in data:
                p, _ = _project(cams[i], Tcw, (X * sc) @ Tn[:3, :3].T + Tn[:3, 3])
                out.append(np.einsum("ij,ij->i", p - tgt, n2))
            return np.concatenate(out)
        return res

    for rnd, S in enumerate(searches):
        cur = template_spec(L, W, T, r, spec.name if not spec.raw.get("template") else None)
        data = collect(cur, Tw, S)
        if not data:
            break
        res = make_res(data, Tw, True)
        x0 = np.zeros(9)
        r0 = res(x0)
        sol = least_squares(res, x0, loss="huber", f_scale=huber_px, max_nfev=60,
                            x_scale=[1e-3] * 3 + [0.1] * 3 + [3e-3] * 3)
        x = sol.x
        Tw = Tw @ se3_exp(x[:6])
        sc = np.exp(x[6:9])
        W, L = W * sc[0], L * sc[1]
        if fit_thickness:
            T = T * sc[2]
        r1 = res(x)
        rounds.append({"round": rnd, "search_px": S, "n_points": int(len(r0)), "median_px_before": float(np.median(np.abs(r0))),
                       "median_px_after": float(np.median(np.abs(r1))), "size_mm": [round(L, 3), round(W, 3), round(T, 3)]})
        log(f"  template fit round {rnd}: {len(r0)} edge points, |r| {rounds[-1]['median_px_before']:.2f} -> "
            f"{rounds[-1]['median_px_after']:.2f} px, size {L:.2f} x {W:.2f} x {T:.2f}")
    # corner radius: the radius whose model leaves the smallest residual (pose refit for each)
    best = None
    for rr in radii:
        cur = template_spec(L, W, T, rr)
        data = collect(cur, Tw, searches[-1])
        if not data:
            continue
        res = make_res(data, Tw, False)
        sol = least_squares(res, np.zeros(6), loss="huber", f_scale=huber_px, max_nfev=30, x_scale=[1e-3] * 3 + [0.1] * 3)
        med = float(np.median(np.abs(res(sol.x))))
        if best is None or med < best[0]:
            best = (med, rr, Tw @ se3_exp(sol.x))
    if best is not None:
        r, Tw = best[1], best[2]
    out = template_spec(L, W, T, r)
    info = {"rounds": rounds, "size_mm": [L, W, T], "corner_radius": r, "median_px": best[0] if best else None}
    log(f"  template: {L:.2f} x {W:.2f} x {T:.2f}, corner radius {r:.0f}, residual {info['median_px']}")
    return out, Tw, info


def identify(length: float, width: float, tol_mm: float = 1.0) -> DeviceSpec | None:
    """The drawing-based device whose length and width are both within ``tol_mm`` of the fit."""
    from ..utils.io import repo_root

    best = None
    for p in sorted((repo_root() / "configs" / "devices").glob("*.yaml")):
        try:
            s = DeviceSpec.load(p)
        except Exception:
            continue
        d = max(abs(s.length - length), abs(s.width - width))
        if d <= tol_mm and (best is None or d < best[0]):
            best = (d, s)
    return best[1] if best else None


def plate_box(points: np.ndarray, cam_centres: np.ndarray, face_up: str = "screen") -> tuple[np.ndarray, np.ndarray]:
    """Initial ``T_world_cad`` and (length, width, thickness) of a plate from object points.

    The face plane is fitted robustly (SVD, points beyond 3 MAD of the plane dropped, 4 rounds), and
    length and width come from the smallest rectangle around the in-plane points (2 % farthest from the
    centre dropped). Principal axes of all points were used first; on uCO3D 29776-61880-12178 the
    segmented cloud carries off-plane noise, the longest axis came out across the phone (length/width
    0.57) and the fit settled on a 40 x 71 mm plate. Cad z (screen normal) points to the cameras when
    the screen is up."""
    import cv2

    c = np.median(points, 0)
    q = points - c
    if len(q) > 60000:
        q = q[np.random.default_rng(0).choice(len(q), 60000, replace=False)]
    keep = np.ones(len(q), bool)
    cc = np.zeros(3)
    for _ in range(4):
        cc = np.median(q[keep], 0)
        _, _, Vt = np.linalg.svd(q[keep] - cc, full_matrices=False)
        d = (q - cc) @ Vt[2]
        mad = np.median(np.abs(d[keep])) + 1e-12
        keep = np.abs(d) < 3.0 * mad
    a1, a2, e_n = Vt[0], Vt[1], Vt[2]
    uv = np.stack([(q - cc) @ a1, (q - cc) @ a2], 1)[keep]
    r = np.linalg.norm(uv - np.median(uv, 0), axis=1)
    uv = uv[r <= np.percentile(r, 98)]
    (rx, ry), (w, h), ang = cv2.minAreaRect(uv.astype(np.float32))
    t = np.radians(ang)
    dir_w = np.cos(t) * a1 + np.sin(t) * a2           # direction of the rectangle side with size w
    dir_h = -np.sin(t) * a1 + np.cos(t) * a2
    e_len, L, W = (dir_w, w, h) if w >= h else (dir_h, h, w)
    toward = (cam_centres - (c + cc)).mean(0)
    if (e_n @ toward > 0) != (face_up == "screen"):
        e_n = -e_n
    e_wid = np.cross(e_len, e_n)                       # x = y cross z keeps the frame right-handed
    R = np.column_stack([e_wid, e_len, e_n])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = c + cc + rx * a1 + ry * a2
    thick = float(2.0 * np.percentile(np.abs(d[keep]), 98))
    return T, np.array([float(L), float(W), thick])


def axes_box(points: np.ndarray, cam_centres: np.ndarray, face_up: str = "screen") -> tuple[np.ndarray, np.ndarray]:
    """The first initialisation: principal axes of all points, 1st-99th percentile extents. Kept as a
    second start next to ``plate_box`` (on uCO3D 59015-27660-21826 it gave the better plate)."""
    c = np.median(points, 0)
    q = points - c
    _, _, Vt = np.linalg.svd(q[np.random.default_rng(0).choice(len(q), min(len(q), 20000), replace=False)], full_matrices=False)
    e_len, e_n = Vt[0], Vt[2]
    toward = (cam_centres - c).mean(0)
    if (e_n @ toward > 0) != (face_up == "screen"):
        e_n = -e_n
    e_wid = np.cross(e_len, e_n)
    R = np.column_stack([e_wid, e_len, e_n])
    proj = q @ R
    lo, hi = np.percentile(proj, 1, axis=0), np.percentile(proj, 99, axis=0)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = c + R @ ((lo + hi) / 2)
    size = hi - lo
    return T, np.array([size[1], size[0], size[2]])


def mask_iou(model, T_world_cad: np.ndarray, cams: dict, T_world_cam: dict, masks: dict, scale: float = 0.25) -> float:
    """Median IoU between the model's silhouette and given object masks (bool, full resolution)."""
    import cv2

    scene = RaycastScene(model.mesh, None)
    vals = []
    for i, m in masks.items():
        h = scene.cast(cams[i].scaled(scale), inv_T(T_world_cad) @ T_world_cam[i], step=1)
        mm = cv2.resize(m.astype(np.uint8), (h.hit.shape[1], h.hit.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        u = (h.hit | mm).sum()
        if u:
            vals.append((h.hit & mm).sum() / u)
    return float(np.median(vals)) if vals else 0.0


pca_box = plate_box     # the first name, kept for callers
