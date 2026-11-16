"""Register the reference model to a capture by aligning its silhouette with image edges.

Cameras are fixed (board poses, ~0.1 mm). The device pose ``T_world_cad`` starts from the printed
outline (``parametric.rest_pose``) and is refined in a few rounds:

1. cast the model into each selected view at half resolution and take the silhouette pixels with
   their 3D surface points and outward 2D normals;
2. at full resolution, search along each normal for the strongest intensity edge (sub-pixel by a
   parabola fit), preferring offsets near the prediction;
3. solve the 6-DoF update that moves the projected silhouette points onto those edges (robust
   least squares, Huber loss in pixels).

This is model-based edge tracking. IoU of masks was the first plan, but it is flat near the optimum
and SAM 2 masks are 256x256 internally (1-1.5 mm per mask pixel on a 1920x1440 frame).
Per-view residuals (px and mm) are kept: they measure registration quality and, along the frame,
are the raw signal for outline-changing dents.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares

from ..geometry.camera import Camera
from ..geometry.transforms import inv_T, se3_exp
from ..surface.raycast import RaycastScene
from .parametric import DeviceModel


@dataclass
class ViewEdges:
    frame: int
    X: np.ndarray          # (n,3) silhouette points, cad frame
    n2: np.ndarray         # (n,2) outward unit normals in the image
    target: np.ndarray     # (n,2) matched edge positions (px), NaN where none
    strength: np.ndarray   # (n,) edge strength at the match


@dataclass
class Registration:
    T_world_cad: np.ndarray
    rounds: list = field(default_factory=list)
    per_view: dict = field(default_factory=dict)       # frame -> {median_px, median_mm, n}
    edges: list = field(default_factory=list)          # final round: per view {frame, X (cad), r_px, r_mm}

    def to_json(self) -> dict:
        return {"T_world_cad": self.T_world_cad, "rounds": self.rounds, "per_view": self.per_view}


def _grad(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) if img.ndim == 3 else img.astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    return cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0, cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0


def _bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = img.shape
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0, y0 = x.astype(np.int64), y.astype(np.int64)
    fx, fy = x - x0, y - y0
    return (img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx) * (1 - fy) + (img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx) * fy


def _project(cam: Camera, T_cam_world: np.ndarray, Xw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pc = Xw @ T_cam_world[:3, :3].T + T_cam_world[:3, 3]
    z = pc[:, 2]
    return np.stack([cam.fx * pc[:, 0] / z + cam.cx, cam.fy * pc[:, 1] / z + cam.cy], axis=1), z


def cast_pixels(scene_cad: RaycastScene, cam: Camera, T_cad_cam: np.ndarray, uv: np.ndarray):
    """Cast rays through continuous pixel coordinates ``uv`` (n,2); returns (hit, point_cad)."""
    d = cam.ray_dirs_cam(uv[:, 0], uv[:, 1])
    d = d / np.linalg.norm(d, axis=1, keepdims=True) @ T_cad_cam[:3, :3].T
    o = np.broadcast_to(T_cad_cam[:3, 3], d.shape)
    t, _, _, hit = scene_cad._cast(o, d)
    return hit, o + d * np.where(hit, t, 0.0)[:, None]


def silhouette_points(scene_cad: RaycastScene, cam: Camera, T_cad_cam: np.ndarray, scale: float = 0.5,
                      every: int = 3, bisect: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """3D points on the silhouette generator (cad frame) and outward image normals.

    The contour is found on a cast at ``scale`` and then located precisely by bisection along each
    normal at full resolution (hit -> miss), so the 3D point is the grazing point itself and not a
    surface point up to a few pixels inside the contour (that bias gave 0.19 mm / 0.07 deg pose
    errors on the synthetic captures)."""
    c = cam.scaled(scale)
    h = scene_cad.cast(c, T_cad_cam, step=1)
    m = h.hit.astype(np.uint8)
    if m.sum() < 50:
        return np.zeros((0, 3)), np.zeros((0, 2))
    border = m & (1 - cv2.erode(m, np.ones((3, 3), np.uint8)))
    blur = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.5)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    vs, us = np.nonzero(border)
    keep = (np.arange(len(vs)) % every) == 0
    vs, us = vs[keep], us[keep]
    n = -np.stack([gx[vs, us], gy[vs, us]], axis=1)
    nn = np.linalg.norm(n, axis=1)
    ok = nn > 1e-3
    vs, us, n = vs[ok], us[ok], n[ok] / nn[ok, None]
    # full-resolution pixel of the (inside) contour pixel centre
    q = np.stack([(us + 0.5) / scale - 0.5, (vs + 0.5) / scale - 0.5], axis=1)
    lo = np.full(len(q), -1.0 / scale)          # inside
    hi = np.full(len(q), 2.0 / scale)           # outside
    hit_lo, X = cast_pixels(scene_cad, cam, T_cad_cam, q + lo[:, None] * n)
    hit_hi, _ = cast_pixels(scene_cad, cam, T_cad_cam, q + hi[:, None] * n)
    valid = hit_lo & ~hit_hi
    for _ in range(bisect):
        mid = 0.5 * (lo + hi)
        hm, Xm = cast_pixels(scene_cad, cam, T_cad_cam, q + mid[:, None] * n)
        lo = np.where(hm, mid, lo)
        hi = np.where(hm, hi, mid)
        X = np.where(hm[:, None], Xm, X)
    return X[valid], n[valid]


def match_edges(gx: np.ndarray, gy: np.ndarray, p: np.ndarray, n2: np.ndarray, search: float,
                step: float = 0.5, min_strength: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """For each predicted point p and normal n2, the sub-pixel position of the strongest edge along
    the normal within +-search px (prior favouring small offsets)."""
    ts = np.arange(-search, search + 1e-9, step)
    X = p[:, None, 0] + ts[None, :] * n2[:, None, 0]
    Y = p[:, None, 1] + ts[None, :] * n2[:, None, 1]
    d = np.abs(_bilinear(gx, X.ravel(), Y.ravel()) * np.repeat(n2[:, 0], len(ts)) +
               _bilinear(gy, X.ravel(), Y.ravel()) * np.repeat(n2[:, 1], len(ts))).reshape(X.shape)
    prior = np.exp(-0.5 * (ts / (0.6 * search)) ** 2)
    k = np.argmax(d * prior[None, :], axis=1)
    peak = d[np.arange(len(k)), k]
    k0, k1 = np.clip(k - 1, 0, len(ts) - 1), np.clip(k + 1, 0, len(ts) - 1)
    a, b, c = d[np.arange(len(k)), k0], peak, d[np.arange(len(k)), k1]
    denom = a - 2 * b + c
    with np.errstate(divide="ignore", invalid="ignore"):
        off = np.where(np.abs(denom) > 1e-9, 0.5 * (a - c) / denom, 0.0)
    off = np.clip(off, -0.5, 0.5)
    t = ts[k] + off * step
    tgt = p + t[:, None] * n2
    bad = (peak < min_strength) | (k == 0) | (k == len(ts) - 1)
    tgt[bad] = np.nan
    return tgt, peak


def register_device(model: DeviceModel, frames: dict[int, np.ndarray], cams: dict[int, Camera],
                    T_world_cam: dict[int, np.ndarray], T_init: np.ndarray, views: list[int] | None = None,
                    searches: tuple[float, ...] = (12.0, 6.0, 4.0, 3.0, 3.0), huber_px: float = 1.0,
                    max_points_per_view: int = 600, log=print) -> Registration:
    scene = RaycastScene(model.mesh, None)
    views = views or sorted(frames)
    grads = {i: _grad(frames[i]) for i in views}
    T = T_init.copy()
    reg = Registration(T)
    rng = np.random.default_rng(0)
    for rnd, S in enumerate(searches):
        data = []
        for i in views:
            T_cad_cam = inv_T(T) @ T_world_cam[i]
            X, n2 = silhouette_points(scene, cams[i], T_cad_cam)
            if len(X) == 0:
                continue
            if len(X) > max_points_per_view:
                sel = rng.choice(len(X), max_points_per_view, replace=False)
                X, n2 = X[sel], n2[sel]
            T_cam_world = inv_T(T_world_cam[i])
            Xw = X @ T[:3, :3].T + T[:3, 3]
            p, z = _project(cams[i], T_cam_world, Xw)
            tgt, strength = match_edges(*grads[i], p, n2, S)
            ok = np.isfinite(tgt[:, 0])
            data.append((i, X[ok], n2[ok], tgt[ok], T_cam_world))
        if not data:
            break

        def residuals(xi):
            Tn = se3_exp(xi) @ T
            out = []
            for i, X, n2, tgt, Tcw in data:
                Xw = X @ Tn[:3, :3].T + Tn[:3, 3]
                p, _ = _project(cams[i], Tcw, Xw)
                out.append(np.einsum("ij,ij->i", p - tgt, n2))
            return np.concatenate(out)

        r0 = residuals(np.zeros(6))
        # rotate about the device centre, not the board origin: express the update at the device
        c = T[:3, 3].copy()

        def residuals_centred(xi):
            w, v = xi[:3], xi[3:]
            v_eff = v - np.cross(w, c)     # rotation about c: x' = R(x - c) + c + v
            return residuals(np.concatenate([w, v_eff]))

        sol = least_squares(residuals_centred, np.zeros(6), loss="huber", f_scale=huber_px, x_scale=[1e-3] * 3 + [0.1] * 3,
                            max_nfev=60)
        w, v = sol.x[:3], sol.x[3:]
        T = se3_exp(np.concatenate([w, v - np.cross(w, c)])) @ T
        r1 = residuals(np.zeros(6))
        info = {"round": rnd, "search_px": S, "n_points": int(len(r0)), "median_abs_px_before": float(np.median(np.abs(r0))),
                "median_abs_px_after": float(np.median(np.abs(r1))), "step_rot_deg": float(np.degrees(np.linalg.norm(w))),
                "step_mm": float(np.linalg.norm(v))}
        reg.rounds.append(info)
        log(f"  register round {rnd}: {info['n_points']} edge points, |r| median {info['median_abs_px_before']:.2f} -> "
            f"{info['median_abs_px_after']:.2f} px, step {info['step_mm']:.3f} mm / {info['step_rot_deg']:.3f} deg")
    # per-view quality with the final pose; the signed residuals (r > 0: the image edge lies inside
    # the model's silhouette, i.e. material is missing there) are kept for the outline channel
    for i, X, n2, tgt, Tcw in data:
        Xw = X @ T[:3, :3].T + T[:3, 3]
        p, z = _project(cams[i], Tcw, Xw)
        r = np.einsum("ij,ij->i", p - tgt, n2)
        mm = r * z / cams[i].fx
        reg.per_view[int(i)] = {"median_px": float(np.median(np.abs(r))), "median_mm": float(np.median(np.abs(mm))), "n": int(len(r))}
        reg.edges.append({"frame": int(i), "X": X, "r_px": r, "r_mm": mm})
    reg.T_world_cad = T
    return reg


def edge_residuals(model: DeviceModel, frames: dict[int, np.ndarray], cams: dict[int, Camera],
                   T_world_cam: dict[int, np.ndarray], T_world_cad: np.ndarray, views: list[int] | None = None,
                   search: float = 8.0, every: int = 1, max_points_per_view: int = 4000) -> list[dict]:
    """Signed silhouette-to-edge residuals of many views at a fixed (final) pose, for the outline
    channel. The registration itself uses ~36 views x 600 points; a dent a few mm long needs denser
    sampling around the frame (on the two-pass synthetic demo the registration's own points gave
    only 3,327 outline samples). The search reaches +-8 px: a 0.8 mm dent is ~4 px at 300 mm, and
    with the registration's final 3 px window the dent's own edge points fell outside and were
    dropped (no dent found on the synthetic test, 2026-11-15)."""
    scene = RaycastScene(model.mesh, None)
    out = []
    rng = np.random.default_rng(1)
    for i in (views or sorted(frames)):
        T_cad_cam = inv_T(T_world_cad) @ T_world_cam[i]
        X, n2 = silhouette_points(scene, cams[i], T_cad_cam, every=every)
        if len(X) == 0:
            continue
        if len(X) > max_points_per_view:
            sel = rng.choice(len(X), max_points_per_view, replace=False)
            X, n2 = X[sel], n2[sel]
        Tcw = inv_T(T_world_cam[i])
        Xw = X @ T_world_cad[:3, :3].T + T_world_cad[:3, 3]
        p, z = _project(cams[i], Tcw, Xw)
        gx, gy = _grad(frames[i])
        tgt, _ = match_edges(gx, gy, p, n2, search)
        ok = np.isfinite(tgt[:, 0])
        r = np.einsum("ij,ij->i", p[ok] - tgt[ok], n2[ok])
        out.append({"frame": int(i), "X": X[ok], "r_px": r, "r_mm": r * z[ok] / cams[i].fx})
    return out
