"""Find the device on any surface, without assuming that it differs in colour from the floor.

``anchor.floor_outline`` measures the device on a floor mosaic by colour. That needs a floor whose
colour differs from the device's, and it lets a shadow join the device (the iPad back on a wood
floor: 327 x 294 mm, IoU 0.668, 2026-11-17). Here the device is found by what does not depend on the
floor's colour:

1. **Floor plane** (``floor_plane``): RANSAC on the LiDAR points around the device, in the
   gravity-aligned world of ``poses.boardless`` (z up, mm). The floor's height comes from it; its
   tilt is reported (and the world is levelled when it is not horizontal).
2. **Where the texture lines up** (``consistency_map``): every view is projected onto the floor
   plane and onto a plane a few millimetres above it. Floor texture only agrees across views at the
   floor's height; whatever stands above the floor (the device's outline, logo, camera) agrees
   better higher up. The cells with that parallax, closed and filled near where the cameras aim,
   are the device candidate. No colour contrast is needed. The same spread, minimised over the
   height, also gives the floor's height from the images themselves (``refine_floor``): on the iPad
   recordings the LiDAR plane lay 5-6.5 mm below it, and the top face then sat 5 mm above the
   photogrammetric floor (the iPad Pro 13" is 5.1 mm thick).
3. **SAM 2** (``segment.device_masks.track_device``) is prompted with positive points at the aim
   point and the candidate's centre on a few keyframes and follows the device through all
   keyframes; a second round is prompted with the box of the first footprint and negative points
   on the floor just outside it.
4. **Footprint** (``footprint``): the 2D visual hull of the masks. A point of a horizontal plane
   inside the device is inside every view's mask; a point outside is outside it in the views from
   its side. A shadow is not in the masks, so it is not in the footprint either. A rounded
   rectangle fitted to the footprint gives size, corner radius, centre and heading
   (``floor_outline.FloorPlate``), and a second SAM 2 round prompted with that box cleans the masks.

The masks stay with the pass: the registration uses them next to the image edges
(``anchor.register``), the colour samples are gated by them, and they judge pose alternatives.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage as ndi

from ..geometry.camera import Camera
from ..geometry.transforms import inv_T
from .floor_outline import FloorPlate, _fit_rounded_rect, _hp_spread, floor_aim_point


@dataclass
class Located:
    plate: FloorPlate | None
    masks: dict[int, np.ndarray]          # frame -> bool device mask at ``mask_scale`` of the frame
    mask_scale: float
    z_floor: float
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------- 1. floor plane
def lidar_points(cap, ids: list[int], T_world_cam: dict[int, np.ndarray], min_conf: int = 2, stride: int = 2) -> np.ndarray:
    """LiDAR points (world mm) of the frames ``ids``, every ``stride``-th depth pixel."""
    pts = []
    for i in ids:
        d = cap.depth(i)
        if d is None:
            continue
        conf = cap.confidence(i)
        h, w = d.shape[:2]
        K = cap.K[i]
        sx, sy = w / cap.width, h / cap.height
        fx, fy = K[0, 0] * sx, K[1, 1] * sy
        cx, cy = (K[0, 2] + 0.5) * sx - 0.5, (K[1, 2] + 0.5) * sy - 0.5
        vs, us = np.mgrid[0:h:stride, 0:w:stride]
        z = d[::stride, ::stride].astype(np.float32)
        ok = z > 0
        if conf is not None and conf.shape == d.shape:
            ok &= conf[::stride, ::stride] >= min_conf
        pc = np.stack([(us - cx) / fx * z, (vs - cy) / fy * z, z], -1)[ok]
        T = T_world_cam[i]
        pts.append(pc @ T[:3, :3].T + T[:3, 3])
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def floor_plane(cap, ids: list[int], T_world_cam: dict[int, np.ndarray], centre_xy: np.ndarray, radius: float = 300.0,
                iters: int = 400, tol: float = 2.5, max_tilt_deg: float = 20.0, seed: int = 0) -> dict | None:
    """RANSAC plane of the LiDAR points within ``radius`` of the device (world z up, mm): its normal,
    tilt from the gravity axis, height under ``centre_xy`` and inlier share. None without depth."""
    P = lidar_points(cap, ids, T_world_cam)
    if len(P) < 500:
        return None
    P = P[np.hypot(P[:, 0] - centre_xy[0], P[:, 1] - centre_xy[1]) < radius]
    cam_z = min(float(T_world_cam[i][2, 3]) for i in ids)
    P = P[P[:, 2] < cam_z - 30.0]                       # the surface lies below every camera
    if len(P) < 500:
        return None
    rng = np.random.default_rng(seed)
    if len(P) > 40000:
        P = P[rng.choice(len(P), 40000, replace=False)]
    best = (0, None)
    cos_max = np.cos(np.radians(max_tilt_deg))
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n /= nn
        if n[2] < 0:
            n = -n
        if n[2] < cos_max:
            continue
        inl = np.abs((P - a) @ n) < tol
        k = int(inl.sum())
        if k > best[0]:
            best = (k, inl)
    if best[1] is None:
        return None
    Q = P[best[1]]
    c = Q.mean(0)
    _, _, Vt = np.linalg.svd(Q - c, full_matrices=False)
    n = Vt[2] if Vt[2][2] > 0 else -Vt[2]
    z_at = c[2] - (n[0] * (centre_xy[0] - c[0]) + n[1] * (centre_xy[1] - c[1])) / n[2]
    resid = (Q - c) @ n
    return {"normal": n, "point": c, "z": float(z_at), "tilt_deg": float(np.degrees(np.arccos(np.clip(n[2], -1, 1)))),
            "inlier_share": round(best[0] / len(P), 3), "rms_mm": round(float(np.sqrt(np.mean(resid ** 2))), 2), "n_points": int(len(P))}


def levelling(plane: dict) -> np.ndarray:
    """World-frame rotation (about the plane point) that makes the floor plane horizontal."""
    from scipy.spatial.transform import Rotation

    n = plane["normal"]
    z = np.array([0.0, 0.0, 1.0])
    ax = np.cross(n, z)
    s = np.linalg.norm(ax)
    T = np.eye(4)
    if s < 1e-9:
        return T
    R = Rotation.from_rotvec(ax / s * np.arcsin(np.clip(s, -1, 1))).as_matrix()
    c = plane["point"]
    T[:3, :3] = R
    T[:3, 3] = c - R @ c
    return T


# ---------------------------------------------------------------------------- 2. where the floor texture lines up
def consistency_map(frames: dict[int, np.ndarray], cams: dict[int, Camera], T_world_cam: dict[int, np.ndarray],
                    ids: list[int], z_floor: float, centre_xy: np.ndarray, half_mm: float = 230.0, res: float = 1.0,
                    probe_mm: float = 6.0, scale: float = 0.5, tex_thr: float = 1.5) -> dict:
    """Per floor cell (``res`` mm, rows along -y): spread across views of high-passed grey at the
    floor height (``sd0``) and ``probe_mm`` above it (``sdh``). ``floorlike``: texture that lines up
    at the floor; ``parallax``: structure that lines up above it (``sd0 - sdh`` well above zero)."""
    n = int(round(2 * half_mm / res))
    x0, y0 = centre_xy[0] - half_mm, centre_xy[1] + half_mm
    sd0 = _hp_spread(frames, cams, T_world_cam, ids, z_floor, x0, y0, res, n, scale)
    sdh = _hp_spread(frames, cams, T_world_cam, ids, z_floor + probe_mm, x0, y0, res, n, scale)
    ok = np.isfinite(sd0) & np.isfinite(sdh)
    s0, sh = np.nan_to_num(sd0, nan=0.0), np.nan_to_num(sdh, nan=0.0)
    textured = ok & (np.maximum(s0, sh) > tex_thr)
    floorlike = textured & (s0 < 0.7 * sh)
    d = s0 - sh
    spread = np.median(np.abs(d[ok])) * 1.4826 + 1e-6 if ok.any() else 1.0
    parallax = ok & (d > max(3.0 * spread, 0.5))
    return {"sd0": sd0, "sdh": sdh, "floorlike": floorlike, "parallax": parallax, "valid": ok, "x0": x0, "y0": y0,
            "res": res, "n": n}


def _cell_to_world(cm: dict, r: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.stack([cm["x0"] + (c + 0.5) * cm["res"], cm["y0"] - (r + 0.5) * cm["res"]], -1)


def candidate_region(cm: dict, aim_xy: np.ndarray, min_area_mm2: float = 1500.0, close_mm: float = 12.0) -> dict | None:
    """The parallax cells, closed over ``close_mm`` and filled: the component that is large and close
    to where the cameras aim."""
    res = cm["res"]
    m = cm["parallax"].astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    k = max(3, int(round(close_mm / res)) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m = ndi.binary_fill_holes(m).astype(np.uint8)
    k2 = max(3, int(round(4.0 / res)) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2)))
    nlab, lab, st, cen = cv2.connectedComponentsWithStats(m)
    if nlab <= 1:
        return None
    ac = np.array([(aim_xy[0] - cm["x0"]) / res - 0.5, (cm["y0"] - aim_xy[1]) / res - 0.5])     # (col, row)
    best = None
    for j in range(1, nlab):
        area = st[j, cv2.CC_STAT_AREA] * res * res
        if area < min_area_mm2:
            continue
        d = np.hypot(*(cen[j] - ac)) * res
        score = area / (1.0 + (d / 60.0) ** 2)
        if best is None or score > best[0]:
            best = (score, j)
    if best is None:
        return None
    j = best[1]
    comp = lab == j
    filled = ndi.binary_fill_holes(comp)
    rr, cc = np.nonzero(filled)
    xy = _cell_to_world(cm, rr, cc)
    c = xy.mean(0)
    _, sv, Vt = np.linalg.svd(xy - c, full_matrices=False)
    ext = (xy - c) @ Vt.T
    half = np.percentile(np.abs(ext), 95, axis=0)
    area = float(filled.sum() * res * res)
    # a device-like region: wide in both directions and filling most of its box (not a floor seam)
    compact = bool(half.min() >= 20.0 and area >= 0.6 * 4 * half[0] * half[1] and area >= 4000.0)
    return {"mask": filled, "centre_xy": c, "axes": Vt, "half_extent": half, "area_mm2": area, "compact": compact}


def lidar_is_clean(pl: dict | None, min_inliers: float = 0.8, max_rms_mm: float = 0.8, max_tilt_deg: float = 0.6) -> bool:
    """A LiDAR floor plane worth keeping as it is: most points on it, thin, and level with gravity."""
    return pl is not None and pl["inlier_share"] >= min_inliers and pl["rms_mm"] <= max_rms_mm and pl["tilt_deg"] <= max_tilt_deg


def refine_floor(frames, cams, T_world_cam, ids, z0: float, plate: FloorPlate, half_mm: float = 230.0, res: float = 1.0,
                 search_mm: float = 12.0, margin_mm: float = 25.0, scale: float = 1.0) -> tuple[float, dict]:
    """The floor's height where its texture agrees best across views: mean high-pass spread over the
    floor around the device (``margin_mm`` outside ``plate``) at heights ``z0 +- search_mm``, parabola
    through the minimum. Returns (z, record); ``z0`` when the curve has no inner minimum."""
    c = plate.centre_xy
    n = int(round(2 * half_mm / res))
    x0, y0 = c[0] - half_mm, c[1] + half_mm
    xs = x0 + (np.arange(n) + 0.5) * res
    ys = y0 - (np.arange(n) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    a = np.radians(plate.long_axis_deg)
    u = (X - c[0]) * np.cos(a) + (Y - c[1]) * np.sin(a)
    v = -(X - c[0]) * np.sin(a) + (Y - c[1]) * np.cos(a)
    ring = (np.abs(u) > plate.length / 2 + margin_mm) | (np.abs(v) > plate.width / 2 + margin_mm)
    heights = np.arange(-search_mm, search_mm + 1e-9, 1.0)
    curve = []
    for h in heights:
        sd = _hp_spread(frames, cams, T_world_cam, ids, z0 + h, x0, y0, res, n, scale)
        with np.errstate(all="ignore"):
            curve.append(float(np.nanmean(sd[ring & np.isfinite(sd)])))
    a_ = np.array(curve)
    k = int(np.nanargmin(a_))
    rec = {"z_lidar": round(z0, 2), "curve": [round(x, 3) for x in curve], "depth": round(float((np.nanmax(a_) - a_[k]) / max(a_[k], 1e-6)), 3)}
    if not 0 < k < len(a_) - 1:
        rec["used"] = False
        return z0, rec
    y_0, y_1, y_2 = a_[k - 1:k + 2]
    off = float(heights[k] + 0.5 * (y_0 - y_2) / (y_0 - 2 * y_1 + y_2))
    rec.update(used=True, offset_mm=round(off, 2))
    return z0 + off, rec


# ---------------------------------------------------------------------------- 3. prompts
def _project(cam: Camera, T_world_cam: np.ndarray, Xw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Tcw = inv_T(T_world_cam)
    pc = Xw @ Tcw[:3, :3].T + Tcw[:3, 3]
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack([cam.fx * pc[:, 0] / z + cam.cx, cam.fy * pc[:, 1] / z + cam.cy], 1), z


def _inside(cam: Camera, uv: np.ndarray, z: np.ndarray, margin: float = 20.0) -> np.ndarray:
    return (z > 1.0) & (uv[:, 0] > margin) & (uv[:, 1] > margin) & (uv[:, 0] < cam.width - margin) & (uv[:, 1] < cam.height - margin)


def point_prompts(cams: dict[int, Camera], T_world_cam: dict[int, np.ndarray], ids: list[int], pos_xyz: np.ndarray,
                  neg_xyz: np.ndarray) -> dict[int, dict]:
    """Positive and negative point prompts (full-frame px) on the frames ``ids``; points outside the
    image are left out, and a frame without a visible positive point gets no prompt."""
    out = {}
    for i in ids:
        up, zp = _project(cams[i], T_world_cam[i], pos_xyz)
        okp = _inside(cams[i], up, zp, 40.0)
        if not okp.any():
            continue
        un, zn = _project(cams[i], T_world_cam[i], neg_xyz) if len(neg_xyz) else (np.zeros((0, 2)), np.zeros(0))
        okn = _inside(cams[i], un, zn) if len(neg_xyz) else np.zeros(0, bool)
        pts = np.concatenate([up[okp], un[okn]])
        lab = np.concatenate([np.ones(okp.sum(), np.int32), np.zeros(okn.sum(), np.int32)])
        out[i] = {"points": pts, "labels": lab}
    return out


def box_prompts(cams: dict[int, Camera], T_world_cam: dict[int, np.ndarray], ids: list[int], plate: FloorPlate,
                z_floor: float, height: float, pad_mm: float = 3.0, neg_mm: float = 25.0) -> dict[int, dict]:
    """Box prompts from a fitted plate: its 3D box (floor to ``height``) projected, its centre as a
    positive point, and negative points on the floor ``neg_mm`` outside its sides and corners (a
    shadow or a neighbouring object stays out). Frames where the box leaves the image are skipped."""
    a = np.radians(plate.long_axis_deg)
    e_l, e_w = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    hl, hw = plate.length / 2 + pad_mm, plate.width / 2 + pad_mm
    corners = []
    for sl in (-1, 1):
        for sw in (-1, 1):
            xy = plate.centre_xy + sl * hl * e_l + sw * hw * e_w
            corners += [[xy[0], xy[1], z_floor], [xy[0], xy[1], z_floor + height]]
    corners = np.array(corners)
    centre = np.array([[plate.centre_xy[0], plate.centre_xy[1], z_floor + height]])
    neg = []
    for sl, sw in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        xy = plate.centre_xy + sl * (plate.length / 2 + neg_mm) * e_l + sw * (plate.width / 2 + neg_mm) * e_w
        neg.append([xy[0], xy[1], z_floor])
    neg = np.array(neg)
    out = {}
    for i in ids:
        uv, z = _project(cams[i], T_world_cam[i], corners)
        if not _inside(cams[i], uv, z, 5.0).all():
            continue
        c, zc = _project(cams[i], T_world_cam[i], centre)
        un, zn = _project(cams[i], T_world_cam[i], neg)
        okn = _inside(cams[i], un, zn, 10.0)
        out[i] = {"box": np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()], np.float32),
                  "points": np.concatenate([c, un[okn]]), "labels": np.r_[np.ones(1, np.int32), np.zeros(okn.sum(), np.int32)]}
    return out


def floor_extent(mask: np.ndarray, cam: Camera, T_world_cam: np.ndarray, z: float) -> dict | None:
    """Size on the plane at height ``z`` of what a mask covers: its outline cast onto the plane, then
    length and width along the principal axes (95th percentiles) and the enclosed area (mm2)."""
    m = mask.astype(np.uint8)
    if m.sum() < 50:
        return None
    sx, sy = m.shape[1] / cam.width, m.shape[0] / cam.height
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    uv = np.stack([(c[:, 0] + 0.5) / sx - 0.5, (c[:, 1] + 0.5) / sy - 0.5], 1)
    d = cam.ray_dirs_cam(uv[:, 0], uv[:, 1]) @ T_world_cam[:3, :3].T
    o = T_world_cam[:3, 3]
    ok = d[:, 2] < -0.05                                   # rays that reach the plane in front of the camera
    if ok.sum() < 0.9 * len(d):
        return None
    t = (z - o[2]) / d[ok, 2]
    xy = o[:2] + t[:, None] * d[ok, :2]
    cen = xy.mean(0)
    _, _, Vt = np.linalg.svd(xy - cen, full_matrices=False)
    e = (xy - cen) @ Vt.T
    L, W = 2 * np.percentile(np.abs(e), 95, axis=0)
    area = 0.5 * abs(np.dot(xy[:, 0], np.roll(xy[:, 1], 1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], 1)))
    return {"length": float(L), "width": float(W), "area": float(area), "centre_xy": cen}


def whole_object(hyps: list, cam: Camera, T_world_cam: np.ndarray, z: float, min_len: float = 40.0,
                 max_len: float = 420.0) -> tuple[np.ndarray | None, dict | None]:
    """Of SAM 2's nested answers to one point, the largest whose size on the floor fits a phone or
    a tablet (the logo is too small, the floor too large)."""
    best = (None, None)
    for m, score in hyps:
        ext = floor_extent(m, cam, T_world_cam, z)
        if ext is None or not (min_len <= ext["length"] <= max_len) or ext["width"] < 0.25 * min_len:
            continue
        if best[1] is None or ext["area"] > best[1]["area"]:
            best = (m, {**ext, "score": score})
    return best


# ---------------------------------------------------------------------------- 4. footprint
def footprint(masks: dict[int, np.ndarray], cams: dict[int, Camera], T_world_cam: dict[int, np.ndarray], z: float,
              x0: float, y0: float, res: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Per cell of a horizontal grid at height ``z``: the share of the views that see it in which it is
    inside the device mask, and the number of those views."""
    xs = x0 + (np.arange(n) + 0.5) * res
    ys = y0 - (np.arange(n) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1)
    inside = np.zeros(X.size, np.int32)
    seen = np.zeros(X.size, np.int32)
    for i, m in masks.items():
        cam = cams[i]
        uv, zc = _project(cam, T_world_cam[i], P)
        ok = _inside(cam, uv, zc, 2.0)
        sx, sy = m.shape[1] / cam.width, m.shape[0] / cam.height
        mu = np.clip(((uv[ok, 0] + 0.5) * sx).astype(np.int64), 0, m.shape[1] - 1)
        mv = np.clip(((uv[ok, 1] + 0.5) * sy).astype(np.int64), 0, m.shape[0] - 1)
        seen[ok] += 1
        inside[np.flatnonzero(ok)[m[mv, mu]]] += 1
    share = np.where(seen > 0, inside / np.maximum(seen, 1), 0.0).reshape(n, n)
    return share.astype(np.float32), seen.reshape(n, n)


def fit_footprint(share: np.ndarray, seen: np.ndarray, x0: float, y0: float, res: float, min_share: float = 0.6,
                  min_views: int = 8, size: tuple | None = None) -> tuple[FloorPlate | None, np.ndarray]:
    """Rounded rectangle fitted to the cells inside the masks in at least ``min_share`` of their views
    (the largest such component). ``size`` = (length, width, corner radius) mm fixes the shape.

    Just outside the device the floor is hidden behind it from the opposite side, so the share falls
    from 1 to about 0.5 at the outline and to 0 further out. With raw ARKit poses (drift of several
    millimetres) the stricter 0.9 left the iPad 4-7 mm short; at 0.6, half way up the device and on
    the photogrammetric floor, it came within 1 mm on the back pass (281.2 x 216.1 vs 281.6 x 215.5)."""
    fp = (share >= min_share) & (seen >= min_views)
    fp = cv2.morphologyEx(fp.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    nlab, lab, st, _ = cv2.connectedComponentsWithStats(fp)
    if nlab <= 1:
        return None, fp > 0
    j = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    fp = ndi.binary_fill_holes(lab == j)
    if fp.sum() < 400:
        return None, fp
    cnts, _ = cv2.findContours(fp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    (cx, cy), (w, h), ang = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    if w < h:
        w, h, ang = h, w, ang + 90.0
    if size is not None:
        # the shape is held: both quarter turns of the starting heading (a footprint that is not the whole
        # device can make the smallest rectangle's long side the device's short one: 6c417d24a0, 2026-11-18)
        best = None
        for a0 in (ang, ang + 90.0):
            p_, iou_ = _fit_rounded_rect(fp, {"cx": cx, "cy": cy, "ang": a0, "L": size[0] / res, "W": size[1] / res, "r": size[2] / res},
                                         keys=("cx", "cy", "ang"))
            if best is None or iou_ > best[1]:
                best = (p_, iou_)
        p, iou = best
    else:
        p, iou = _fit_rounded_rect(fp, {"cx": cx, "cy": cy, "ang": ang, "L": w, "W": h, "r": 0.06 * h})
    if p["W"] > p["L"]:
        # the fit may turn the rectangle by a quarter and trade the sides: the long side stays the length
        p = {**p, "L": p["W"], "W": p["L"], "ang": p["ang"] + 90.0}
    L, W, r = p["L"] * res, p["W"] * res, p["r"] * res
    centre = np.array([x0 + (p["cx"] + 0.5) * res, y0 - (p["cy"] + 0.5) * res])
    info = {"size_mm": [round(L, 2), round(W, 2)], "corner_radius_mm": round(r, 2), "iou": round(iou, 4), "from": "masks",
            "footprint_area_mm2": round(float(fp.sum()) * res * res, 1)}
    return FloorPlate(L, W, r, None, centre, float(-p["ang"]) % 180.0, float(iou), info), fp



def plate_outline(plate: FloorPlate, z: float, n_arc: int = 10) -> np.ndarray:
    """Points (world mm) around a plate's rounded-rectangle outline at height ``z``."""
    hl, hw = plate.length / 2, plate.width / 2
    r = float(np.clip(plate.corner_radius, 0.5, min(hl, hw) - 0.5))
    pts = []
    for sx, sy, a0 in ((1, 1, 0.0), (-1, 1, 90.0), (-1, -1, 180.0), (1, -1, 270.0)):
        cx, cy = sx * (hl - r), sy * (hw - r)
        t = np.radians(a0 + np.linspace(0.0, 90.0, n_arc))
        pts.append(np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)]))
    pts = np.concatenate(pts)
    a = np.radians(plate.long_axis_deg)
    e_l, e_w = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    xy = plate.centre_xy + pts[:, :1] * e_l + pts[:, 1:] * e_w
    return np.column_stack([xy, np.full(len(xy), z)])


def box_masks(plate: FloorPlate, cams: dict[int, Camera], T_world_cam: dict[int, np.ndarray], ids, z_floor: float,
              height: float, scale: float = 0.25) -> dict[int, np.ndarray]:
    """Exact silhouettes, at ``scale`` x the frame size, of the plate as a box from the floor to
    ``height``: the convex hull of its outline at both heights, projected. Views the box leaves are skipped."""
    P = np.concatenate([plate_outline(plate, z_floor), plate_outline(plate, z_floor + height)])
    out = {}
    for i in ids:
        cam = cams[i]
        uv, zc = _project(cam, T_world_cam[i], P)
        if (zc <= 1.0).any():
            continue
        m = np.zeros((int(round(cam.height * scale)), int(round(cam.width * scale))), np.uint8)
        hull = cv2.convexHull(((uv + 0.5) * scale - 0.5).astype(np.float32))
        cv2.fillConvexPoly(m, np.round(hull * 16).astype(np.int32), 1, lineType=cv2.LINE_8, shift=4)
        out[i] = m > 0
    return out


def debias_footprint(plate: FloorPlate, masks: dict[int, np.ndarray], cams, T_world_cam, z_floor: float, height: float,
                     z_mid: float, x0: float, y0: float, res: float, n: int, **fit_kw) -> tuple[FloorPlate, dict | None]:
    """The plate whose own visual hull, through the same cameras and by the same rule, gives the
    footprint measured: the measured plate minus the excess the hull of that plate shows (one
    fixed-point step).

    Just outside the device the floor at ``z_mid`` is hidden behind it from the far side, so the 0.6
    share grows the footprint by up to (height - mid) x cot(elevation) per side. On the synthetic
    iPhone 12 orbits (22 and 50 deg) the plate came out 3-4 mm long and wide, and the boardless joint
    fit, whose drift can move every camera along its axis, bent the poses to fit it; the size fit
    started from those poses kept most of the excess (148.4 x 73.5 vs 146.7 x 71.5, 2026-11-17)."""
    sim = box_masks(plate, cams, T_world_cam, list(masks), z_floor, height)
    if len(sim) < 8:
        return plate, None
    share, seen = footprint(sim, cams, T_world_cam, z_mid, x0, y0, res, n)
    ps, _ = fit_footprint(share, seen, x0, y0, res, **fit_kw)
    if ps is None:
        return plate, None
    bL, bW, br = ps.length - plate.length, ps.width - plate.width, ps.corner_radius - plate.corner_radius
    rec = {"excess_mm": [round(bL, 2), round(bW, 2)], "corner_excess_mm": round(br, 2), "views": len(sim)}
    if not (-2.0 < bL < 15.0 and -2.0 < bW < 15.0):
        rec["used"] = False
        return plate, rec
    rec["used"] = True
    out = FloorPlate(plate.length - bL, plate.width - bW, float(max(1.0, plate.corner_radius - br)), plate.top_height,
                     plate.centre_xy, plate.long_axis_deg, plate.iou, dict(plate.info))
    out.info.update(size_mm=[round(out.length, 2), round(out.width, 2)], corner_radius_mm=round(out.corner_radius, 2),
                    measured_size_mm=[round(plate.length, 2), round(plate.width, 2)], debias=rec)
    return out, rec


def top_height(frames, cams, T_world_cam, ids, z_floor: float, fp: np.ndarray, x0: float, y0: float, res: float,
               heights=tuple(np.arange(-1.0, 12.01, 1.0)), scale: float = 0.5) -> tuple[float | None, list]:
    """Height of the device's top face above the floor where its texture agrees best across views
    (``floor_outline.measure_plate`` step 1, inside the footprint); None for a textureless face."""
    inner = cv2.erode(fp.astype(np.uint8), np.ones((int(15 / res) | 1, int(15 / res) | 1), np.uint8)) > 0
    if inner.sum() < 200:
        return None, []
    rr, cc = np.nonzero(fp)
    bx0, by0 = x0 + cc.min() * res - 5.0, y0 - rr.min() * res + 5.0
    nb = int(round(max(cc.max() - cc.min(), rr.max() - rr.min()) * res + 10.0))
    xs = bx0 + np.arange(nb) + 0.5
    ys = by0 - np.arange(nb) - 0.5
    ci = np.clip(((xs - x0) / res).astype(int), 0, fp.shape[1] - 1)
    ri = np.clip(((y0 - ys) / res).astype(int), 0, fp.shape[0] - 1)
    inner_b = inner[np.ix_(ri, ci)]
    curve = []
    for hgt in heights:
        sd = _hp_spread(frames, cams, T_world_cam, ids, z_floor + hgt, bx0, by0, 1.0, nb, scale)
        with np.errstate(all="ignore"):
            curve.append(float(np.nanmean(sd[inner_b])))
    a = np.array(curve)
    k = int(np.nanargmin(a))
    depth = (np.nanmedian(a) - a[k]) / max(np.nanmedian(a), 1e-6)
    if 0 < k < len(a) - 1 and depth > 0.03:
        y_0, y_1, y_2 = a[k - 1:k + 2]
        return float(heights[k] + 0.5 * (y_0 - y_2) / (y_0 - 2 * y_1 + y_2) * (heights[1] - heights[0])), curve
    return None, curve


# ---------------------------------------------------------------------------- all of it
def plausible(plate: FloorPlate | None, min_iou: float = 0.85) -> bool:
    """A phone- or tablet-like plate: 40-420 mm long, 1.0-3.0 times as long as wide, a clean rectangle."""
    if plate is None:
        return False
    return 40.0 <= plate.length <= 420.0 and 1.0 <= plate.length / max(plate.width, 1e-6) <= 3.0 and plate.iou >= min_iou


def locate_device(cap, frames: dict[int, np.ndarray], cams: dict[int, Camera], T_world_cam: dict[int, np.ndarray],
                  half_mm: float = 230.0, n_prompt: int = 5, height_guess: float = 6.0, size: tuple | None = None,
                  log=print) -> Located | None:
    """The device in a pass with metric poses (ARKit) and LiDAR: floor plane, SAM 2 masks of every
    keyframe in ``frames``, and the plate fitted to the masks' footprint. ``size`` fixes the plate's
    shape (from a cleaner pass of the same device). None when SAM 2 is not available; a Located with
    ``plate=None`` when nothing plate-like was found."""
    from ..poses.boardless import axes_meeting_point
    from ..segment.device_masks import available, track_device

    if not available():
        log("  locate: SAM 2 / torch / checkpoint not available")
        return None
    ids = sorted(frames)
    info: dict = {}
    centre = axes_meeting_point([T_world_cam[i] for i in ids])
    early = ids[: max(8, len(ids) * 3 // 10)]
    # 1. floor plane from LiDAR (coarse; refined from the images below)
    pl = floor_plane(cap, early[:: max(1, len(early) // 10)], T_world_cam, centre[:2])
    if pl is not None:
        aim0 = floor_aim_point(T_world_cam, pl["z"])
        if aim0 is not None:
            pl = floor_plane(cap, early[:: max(1, len(early) // 10)], T_world_cam, aim0) or pl
        z0 = pl["z"]
        info["floor_lidar"] = {"z": round(z0, 2), "tilt_deg": round(pl["tilt_deg"], 2), "rms_mm": pl["rms_mm"],
                               "inlier_share": pl["inlier_share"], "normal": [round(float(x), 4) for x in pl["normal"]]}
        log(f"  locate: LiDAR floor plane z {z0:.1f} mm, tilt {pl['tilt_deg']:.2f} deg, rms {pl['rms_mm']} mm, inliers {pl['inlier_share']:.0%}")
    else:
        z0 = float(centre[2] - height_guess)
        info["floor_lidar"] = None
        log(f"  locate: no LiDAR floor; taken {height_guess} mm under the axes' meeting point")
    aim = floor_aim_point(T_world_cam, z0)
    aim = centre[:2] if aim is None else aim
    info["aim_xy"] = [round(float(x), 1) for x in aim]
    # 2. parallax candidate (frames at half size)
    sub = ids[:: max(1, len(ids) // 30)]
    small = {i: cv2.resize(frames[i], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) for i in sub}
    scams = {i: cams[i].scaled(0.5) for i in sub}
    cm = consistency_map(small, scams, T_world_cam, sub, z0, aim, half_mm=half_mm, scale=1.0)
    cand = candidate_region(cm, aim)
    if cand is not None:
        info["candidate"] = {"centre_xy": [round(float(x), 1) for x in cand["centre_xy"]], "area_mm2": round(cand["area_mm2"]),
                             "half_extent_mm": [round(float(x), 1) for x in cand["half_extent"]], "compact": cand["compact"]}
        log(f"  locate: parallax candidate at {np.round(cand['centre_xy'], 1)} mm, {cand['area_mm2']:.0f} mm2, "
            f"half extent {np.round(cand['half_extent'], 1)} mm, {'device-like' if cand['compact'] else 'not device-like'} "
            f"(aim {np.round(aim, 1)})")
    else:
        log(f"  locate: no parallax candidate (aim {np.round(aim, 1)})")
    # the cameras' aim point first (the seller keeps the device in the picture's middle); the parallax
    # candidate when that gives nothing plate-like. A floor seam once passed as the candidate on the
    # iPad back, and a positive point on it spoilt the masks (2026-11-17), so it is never mixed in.
    starts = [("aim", aim)]
    if cand is not None and cand["compact"] and np.hypot(*(cand["centre_xy"] - aim)) > 20.0:
        starts.append(("candidate", cand["centre_xy"]))
    pids = [ids[int(k)] for k in np.linspace(0, len(ids) - 1, min(n_prompt, len(ids))).round()]
    res_fp = 0.5
    n = int(round(2 * half_mm / res_fp))
    x0, y0 = aim[0] - half_mm, aim[1] + half_mm
    rounds, plate = [], None
    from ..segment.device_masks import object_hypotheses

    for how, xy in starts:
        # one point is ambiguous (the logo, the device, the floor): SAM 2's nested answers are cast onto
        # the floor and the largest one of phone-to-tablet size becomes the prompt of that keyframe
        pos = np.array([[xy[0], xy[1], z0 + height_guess / 2]])
        pp = point_prompts(cams, T_world_cam, pids, pos, np.zeros((0, 3)))
        hyps = object_hypotheses(frames, {i: pr["points"][0] for i, pr in pp.items()})
        prompts, chosen = {}, []
        for i, hs in hyps.items():
            m, ext = whole_object(hs, cams[i], T_world_cam[i], z0 + height_guess / 2)
            if m is not None:
                prompts[i] = {"mask": m}
                chosen.append([round(ext["length"]), round(ext["width"])])
        info.setdefault("hypotheses", {})[how] = chosen
        if not prompts:
            rounds.append({"prompt": f"point at {how}", "error": "no hypothesis of device size"})
            continue
        log(f"  locate: {how}: SAM 2 hypotheses of device size on {len(prompts)}/{len(pp)} keyframes, sizes {chosen} mm")
        masks = track_device(frames, prompts, log=log)
        share, seen = footprint(masks, cams, T_world_cam, z0 + height_guess / 2, x0, y0, res_fp, n)
        plate, fp = fit_footprint(share, seen, x0, y0, res_fp, size=size)
        rounds.append({"prompt": f"point at {how}", **(plate.info if plate is not None else {})})
        if plausible(plate):
            info["start"] = how
            break
    if not plausible(plate):
        log(f"  locate: no plate-like footprint ({rounds})")
        return Located(None, {}, 0.25, z0, {**info, "rounds": rounds})
    log(f"  locate: footprint {plate.length:.1f} x {plate.width:.1f} mm, r {plate.corner_radius:.1f}, IoU {plate.iou:.3f} (point prompts)")
    # 3. the floor's height. A clean LiDAR plane is kept; the images' height replaces only a poor one.
    #    Clean: the synthetic floors' depth (inliers 94-95 %, rms 0.5 mm, tilt 0.1 deg) lay within 1.5 mm
    #    of the true floor, while the images' curve, with the ARKit drift still in the poses, ran 4-8 mm
    #    high (wood +4.2, dark +7.5: above the device's top face). Poor: the real iPad's (inliers 49-51 %,
    #    rms 1.3 mm, tilt 1.2-1.4 deg) lay 5-6 mm low, and the images' height gave Apple's size (11-17).
    z_img, frec = refine_floor(small, scams, T_world_cam, sub, z0, plate, half_mm=half_mm)
    lidar_clean = lidar_is_clean(pl)
    z_floor = z_img if frec.get("used") and not lidar_clean else z0
    frec.update(lidar_clean=lidar_clean, z_images=round(z_img, 2), source="images" if z_floor != z0 else "lidar")
    info["floor"] = {**frec, "z": round(z_floor, 2)}
    log(f"  locate: floor z {z_floor:.2f} mm from the {frec['source']} (LiDAR plane {'clean' if lidar_clean else 'poor'}; "
        f"images {'LiDAR ' + format(z_img - z0, '+.2f') + ' mm' if frec.get('used') else 'no minimum'})")
    # 4. second round: box prompts and floor negatives from the first footprint
    pids = [ids[int(k)] for k in np.linspace(0, len(ids) - 1, min(2 * n_prompt, len(ids))).round()]
    prompts = box_prompts(cams, T_world_cam, pids, plate, z_floor, height_guess)
    if len(prompts) >= 2:
        masks2 = track_device(frames, prompts, log=log)
        share2, seen2 = footprint(masks2, cams, T_world_cam, z_floor + height_guess / 2, x0, y0, res_fp, n)
        plate2, fp2 = fit_footprint(share2, seen2, x0, y0, res_fp, size=size)
        rounds.append({"prompt": "box", **(plate2.info if plate2 is not None else {})})
        if plausible(plate2) and plate2.iou >= plate.iou - 0.01:
            masks, plate, fp = masks2, plate2, fp2
    # 5. top face height, then the footprint half way up the device
    top, curve = top_height(small, scams, T_world_cam, sub, z_floor, fp, x0, y0, res_fp, scale=1.0)
    if z_floor != z0 and top is not None and top < 1.5:
        # the images' floor left no room for a device under its top face: back to the LiDAR plane
        log(f"  locate: top face {top:.2f} mm above the images' floor; LiDAR floor kept instead")
        z_floor = z0
        info["floor"].update(z=round(z0, 2), source="lidar", reverted=True)
        top, curve = top_height(small, scams, T_world_cam, sub, z_floor, fp, x0, y0, res_fp, scale=1.0)
    h_mid = (top if top is not None and 1.5 < top < 15.0 else height_guess) / 2
    share, seen = footprint(masks, cams, T_world_cam, z_floor + h_mid, x0, y0, res_fp, n)
    plate3, fp3 = fit_footprint(share, seen, x0, y0, res_fp, size=size)
    if plausible(plate3):
        plate, fp = plate3, fp3
    plate.top_height = top
    plate.info.update({"top_height_mm": None if top is None else round(top, 2), "height_curve": [round(v, 3) for v in curve],
                       "z_floor": round(z_floor, 2), "footprint_height_mm": round(h_mid, 2)})
    if size is None:
        # 6. the hull's own excess, from the plate's exact silhouettes through the same cameras
        deb, drec = debias_footprint(plate, masks, cams, T_world_cam, z_floor, 2 * h_mid, z_floor + h_mid, x0, y0, res_fp, n)
        info["debias"] = drec
        if drec is not None and drec.get("used"):
            log(f"  locate: the plate's own hull is {drec['excess_mm'][0]:+.2f} x {drec['excess_mm'][1]:+.2f} mm larger "
                f"(corner {drec['corner_excess_mm']:+.2f}); plate {deb.length:.1f} x {deb.width:.1f} mm, r {deb.corner_radius:.1f}")
            plate = deb
    log(f"  locate: footprint {plate.length:.1f} x {plate.width:.1f} mm, r {plate.corner_radius:.1f}, IoU {plate.iou:.3f}; top face "
        + ("not measurable (no texture)" if top is None else f"{top:.2f} mm above the floor"))
    info["rounds"] = rounds
    loc = Located(plate, masks, 0.25, z_floor, info)
    loc._footprint = (share, x0, y0, res_fp)
    return loc
