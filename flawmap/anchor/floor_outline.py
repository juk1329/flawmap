"""A flat device on the floor, measured from a floor-projected mosaic (no drawing, no board).

The boardless path finds the device by projecting a model outline into the images, so it needs a
model of about the right size first; the generic phone plate misses a tablet entirely. This module
measures the device without a model:

1. every keyframe is projected onto the floor plane (LiDAR height, ARKit poses; z-up world of
   ``poses.boardless``), and the median over views gives a top-down mosaic: glare that moves with
   the viewpoint drops out, the floor and the device stay;
2. the device is the large central region whose colour differs from the floor (the floor's colour
   from the mosaic's border ring), filled and fitted with a rounded rectangle by maximising IoU;
3. the height of the top face is where the views agree best inside the outline (projected at a
   height ``h`` above the floor, a textured surface only lines up across views at its real height).
   A textureless face (black glass) gives a flat curve; the height is then left to the edge fit.

First used on an iPad Pro 13" (2026-11-17, two real iPhone recordings on a wood floor): 281.0 x 213.5
mm against Apple's 281.6 x 215.5, top of the back face 5.8 mm above the floor (5.1 mm thick).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..geometry.camera import Camera
from ..geometry.transforms import inv_T


@dataclass
class FloorPlate:
    length: float                 # mm
    width: float
    corner_radius: float
    top_height: float | None      # top face above the floor (mm); None when the views do not tell
    centre_xy: np.ndarray         # device centre on the floor, world mm
    long_axis_deg: float          # direction of the long side in the world xy plane
    iou: float                    # rounded-rectangle fit to the mask
    info: dict


def _project_plane(frames, cams, T_world_cam, ids, z, x0, y0, res, nx, ny, scale):
    """Per-view samples (V, ny, nx, 3) float32 with NaN of the plane at height ``z``; rows go along -y."""
    xs = x0 + (np.arange(nx) + 0.5) * res
    ys = y0 - (np.arange(ny) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z), np.ones(X.size)])
    out = []
    for i in ids:
        cam = cams[i].scaled(scale)
        img = frames[i] if scale == 1.0 else cv2.resize(frames[i], (cam.width, cam.height), interpolation=cv2.INTER_AREA)
        Xc = (inv_T(T_world_cam[i]) @ P)[:3]
        zc = Xc[2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (cam.fx * Xc[0] / zc + cam.cx).reshape(X.shape).astype(np.float32)
            v = (cam.fy * Xc[1] / zc + cam.cy).reshape(X.shape).astype(np.float32)
        ok = (zc.reshape(X.shape) > 0) & (u >= 1) & (v >= 1) & (u < img.shape[1] - 2) & (v < img.shape[0] - 2)
        s = cv2.remap(img, np.nan_to_num(u, nan=-10), np.nan_to_num(v, nan=-10), cv2.INTER_LINEAR).astype(np.float32)
        s[~ok] = np.nan
        out.append(s)
    return np.stack(out)


def _rr_mask(shape, cx, cy, ang, L, W, r):
    a = np.radians(ang)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    u = dx * np.cos(a) + dy * np.sin(a)
    v = -dx * np.sin(a) + dy * np.cos(a)
    qx, qy = np.abs(u) - (L / 2 - r), np.abs(v) - (W / 2 - r)
    d = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r
    return d <= 0


def _fit_rounded_rect(mask: np.ndarray, p0: dict, keys=("cx", "cy", "ang", "L", "W", "r")) -> tuple[dict, float]:
    """Coordinate search over ``keys`` of centre, angle, size and corner radius (pixels) maximising IoU."""
    def iou(p):
        m = _rr_mask(mask.shape, **p)
        return (m & mask).sum() / max((m | mask).sum(), 1)

    p, best = dict(p0), iou(p0)
    steps = {k: v for k, v in {"cx": 4.0, "cy": 4.0, "ang": 1.0, "L": 4.0, "W": 4.0, "r": 4.0}.items() if k in keys}
    for _ in range(60):
        moved = False
        for k in steps:
            for sgn in (1, -1):
                q = dict(p)
                q[k] += sgn * steps[k]
                q["r"] = float(np.clip(q["r"], 1.0, min(q["L"], q["W"]) / 2 - 1.0))
                s = iou(q)
                if s > best:
                    best, p, moved = s, q, True
        if not moved:
            steps = {k: v / 2 for k, v in steps.items()}
            if steps["cx"] < 0.1:
                break
    return p, float(best)


def _hp_spread(frames, cams, T_world_cam, ids, z, x0, y0, res, n, scale) -> np.ndarray:
    """Per-pixel spread across views of high-passed grey samples of the plane at height ``z``: low
    where the plane holds a texture that really lies at that height."""
    g = _project_plane(frames, cams, T_world_cam, ids, z, x0, y0, res, n, n, scale).mean(-1)
    out = []
    for s in g:
        # normalised blur over the view's own pixels only, and nothing near the image border:
        # blurring a zero-filled outside turned every frame edge into a bright line (2026-11-17)
        v = np.isfinite(s).astype(np.float32)
        f = np.nan_to_num(s).astype(np.float32)
        bv = cv2.GaussianBlur(v, (0, 0), 3.0)
        with np.errstate(all="ignore"):
            hp = f - cv2.GaussianBlur(f * v, (0, 0), 3.0) / bv
        hp[bv < 0.98] = np.nan
        out.append(hp)
    with np.errstate(all="ignore"):
        return np.nanstd(np.stack(out), 0)


def _device_mask(M: np.ndarray, valid: np.ndarray, exclude: np.ndarray | None = None) -> np.ndarray:
    """The large central region of the mosaic whose colour differs from the floor's, leaving out the
    ``exclude`` pixels (what lies on the floor)."""
    lab = cv2.cvtColor(M, cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w = M.shape[:2]
    ring = np.zeros((h, w), bool)
    b = max(4, min(h, w) // 12)
    ring[:b], ring[-b:], ring[:, :b], ring[:, -b:] = True, True, True, True
    ring &= valid
    floor = np.median(lab[ring], 0)
    spread = np.median(np.abs(lab[ring] - floor), 0) * 1.4826 + 1.0
    # lightness and chroma apart: black glass differs in lightness, silver aluminium on wood only in
    # chroma (neutral vs yellowish), and a shadow in the border ring inflates the lightness spread
    # (iPad back on a wood floor, 2026-11-17: lightness spread 38, chroma difference ~3 sigma)
    d_l = np.abs(lab[..., 0] - floor[0]) / spread[0]
    d_c = np.sqrt(((lab[..., 1] - floor[1]) / spread[1]) ** 2 + ((lab[..., 2] - floor[2]) / spread[2]) ** 2)
    d = np.maximum(d_l, d_c)
    dv = np.clip(d[valid] * 32.0, 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(dv.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (d > max(2.0, thr / 32.0)) & valid
    if exclude is not None:
        m &= ~exclude
    m = m.astype(np.uint8)
    k = max(3, int(min(h, w) / 120)) | 1
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, lbl, st, cen = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.zeros((h, w), bool)
    score = [st[j, cv2.CC_STAT_AREA] / (1.0 + (np.hypot(*(cen[j] - [w / 2, h / 2])) / (0.25 * min(h, w))) ** 2) for j in range(1, n)]
    m = (lbl == 1 + int(np.argmax(score))).astype(np.uint8)
    kc = max(5, int(min(h, w) / 30)) | 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((kc, kc), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(m)
    cv2.drawContours(out, cnts, -1, 1, -1)
    return out > 0


def floor_aim_point(T_world_cam: dict[int, np.ndarray], z_floor: float) -> np.ndarray | None:
    """Median point where the optical axes hit the floor (xy, mm). Unlike the axes' least-squares
    meeting point it stays put when the axes are nearly parallel (filming from above)."""
    hits = []
    for T in T_world_cam.values():
        c, d = T[:3, 3], T[:3, 2]
        if d[2] < -0.2:                                 # looking down
            t = (z_floor - c[2]) / d[2]
            hits.append(c[:2] + t * d[:2])
    return np.median(np.array(hits), 0) if len(hits) >= 5 else None


def measure_plate(frames: dict[int, np.ndarray], cams: dict[int, Camera], T_world_cam: dict[int, np.ndarray],
                  z_floor: float, centre_xy: np.ndarray, half_mm: float = 230.0, res: float = 0.5, scale: float = 0.5,
                  max_views: int = 60, heights=tuple(np.arange(-1.0, 12.01, 1.0)), size: tuple | None = None,
                  top_hint: float | None = None, log=print) -> FloorPlate | None:
    """Size, corner radius, position and heading of a flat device lying on the floor ``z_floor``
    (world z-up, mm), from keyframes with metric poses. None when no device-like region is found.
    ``size`` = (length, width, corner radius) in mm fixes the shape (measured on a cleaner pass of the
    same device) and fits only position and heading: a shadow next to the device cannot stretch it.
    ``top_hint``: the top face's height (mm) when this pass cannot measure it (black glass)."""
    ids = sorted(frames)
    ids = [ids[int(k)] for k in np.linspace(0, len(ids) - 1, min(max_views, len(ids))).round()]
    aim = floor_aim_point({i: T_world_cam[i] for i in ids}, z_floor)
    if aim is not None:
        centre_xy = aim
    n = int(round(2 * half_mm / res))
    x0, y0 = centre_xy[0] - half_mm, centre_xy[1] + half_mm
    S = _project_plane(frames, cams, T_world_cam, ids, z_floor, x0, y0, res, n, n, scale)
    valid = np.isfinite(S[..., 0]).sum(0) >= 3
    M = np.nan_to_num(np.nanmedian(S, 0)).astype(np.uint8)
    del S
    # what lies on the floor (shadows included) lines up across views at the floor height and not a
    # few mm above it; the device's textured top does the opposite. Colour alone joined the
    # photographer's shadow to the iPad's silver back (2026-11-17).
    sub = ids[:: max(1, len(ids) // 30)]
    n1 = int(round(2 * half_mm))
    probe = top_hint if top_hint else 7.0
    sd0 = _hp_spread(frames, cams, T_world_cam, sub, z_floor, x0, y0, 1.0, n1, scale)
    sdh = _hp_spread(frames, cams, T_world_cam, sub, z_floor + probe, x0, y0, 1.0, n1, scale)
    floorlike = (sd0 < 0.6 * sdh) & np.isfinite(sd0) & np.isfinite(sdh)
    floorlike = cv2.resize(floorlike.astype(np.uint8), (n, n), interpolation=cv2.INTER_NEAREST) > 0
    mask = _device_mask(M, valid, exclude=floorlike)
    if mask.sum() < 1000:
        log("  floor plate: no device-like region in the floor mosaic")
        return None
    # 1. height of the top face: cross-view agreement of high-passed samples inside the coarse
    #    outline (1 mm grid); a textureless face gives a flat curve and keeps ``top_hint``
    inner = cv2.erode(mask.astype(np.uint8), np.ones((int(20 / res) | 1, int(20 / res) | 1), np.uint8)) > 0
    top, curve = None, []
    if inner.sum() > 200:
        rr, cc = np.nonzero(mask)
        bx0, by0 = x0 + cc.min() * res - 10.0, y0 - rr.min() * res + 10.0
        nb = int(round(max(cc.max() - cc.min(), rr.max() - rr.min()) * res + 20.0))
        xs = bx0 + np.arange(nb) + 0.5
        ys = by0 - np.arange(nb) - 0.5
        ci = np.clip(((xs - x0) / res).astype(int), 0, n - 1)
        ri = np.clip(((y0 - ys) / res).astype(int), 0, n - 1)
        inner_b = inner[np.ix_(ri, ci)]
        for hgt in heights:
            sd = _hp_spread(frames, cams, T_world_cam, sub, z_floor + hgt, bx0, by0, 1.0, nb, scale)
            with np.errstate(all="ignore"):
                curve.append(float(np.nanmean(sd[inner_b])))
        a = np.array(curve)
        k = int(np.argmin(a))
        depth = (np.median(a) - a[k]) / max(np.median(a), 1e-6)
        if 0 < k < len(a) - 1 and depth > 0.03:      # a real minimum, not a flat curve
            y_0, y_1, y_2 = a[k - 1:k + 2]
            top = float(heights[k] + 0.5 * (y_0 - y_2) / (y_0 - 2 * y_1 + y_2) * (heights[1] - heights[0]))
    # 2. the outline at the top face's height: projected at the floor, the raised top edge spreads
    #    outwards view by view (+7 mm on the iPad screen at 5 mm height)
    z_top = top if top is not None else top_hint
    if z_top:
        S = _project_plane(frames, cams, T_world_cam, ids, z_floor + z_top, x0, y0, res, n, n, scale)
        valid = np.isfinite(S[..., 0]).sum(0) >= 3
        M = np.nan_to_num(np.nanmedian(S, 0)).astype(np.uint8)
        del S
        m2 = _device_mask(M, valid, exclude=floorlike)
        if m2.sum() >= 1000:
            mask = m2
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    (cx, cy), (w, h), ang = cv2.minAreaRect(c)
    if w < h:
        w, h, ang = h, w, ang + 90.0
    if size is not None:
        p, iou = _fit_rounded_rect(mask, {"cx": cx, "cy": cy, "ang": ang, "L": size[0] / res, "W": size[1] / res,
                                          "r": size[2] / res}, keys=("cx", "cy", "ang"))
    else:
        p, iou = _fit_rounded_rect(mask, {"cx": cx, "cy": cy, "ang": ang, "L": w, "W": h, "r": 0.06 * h})
    L, W, r = p["L"] * res, p["W"] * res, p["r"] * res
    # pixel -> world: column -> +x, row -> -y
    cxw, cyw = x0 + (p["cx"] + 0.5) * res, y0 - (p["cy"] + 0.5) * res
    long_deg = float(-p["ang"]) % 180.0               # image angle (rows down) -> world angle (y up)
    info = {"size_mm": [round(L, 2), round(W, 2)], "corner_radius_mm": round(r, 2), "iou": round(iou, 4),
            "top_height_mm": None if top is None else round(top, 2), "outline_at_height_mm": z_top,
            "height_curve": [round(v, 3) for v in curve], "n_views": len(ids), "mosaic_res_mm": res,
            "floorlike_share": round(float(floorlike.mean()), 3)}
    log(f"  floor plate: {L:.1f} x {W:.1f} mm, corner r {r:.1f} mm, IoU {iou:.3f}, top "
        + ("not measurable (flat curve)" if top is None else f"{top:.2f} mm above the floor")
        + (f", outline at {z_top:.1f} mm" if z_top else ""))
    return FloorPlate(L, W, r, top, np.array([cxw, cyw]), long_deg, iou, info)
