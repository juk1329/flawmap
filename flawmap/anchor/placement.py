"""Finding a device that does not lie flat on the floor: on a stand, leaning on furniture.

Everything in ``anchor.locate`` and ``poses.boardless`` assumes a plate lying on a horizontal floor
(a floor plane within 20 degrees of up, footprints sliced at the floor's height, a 4-DoF pose). The
11-18 recordings break that: 509e5ece1b stands in a slotted stand at about 64 degrees (the stand
hides part of the lower edge), 6c417d24a0 leans on a cabinet at about 76 degrees. Here:

1. ``detect_placement``: the plane the cameras aim at (RANSAC on the LiDAR points of the central
   third of each depth map) against the floor (the lowest large near-horizontal plane). Tilted when
   the aimed plane leans more than 10 degrees and its centre is more than 15 mm above the floor.
2. ``locate_tilted``:
   - SAM 2 is prompted with the aimed plane's centre on a few keyframes; of its nested answers,
     the one whose LiDAR points lie on the aimed plane and form a phone- or tablet-sized rectangle
     wins, and its masks are tracked through every keyframe. The stand is not in the device's mask,
     so it is left out without anyone marking it.
   - Size and first pose: the floor path's footprint (``locate.footprint``, ``fit_footprint``,
     ``debias_footprint``) on the aimed plane instead of the floor. The cameras are moved into a
     frame in which that plane is z = 0, so the share of views in which each plane cell is inside the
     device mask is fitted by a rounded rectangle exactly as for a device lying on the floor. The box
     of the masked LiDAR points (``template.plate_box``) is only the fallback: edge pixels mixing
     device and background made it 18 x 14 mm too large on 509e5ece1b (299.7 x 229.6 for an iPad of
     281.6 x 215.5), too far for the outline fit's windows, and the registration failed (2026-11-18).
3. The caller (``tilted_inits``) turns the plate into the two 180-degree twins of ``T_world_cad`` with
   the model's thickness and registers from there with 6-DoF and wide windows
   (``register.register_device``) on the ARKit poses (no floor, so no floor-based drift fit).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..geometry.transforms import inv_T, make_T, rot_z


@dataclass
class TiltedPlate:
    """What ``estimate_device`` needs of a located plate (like ``floor_outline.FloorPlate``)."""
    length: float
    width: float
    corner_radius: float
    top_height: float | None
    iou: float
    info: dict


@dataclass
class TiltedLocated:
    plate: TiltedPlate | None
    masks: dict[int, np.ndarray]
    mask_scale: float
    z_floor: float | None
    T_init: np.ndarray | None                 # world <- cad from the LiDAR box (fallback when there is no footprint)
    plane: dict
    info: dict = field(default_factory=dict)
    placement: str = "tilted"
    T_plane_world: np.ndarray | None = None   # the device's plane as z = 0
    footprint: object = None                  # floor_outline.FloorPlate in that frame (centre, heading, size)
    mid_plane: bool = False                   # z = 0 is the device's mid-plane (both faces filmed), else its filmed face


def _central_points(cap, ids, T_world_cam, window: float = 0.34, stride: int = 2) -> np.ndarray:
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
        y0, y1 = int(h * (0.5 - window / 2)), int(h * (0.5 + window / 2))
        x0, x1 = int(w * (0.5 - window / 2)), int(w * (0.5 + window / 2))
        vs, us = np.mgrid[y0:y1:stride, x0:x1:stride]
        z = d[y0:y1:stride, x0:x1:stride].astype(np.float32)
        ok = (z > 50) & (z < 1500)
        if conf is not None and conf.shape == d.shape:
            ok &= conf[y0:y1:stride, x0:x1:stride] >= 2
        pc = np.stack([(us - cx) / fx * z, (vs - cy) / fy * z, z], -1)[ok]
        T = T_world_cam[i]
        pts.append(pc @ T[:3, :3].T + T[:3, 3])
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def _ransac_plane(X: np.ndarray, th: float = 3.0, iters: int = 600, seed: int = 0) -> tuple[np.ndarray, float, np.ndarray]:
    rng = np.random.default_rng(seed)
    best, bestn = None, -1
    for _ in range(iters):
        a, b, c = X[rng.choice(len(X), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        k = int((np.abs((X - a) @ n) < th).sum())
        if k > bestn:
            best, bestn = (n, a), k
    n, a = best
    inl = np.abs((X - a) @ n) < th
    Y = X[inl]
    c = Y.mean(0)
    _, _, vt = np.linalg.svd(Y - c, full_matrices=False)
    n = vt[2]
    return n, float(-n @ c), np.abs(X @ n - (n @ c)) < th


def detect_placement(cap, ids: list[int], T_world_cam: dict[int, np.ndarray], tilt_deg: float = 10.0, min_height_mm: float = 15.0) -> dict:
    """Is the device lying flat or standing/leaning? World z is up (``poses.boardless.T_W_ARK``)."""
    sel = [ids[int(k)] for k in np.linspace(0, len(ids) - 1, min(24, len(ids))).round()]
    X = _central_points(cap, sel, T_world_cam)
    if len(X) < 500:
        return {"tilted": False, "reason": "too few LiDAR points"}
    n, d, inl = _ransac_plane(X)
    cams = np.stack([T_world_cam[i][:3, 3] for i in sel])
    if ((cams.mean(0) @ n) + d) < 0:                      # normal towards the cameras
        n, d = -n, -d
    c = X[inl].mean(0)
    tilt = float(np.degrees(np.arccos(np.clip(abs(n[2]), 0, 1))))
    # the floor: the lowest large near-horizontal plane among all depth points
    from .locate import lidar_points

    A = lidar_points(cap, sel, T_world_cam, stride=4)
    horiz = None
    if len(A) > 500:
        z = A[:, 2]
        lo = np.percentile(z, 5)
        band = A[z < lo + 40]
        if len(band) > 300:
            nf, df, inf_ = _ransac_plane(band, th=6.0)
            if abs(nf[2]) > 0.95:
                horiz = float(np.median(band[inf_, 2]))
    height = None if horiz is None else float(c[2] - horiz)
    tilted = tilt > tilt_deg and (height is None or height > min_height_mm)
    return {"tilted": bool(tilted), "tilt_deg": round(tilt, 1), "plane_normal": n.round(4).tolist(), "plane_d": round(float(d), 2),
            "centre": c.round(1).tolist(), "floor_z": None if horiz is None else round(horiz, 1),
            "height_mm": None if height is None else round(height, 1), "inlier_share": round(float(inl.mean()), 3)}


def _mask_points(cap, i: int, mask: np.ndarray, T_world_cam: np.ndarray, erode: int = 1) -> np.ndarray:
    d = cap.depth(i)
    if d is None:
        return np.zeros((0, 3))
    conf = cap.confidence(i)
    h, w = d.shape[:2]
    m = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    if erode:
        m = cv2.erode(m, np.ones((2 * erode + 1,) * 2, np.uint8))
    ok = (m > 0) & (d > 50)
    if conf is not None and conf.shape == d.shape:
        ok &= conf >= 2
    v, u = np.nonzero(ok)
    K = cap.K[i]
    sx, sy = w / cap.width, h / cap.height
    fx, fy = K[0, 0] * sx, K[1, 1] * sy
    cx, cy = (K[0, 2] + 0.5) * sx - 0.5, (K[1, 2] + 0.5) * sy - 0.5
    z = d[v, u].astype(np.float32)
    pc = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], 1)
    return pc @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]


def locate_tilted(cap, frames: dict[int, np.ndarray], cams: dict, T_world_cam: dict[int, np.ndarray], face_up: str, plane: dict,
                  n_prompt: int = 6, log=print) -> TiltedLocated:
    from ..segment.device_masks import object_hypotheses, track_device
    from .template import plate_box

    ids = sorted(frames)
    n = np.asarray(plane["plane_normal"], float)
    c = np.asarray(plane["centre"], float)
    # 1. prompts: the aimed plane's centre on a few keyframes where it is well inside the image
    pts = {}
    for i in ids:
        Tc = inv_T(T_world_cam[i])
        pc = Tc[:3, :3] @ c + Tc[:3, 3]
        if pc[2] < 50:
            continue
        cam = cams[i]
        u, v = cam.fx * pc[0] / pc[2] + cam.cx, cam.fy * pc[1] / pc[2] + cam.cy
        if 0.2 * cam.width < u < 0.8 * cam.width and 0.2 * cam.height < v < 0.8 * cam.height:
            pts[i] = (u, v)
    if not pts:
        return TiltedLocated(None, {}, 0.25, None, None, plane, {"error": "the aimed plane's centre is never inside the frame"})
    keys = sorted(pts)
    pick = [keys[int(k)] for k in np.linspace(0, len(keys) - 1, min(n_prompt, len(keys))).round()]
    hyps = object_hypotheses({i: frames[i] for i in pick}, {i: pts[i] for i in pick})
    # 2. the hypothesis whose points lie on the aimed plane and make a device-sized rectangle
    prompts, sizes = {}, []
    for i in pick:
        best = None
        for mask, score in hyps.get(i, []):
            P = _mask_points(cap, i, mask, T_world_cam[i])
            if len(P) < 200:
                continue
            on = np.abs(P @ n + plane["plane_d"]) < 12.0
            if on.mean() < 0.6:
                continue
            T, size = plate_box(P[on], np.stack([T_world_cam[i][:3, 3]]), face_up)
            L, W = float(size[0]), float(size[1])
            if not (40 <= L <= 420 and 20 <= W <= 320 and 1.0 <= L / max(W, 1e-6) <= 3.2):
                continue
            area = float(mask.sum())
            if best is None or area > best[0]:
                best = (area, mask, L, W)
        if best is not None:
            prompts[i] = {"mask": best[1]}
            sizes.append((best[2], best[3]))
    if not prompts:
        return TiltedLocated(None, {}, 0.25, None, None, plane, {"error": "no SAM 2 answer lies on the aimed plane at device size"})
    log(f"  tilted: SAM 2 prompts on {len(prompts)} keyframes, first sizes {np.median(sizes, axis=0).round(1).tolist()} mm")
    masks = track_device({i: frames[i] for i in ids}, prompts, log=log)
    # 3. the device's own plane from the masked LiDAR points. The aimed plane (RANSAC on the central
    #    third of the depth maps, 26 % inliers on 509e5ece1b) can be tilted against the device by
    #    several degrees; projected through it, the views from the far side stretched the device to
    #    276 mm wide (2026-11-18). Its normal points to where the recording starts (the declared face).
    P = np.concatenate([_mask_points(cap, i, m, T_world_cam[i]) for i, m in list(masks.items())[:: max(1, len(masks) // 40)]])
    cams_c = np.stack([T_world_cam[i][:3, 3] for i in ids])
    early = cams_c[: max(3, len(ids) // 5)].mean(0)
    dev = _robust_plane(P, early)
    if dev is not None:
        info_plane = {"aimed_vs_device_deg": round(float(np.degrees(np.arccos(np.clip(abs(dev["plane_normal"] @ n), 0, 1)))), 2)}
        plane = {**plane, **dev}
        n = np.asarray(plane["plane_normal"], float)
    else:
        info_plane = {}
    P = P[np.abs(P @ n + plane["plane_d"]) < 12.0]
    T0, size = plate_box(P, cams_c, face_up)
    L, W = float(size[0]), float(size[1])
    r = float(np.clip(0.065 * W, 6.0, 16.0))
    info = {"prompts": len(prompts), "masks": len(masks), "tilt_deg": plane.get("tilt_deg"),
            "lidar_box_mm": [round(L, 1), round(W, 1)], "points": int(len(P)), **info_plane}
    # 4. the size view by view, from each frame's own LiDAR plane and mask outline (camera frame: no
    #    ARKit drift in it; around the stand on 509e5ece1b the drift between the two sides was 36 mm
    #    and the views' masks, put together on one plane, agreed only on a plate 22 mm too narrow)
    pv = per_view_size(cap, masks, T_world_cam=T_world_cam, n_world=np.asarray(plane["plane_normal"], float))
    if pv is not None:
        info["per_view"] = pv
    # 5. position and heading: the footprint on the device's plane (the floor path's rule, in a frame
    #    where that plane is z = 0) from the first 35 % of the views (before the drift grows), the shape
    #    held at the per-view size. Filmed from both sides, the plane through all its points is the mid-plane
    T_pw = plane_frame(plane)
    Tp = {i: T_pw @ T_world_cam[i] for i in masks}
    side = np.array([Tp[i][2, 3] > 0 for i in sorted(Tp)])
    mid = bool(min(side.mean(), 1 - side.mean()) >= 0.15)
    info.update(views_front=int(side.sum()), views_back=int((~side).sum()), plane_is="mid-plane" if mid else "filmed face")
    early_ids = sorted(masks)[: max(12, int(round(0.35 * len(masks))))]
    if pv is not None:
        size, how = (pv["length_mm"], pv["width_mm"], float(np.clip(0.065 * pv["width_mm"], 6.0, 16.0))), "per-view LiDAR size"
    else:
        # no single view to measure (6c417d24a0's close views all touch the image border): the footprint of
        # all views (few views, little drift there); its shape, then the early views for position and heading
        whole = _plane_footprint(masks, cams, Tp, log, mid=mid)
        size, how = ((whole.length, whole.width, whole.corner_radius), "footprint of all views") if whole is not None else (None, "")
    fp = None
    if size is not None:
        # position and heading from the early views (before the drift grows) or from all of them (little
        # drift, more of the device seen), whichever the held shape fits better
        cands = [(_plane_footprint({i: masks[i] for i in early_ids}, cams, Tp, log, mid=mid, size=size), "early"),
                 (_plane_footprint(masks, cams, Tp, log, mid=mid, size=size), "all")]
        cands = [(f, w) for f, w in cands if f is not None]
        if cands:
            fp, which = max(cands, key=lambda c: c[0].iou)
            info["footprint_views"] = which
    if fp is not None:
        L, W, r = fp.length, fp.width, fp.corner_radius
        info.update(fp.info, **{"from": f"{how}, {info['footprint_views']} views' footprint on the device's plane"})
    else:
        info["from"] = "tilted LiDAR plate box"
    plate = TiltedPlate(L, W, r, None, float(fp.iou) if fp is not None else 0.0, {**info, "size_mm": [round(L, 2), round(W, 2)]})
    log(f"  tilted: plate {L:.1f} x {W:.1f} mm, corner r {r:.1f} ({info['from']}; LiDAR box {info['lidar_box_mm']} from {len(P)} points, "
        f"tilt {plane.get('tilt_deg')} deg)")
    return TiltedLocated(plate, masks, 0.25, None, T0, plane, plate.info, T_plane_world=T_pw, footprint=fp, mid_plane=mid)


def _robust_plane(P: np.ndarray, towards: np.ndarray, rounds: int = 4) -> dict | None:
    """Plane through object points (SVD, points beyond 3 MAD dropped each round, like
    ``template.plate_box``), its normal towards ``towards``; with ``tilt_deg`` from the vertical."""
    if len(P) < 500:
        return None
    if len(P) > 80000:
        P = P[np.random.default_rng(0).choice(len(P), 80000, replace=False)]
    keep = np.ones(len(P), bool)
    for _ in range(rounds):
        c = np.median(P[keep], 0)
        _, _, Vt = np.linalg.svd(P[keep] - c, full_matrices=False)
        dist = (P - c) @ Vt[2]
        mad = np.median(np.abs(dist[keep])) + 1e-9
        keep = np.abs(dist) < 3.0 * max(mad, 0.5)
    nn = Vt[2] / np.linalg.norm(Vt[2])
    if (towards - c) @ nn < 0:
        nn = -nn
    return {"plane_normal": nn.round(5).tolist(), "plane_d": round(float(-nn @ c), 3), "centre": c.round(2).tolist(),
            "tilt_deg": round(float(np.degrees(np.arccos(np.clip(abs(nn[2]), 0, 1)))), 1), "plane_from": "masked LiDAR",
            "plane_rms_mm": round(float(np.sqrt(np.mean(((P[keep] - c) @ nn) ** 2))), 2)}


def plane_frame(plane: dict) -> np.ndarray:
    """``T_plane_world``: a frame whose z = 0 is the aimed plane and whose +z is its normal (towards the cameras)."""
    n = np.asarray(plane["plane_normal"], float)
    n /= np.linalg.norm(n)
    c = np.asarray(plane["centre"], float)
    c = c - (c @ n + plane["plane_d"]) * n                 # onto the plane
    a = np.cross([0.0, 0.0, 1.0], n)
    if np.linalg.norm(a) < 1e-6:
        a = np.array([1.0, 0.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    R = np.stack([a, b, n])                                # rows: the plane's axes in world coordinates
    return make_T(R, -R @ c)


def per_view_size(cap, masks: dict[int, np.ndarray], thick_guess: float = 5.0, theta_range=(20.0, 60.0), min_views: int = 4,
                  T_world_cam: dict[int, np.ndarray] | None = None, n_world: np.ndarray | None = None, max_normal_deg: float = 12.0) -> dict | None:
    """Length and width of the device from single views: the frame's masked LiDAR points give its plane
    in the camera frame, the mask's outline is cast onto it, and the smallest rectangle around it is the
    view's size. The outline is the silhouette of a slab, which reaches past the filmed face by the
    side band; a box of the measured size and ``thick_guess`` behind the plane, seen by the same camera,
    gives that excess, and it is taken off (one fixed-point step). Views between ``theta_range`` from the
    plane's normal (flatter ones fit the plane badly, steeper ones see mostly the band), not touching the
    image border, and (with ``T_world_cam`` and ``n_world``) whose plane turned into the world agrees with
    the device's plane within ``max_normal_deg``: over the black screen of 509e5ece1b the LiDAR plane was
    wrong and the outline cast onto it came out 540 x 160 mm. The median. The iPad's back views gave
    about 286 x 218 mm before the correction (iPad 281.6 x 215.5). None with fewer than ``min_views``."""
    rows = []
    for i, m in masks.items():
        if m.sum() < 200:
            continue
        if m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any():
            continue
        P = _mask_points(cap, i, m, np.eye(4), erode=2)                 # camera frame, mm
        pl = _robust_plane(P, np.zeros(3))
        if pl is None or pl["plane_rms_mm"] > 3.0:
            continue
        n, c = np.asarray(pl["plane_normal"]), np.asarray(pl["centre"])
        theta = float(np.degrees(np.arccos(np.clip(abs(n @ c) / np.linalg.norm(c), 0, 1))))
        if not theta_range[0] <= theta <= theta_range[1]:
            continue
        if T_world_cam is not None and n_world is not None and i in T_world_cam:
            nw = T_world_cam[i][:3, :3] @ n
            if np.degrees(np.arccos(np.clip(abs(float(nw @ n_world)), 0, 1))) > max_normal_deg:
                continue
        cam = cap.camera(i)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea)[:, 0, :].astype(np.float64)
        s = cam.width / m.shape[1]
        uv_px = np.stack([(cnt[:, 0] + 0.5) * s - 0.5, (cnt[:, 1] + 0.5) * s - 0.5], 1)
        a = np.cross(n, [1.0, 0, 0]) if abs(n[0]) < 0.9 else np.cross(n, [0, 1.0, 0])
        a /= np.linalg.norm(a)
        b = np.cross(n, a)

        def on_plane(px):
            rays = np.stack([(px[:, 0] - cam.cx) / cam.fx, (px[:, 1] - cam.cy) / cam.fy, np.ones(len(px))], 1)
            X = rays * ((n @ c) / (rays @ n))[:, None]
            return np.stack([(X - c) @ a, (X - c) @ b], 1).astype(np.float32)

        (cx, cy), (w, h), ang = cv2.minAreaRect(on_plane(uv_px))
        # the excess a box of this size shows through this camera
        t = np.radians(ang)
        ea, eb = np.cos(t) * a + np.sin(t) * b, -np.sin(t) * a + np.cos(t) * b
        ctr = c + cx * a + cy * b
        corners = [ctr + sx * w / 2 * ea + sy * h / 2 * eb + dz * n for sx in (-1, 1) for sy in (-1, 1) for dz in (0.0, -thick_guess)]
        C = np.array(corners)
        px = np.stack([cam.fx * C[:, 0] / C[:, 2] + cam.cx, cam.fy * C[:, 1] / C[:, 2] + cam.cy], 1)
        hull = cv2.convexHull(px.astype(np.float32))[:, 0, :].astype(np.float64)
        (_, _), (w2, h2), ang2 = cv2.minAreaRect(on_plane(hull))
        if abs(((ang2 - ang) + 45) % 90 - 45) > 10:           # the box's rectangle turned: sides swapped
            w2, h2 = h2, w2
        wc, hc = w - (w2 - w), h - (h2 - h)
        rows.append((max(wc, hc), min(wc, hc), max(w, h), min(w, h), theta))
    if len(rows) < min_views:
        return None
    R = np.array(rows)
    L, W = float(np.median(R[:, 0])), float(np.median(R[:, 1]))
    return {"length_mm": round(L, 2), "width_mm": round(W, 2), "views": len(rows),
            "raw_median_mm": [round(float(np.median(R[:, 2])), 2), round(float(np.median(R[:, 3])), 2)],
            "iqr_mm": [round(float(np.subtract(*np.percentile(R[:, 0], [75, 25]))), 2), round(float(np.subtract(*np.percentile(R[:, 1], [75, 25]))), 2)],
            "theta_median_deg": round(float(np.median(R[:, 4])), 1)}


def _plane_footprint(masks, cams, T_plane_cam, log, half: float = 260.0, res: float = 1.0, thick_guess: float = 6.0, mid: bool = False,
                     size: tuple | None = None):
    """Rounded rectangle fitted to the masks' footprint on z = 0 of ``T_plane_cam``'s frame, debiased like
    the floor path's: the device as a box from z = -``thick_guess`` to 0 (the filmed face is the plane)
    or, with ``mid``, centred on it."""
    from .locate import debias_footprint, fit_footprint, footprint

    n = int(round(2 * half / res))
    x0, y0 = -half, half
    share, seen = footprint(masks, cams, T_plane_cam, 0.0, x0, y0, res, n)
    if size is not None:
        # the shape is known (per-view size): position and heading only
        pl, _ = fit_footprint(share, seen, x0, y0, res, min_views=4, size=size)
        if pl is not None:
            log(f"  tilted: early footprint on the device's plane: heading {pl.long_axis_deg:.1f} deg, IoU {pl.iou:.3f} "
                f"with the per-view size {size[0]:.1f} x {size[1]:.1f} mm")
            return pl
    pl, _ = fit_footprint(share, seen, x0, y0, res)
    if pl is None or not (40.0 <= pl.length <= 420.0 and 20.0 <= pl.width <= 320.0):
        log(f"  tilted: no device-sized footprint on the device's plane ({None if pl is None else [round(pl.length, 1), round(pl.width, 1)]})")
        return None
    measured = (pl.length, pl.width)
    z0 = -thick_guess / 2 if mid else -thick_guess
    pl, rec = debias_footprint(pl, masks, cams, T_plane_cam, z0, thick_guess, 0.0, x0, y0, res, n)
    log(f"  tilted: footprint on the aimed plane {measured[0]:.1f} x {measured[1]:.1f} mm (IoU {pl.iou:.3f})"
        + (f", debiased {pl.length:.1f} x {pl.width:.1f}" if rec and rec.get("used") else ""))
    return pl


def tilted_inits(loc: TiltedLocated, model, face_up: str) -> list[np.ndarray]:
    """The two 180-degree twins of ``T_world_cad`` for a tilted plate: the device as if lying with its
    filmed face up on the aimed plane (``parametric.rest_pose``, then the footprint's heading and centre),
    taken back to the world. Without a footprint, the LiDAR box and its twin."""
    from .parametric import rest_pose

    if loc.footprint is None or loc.T_plane_world is None:
        return [loc.T_init, loc.T_init @ make_T(np.diag([-1.0, -1.0, 1.0]))]
    fp = loc.footprint
    R0 = rest_pose(model, face_up)                           # long side along +y, lowest point at z = 0
    top = float(np.max((model.mesh.vertices @ R0[:3, :3].T + R0[:3, 3])[:, 2]))
    T_wp = inv_T(loc.T_plane_world)
    yaw = fp.long_axis_deg - 90.0
    z = -top / 2 if loc.mid_plane else -top
    return [T_wp @ make_T(rot_z(yaw + tw), [fp.centre_xy[0], fp.centre_xy[1], z]) @ R0 for tw in (0.0, 180.0)]
