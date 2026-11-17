"""Evidence frames: for each defect, the original frames that show it best.

Every view's samples of the defect's texels are already in memory (``surface.maps``), so each view
gets a score without re-reading video:

    score = (share of the defect's texels visible) x (mean incidence cos)
            x contrast / (contrast + 4)        # Lab difference to a ring of texels around it
            x sharpness of the frame (normalised)

The best view is taken first, then up to two more whose viewing directions differ by at least
``min_sep_deg`` from those already chosen, so the buyer sees the defect from different angles. The
same per-view contrast also tells how persistent a glass line is across views, which separates
cracks (seen in most views) from scratches (seen only at some angles).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi

from ..geometry.camera import Camera
from ..geometry.transforms import inv_T
from ..surface.maps import ChartMaps, lab_of
from ..utils.image import imwrite


@dataclass
class ViewScore:
    frame: int
    score: float
    visible: float
    cos: float
    contrast: float
    direction: np.ndarray


def ring_texels(cm: ChartMaps, texels: np.ndarray, inner_mm: float = 0.4, outer_mm: float = 1.5) -> np.ndarray:
    g = cm.geom
    m = np.zeros(g.shape[0] * g.shape[1], bool)
    m[g.flat_index[texels]] = True
    m = m.reshape(g.shape)
    a = ndi.binary_dilation(m, iterations=max(int(round(outer_mm / g.res)), 1))
    b = ndi.binary_dilation(m, iterations=max(int(round(inner_mm / g.res)), 1))
    ring = (a & ~b).ravel()
    flat_to_valid = np.full(g.shape[0] * g.shape[1], -1, np.int64)
    flat_to_valid[g.flat_index] = np.arange(len(g.flat_index))
    idx = flat_to_valid[np.flatnonzero(ring)]
    return idx[idx >= 0]


def view_scores(cm: ChartMaps, texels: np.ndarray, T_world_cad: np.ndarray, T_world_cam: dict[int, np.ndarray],
                sharpness: dict[int, float], min_cos: float = 0.3) -> list[ViewScore]:
    s = cm.samples
    ring = ring_texels(cm, texels)
    cos_r = s.cos[:, texels].astype(np.float32)
    cos_g = s.cos[:, ring].astype(np.float32) if len(ring) else np.zeros((len(s.frames), 0), np.float32)
    lab_r = lab_of(s.rgb[:, texels])
    lab_g = lab_of(s.rgb[:, ring]) if len(ring) else None
    centre = cm.geom.point[texels].mean(0) @ T_world_cad[:3, :3].T + T_world_cad[:3, 3]
    sh = np.array([sharpness.get(f, 0.0) for f in s.frames], float)
    sh_n = sh / (np.percentile(sh, 90) + 1e-6) if len(sh) else sh
    out = []
    for vi, f in enumerate(s.frames):
        seen = cos_r[vi] >= min_cos
        if seen.sum() < max(3, 0.3 * len(texels)):
            continue
        gseen = cos_g[vi] >= min_cos if lab_g is not None else np.zeros(0, bool)
        if gseen.sum() < 5:
            continue
        c = float(np.linalg.norm(lab_r[vi][seen].mean(0) - lab_g[vi][gseen].mean(0)))
        vis = float(seen.mean())
        mc = float(cos_r[vi][seen].mean())
        d = T_world_cam[f][:3, 3] - centre
        d /= np.linalg.norm(d)
        score = vis * mc * (c / (c + 4.0)) * float(np.clip(sh_n[vi], 0.2, 1.0))
        out.append(ViewScore(int(f), score, vis, mc, c, d))
    return out


def pick_diverse(scores: list[ViewScore], n: int = 3, min_sep_deg: float = 25.0) -> list[ViewScore]:
    order = sorted(scores, key=lambda v: -v.score)
    chosen: list[ViewScore] = []
    for v in order:
        if all(np.degrees(np.arccos(np.clip(v.direction @ c.direction, -1, 1))) >= min_sep_deg for c in chosen):
            chosen.append(v)
        if len(chosen) == n:
            break
    return chosen


def persistence(scores: list[ViewScore], thr: float = 3.0) -> float:
    """Share of the views that saw the defect where it stood out from its surroundings."""
    if not scores:
        return 0.0
    return float(np.mean([v.contrast > thr for v in scores]))


def project_region(cm: ChartMaps, texels: np.ndarray, T_world_cad: np.ndarray, cam: Camera, T_world_cam: np.ndarray) -> np.ndarray:
    P = cm.geom.point[texels] @ T_world_cad[:3, :3].T + T_world_cad[:3, 3]
    Tcw = inv_T(T_world_cam)
    pc = P @ Tcw[:3, :3].T + Tcw[:3, 3]
    return np.stack([cam.fx * pc[:, 0] / pc[:, 2] + cam.cx, cam.fy * pc[:, 1] / pc[:, 2] + cam.cy], 1)


def region_outline(px: np.ndarray, shape: tuple[int, int], dilate: int = 3) -> list[np.ndarray]:
    """Outer contour(s) of the projected texels as polygons in pixel coordinates."""
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    q = np.round(px).astype(int)
    ok = (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
    m[q[ok, 1], q[ok, 0]] = 255
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1)))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # simplified to ~1.5 px: full contours made the viewer's JSON 6.8 MB for a two-pass capture
    cs = [cv2.approxPolyDP(c, 1.5, True) for c in cs]
    return [c.reshape(-1, 2) for c in cs if len(c) >= 3]


def render_evidence(img: np.ndarray, px: np.ndarray, out_crop, out_full, crop_px: int = 512, full_w: int = 1280) -> dict:
    """Write a red-outlined close-up and a full-frame thumbnail; return the boxes used."""
    h, w = img.shape[:2]
    polys = region_outline(px, (h, w))
    x0, y0 = px.min(0)
    x1, y1 = px.max(0)
    size = max(x1 - x0, y1 - y0)
    half = max(70.0, 0.9 * size + 30.0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bx0, by0 = int(max(0, cx - half)), int(max(0, cy - half))
    bx1, by1 = int(min(w, cx + half)), int(min(h, cy + half))
    crop = img[by0:by1, bx0:bx1].copy()
    th = max(2, int(round((bx1 - bx0) / crop_px * 2.5)))
    for p in polys:
        cv2.polylines(crop, [(p - [bx0, by0]).astype(np.int32)], True, (230, 30, 30), th, cv2.LINE_AA)
    sc = crop_px / max(crop.shape[1], 1)
    crop = cv2.resize(crop, (crop_px, max(int(round(crop.shape[0] * sc)), 1)), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
    imwrite(str(out_crop), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    full = img.copy()
    pad = int(0.25 * half)
    cv2.rectangle(full, (int(x0) - pad, int(y0) - pad), (int(x1) + pad, int(y1) + pad), (230, 30, 30), max(3, w // 400), cv2.LINE_AA)
    fs = full_w / w
    full = cv2.resize(full, (full_w, int(round(h * fs))), interpolation=cv2.INTER_AREA)
    imwrite(str(out_full), cv2.cvtColor(full, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {"bbox_px": [float(x0), float(y0), float(x1), float(y1)], "crop_box": [bx0, by0, bx1, by1]}
