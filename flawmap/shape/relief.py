"""The face's relief above the plate model (plan C): what stands proud of a flat face, from the views.

A plate fitted to the outline has flat faces, but a back face carries a camera module 1-2 mm high
and a phone's lens rings stand higher still. Two measurements find them without a drawing:

1. **Plane + parallax** (``consistency_heights``): every keyframe of the pass that sees the face is
   first aligned to the face's surface colour (``texture.decompose``) on the face plane (ECC,
   affine): that removes the poses' own error, 1.9 mm in the face on the iPad back, which is as
   large as the parallax of a 1-2 mm step and made the first version call the flat print 2.5 mm
   high. With that correction kept, the face's points are lifted by ``h`` along its normal and
   sampled again; texture that really lies at that height agrees across views (low spread of the
   high-passed grey). Per 1.2 mm block the height with the lowest spread wins (parabola through the
   minimum) where the curve has a clear minimum and enough texture; heights are relative to the
   median of the flat face. A textureless face keeps NaN: nothing is invented.
2. **Silhouettes** (``hull_heights``): a point at height ``h`` above the face that every view sees
   inside the device mask (SAM 2, ``anchor.locate``) belongs to the device: the visual hull's top
   per block. The masks are ~1 mm per pixel, so only what stands a millimetre or more proud counts.

``relief_of`` combines them: the consistency height where it is known, the hull's height where it
says the surface is raised by at least ``hull_min_mm`` and the consistency says nothing; raised
regions (``raised``) are the compact components standing ``raised_mm`` or more. They become design
elements for the detectors and a displaced patch in the exported model (``export.bundle``).
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from ..geometry.transforms import inv_T
from ..surface.maps import ChartGeometry, chart_geometry


def _project(cam, T_cad_cam: np.ndarray, P: np.ndarray):
    Tcc = inv_T(T_cad_cam)
    pc = P @ Tcc[:3, :3].T + Tcc[:3, 3]
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cam.fx * pc[:, 0] / z + cam.cx
        v = cam.fy * pc[:, 1] / z + cam.cy
    ok = (z > 1.0) & (u >= 1) & (v >= 1) & (u < cam.width - 2) & (v < cam.height - 2)
    return u.astype(np.float32), v.astype(np.float32), ok


def _visible(geom: ChartGeometry, T_cad_cam: np.ndarray, min_cos: float = 0.3) -> np.ndarray:
    c = T_cad_cam[:3, 3]
    d = c[None, :] - geom.point
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return np.einsum("ij,ij->i", d, geom.normal) > min_cos


def _box(img: np.ndarray, m: np.ndarray, k: int) -> np.ndarray:
    num = cv2.boxFilter(img * m, -1, (k, k), normalize=False)
    den = cv2.boxFilter(m, -1, (k, k), normalize=False)
    return num / np.maximum(den, 1e-6)


def _sample_at(geom: ChartGeometry, grey: np.ndarray, cam, T_cad_cam: np.ndarray, vis: np.ndarray, h: float):
    rows, cols = geom.shape
    P = geom.point + h * geom.normal
    u, v, ok = _project(cam, T_cad_cam, P)
    ok &= vis
    mu = np.full(rows * cols, -10.0, np.float32)
    mv = np.full(rows * cols, -10.0, np.float32)
    mu[geom.flat_index[ok]], mv[geom.flat_index[ok]] = u[ok], v[ok]
    img = cv2.remap(grey, mu.reshape(rows, cols), mv.reshape(rows, cols), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    m = np.zeros(rows * cols, np.float32)
    m[geom.flat_index[ok]] = 1.0
    m = m.reshape(rows, cols)
    return img * m, m


def plane_alignments(geom: ChartGeometry, grey: dict, cams: dict, T_cad_cam: dict, views: list, vis: dict,
                     reference: np.ndarray, max_shift_mm: float = 4.0) -> dict:
    """Per view, the affine (chart pixels) that aligns its face-plane resampling to ``reference``
    (grey, on ``geom``'s grid), or None (``texture.allframes._align``)."""
    from ..texture.allframes import _align

    ref_valid = reference > 0
    out = {}
    for i in views:
        img, m = _sample_at(geom, grey[i], cams[i], T_cad_cam[i], vis[i], 0.0)
        M = _align(img, m > 0, reference, ref_valid, 1)
        if M is not None and np.hypot(M[0, 2], M[1, 2]) * geom.res <= max_shift_mm:
            out[i] = M
        else:
            out[i] = None
    return out


def consistency_heights(geom: ChartGeometry, frames: dict, cams: dict, T_cad_cam: dict, views: list,
                        heights=tuple(np.arange(-1.0, 3.01, 0.25)), hp_mm: float = 1.0, block_mm: float = 1.2,
                        min_depth: float = 0.12, min_texture: float = 2.0, reference: np.ndarray | None = None) -> dict:
    """Per texel of ``geom`` (a coarse face chart): the height above the face where the views agree
    best, NaN where the spread-vs-height curve has no clear inner minimum (``min_depth`` of its
    median) or the face has little texture there (median spread below ``min_texture`` grey levels).
    ``reference``: the face's grey surface colour on ``geom``'s grid; views are aligned to it on the
    face plane first (plane + parallax), and views that do not align are left out."""
    rows, cols = geom.shape
    grey = {i: cv2.GaussianBlur(cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 0.7) for i in views}
    vis = {i: _visible(geom, T_cad_cam[i]) for i in views}
    aff = plane_alignments(geom, grey, cams, T_cad_cam, views, vis, reference) if reference is not None else {i: None for i in views}
    if reference is not None:
        views = [i for i in views if aff[i] is not None]
    k_hp = max(3, int(round(hp_mm / geom.res)) | 1)
    k_bl = max(3, int(round(block_mm / geom.res)) | 1)
    curves = []
    for h in heights:
        acc = np.zeros(rows * cols, np.float64)
        acc2 = np.zeros(rows * cols, np.float64)
        cnt = np.zeros(rows * cols, np.float64)
        for i in views:
            img, m = _sample_at(geom, grey[i], cams[i], T_cad_cam[i], vis[i], float(h))
            if aff[i] is not None:
                img = cv2.warpAffine(img, aff[i], (cols, rows), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                m = (cv2.warpAffine(m, aff[i], (cols, rows), flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP) > 0.5).astype(np.float32)
                img = img * m
            hp = (img - _box(img, m, k_hp)) * (m > 0)
            acc += hp.ravel()
            acc2 += (hp * hp).ravel()
            cnt += m.ravel()
        with np.errstate(all="ignore"):
            mean = acc / np.maximum(cnt, 1)
            var = acc2 / np.maximum(cnt, 1) - mean ** 2
        var = np.where(cnt >= 3, np.maximum(var, 0), np.nan).reshape(rows, cols).astype(np.float32)
        okv = np.isfinite(var).astype(np.float32)
        curves.append(np.sqrt(_box(np.nan_to_num(var), okv, k_bl)) * np.where(okv > 0, 1.0, np.nan))
    C = np.stack(curves)                                     # (H, rows, cols)
    hs = np.asarray(heights, float)
    with np.errstate(all="ignore"):
        k = np.nanargmin(np.where(np.isfinite(C), C, np.inf), 0)
        med = np.nanmedian(C, 0)
    kk = np.clip(k, 1, len(hs) - 2)
    y0 = np.take_along_axis(C, (kk - 1)[None], 0)[0]
    y1 = np.take_along_axis(C, kk[None], 0)[0]
    y2 = np.take_along_axis(C, (kk + 1)[None], 0)[0]
    den = y0 - 2 * y1 + y2
    with np.errstate(all="ignore"):
        off = np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / den, 0.0)
        h_star = hs[kk] + np.clip(off, -0.5, 0.5) * (hs[1] - hs[0])
        depth = (med - y1) / np.maximum(med, 1e-6)
    inner = (k > 0) & (k < len(hs) - 1)
    good = inner & (depth >= min_depth) & (med >= min_texture) & geom.valid
    return {"height": np.where(good, h_star, np.nan).astype(np.float32), "depth": depth.astype(np.float32),
            "texture": med.astype(np.float32), "heights": hs, "views_used": len(views)}


def hull_heights(geom: ChartGeometry, masks: dict, cams: dict, T_cad_cam: dict, views: list,
                 heights=tuple(np.arange(0.0, 4.01, 0.25)), min_views: int = 6, share: float = 0.95) -> np.ndarray:
    """Per texel: the highest ``h`` at which the point lifted by ``h`` along the face normal lies
    inside the device mask in at least ``share`` of the views that see it (at least ``min_views``),
    i.e. the top of the visual hull above the face; 0 where even the face itself is not."""
    rows, cols = geom.shape
    top = np.zeros(rows * cols, np.float32)
    for h in heights:
        P = geom.point + h * geom.normal
        inside = np.zeros(len(P), np.int32)
        seen = np.zeros(len(P), np.int32)
        for i in views:
            if i not in masks:
                continue
            m = masks[i]
            u, v, ok = _project(cams[i], T_cad_cam[i], P)
            sx, sy = m.shape[1] / cams[i].width, m.shape[0] / cams[i].height
            mu = np.clip(((u + 0.5) * sx).astype(np.int64), 0, m.shape[1] - 1)
            mv = np.clip(((v + 0.5) * sy).astype(np.int64), 0, m.shape[0] - 1)
            seen += ok
            inside += ok & m[mv, mu]
        good = (seen >= min_views) & (inside >= share * np.maximum(seen, 1))
        top[geom.flat_index[good]] = np.maximum(top[geom.flat_index[good]], h)
    return top.reshape(rows, cols)


def relief_of(model, chart: str, frames: dict, cams: dict, T_cad_cam: dict, masks: dict | None = None, res: float = 0.3,
              n_views: int = 30, raised_mm: float = 0.6, hull_min_mm: float = 1.0, min_area_mm2: float = 12.0,
              reference: np.ndarray | None = None, log=None) -> dict:
    """Relief of one face chart (``screen`` or ``back``) from the keyframes of the pass that sees it.
    Returns ``height`` (rows, cols) mm on a ``res`` grid (0 where flat or unknown), the ``raised``
    mask on that grid, the coarse geometry, and a summary."""
    log = log or (lambda m: None)
    geom = chart_geometry(model, chart, res=res)
    ids = sorted(i for i in frames if _visible(geom, T_cad_cam[i]).mean() > 0.3)
    views = [ids[int(k)] for k in np.linspace(0, len(ids) - 1, min(n_views, len(ids))).round()] if ids else []
    if len(views) < 6:
        return {"height": np.zeros(geom.shape, np.float32), "raised": np.zeros(geom.shape, bool), "geom": geom,
                "info": {"views": len(views), "used": False}}
    ref = None
    if reference is not None:
        ref = cv2.resize(reference.astype(np.float32), (geom.shape[1], geom.shape[0]), interpolation=cv2.INTER_AREA)
        if ref.ndim == 3:
            ref = ref.mean(-1)
        ref = np.where(geom.valid, ref, 0.0).astype(np.float32)
    con = consistency_heights(geom, frames, cams, T_cad_cam, views, reference=ref)
    h = con["height"]
    known0 = np.isfinite(h)
    if known0.sum() > 50:
        # heights relative to the flat face (the fitted plate's face can sit a few tenths off)
        h = h - float(np.median(h[known0]))
    hull = hull_heights(geom, masks, cams, T_cad_cam, views) if masks else np.zeros(geom.shape, np.float32)
    known = np.isfinite(h)
    height = np.where(known, h, 0.0)
    from_hull = ~known & (hull >= hull_min_mm)
    height = np.where(from_hull, hull, height).astype(np.float32)
    height = np.where(geom.valid, height, 0.0)
    # raised regions: compact components standing raised_mm or more (closing bridges the lens holes)
    from skimage.measure import label, regionprops

    up = ndi.binary_closing(height >= raised_mm, iterations=max(1, int(round(1.5 / res))))
    raised = np.zeros(geom.shape, bool)
    comps = []
    for rp in regionprops(label(up, connectivity=2)):
        area = rp.area * res * res
        if area < min_area_mm2 or rp.solidity < 0.5:
            continue
        rr, cc = rp.coords[:, 0], rp.coords[:, 1]
        hull_pts = cv2.convexHull(np.stack([cc, rr], 1).astype(np.int32))
        cv2.fillConvexPoly(raised.view(np.uint8), hull_pts, 1)
        comps.append({"centre_mm": [round(float(rp.centroid[1] * res), 1), round(float(rp.centroid[0] * res), 1)],
                      "size_mm": [round(float((rp.bbox[3] - rp.bbox[1]) * res), 1), round(float((rp.bbox[2] - rp.bbox[0]) * res), 1)],
                      "height_mm_p90": round(float(np.percentile(height[rr, cc], 90)), 2), "area_mm2": round(area, 1)})
    info = {"views": len(views), "views_aligned": con["views_used"], "used": True, "known_share": round(float(known[geom.valid].mean()), 3),
            "hull_raised_share": round(float(from_hull[geom.valid].mean()), 4),
            "flat_p50_mm": round(float(np.median(h[known & ~raised])), 3) if (known & ~raised).any() else None,
            "raised": comps}
    log(f"  relief {chart}: height known on {info['known_share']:.0%} of the face (flat parts p50 {info['flat_p50_mm']} mm); "
        f"raised: {[(c['centre_mm'], c['size_mm'], c['height_mm_p90']) for c in comps]}")
    return {"height": height, "raised": raised, "geom": geom, "info": info}


def resample_to(src: np.ndarray, src_geom: ChartGeometry, dst_geom: ChartGeometry, nearest: bool = False) -> np.ndarray:
    """A chart image on ``src_geom``'s grid resampled onto ``dst_geom``'s (same chart, other texel size)."""
    W, H = dst_geom.shape[1], dst_geom.shape[0]
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(src.astype(np.float32), (W, H), interpolation=interp)


def normal_map(height: np.ndarray, res: float, strength: float = 1.0) -> np.ndarray:
    """Tangent-space normal map (uint8 RGB) of a height field on a chart grid (u right, v down)."""
    gy, gx = np.gradient(cv2.GaussianBlur(height.astype(np.float32), (0, 0), 1.0), res)
    n = np.dstack([-gx * strength, gy * strength, np.ones_like(gx)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return np.clip((n * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)


def element_heights(geom: ChartGeometry, elements: list[np.ndarray], frames: dict, cams: dict, T_cad_cam: dict, views: list,
                    reference: np.ndarray, heights=tuple(np.arange(-0.5, 3.01, 0.25)), band_mm: float = 1.0,
                    min_depth: float = 0.05) -> list[dict]:
    """One height per element (bool masks on ``geom``'s grid, e.g. the design elements of
    ``detect.multiview``): views aligned on the face plane (plane + parallax, ``plane_alignments``),
    then the height at which the gradient magnitude of the element's rim (a ``band_mm`` band around
    its outline) agrees best across views. A rim is an edge in every view even where the part is
    glossy; aggregated over the whole rim the curve is far less noisy than per block."""
    rows, cols = geom.shape
    grey = {i: cv2.GaussianBlur(cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 0.7) for i in views}
    vis = {i: _visible(geom, T_cad_cam[i]) for i in views}
    ref = cv2.resize(reference.astype(np.float32), (cols, rows), interpolation=cv2.INTER_AREA)
    ref = (ref.mean(-1) if ref.ndim == 3 else ref) * geom.valid
    aff = plane_alignments(geom, grey, cams, T_cad_cam, views, vis, ref.astype(np.float32))
    views = [i for i in views if aff[i] is not None]
    k = max(1, int(round(band_mm / geom.res)))
    bands = [ndi.binary_dilation(e, iterations=k) & ~ndi.binary_erosion(e, iterations=k) & geom.valid for e in elements]
    curves = np.zeros((len(elements), len(heights)), np.float64)
    for hi, h in enumerate(heights):
        acc = np.zeros((rows, cols), np.float64)
        acc2 = np.zeros((rows, cols), np.float64)
        cnt = np.zeros((rows, cols), np.float64)
        for i in views:
            img, m = _sample_at(geom, grey[i], cams[i], T_cad_cam[i], vis[i], float(h))
            img = cv2.warpAffine(img, aff[i], (cols, rows), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            m = cv2.warpAffine(m, aff[i], (cols, rows), flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP) > 0.5
            gm = np.hypot(cv2.Sobel(img, cv2.CV_32F, 1, 0), cv2.Sobel(img, cv2.CV_32F, 0, 1)) * cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8))
            acc += gm
            acc2 += gm * gm
            cnt += m
        with np.errstate(all="ignore"):
            mean = acc / np.maximum(cnt, 1)
            sd = np.sqrt(np.maximum(acc2 / np.maximum(cnt, 1) - mean ** 2, 0))
            rel = sd / np.maximum(mean, 1e-3)
        for e, b in enumerate(bands):
            sel = b & (cnt >= 3)
            curves[e, hi] = float(np.median(rel[sel])) if sel.sum() > 20 else np.nan
    hs = np.asarray(heights, float)
    out = []
    for e, c in enumerate(curves):
        if not np.isfinite(c).any():
            out.append({"height_mm": None, "depth": None, "curve": []})
            continue
        k_ = int(np.nanargmin(c))
        med = float(np.nanmedian(c))
        depth = (med - c[k_]) / max(med, 1e-6)
        h_ = hs[k_]
        if 0 < k_ < len(hs) - 1:
            y0, y1, y2 = c[k_ - 1:k_ + 2]
            den = y0 - 2 * y1 + y2
            if abs(den) > 1e-9:
                h_ = hs[k_] + float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5)) * (hs[1] - hs[0])
        ok = 0 < k_ < len(hs) - 1 and depth >= min_depth
        out.append({"height_mm": round(float(h_), 2) if ok else None, "depth": round(float(depth), 3),
                    "curve": [round(float(x), 4) for x in c], "views": len(views)})
    return out
