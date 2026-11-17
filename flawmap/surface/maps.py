"""Unwrapped appearance maps: what every texel of the device looked like in every view.

Each chart (``screen``, ``back``, ``frame``) is a regular grid in the device's unwrapped
coordinates (mm). For every texel the visible surface point of the reference model is found once by
casting a ray "from outside" (down onto the screen, up onto the back, inwards onto the frame band).
Every registered frame then **pulls** colour into the texels it sees: texel points are projected,
kept if they face the camera and pass a depth test against the model rendered from that view, and
the frame is sampled bilinearly. The per-view samples stay in memory for the detectors and the
evidence frames; the maps keep robust statistics over views:

* ``median`` Lab and RGB (the texture of the inspection model; glare that appears in a few views
  is rejected),
* ``low`` (25th percentile) and ``high`` (80th percentile) of lightness,
* ``n_good``: number of views with incidence cos > 0.5 (coverage).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..anchor.parametric import CHARTS, DeviceModel
from ..geometry.camera import Camera
from ..geometry.transforms import inv_T
from .raycast import RaycastScene

DEFAULT_RES = {"screen": 0.1, "back": 0.1, "frame": 0.05}
MAX_FACE_TEXELS = 3.0e6     # per face; per-view samples cost ~5 bytes per texel and view
MAX_BAND_TEXELS = 1.5e6     # the frame band at 0.05 mm: an iPhone 12 ~1.3 M, a 13-inch tablet ~2.4 M (stopped
                            # by the memory guard on a 32 GB PC with two passes of 141 views, 2026-11-17)


def chart_res(model: DeviceModel, chart: str) -> float:
    """Texel size for ``chart``: the default, coarsened in 0.05 mm steps for large faces (a 13-inch
    tablet face at 0.1 mm is 6 M texels, over 4 GB of samples for 140 views). Phones keep 0.1 mm;
    the capture itself resolves ~0.27 mm per pixel at 40 cm."""
    res = DEFAULT_RES[chart]
    w, h = model.chart_size(chart)
    cap = MAX_FACE_TEXELS if chart in ("screen", "back") else MAX_BAND_TEXELS
    need = float(np.sqrt(w * h / cap))
    if need > res:
        step = 0.05 if chart in ("screen", "back") else 0.01
        res = float(np.ceil(need / step) * step)
    return res


@dataclass
class ChartGeometry:
    chart: str
    res: float                    # mm per texel
    shape: tuple[int, int]        # (rows, cols)
    valid: np.ndarray             # (rows, cols) bool
    point: np.ndarray             # (n_valid, 3) cad frame
    normal: np.ndarray            # (n_valid, 3) cad frame
    part: np.ndarray              # (n_valid,) part index
    face: np.ndarray              # (n_valid,) face index
    flat_index: np.ndarray        # (n_valid,) index into rows*cols

    def to_image(self, values: np.ndarray, fill=0) -> np.ndarray:
        out = np.full((self.shape[0] * self.shape[1],) + values.shape[1:], fill, dtype=values.dtype)
        out[self.flat_index] = values
        return out.reshape(self.shape + values.shape[1:])

    def uv_of(self, idx: np.ndarray) -> np.ndarray:
        """Chart coordinates (mm) of texel indices into the valid list."""
        r, c = np.divmod(self.flat_index[idx], self.shape[1])
        return np.stack([(c + 0.5) * self.res, (r + 0.5) * self.res], axis=1)


def chart_geometry(model: DeviceModel, chart: str, res: float | None = None) -> ChartGeometry:
    res = res or chart_res(model, chart)
    w, h = model.chart_size(chart)
    cols, rows = int(np.ceil(w / res)), int(np.ceil(h / res))
    cu = (np.arange(cols) + 0.5) * res
    rv = (np.arange(rows) + 0.5) * res
    U, V = np.meshgrid(cu, rv)
    W2, L2 = model.spec.width / 2.0, model.spec.length / 2.0
    if chart == "screen":
        xy = np.stack([U.ravel() - W2, L2 - V.ravel()], 1)
        o = np.column_stack([xy, np.full(len(xy), model.z_front + 30.0)])
        d = np.tile([0.0, 0.0, -1.0], (len(xy), 1))
    elif chart == "back":
        xy = np.stack([W2 - U.ravel(), L2 - V.ravel()], 1)
        o = np.column_stack([xy, np.full(len(xy), model.z_back - 30.0)])
        d = np.tile([0.0, 0.0, 1.0], (len(xy), 1))
    else:
        s = U.ravel()
        P = model.perimeter
        xs = np.interp(s, np.r_[model.outline_s, P], np.r_[model.outline_xy[:, 0], model.outline_xy[0, 0]])
        ys = np.interp(s, np.r_[model.outline_s, P], np.r_[model.outline_xy[:, 1], model.outline_xy[0, 1]])
        from ..anchor.parametric import outward_normals
        nrm = outward_normals(model.outline_xy)
        nx = np.interp(s, np.r_[model.outline_s, P], np.r_[nrm[:, 0], nrm[0, 0]])
        ny = np.interp(s, np.r_[model.outline_s, P], np.r_[nrm[:, 1], nrm[0, 1]])
        n2 = np.stack([nx, ny], 1)
        n2 /= np.linalg.norm(n2, axis=1, keepdims=True)
        z = model.z_front - V.ravel()
        o = np.column_stack([np.stack([xs, ys], 1) + 6.0 * n2, z])
        d = np.column_stack([-n2, np.zeros(len(n2))])
    scene = RaycastScene(model.mesh, None)
    t, face, bary, hit = scene._cast(o, d)
    if chart == "frame":   # the band ray must land within a few mm of the outline
        hit &= t < 12.0
    point = o + d * np.where(hit, t, 0.0)[:, None]
    idx = np.flatnonzero(hit)
    face_h = face[idx].astype(np.int64)
    normal = scene.face_normals_obj[face_h]
    normal = np.where((np.einsum("ij,ij->i", normal, d[idx]) > 0)[:, None], -normal, normal)   # face the viewer side
    return ChartGeometry(chart, res, (rows, cols), hit.reshape(rows, cols), point[idx], normal,
                         model.mesh.face_label[face_h], face_h, idx)


@dataclass
class ViewSamples:
    """Per-view colour samples of one chart: (n_views, n_texels) arrays, NaN-free with a mask."""
    frames: list[int]
    rgb: np.ndarray        # uint8 (V, N, 3)
    cos: np.ndarray        # float16 (V, N); 0 where not visible (pixel positions are re-projected when needed)

    @property
    def visible(self) -> np.ndarray:
        return self.cos > 0


@dataclass
class ChartMaps:
    geom: ChartGeometry
    samples: ViewSamples
    stats: dict = field(default_factory=dict)


def _bilinear_rgb(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x = np.clip(u, 0, w - 1.001)
    y = np.clip(v, 0, h - 1.001)
    x0, y0 = x.astype(np.int64), y.astype(np.int64)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    f = img.astype(np.float32)
    return (f[y0, x0] * (1 - fx) + f[y0, x0 + 1] * fx) * (1 - fy) + (f[y0 + 1, x0] * (1 - fx) + f[y0 + 1, x0 + 1] * fx) * fy


def collect_samples(model: DeviceModel, geoms: dict[str, ChartGeometry], frames: dict[int, np.ndarray],
                    cams: dict[int, Camera], T_world_cam: dict[int, np.ndarray], T_world_cad: np.ndarray,
                    min_cos: float = 0.15, depth_scale: float = 0.25, masks: dict[int, np.ndarray] | None = None,
                    mask_grow_px: int = 4) -> dict[str, ChartMaps]:
    """Per-view colour samples of each chart. ``masks`` (bool device masks per frame, e.g. SAM 2)
    leave out samples that project outside the device in that frame (grown by ``mask_grow_px`` for
    the mask's coarse edge), so a slightly misregistered frame cannot bring floor into the texture."""
    scene = RaycastScene(model.mesh, T_world_cad)
    order = sorted(frames)
    V = len(order)
    out = {}
    buf = {c: (np.zeros((V, len(g.point), 3), np.uint8), np.zeros((V, len(g.point)), np.float16)) for c, g in geoms.items()}
    Pw = {c: g.point @ T_world_cad[:3, :3].T + T_world_cad[:3, 3] for c, g in geoms.items()}
    Nw = {c: g.normal @ T_world_cad[:3, :3].T for c, g in geoms.items()}
    for vi, i in enumerate(order):
        cam, Twc = cams[i], T_world_cam[i]
        cq = cam.scaled(depth_scale)
        hq = scene.cast(cq, Twc, step=1)
        dmap = np.where(hq.hit, hq.depth, np.inf).astype(np.float32)
        Tcw = inv_T(Twc)
        img = frames[i]
        mk = None
        if masks is not None and i in masks:
            mk = masks[i]
            g = int(np.ceil(mask_grow_px * mk.shape[1] / cam.width))       # grow in full-frame pixels
            if g > 0:
                mk = cv2.dilate(mk.astype(np.uint8), np.ones((2 * g + 1,) * 2, np.uint8)) > 0
        for c, g in geoms.items():
            pc = Pw[c] @ Tcw[:3, :3].T + Tcw[:3, 3]
            z = pc[:, 2]
            u = cam.fx * pc[:, 0] / z + cam.cx
            v = cam.fy * pc[:, 1] / z + cam.cy
            view = Twc[:3, 3] - Pw[c]
            view /= np.linalg.norm(view, axis=1, keepdims=True)
            cosv = np.einsum("ij,ij->i", Nw[c], view)
            ok = (z > 0) & (cosv > min_cos) & (u >= 1) & (u < cam.width - 2) & (v >= 1) & (v < cam.height - 2)
            if not ok.any():
                continue
            # depth test against the model seen from this view (quarter resolution, nearest pixel)
            uq = np.clip(np.round((u[ok] + 0.5) * depth_scale - 0.5).astype(int), 0, dmap.shape[1] - 1)
            vq = np.clip(np.round((v[ok] + 0.5) * depth_scale - 0.5).astype(int), 0, dmap.shape[0] - 1)
            dref = dmap[vq, uq]
            # neighbourhood minimum guards the quantised depth at silhouettes
            vis = z[ok] <= dref + 0.6 + 0.004 * z[ok]
            idx = np.flatnonzero(ok)[vis]
            if mk is not None:
                mu = np.clip(np.floor((u[idx] + 0.5) * mk.shape[1] / cam.width).astype(int), 0, mk.shape[1] - 1)
                mv = np.clip(np.floor((v[idx] + 0.5) * mk.shape[0] / cam.height).astype(int), 0, mk.shape[0] - 1)
                idx = idx[mk[mv, mu]]
            rgb, cs = buf[c]
            rgb[vi, idx] = np.clip(_bilinear_rgb(img, u[idx], v[idx]), 0, 255).astype(np.uint8)
            cs[vi, idx] = cosv[idx].astype(np.float16)
    for c, g in geoms.items():
        rgb, cs = buf[c]
        out[c] = ChartMaps(g, ViewSamples(order, rgb, cs))
    return out


def prefer_sharp_views(cm: ChartMaps, keep_share: float = 0.3, min_views: int = 3, min_cos: float = 0.3) -> dict:
    """Keep only the sharpest views of a chart where enough of them saw a texel.

    A hand-held orbit is mostly motion-blurred: on the first real iPad capture the views' detail
    varied five-fold, and the median over all of them washed out the logo's edge (gradient energy
    17, sharpest single view 509; median of the sharpest 20 % of views 122, 173 with COLMAP poses).
    Each view's detail is measured on the chart itself (rectified samples), as the gradient energy
    over the texels it saw divided by that of the mean image over the same texels, so what the view
    happens to cover does not count. Views in the top ``keep_share`` are kept; at texels that fewer
    than ``min_views`` kept views saw, every view stays. Works in place on ``cm.samples.cos`` (0 =
    not used), so statistics, detectors and coverage all follow it. Returns a summary."""
    s = cm.samples
    g = cm.geom
    V, N = s.cos.shape
    if V < 2 * min_views:
        return {"views": V, "kept": V, "applied": False}
    grey = s.rgb.mean(-1, dtype=np.float32)                       # (V, N)
    vis = s.cos.astype(np.float32) >= min_cos
    cnt = vis.sum(0)
    mean = np.where(cnt > 0, (grey * vis).sum(0) / np.maximum(cnt, 1), 0).astype(np.float32)
    mean_img = g.to_image(mean)
    k = np.ones((5, 5), np.uint8)

    def energy(img, m):
        gx, gy = cv2.Sobel(img, cv2.CV_32F, 1, 0), cv2.Sobel(img, cv2.CV_32F, 0, 1)
        e = gx * gx + gy * gy
        return float(e[m].mean()) if m.any() else np.nan

    ratio = np.full(V, np.nan)
    for v in range(V):
        m = cv2.erode(g.to_image(vis[v].astype(np.uint8)), k) > 0     # away from the view's own border
        if m.sum() < 500:
            continue
        ratio[v] = energy(g.to_image(grey[v]), m) / max(energy(mean_img, m), 1e-6)
    ok = np.isfinite(ratio)
    if ok.sum() < 2 * min_views:
        return {"views": V, "kept": V, "applied": False}
    cut = np.quantile(ratio[ok], 1.0 - keep_share)
    keep = ok & (ratio >= cut)
    sharp_cnt = (vis & keep[:, None]).sum(0)
    use_sharp = sharp_cnt >= min_views
    drop = (~keep)[:, None] & use_sharp[None, :]
    s.cos[drop] = 0
    return {"views": V, "kept": int(keep.sum()), "applied": True, "ratio_cut": round(float(cut), 3),
            "ratio_p50": round(float(np.median(ratio[ok])), 3), "ratio_max": round(float(np.max(ratio[ok])), 3),
            "texels_sharp_only": round(float(use_sharp[cnt > 0].mean()), 3) if (cnt > 0).any() else 0.0}


def lab_of(rgb_u8: np.ndarray) -> np.ndarray:
    """CIE Lab (L 0-100, a/b about -128..127) of uint8 RGB with any leading shape."""
    shp = rgb_u8.shape
    lab = cv2.cvtColor(rgb_u8.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(shp).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return lab


def masked_percentiles(values: np.ndarray, valid: np.ndarray, qs: tuple[float, ...]) -> list[np.ndarray]:
    """Per-column percentiles of ``values`` (V, N) over rows where ``valid``; NaN where no row is
    valid. Sorting with +inf for invalid rows keeps it vectorised (np.nanpercentile loops in Python
    over columns and took >10 min for 1.2M texels)."""
    x = np.where(valid, values.astype(np.float32), np.inf)
    x.sort(axis=0)
    n = valid.sum(0)
    out = []
    for q in qs:
        k = np.clip(np.round((q / 100.0) * (n - 1)).astype(np.int64), 0, len(x) - 1)
        v = np.take_along_axis(x, k[None, :], axis=0)[0]
        out.append(np.where(n > 0, v, np.nan))
    return out


def compute_stats(cm: ChartMaps, good_cos: float = 0.5, min_cos: float = 0.3, chunk: int = 150_000) -> dict:
    """Robust per-texel statistics over the views that saw the texel with cos >= ``min_cos``.

    Done in chunks of texels: converting all samples to float Lab at once needs ~4 GB for the
    frame chart of a two-pass capture (280 views x 1.2M texels)."""
    s = cm.samples
    V, N = s.cos.shape
    out = {k: np.empty(N, np.float32) for k in ("L50", "L25", "L80", "a50", "b50")}
    rgb50 = np.empty((N, 3), np.float32)
    n_seen = np.empty(N, np.int32)
    n_good = np.empty(N, np.int32)
    for a in range(0, N, chunk):
        b = min(N, a + chunk)
        cosf = s.cos[:, a:b].astype(np.float32)
        vis = cosf >= min_cos
        lab = lab_of(s.rgb[:, a:b])
        out["L50"][a:b], out["L25"][a:b], out["L80"][a:b] = masked_percentiles(lab[..., 0], vis, (50, 25, 80))
        out["a50"][a:b], = masked_percentiles(lab[..., 1], vis, (50,))
        out["b50"][a:b], = masked_percentiles(lab[..., 2], vis, (50,))
        for k in range(3):
            rgb50[a:b, k], = masked_percentiles(s.rgb[:, a:b, k], vis, (50,))
        n_seen[a:b] = vis.sum(0)
        n_good[a:b] = (cosf >= good_cos).sum(0)
    stats = {
        "n_seen": n_seen, "n_good": n_good,
        "med_lab": np.stack([out["L50"], out["a50"], out["b50"]], -1), "med_rgb": rgb50,
        "low_L": out["L25"], "high_L": out["L80"],
    }
    cm.stats = stats
    return stats


def coverage(maps: dict[str, ChartMaps], model: DeviceModel, geoms: dict[str, ChartGeometry] | None = None,
             good_views: int = 2) -> dict:
    """Share of the device's surface area seen in at least ``good_views`` views with cos > 0.5.

    ``geoms`` lists every chart of the device: a chart that no pass looked at (the screen of a
    back-up-only capture) counts as unseen. Before 2026-11-16 such charts were simply missing from
    ``maps`` and the overall share came out as 97 % for a one-sided capture. Texels are weighted by
    their area (``res**2``): the frame band is sampled at 0.05 mm and the faces at 0.1 mm, so
    counting texels gave the frame four times its real weight."""
    from ..anchor.parametric import PARTS

    geoms = dict(geoms or {})
    for c, cm in maps.items():
        geoms.setdefault(c, cm.geom)
    by_part: dict[str, list[float]] = {}
    by_chart = {}
    tot_seen = tot_all = 0.0
    for c, g in geoms.items():
        a_tex = g.res ** 2
        good = maps[c].stats["n_good"] >= good_views if c in maps else np.zeros(len(g.point), bool)
        for pi, pname in enumerate(PARTS):
            sel = g.part == pi
            if sel.any():
                a = by_part.setdefault(pname, [0.0, 0.0])
                a[0] += float(good[sel].sum()) * a_tex
                a[1] += float(sel.sum()) * a_tex
        seen, total = float(good.sum()) * a_tex, float(len(good)) * a_tex
        by_chart[c] = {"seen": seen / max(total, 1e-9), "area_mm2": round(total, 1), "captured": c in maps}
        tot_seen += seen
        tot_all += total
    return {"overall": tot_seen / max(tot_all, 1e-9), "by_part": {k: v[0] / max(v[1], 1e-9) for k, v in by_part.items()},
            "by_chart": by_chart, "good_views": good_views}
