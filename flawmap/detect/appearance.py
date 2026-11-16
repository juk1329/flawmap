"""Training-free appearance detectors on the unwrapped maps.

No public data covers frame dents/chips or back-glass defects, so the main detectors use only the
device's own maps:

* **Frame chips / dents** (``frame_anomaly``): along the frame band every height row should look
  the same all the way round, apart from slow shading changes. Each texel's median Lab colour is
  compared with a rolling median along the row (12 mm window, wrapping round the device); the
  difference is scaled by the row's robust spread. A paint chip exposes bright bare metal; a dent
  breaks the reflection along the band. Buttons and openings are masked out (they are proud of
  the frame or holes and would always fire).
* **Glass cracks and scratches** (``glass_lines``): thin bright lines on the screen/back maps,
  found with a multi-scale ridge filter on lightness after removing the local background. Cracks
  use the median over views (present in most views); scratches use the 80th percentile (visible
  only at some angles) and are reported with lower confidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from ..anchor.parametric import PARTS, DeviceModel
from ..surface.maps import ChartMaps


@dataclass
class Region:
    chart: str
    kind: str                      # chip | crack | scratch
    texels: np.ndarray             # indices into the chart's valid-texel list
    score: float
    extra: dict = field(default_factory=dict)


def _fill_nan_along_rows(a: np.ndarray) -> np.ndarray:
    """Replace NaNs by the nearest valid value along each row (circular)."""
    out = a.copy()
    rows, cols = a.shape
    for r in range(rows):
        row = out[r]
        ok = np.isfinite(row)
        if not ok.any():
            continue
        idx = np.flatnonzero(ok)
        allx = np.arange(cols)
        out[r] = np.interp(allx, np.r_[idx - cols, idx, idx + cols], np.r_[row[idx], row[idx], row[idx]])
    return out


def _rolling_median_rows(a: np.ndarray, window: int, step: int = 8) -> np.ndarray:
    """Rolling median along rows (wrapping), computed on a ``step``-decimated copy for speed."""
    sub = a[:, ::step]
    w = max(int(round(window / step)) | 1, 3)
    med = ndi.median_filter(sub, size=(1, w), mode="wrap")
    x_sub = np.arange(sub.shape[1]) * step
    out = np.empty_like(a)
    xs = np.arange(a.shape[1])
    for r in range(a.shape[0]):
        out[r] = np.interp(xs, np.r_[x_sub, a.shape[1]], np.r_[med[r], med[r, 0]])
    return out


def frame_anomaly(cm: ChartMaps, model: DeviceModel, window_mm: float = 12.0, exclusion_margin: float = 3.0,
                  min_views: int = 2) -> dict:
    g = cm.geom
    lab = g.to_image(cm.stats["med_lab"].astype(np.float32), fill=np.nan)
    n_good = g.to_image(cm.stats["n_seen"].astype(np.int32), fill=0)
    valid = np.isfinite(lab[..., 0]) & (n_good >= min_views)
    win = int(round(window_mm / g.res))
    base = np.stack([_rolling_median_rows(_fill_nan_along_rows(np.where(valid, lab[..., k], np.nan)), win) for k in range(3)], -1)
    d = lab - base
    dE = np.sqrt((d ** 2).sum(-1))
    dE[~valid] = 0.0
    # robust spread per row
    z = np.zeros_like(dE)
    for r in range(dE.shape[0]):
        v = dE[r][valid[r]]
        if len(v) > 50:
            mad = np.median(np.abs(v - np.median(v))) * 1.4826
            z[r] = dE[r] / max(mad, 0.6)
    mask_ex = np.zeros(dE.shape[1], bool)
    s_cols = (np.arange(dE.shape[1]) + 0.5) * g.res
    for e in model.frame_exclusions(margin=exclusion_margin):
        s0, s1 = e["s0"], e["s1"]
        if s0 <= s1:
            mask_ex |= (s_cols >= s0) & (s_cols <= s1)
        else:
            mask_ex |= (s_cols >= s0) | (s_cols <= s1)
    z[:, mask_ex] = 0.0
    dE[:, mask_ex] = 0.0
    return {"z": z, "dE": dE, "dL": np.where(valid, d[..., 0], 0.0), "valid": valid, "excluded_cols": mask_ex}


def glass_lines(cm: ChartMaps, edge_margin_mm: float = 1.5, use: str = "med") -> dict:
    """Ridge response of thin bright lines on a glass chart (screen or back)."""
    from skimage.filters import sato

    g = cm.geom
    L = g.to_image((cm.stats["med_lab"][:, 0] if use == "med" else cm.stats["high_L"]).astype(np.float32), fill=np.nan)
    part = g.to_image(g.part.astype(np.int32), fill=-1)
    glass = (part == PARTS.index("screen")) | (part == PARTS.index("back"))
    valid = np.isfinite(L) & glass
    m = int(round(edge_margin_mm / g.res))
    valid = ndi.binary_erosion(valid, iterations=max(m, 1))
    Lf = np.where(np.isfinite(L), L, np.nanmedian(L))
    bg = ndi.median_filter(Lf, size=int(round(3.0 / g.res)) | 1)
    hp = np.where(valid, Lf - bg, 0.0)
    ridge = sato(hp, sigmas=[1.0, 1.5, 2.0], black_ridges=False)
    ridge[~valid] = 0.0
    v = ridge[valid]
    mad = np.median(np.abs(v - np.median(v))) * 1.4826 if len(v) else 1.0
    return {"z": ridge / max(mad, 1e-3), "ridge": ridge, "hp": hp, "valid": valid}


def regions_from_score(cm: ChartMaps, z: np.ndarray, thr: float, min_area_mm2: float, kind: str,
                       support: np.ndarray | None = None, open_iter: int = 1, close_iter: int = 1,
                       min_length_mm: float = 0.0) -> list[Region]:
    from skimage.measure import label, regionprops

    g = cm.geom
    m = z > thr
    if support is not None:
        m &= support
    if open_iter:
        m = ndi.binary_opening(m, iterations=open_iter)
    if close_iter:
        m = ndi.binary_closing(m, iterations=close_iter)
    lab = label(m, connectivity=2)
    flat_to_valid = np.full(g.shape[0] * g.shape[1], -1, np.int64)
    flat_to_valid[g.flat_index] = np.arange(len(g.flat_index))
    out = []
    for rp in regionprops(lab):
        area = rp.area * g.res ** 2
        length = rp.axis_major_length * g.res
        if area < min_area_mm2 or length < min_length_mm:
            continue
        rr, cc = rp.coords[:, 0], rp.coords[:, 1]
        tex = flat_to_valid[rr * g.shape[1] + cc]
        tex = tex[tex >= 0]
        if len(tex) == 0:
            continue
        out.append(Region(cm.geom.chart, kind, tex, float(z[rr, cc].max()),
                          {"area_mm2": float(area), "length_mm": float(length), "mean_z": float(z[rr, cc].mean())}))
    return out


def detect_all(maps: dict[str, ChartMaps], model: DeviceModel, cfg: dict | None = None) -> tuple[list[Region], dict]:
    """Run the detectors on whatever charts the capture saw. Returns regions and the score maps."""
    cfg = {"frame_z": 8.0, "frame_dE": 6.0, "frame_min_area": 0.15,
           "crack_z": 12.0, "crack_min_len": 3.0, "scratch_z": 14.0, "scratch_min_len": 4.0, **(cfg or {})}
    regions, scores = [], {}
    if "frame" in maps:
        fa = frame_anomaly(maps["frame"], model)
        scores["frame"] = fa
        regions += regions_from_score(maps["frame"], fa["z"], cfg["frame_z"], cfg["frame_min_area"], "chip",
                                      support=fa["dE"] > cfg["frame_dE"])
    for c in ("screen", "back"):
        if c not in maps:
            continue
        gl = glass_lines(maps[c], use="med")
        scores[c] = gl
        cracks = regions_from_score(maps[c], gl["z"], cfg["crack_z"], 0.3, "crack", open_iter=0, close_iter=2,
                                    min_length_mm=cfg["crack_min_len"])
        regions += cracks
        gs = glass_lines(maps[c], use="high")
        scores[c + "_high"] = gs
        taken = np.zeros(len(maps[c].geom.point), bool)
        for r in cracks:
            taken[r.texels] = True
        for r in regions_from_score(maps[c], gs["z"], cfg["scratch_z"], 0.2, "scratch", open_iter=0, close_iter=2,
                                    min_length_mm=cfg["scratch_min_len"]):
            if taken[r.texels].mean() < 0.3:
                regions.append(r)
    return regions, scores
