"""Outline channel for frame dents: where the device's real outline lies inside the reference model.

The edge registration leaves, for every silhouette point of every view, the signed distance between
the model's projected silhouette and the image edge, converted to mm (``r_mm > 0``: the image edge is
inside the model, i.e. material is missing). Points on the outermost part of the frame band are
binned by their position ``s`` around the outline; the median per bin, minus a 20 mm running median
(slow differences between the drawing-based model and the real phone, ~0.2-0.5 mm, are expected and
are not defects), divided by a robust spread, marks dents. Buttons and openings are masked.

Limits (see the plan review): a dent only changes the outline when it sits on the widest part of the
frame as seen from some view; a dent in the middle of a flat side face is invisible here and is left
to the appearance channel. Depth is not reported, only position and extent.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from ..anchor.parametric import DeviceModel


def _dense_outline(model: DeviceModel, step: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    P = model.perimeter
    ss = np.arange(0.0, P, step)
    xs = np.interp(ss, np.r_[model.outline_s, P], np.r_[model.outline_xy[:, 0], model.outline_xy[0, 0]])
    ys = np.interp(ss, np.r_[model.outline_s, P], np.r_[model.outline_xy[:, 1], model.outline_xy[0, 1]])
    return np.stack([xs, ys], 1), ss


def outline_samples(model: DeviceModel, edges: list[dict], max_inset_mm: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    """(s, r_mm) of silhouette points that lie on the outermost frame band. ``s`` comes from the
    nearest point of the outline resampled every 0.1 mm (snapping to the model's own outline
    vertices, up to 4 mm apart on straight edges, lumped samples together and hid dents)."""
    from scipy.spatial import cKDTree

    dense, ss = _dense_outline(model)
    tree = cKDTree(dense)
    S, R = [], []
    for e in edges:
        X = np.asarray(e["X"])
        if len(X) == 0:
            continue
        d, k = tree.query(X[:, :2])
        band = (X[:, 2] < model.z_front - 0.5) & (X[:, 2] > model.z_back + 0.5)
        sel = (d < max_inset_mm) & band
        S.append(ss[k[sel]])
        R.append(np.asarray(e["r_mm"])[sel])
    if not S:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(S), np.concatenate(R)


def outline_profile(model: DeviceModel, s: np.ndarray, r: np.ndarray, bin_mm: float = 0.5, trend_mm: float = 20.0,
                    min_count: int = 3) -> dict:
    P = model.perimeter
    nb = int(np.ceil(P / bin_mm))
    b = np.clip((s / bin_mm).astype(int), 0, nb - 1)
    med = np.full(nb, np.nan)
    cnt = np.bincount(b, minlength=nb)
    order = np.argsort(b)
    bs, rs = b[order], r[order]
    starts = np.searchsorted(bs, np.arange(nb))
    ends = np.searchsorted(bs, np.arange(nb), side="right")
    for k in range(nb):
        if ends[k] - starts[k] >= min_count:
            # 25th percentile, not the median: where two edges lie ~1.5 px apart (a highlight on the
            # rounded rim) half of a bin's samples snap to the inner one and the median flips
            # between them (0.3 mm false dents on the clean synthetic capture, 2026-11-15). A real
            # dent moves the edge inwards in nearly every view, so its low quantile stays high.
            med[k] = np.percentile(rs[starts[k]:ends[k]], 25)
    ok = np.isfinite(med)
    filled = med.copy()
    if ok.any():
        idx = np.flatnonzero(ok)
        filled = np.interp(np.arange(nb), np.r_[idx - nb, idx, idx + nb], np.r_[med[idx], med[idx], med[idx]])
    trend = ndi.median_filter(filled, size=max(int(round(trend_mm / bin_mm)) | 1, 3), mode="wrap")
    # a 3-bin (1.5 mm) median removes single-bin spikes; real dents span several bins
    resid = np.where(ok, ndi.median_filter(filled - trend, size=3, mode="wrap"), 0.0)
    v = resid[ok]
    sigma = 1.4826 * np.median(np.abs(v - np.median(v))) if len(v) > 20 else 0.1
    excl = np.zeros(nb, bool)
    centres = (np.arange(nb) + 0.5) * bin_mm
    for e in model.frame_exclusions(margin=3.0):
        if e["s0"] <= e["s1"]:
            excl |= (centres >= e["s0"]) & (centres <= e["s1"])
        else:
            excl |= (centres >= e["s0"]) | (centres <= e["s1"])
    z = np.where(ok & ~excl, resid / max(sigma, 0.02), 0.0)
    return {"s": centres, "median_mm": med, "trend_mm": trend, "resid_mm": resid, "z": z, "count": cnt,
            "sigma_mm": float(sigma), "excluded": excl}


def outline_dents(prof: dict, z_thr: float = 6.0, min_amp_mm: float = 0.15, min_width_mm: float = 1.0) -> list[dict]:
    from skimage.measure import label

    m = (prof["z"] > z_thr) & (prof["resid_mm"] > min_amp_mm)
    lab = label(np.r_[m, m[:1]].astype(int))[:-1]       # simple 1D labelling
    out = []
    bin_mm = prof["s"][1] - prof["s"][0]
    for k in range(1, lab.max() + 1):
        idx = np.flatnonzero(lab == k)
        if len(idx) * bin_mm < min_width_mm:
            continue
        j = idx[np.argmax(prof["resid_mm"][idx])]
        out.append({"s0": float(prof["s"][idx[0]] - bin_mm / 2), "s1": float(prof["s"][idx[-1]] + bin_mm / 2),
                    "s_peak": float(prof["s"][j]), "peak_mm": float(prof["resid_mm"][j]), "z": float(prof["z"][j])})
    return out


def dents_to_regions(dents: list[dict], cm_frame, model: DeviceModel):
    """Frame-chart regions for outline dents: the dent's s range over the middle of the band."""
    from .appearance import Region

    g = cm_frame.geom
    uv = g.uv_of(np.arange(len(g.point)))
    mid = (model.z_front + model.z_back) / 2.0
    half = (model.z_front - model.z_back) / 2.0
    band = np.abs(g.point[:, 2] - mid) < 0.6 * half
    out = []
    for d in dents:
        if d["s0"] <= d["s1"]:
            sel = (uv[:, 0] >= d["s0"]) & (uv[:, 0] <= d["s1"])
        else:
            sel = (uv[:, 0] >= d["s0"]) | (uv[:, 0] <= d["s1"])
        tex = np.flatnonzero(sel & band)
        if len(tex) == 0:
            continue
        L = (d["s1"] - d["s0"]) % model.perimeter
        out.append(Region("frame", "dent", tex, float(d["z"]),
                          {"outline_peak_mm": d["peak_mm"], "s0": d["s0"], "s1": d["s1"], "length_mm": float(L),
                           "area_mm2": float(len(tex) * g.res ** 2)}))
    return out


def merge_dents_with_chips(regions: list, dent_regions: list, maps, gap_mm: float = 1.0) -> list:
    """A colour chip that overlaps an outline dent becomes part of that dent (a dent usually
    breaks the coating too); both channels are recorded."""
    from .appearance import Region

    if not dent_regions:
        return regions
    g = maps["frame"].geom
    P = None
    out, used = [], set()
    for d in dent_regions:
        s0, s1 = d.extra["s0"] - gap_mm, d.extra["s1"] + gap_mm
        tex = [d.texels]
        app = 0.0
        for k, r in enumerate(regions):
            if r.chart != "frame" or r.kind != "chip":
                continue
            su = g.uv_of(r.texels)[:, 0]
            inside = ((su >= s0) & (su <= s1)) if s0 <= s1 else ((su >= s0) | (su <= s1))
            if inside.any():
                used.add(k)
                tex.append(r.texels)
                app = max(app, r.score)
        t = np.unique(np.concatenate(tex))
        extra = dict(d.extra)
        extra["appearance_score"] = app
        extra["area_mm2"] = float(len(t) * g.res ** 2)
        out.append(Region("frame", "dent", t, max(d.score, app), extra))
    out += [r for k, r in enumerate(regions) if k not in used]
    return out
