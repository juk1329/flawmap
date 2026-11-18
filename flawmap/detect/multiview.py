"""Defects from how each spot of the surface behaves across viewing directions (plan E).

Input: the per-texel decomposition of ``texture.decompose`` (surface colour from the best-supported
low lightness over 24 direction bins, and each bin's lightness above it). Every defect candidate
gets scores from independent channels, and design elements are told apart instead of reported:

1. **Surface marks**: thin lines (``line``, a Hessian ridge filter of both polarities) and small
   spots (``spot``, difference of Gaussians) in the *albedo*: the surface colour with the slow
   illumination divided out (``texture.decompose.flatten_shading``). A crack or a paint chip changes
   the surface itself and shows from every direction; reflections are already out of it.
2. **Fixed glints** (``glint``): a scratch or a dent's rim reflects towards a few directions only,
   but always at the same spot. Per direction bin the reflection excess is high-passed (difference
   of Gaussians, bright side); a texel scores by its *second* strongest bin, so a glint must recur
   from two different directions at the same place. A reflected window or ceiling light moves over
   the surface as the viewpoint moves and does not recur in place.
3. **Shape**: the relief's raised areas (``shape.relief``) join the design elements; outline dents
   come from ``detect.outline`` as before.
4. **Design elements** (``design_mask``): logo, print and camera are not defects. Without a drawing
   they are found as albedo standing ``design_dl`` away from the face's median lightness, in
   components that are large and compact (logo, camera module; solidity) or rows of three or more
   small glyphs (print). Regions that lie mostly on them are kept apart as "design (estimated)".
   A band of ``edge_mm`` along the outline is not reported on the faces (the frame band covers
   the edge; masks and outline fit are least certain there).

The thresholds are in lightness units (CIE L, 0-100) or the ridge filter's response to them, so a
face without defects reports none however uniform it is. On the real iPad (2026-11-17) the
statistic tails were 10x apart between the black screen and the silver back (ridge p99.9 0.72 vs
8.0, the back's tail being logo, print and camera), which ruled out per-chart z-scores.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi

from ..anchor.parametric import PARTS
from ..surface.maps import ChartMaps, lab_of
from .appearance import Region, regions_from_score

DEFAULTS = {"line_thr": 2.0, "line_min_len": 2.0, "spot_dl": 3.0, "spot_min_area": 0.2, "glint_dl": 6.0, "glint_min_len": 1.5,
            "glint_min_bins": 2, "edge_mm": 4.0, "part_mm": 2.5, "design_dl": 8.0, "design_min_area": 20.0, "design_solidity": 0.4,
            "design_share": 0.4, "min_bins": 3, "min_mask_views": 2, "print_seed_dl": 6.0}


@dataclass
class ChartScores:
    line: np.ndarray         # (rows, cols) ridge response of thin lines in the albedo
    spot: np.ndarray         # (rows, cols) |difference of Gaussians| of the albedo, lightness units
    glint: np.ndarray        # (rows, cols) second strongest bin's high-passed reflection excess, lightness units
    glint_bins: np.ndarray   # (rows, cols) bins whose high-passed excess reaches ``glint_dl``
    design: np.ndarray       # (rows, cols) bool design-element mask
    edge: np.ndarray         # (rows, cols) bool outline band
    valid: np.ndarray        # (rows, cols) bool texels with enough direction bins
    albedo: np.ndarray       # (rows, cols, 3) uint8


def _contrast(L: np.ndarray, comp: np.ndarray, res: float) -> float:
    """Lightness difference between a component and a ring 0.5-1.5 mm around it."""
    ring = ndi.binary_dilation(comp, iterations=max(1, int(round(1.5 / res)))) & ~ndi.binary_dilation(comp, iterations=max(1, int(round(0.5 / res))))
    if comp.sum() < 3 or ring.sum() < 3:
        return 0.0
    return abs(float(np.median(L[comp])) - float(np.median(L[ring])))


def design_mask(L: np.ndarray, valid: np.ndarray, res: float, dl: float = 8.0, min_area_mm2: float = 20.0,
                solidity: float = 0.4, raised: np.ndarray | None = None, design_contrast: float = 12.0,
                print_seed: float = 6.0) -> tuple[np.ndarray, list[dict]]:
    """Design elements (and stains) on a face: see the module docstring. ``L``: albedo lightness.
    Returns the mask and one record per element (kind ``design`` | ``print`` | ``stain`` | ``raised``,
    centre and size in chart mm)."""
    from skimage.filters import apply_hysteresis_threshold
    from skimage.measure import label, regionprops

    med = float(np.median(L[valid])) if valid.any() else 0.0
    dev = np.where(valid, np.abs(L - med), 0.0)
    # hysteresis: cells ``dl`` away from the median seed an element, cells half as far join when
    # connected (the "Pro" of the iPad's print came out lighter than "iPad" and was reported)
    m = apply_hysteresis_threshold(dev, 0.5 * dl, dl) & valid
    m = ndi.binary_closing(m, iterations=max(1, int(round(0.3 / res))))
    out = np.zeros(m.shape, np.uint8)
    recs = []

    def rec(kind, rp, extra=None):
        return {"kind": kind, "centre_mm": [round(float(rp.centroid[1] * res), 1), round(float(rp.centroid[0] * res), 1)],
                "size_mm": [round(float((rp.bbox[3] - rp.bbox[1]) * res), 1), round(float((rp.bbox[2] - rp.bbox[0]) * res), 1)],
                **(extra or {})}

    for rp in regionprops(label(m, connectivity=2)):
        area = rp.area * res * res
        rr, cc = rp.coords[:, 0], rp.coords[:, 1]
        box = ((rp.bbox[3] - rp.bbox[1]) * res, (rp.bbox[2] - rp.bbox[0]) * res)
        if area >= min_area_mm2 and rp.solidity >= solidity and min(box) >= 2.0:
            comp = np.zeros(m.shape, bool)
            comp[rr, cc] = True
            con = _contrast(L, comp, res)
            # the whole outline of the element (its convex hull): a camera module's flash, microphone and
            # sensors sit between the lenses at lower contrast (they were reported on the iPad back).
            # Strong contrast: a part, the logo or print; weak: a smudge or stain (screen, 2026-11-17)
            hull = cv2.convexHull(np.stack([cc, rr], 1).astype(np.int32))
            cv2.fillConvexPoly(out, hull, 1)
            recs.append(rec("design" if con >= design_contrast else "stain", rp, {"area_mm2": round(area, 1), "contrast": round(con, 1)}))
    # print: glyphs (small components at half the contrast) joined by a 1.25 mm closing into rows of
    # three or more, at least one of them reaching ``print_seed`` (rows of faint smudge dots
    # along the screen's edge passed otherwise); low-contrast blobs that do not form such rows are left alone
    weak = (dev > 0.5 * dl) & valid & (out == 0)
    glyph_lab = label(weak, connectivity=2)
    small = np.zeros(m.shape, bool)
    for rp in regionprops(glyph_lab):
        area = rp.area * res * res
        if 0.05 <= area < min_area_mm2:
            small[rp.coords[:, 0], rp.coords[:, 1]] = True
    if small.any():
        grp = label(ndi.binary_closing(small, structure=np.ones((3, 3)), iterations=max(1, int(round(1.25 / res)))), connectivity=2)
        glyphs = label(small, connectivity=2)
        gprops = {r.label: r for r in regionprops(glyphs)}
        for rp in regionprops(grp):
            rr, cc = rp.coords[:, 0], rp.coords[:, 1]
            ids = [int(i) for i in np.unique(glyphs[rr, cc]) if i > 0]
            n = len(ids)
            # glyphs are about as tall as wide; the pieces of a broken scratch are long and thin (the
            # corner line of the iPad screen came apart into three such pieces and passed as print)
            asp = [max(gprops[i].axis_major_length, 1.0) / max(gprops[i].axis_minor_length, 1.0) for i in ids]
            if asp and float(np.median(asp)) > 2.5:
                continue
            h, w = (rp.bbox[2] - rp.bbox[0]) * res, (rp.bbox[3] - rp.bbox[1]) * res
            strong = bool((dev[rr, cc][small[rr, cc]] >= print_seed).any())
            if n >= 3 and strong and max(h, w) >= 3.0 and max(h, w) >= 2.0 * min(h, w):
                hull = cv2.convexHull(np.stack([cc, rr], 1).astype(np.int32))
                cv2.fillConvexPoly(out, hull, 1)
                recs.append(rec("print", rp, {"glyphs": n}))
    out = out.astype(bool)
    if raised is not None and raised.any():
        out |= raised & valid
        recs.append({"kind": "raised", "area_mm2": round(float((raised & valid).sum() * res * res), 1)})
    out = ndi.binary_dilation(out, iterations=max(1, int(round(2.0 / res))))
    return out, recs


def glint_maps(cm: ChartMaps, dec, valid: np.ndarray, dl: float, lo_px: float = 1.0, hi_px: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Second strongest bright high-pass of the reflection excess over the direction bins, and the
    number of bins reaching ``dl`` (normalised Gaussians, so a bin's unseen texels do not count)."""
    g = cm.geom
    first = np.zeros(g.shape, np.float32)
    second = np.zeros(g.shape, np.float32)
    count = np.zeros(g.shape, np.int16)
    for b in range(dec.excess.shape[0]):
        ex = dec.excess[b].astype(np.float32)
        ok = np.isfinite(ex)
        if ok.sum() < 1000:
            continue
        img = g.to_image(np.where(ok, ex, 0.0), fill=0.0).astype(np.float32)
        m = (g.to_image(ok, fill=False) & valid).astype(np.float32)

        def nblur(s):
            return cv2.GaussianBlur(img * m, (0, 0), s) / np.maximum(cv2.GaussianBlur(m, (0, 0), s), 1e-3)

        d = np.where(m > 0, nblur(lo_px) - nblur(hi_px), 0.0).astype(np.float32)
        d = np.maximum(d, 0.0)
        count += (d >= dl).astype(np.int16)
        second = np.maximum(second, np.minimum(first, d))
        first = np.maximum(first, d)
    return second, count


def chart_scores(cm: ChartMaps, dec, cfg: dict | None = None, raised: np.ndarray | None = None) -> tuple[ChartScores, list[dict]]:
    """Score maps of a face chart (screen or back) from its decomposition ``dec``."""
    from skimage.filters import sato

    from ..texture.decompose import flatten_shading

    cfg = {**DEFAULTS, **(cfg or {})}
    g = cm.geom
    res = g.res
    nb = g.to_image(dec.n_bins.astype(np.int32), fill=0)
    part = g.to_image(g.part.astype(np.int32), fill=-1)
    face = (part == PARTS.index("screen")) | (part == PARTS.index("back"))
    valid = (nb >= cfg["min_bins"]) & face
    if "n_good" in cm.stats:
        # the keyframe samples are gated by the device masks: a texel the masks never put on the device
        # (a plate corner rounder than the model's, lying on the floor) is not surface
        valid &= g.to_image(cm.stats["n_good"].astype(np.int32), fill=0) >= cfg["min_mask_views"]
    inner = ndi.binary_erosion(g.valid, iterations=max(1, int(round(cfg["edge_mm"] / res))))
    edge = g.valid & ~inner
    if cfg["part_mm"] > 0:
        # the same for the rim of a part set into the face (the camera module of a drawing-based model):
        # its raised wall shades and parallaxes onto the glass around it. On the synthetic iPhone 12 the
        # module's rim gave the only false lines (four, 0.5-2.2 mm out, 2026-11-17)
        other = g.valid & (part >= 0) & ~face
        if other.any():
            near = ndi.binary_dilation(other, iterations=max(1, int(round(cfg["part_mm"] / res))))
            edge |= near & face
    alb, _ = flatten_shading(g.to_image(dec.diffuse), valid, res)
    L = lab_of(alb)[..., 0].astype(np.float32)
    dm, recs = design_mask(L, valid & inner, res, cfg["design_dl"], cfg["design_min_area"], cfg["design_solidity"], raised,
                           print_seed=cfg["print_seed_dl"])
    med = float(np.median(L[valid])) if valid.any() else 0.0
    bg = ndi.median_filter(np.where(valid, L, med), size=max(3, int(round(3.0 / res))) | 1)
    hp = np.where(valid, L - bg, 0.0).astype(np.float32)
    sig = [1.0, 1.5, 2.0]
    line = np.maximum(sato(hp, sigmas=sig, black_ridges=False), sato(hp, sigmas=sig, black_ridges=True)).astype(np.float32)
    line[~valid] = 0
    spot = np.abs(cv2.GaussianBlur(hp, (0, 0), 1.2) - cv2.GaussianBlur(hp, (0, 0), 3.5)).astype(np.float32)
    spot[~valid] = 0
    glint, nbins = glint_maps(cm, dec, valid, cfg["glint_dl"])
    return ChartScores(line, spot, glint, nbins, dm, edge, valid, alb), recs


def _merge_overlapping(regions: list[Region], g, gap_mm: float = 0.5) -> list[Region]:
    """One region per place: candidates of different channels that touch (within ``gap_mm``) are
    merged; the kind follows the first channel in order line (crack), glint (scratch), spot."""
    if len(regions) < 2:
        return regions
    from skimage.measure import label

    m = np.zeros(g.shape, np.int32)
    for k, r in enumerate(regions):
        rr, cc = np.divmod(g.flat_index[r.texels], g.shape[1])
        m[rr, cc] = k + 1
    lab = label(ndi.binary_dilation(m > 0, iterations=max(1, int(round(gap_mm / g.res)))), connectivity=2)
    groups: dict[int, list[int]] = {}
    for k, r in enumerate(regions):
        rr, cc = np.divmod(g.flat_index[r.texels[:1]], g.shape[1])
        groups.setdefault(int(lab[rr[0], cc[0]]), []).append(k)
    order = {"crack": 0, "scratch": 1, "spot": 2}
    out = []
    for ks in groups.values():
        rs = [regions[k] for k in ks]
        if len(rs) == 1:
            out.append(rs[0])
            continue
        lead = min(rs, key=lambda r: order.get(r.kind, 9))
        tex = np.unique(np.concatenate([r.texels for r in rs]))
        uv = g.uv_of(tex)
        ext = uv.max(0) - uv.min(0)
        out.append(Region(lead.chart, lead.kind, tex, max(r.score for r in rs),
                          {"area_mm2": float(len(tex) * g.res ** 2), "length_mm": float(max(r.extra.get("length_mm", 0.0) for r in rs)),
                           "channel": " + ".join(sorted({r.extra.get("channel", "") for r in rs})), "merged_from": len(rs),
                           "extent_mm": float(np.hypot(*ext))}))
    return out


def multiview_regions(cm: ChartMaps, dec, cfg: dict | None = None, raised: np.ndarray | None = None
                      ) -> tuple[list[Region], list[Region], ChartScores, list[dict]]:
    """Defect regions of a face chart and the regions set aside as design elements. Returns
    (defects, design_regions, scores, design_records). Region scores are 10 x the strongest
    channel's ratio to its threshold (10 = just at a threshold)."""
    cfg = {**DEFAULTS, **(cfg or {})}
    sc, recs = chart_scores(cm, dec, cfg, raised)
    g = cm.geom
    support = sc.valid & ~sc.edge
    ratio = np.maximum.reduce([sc.line / cfg["line_thr"], sc.spot / cfg["spot_dl"],
                               np.where(sc.glint_bins >= cfg["glint_min_bins"], sc.glint / cfg["glint_dl"], 0.0)]) * 10.0
    cand = []
    for kind, z, thr, area, ln in (("crack", sc.line / cfg["line_thr"] * 10, 10.0, 0.2, cfg["line_min_len"]),
                                   ("scratch", np.where(sc.glint_bins >= cfg["glint_min_bins"], sc.glint / cfg["glint_dl"], 0.0) * 10, 10.0,
                                    0.1, cfg["glint_min_len"]),
                                   ("spot", sc.spot / cfg["spot_dl"] * 10, 10.0, cfg["spot_min_area"], 0.0)):
        rs = regions_from_score(cm, z, thr, area, kind, support=support, open_iter=0 if kind != "spot" else 1, close_iter=2,
                                min_length_mm=ln)
        for r in rs:
            r.extra["channel"] = {"crack": "surface line", "scratch": "fixed glint", "spot": "surface spot"}[kind]
        cand += rs
    cand = _merge_overlapping(cand, g)
    defects, design = [], []
    for r in cand:
        rr, cc = np.divmod(g.flat_index[r.texels], g.shape[1])
        r.score = float(ratio[rr, cc].max())
        r.extra["channels"] = {"line": round(float(sc.line[rr, cc].max()), 2), "spot_dl": round(float(sc.spot[rr, cc].max()), 2),
                               "glint_dl": round(float(sc.glint[rr, cc].max()), 2), "glint_bins": int(sc.glint_bins[rr, cc].max())}
        share = float(sc.design[rr, cc].mean())
        r.extra["design_share"] = round(share, 2)
        (design if share >= cfg["design_share"] else defects).append(r)
    return defects, design, sc, recs
