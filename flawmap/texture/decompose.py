"""Surface colour and reflections, separated per texel from every frame of the capture (plan D).

The keyframe median (``surface.maps``) mixes three things into one colour: the surface itself, the
environment it reflects, and motion blur. On the iPad back that left a blurry texture with diagonal
reflection streaks (logo-window gradient energy 18.6); taking only the sharpest 30 % of keyframes
made it sharp (168) but kept the streaks, because the sharpest frames come from the same slow
moments of the orbit and see the same reflections.

What separates the surface from its reflections is the viewing direction: the surface stays put
in the device's own coordinates while a reflection moves with the viewpoint, and a reflection only
ever adds light. So every texel is looked at from as many *different* directions as the capture
offers, each with the finest view the capture has from that direction:

1. **Poses of every frame** (``texture.allframes.all_frame_poses``): the keyframe poses' correction
   of ARKit, interpolated onto all frames (60 fps; every ``stride``-th frame is used).
2. **Direction bins** (``bin_of``): per texel, the direction to the camera in a local frame of the
   surface (normal, tangent): 8 azimuth sectors x 3 rings of the angle from the normal = 24 bins.
   A scan over the poses only (no images) keeps, per texel and bin, the frame with the finest
   effective resolution (``allframes.frame_resolution``: footprint x motion blur), across passes.
3. **Samples**: only the frames that own a slot are decoded. Each is resampled onto the chart,
   its coarse (1 mm) shading is replaced by the keyframe median's (exposure, broad reflections and
   the photographer's shadow), and on the faces it is aligned to the keyframe median by ECC
   (affine, gradient magnitude; at most ``max_shift_mm``) to take out the poses' residual error.
4. **Per texel over its bins** (``fuse``; weights 1 / resolution^2, bins within ``tau`` of the
   texel's finest resolution): the *surface colour* is the lowest *well-supported* lightness: the
   darkest sample whose lightness (within ``delta``) is shared by at least ``support`` of the
   weight, and the weighted mean colour of the samples within ``delta`` of it. Reflections only add
   light, so the surface is the low end; a single misaligned view of dark print is low too but has no
   support (the plain lower half doubled the iPad's text 4 mm lower, 2026-11-17). The *reflection
   excess* of each bin is its lightness above the surface colour.

Outputs per chart (``ChartDecomposition``): surface colour, the per-bin reflection excess (for the
multi-view detector, ``detect.multiview``), its maximum and mean, the number of usable bins, the
finest resolution and which frame owns each slot (for the sharpest evidence crops).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..geometry.transforms import inv_T
from ..io.stray import iter_frames
from ..surface.maps import ChartGeometry, lab_of
from ..utils.logging import get_logger
from .allframes import _align, _rectify, all_frame_poses, frame_resolution

log = get_logger("flawmap.texture")

N_AZ, RINGS = 8, (25.0, 45.0)          # azimuth sectors; ring edges (degrees from the normal)
N_BINS = N_AZ * (len(RINGS) + 1)


@dataclass
class PassFrames:
    """What the decomposition needs from one pass."""
    name: str
    cap: object
    video: Path
    T_world_cam: dict[int, np.ndarray]     # every frame (``all_frame_poses``)
    T_world_cad: np.ndarray
    charts: list[str]
    undist: str = "none"
    exclude: frozenset = frozenset()       # held-out frames: never slot owners (eval.heldout)


@dataclass
class ChartDecomposition:
    chart: str
    diffuse: np.ndarray        # (N, 3) uint8 surface colour (0 where unseen)
    excess: np.ndarray         # (B, N) float16 lightness above the surface colour per bin, NaN where none
    spec_max: np.ndarray       # (N,) float32
    spec_mean: np.ndarray      # (N,) float32 mean excess over the bins used
    n_bins: np.ndarray         # (N,) uint8 bins with a usable sample
    best_res: np.ndarray       # (N,) float32 finest effective resolution (mm), inf where unseen
    owner_pass: np.ndarray     # (B, N) int8 pass index owning the slot, -1 none
    owner_frame: np.ndarray    # (B, N) int32 frame index (in that pass), -1 none
    res: np.ndarray            # (B, N) float16 effective resolution of the slot (mm)
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------- direction bins
def local_frames(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per texel, two tangents of the surface (cad frame): t1 horizontal where possible."""
    a = np.where((np.abs(normal[:, 2]) < 0.9)[:, None], np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    t1 = np.cross(normal, a)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True) + 1e-12
    t2 = np.cross(normal, t1)
    return t1, t2


def bin_of(d: np.ndarray, n: np.ndarray, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """Direction bin (0..N_BINS-1) of unit view directions ``d`` (texel -> camera)."""
    c = np.clip(np.einsum("ij,ij->i", d, n), -1, 1)
    th = np.degrees(np.arccos(c))
    ring = np.searchsorted(np.asarray(RINGS), th)
    az = (np.degrees(np.arctan2(np.einsum("ij,ij->i", d, t2), np.einsum("ij,ij->i", d, t1))) % 360.0)
    sector = np.minimum((az / (360.0 / N_AZ)).astype(np.int64), N_AZ - 1)
    return (ring * N_AZ + sector).astype(np.int64)


# ---------------------------------------------------------------------------- 2. scan (poses only)
def coarse_map(geom: ChartGeometry, geom_c: ChartGeometry, k: int) -> np.ndarray:
    """For every texel of ``geom``, the index of the ``geom_c`` texel (``k`` times coarser) it lies
    in, -1 where that coarse texel is not valid."""
    rr, cc = np.divmod(geom.flat_index, geom.shape[1])
    rc = np.minimum(rr // k, geom_c.shape[0] - 1)
    ccc = np.minimum(cc // k, geom_c.shape[1] - 1)
    f2v = np.full(geom_c.shape[0] * geom_c.shape[1], -1, np.int64)
    f2v[geom_c.flat_index] = np.arange(len(geom_c.flat_index))
    return f2v[rc * geom_c.shape[1] + ccc]


def _view_bins(geom_c: ChartGeometry, t1: np.ndarray, t2: np.ndarray, c_cad: np.ndarray, idx: np.ndarray) -> np.ndarray:
    d = c_cad[None, :] - geom_c.point[idx]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return bin_of(d, geom_c.normal[idx], t1[idx], t2[idx])


def scan_slots(geom_c: ChartGeometry, passes: list[PassFrames], stride: int = 2, exposure: float = 0.5,
               min_cos: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Per (bin, coarse texel): the finest effective resolution over every ``stride``-th frame of
    every pass that sees the chart, and its owner (pass index, frame index). Done on a coarse grid
    (``coarse_map``): the best frame hardly changes within half a millimetre, and the full grid cost
    0.55 s per frame for the 2.7 M texels of the iPad back (2026-11-17)."""
    N = len(geom_c.point)
    best = np.full((N_BINS, N), np.inf, np.float32)
    own_p = np.full((N_BINS, N), -1, np.int8)
    own_f = np.full((N_BINS, N), -1, np.int32)
    t1, t2 = local_frames(geom_c.normal)
    info = {}
    flat_best, flat_p, flat_f = best.reshape(-1), own_p.reshape(-1), own_f.reshape(-1)
    for pi, pf in enumerate(passes):
        if geom_c.chart not in pf.charts:
            continue
        Rw, tw = pf.T_world_cad[:3, :3], pf.T_world_cad[:3, 3]
        Pw = (geom_c.point @ Rw.T + tw).astype(np.float64)
        Nw = geom_c.normal @ Rw.T
        T_cad_world = inv_T(pf.T_world_cad)
        ids = [i for i in sorted(pf.T_world_cam)[::stride] if i not in pf.exclude]
        # motion blur assumes a continuous hand-held video; frames far apart (the synthetic captures:
        # 32 renders round the device) are not blurred by the camera's path between them
        steps = [np.linalg.norm(pf.T_world_cam[i + 1][:3, 3] - pf.T_world_cam[i][:3, 3]) for i in ids if i + 1 in pf.T_world_cam]
        exp_p = exposure if (steps and float(np.median(steps)) < 15.0) else 0.0
        seen_frames = 0
        for i in ids:
            fr = frame_resolution(geom_c, Pw, Nw, pf.cap.camera(i), pf.T_world_cam[i], pf.T_world_cam.get(i + 1), exp_p,
                                  min_cos=min_cos)
            idx = np.flatnonzero(np.isfinite(fr.eff_mm))
            if len(idx) == 0:
                continue
            seen_frames += 1
            c_cad = T_cad_world[:3, :3] @ pf.T_world_cam[i][:3, 3] + T_cad_world[:3, 3]
            flat = _view_bins(geom_c, t1, t2, c_cad, idx) * N + idx
            e = fr.eff_mm[idx]
            better = e < flat_best[flat]
            fb = flat[better]
            flat_best[fb] = e[better]
            flat_p[fb] = pi
            flat_f[fb] = i
        info[pf.name] = {"frames_scanned": len(ids), "frames_seeing": seen_frames, "exposure": exp_p,
                         "step_mm_p50": round(float(np.median(steps)), 2) if steps else None}
    return best, own_p, own_f, info


# ---------------------------------------------------------------------------- 3. samples
def _coarse_gain(rect: np.ndarray, seen: np.ndarray, base_c: np.ndarray, csz: tuple, shape: tuple) -> np.ndarray:
    """Per-pixel gain that replaces the view's coarse colour by the reference's (``base_c``)."""
    cs = cv2.resize(seen.astype(np.float32), csz, interpolation=cv2.INTER_AREA)
    cr = cv2.resize(rect * seen[..., None], csz, interpolation=cv2.INTER_AREA) / np.maximum(cs, 1e-6)[..., None]
    own = cv2.GaussianBlur(np.where(cs[..., None] > 0.5, cr, base_c), (0, 0), 2.0)
    gain = np.clip(base_c / np.maximum(own, 1.0), 0.4, 2.5)
    return cv2.resize(gain, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def _project_texels(Pw: np.ndarray, Nw: np.ndarray, cam, T: np.ndarray, min_cos: float, margin: int = 2):
    Tcw = inv_T(T)
    pc = Pw @ Tcw[:3, :3].T + Tcw[:3, 3]
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cam.fx * pc[:, 0] / z + cam.cx
        v = cam.fy * pc[:, 1] / z + cam.cy
    view = T[:3, 3] - Pw
    cosv = np.einsum("ij,ij->i", Nw, view) / np.maximum(np.linalg.norm(view, axis=1), 1e-9)
    ok = (z > 1.0) & (cosv > min_cos) & (u >= margin) & (v >= margin) & (u < cam.width - margin - 1) & (v < cam.height - margin - 1)
    return u.astype(np.float32), v.astype(np.float32), ok


def collect_slot_samples(geom: ChartGeometry, geom_c: ChartGeometry, cmap: np.ndarray, passes: list[PassFrames],
                         own_p: np.ndarray, own_f: np.ndarray, reference: np.ndarray, ref_valid: np.ndarray, align: bool = True,
                         align_down: int = 4, max_shift_mm: float = 4.0, min_cos: float = 0.2) -> tuple[np.ndarray, np.ndarray, dict]:
    """(B, N, 3) uint8 samples of every owned slot (owners on the coarse grid, samples per texel) and
    which slots got one. ``reference`` (rows, cols, 3) float32 is the keyframe median on the chart
    grid: the target of the coarse shading and of the alignment."""
    rows, cols = geom.shape
    N = len(geom.point)
    Nc = len(geom_c.point)
    samples = np.zeros((N_BINS, N, 3), np.uint8)
    filled = np.zeros((N_BINS, N), bool)
    t1c, t2c = local_frames(geom_c.normal)
    has_c = cmap >= 0
    cm_safe = np.where(has_c, cmap, 0)
    cdown = max(1, int(round(1.0 / geom.res)))
    csz = (max(cols // cdown, 1), max(rows // cdown, 1))
    seen_ref = ref_valid.astype(np.float32)
    cs = cv2.resize(seen_ref, csz, interpolation=cv2.INTER_AREA)
    base_c = cv2.resize(reference * seen_ref[..., None], csz, interpolation=cv2.INTER_AREA) / np.maximum(cs, 1e-6)[..., None]
    fillv = np.median(reference[ref_valid], 0) if ref_valid.any() else np.full(3, 128.0, np.float32)
    base_c = np.where(cs[..., None] > 0.2, base_c, fillv)
    base_c = cv2.GaussianBlur(base_c.astype(np.float32), (0, 0), 2.0)
    ref_grey = reference.mean(-1)
    stats = {"decoded": 0, "aligned": 0, "shift_mm": []}
    for pi, pf in enumerate(passes):
        if geom.chart not in pf.charts:
            continue
        mine = own_p == pi
        frames_needed = sorted(set(int(f) for f in np.unique(own_f[mine]) if f >= 0))
        if not frames_needed:
            continue
        Rw, tw = pf.T_world_cad[:3, :3], pf.T_world_cad[:3, 3]
        Pw = (geom.point @ Rw.T + tw).astype(np.float64)
        Nw = geom.normal @ Rw.T
        T_cad_world = inv_T(pf.T_world_cad)
        own_p_flat, own_f_flat = own_p.reshape(-1), own_f.reshape(-1)
        lut_cache = {}
        for i, img in iter_frames(pf.video, frames_needed):
            if pf.undist != "none":
                from ..io.stray import distortion_lut, undistort_maps

                lut = distortion_lut(pf.cap, i)
                c = pf.cap.distortion_center[i]
                if lut is not None and np.all(np.isfinite(c)):
                    if c[0] > pf.cap.width or c[1] > pf.cap.height:
                        c = c * (pf.cap.width / (2 * c[0]))
                    key = (round(float(c[0]), 1), round(float(c[1]), 1))
                    if key not in lut_cache:
                        lut_cache[key] = undistort_maps(lut, (float(c[0]), float(c[1])), pf.cap.width, pf.cap.height,
                                                        inverse=(pf.undist == "inverse"))
                    img = cv2.remap(img, *lut_cache[key], cv2.INTER_LINEAR)
            stats["decoded"] += 1
            u, v, ok = _project_texels(Pw, Nw, pf.cap.camera(i), pf.T_world_cam[i], min_cos)
            rect = _rectify(img, geom, u, v)
            seen = np.zeros(rows * cols, bool)
            seen[geom.flat_index] = ok
            seen = seen.reshape(rows, cols)
            rect = rect * _coarse_gain(rect, seen, base_c, csz, (rows, cols))
            if align and geom.chart in ("screen", "back"):
                M = _align(rect.mean(-1), seen, ref_grey, ref_valid, align_down)
                if M is not None and np.hypot(M[0, 2], M[1, 2]) * geom.res <= max_shift_mm:
                    rect = cv2.warpAffine(rect, M, (cols, rows), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
                    seen = cv2.warpAffine(seen.astype(np.uint8), M, (cols, rows), flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP) > 0
                    stats["aligned"] += 1
                    stats["shift_mm"].append(float(np.hypot(M[0, 2], M[1, 2]) * geom.res))
            okv = seen.reshape(-1)[geom.flat_index] & ok & has_c
            idx = np.flatnonzero(okv)
            if len(idx) == 0:
                continue
            # bins of this frame on the coarse grid, handed down to the texels
            c_cad = T_cad_world[:3, :3] @ pf.T_world_cam[i][:3, 3] + T_cad_world[:3, 3]
            bc = np.full(Nc, -1, np.int64)
            cidx = np.unique(cm_safe[idx])
            bc[cidx] = _view_bins(geom_c, t1c, t2c, c_cad, cidx)
            ci = cm_safe[idx]
            b = bc[ci]
            slot_c = b * Nc + ci
            own = (own_p_flat[slot_c] == pi) & (own_f_flat[slot_c] == i)
            sel = idx[own]
            flat = b[own] * N + sel
            vals = rect.reshape(-1, 3)[geom.flat_index[sel]]
            samples.reshape(-1, 3)[flat] = np.clip(vals, 0, 255).astype(np.uint8)
            filled.reshape(-1)[flat] = True
    sh = stats.pop("shift_mm")
    stats["shift_mm_p50"] = round(float(np.median(sh)), 3) if sh else None
    stats["shift_mm_p90"] = round(float(np.percentile(sh, 90)), 3) if sh else None
    return samples, filled, stats


# ---------------------------------------------------------------------------- 4. per texel
def fuse(samples: np.ndarray, res: np.ndarray, have: np.ndarray, tau: float = 1.8, delta: float = 5.0, support: float = 0.3,
         weight: np.ndarray | None = None, chunk: int = 60_000) -> dict:
    """Per texel over its usable bins (``have``, resolution within ``tau`` of the finest; weights
    1 / resolution^2, times ``weight`` when given): the surface colour is the *lowest well-supported
    lightness*: of the samples whose lightness is shared (within ``delta``) by at least ``support``
    of the weight, the darkest one, and the weighted mean colour of the samples within ``delta`` of
    it. A reflection only adds light, so the surface is the low end; but one misaligned view of
    dark print is also low, and taking the plain lower half doubled the iPad's text 4 mm lower
    (2026-11-17); it never has the support of the others. Returns the surface colour, the per-bin
    lightness excess above it, its maximum and mean, and the number of usable bins."""
    B, N = have.shape
    diffuse = np.zeros((N, 3), np.uint8)
    excess = np.full((B, N), np.nan, np.float16)
    spec_max = np.zeros(N, np.float32)
    spec_mean = np.zeros(N, np.float32)
    n_bins = np.zeros(N, np.uint8)
    for a in range(0, N, chunk):
        b = min(N, a + chunk)
        h = have[:, a:b]
        r = np.where(h, res[:, a:b].astype(np.float32), np.inf)
        rbest = r.min(0)
        use = h & (r <= tau * rbest[None, :])
        w = np.where(use, 1.0 / np.maximum(r, 1e-3) ** 2, 0.0).astype(np.float32)
        if weight is not None:
            w *= weight[:, a:b]
        L = lab_of(samples[:, a:b])[..., 0]                       # (B, n) lightness 0-100
        close = np.abs(L[:, None, :] - L[None, :, :]) <= delta    # (B, B, n)
        sup = np.einsum("jkn,kn->jn", close.astype(np.float32), w)
        tot = w.sum(0)
        cand = use & (sup >= support * tot[None, :])
        Lc = np.where(cand, L, np.inf)
        j_low = np.argmin(Lc, 0)
        j_max = np.argmax(np.where(use, sup, -1.0), 0)
        j = np.where(np.isfinite(Lc.min(0)), j_low, j_max)
        Lj = np.take_along_axis(L, j[None, :], 0)[0]
        inmode = use & (np.abs(L - Lj[None, :]) <= delta)
        wb = np.where(inmode, w, 0.0)
        sw = wb.sum(0)
        rgb = samples[:, a:b].astype(np.float32)
        col = (rgb * wb[..., None]).sum(0) / np.maximum(sw, 1e-12)[:, None]
        diffuse[a:b] = np.where((sw > 0)[:, None], np.clip(col, 0, 255), 0).astype(np.uint8)
        Ld = (L * wb).sum(0) / np.maximum(sw, 1e-12)
        ex = np.where(use, L - Ld[None, :], np.nan)
        excess[:, a:b] = ex.astype(np.float16)
        nuse = use.sum(0)
        with np.errstate(all="ignore"):
            spec_max[a:b] = np.where(nuse > 0, np.nanmax(np.where(use, ex, -np.inf), 0), 0.0)
            spec_mean[a:b] = np.where(nuse > 0, np.nansum(np.where(use, np.maximum(ex, 0), 0), 0) / np.maximum(nuse, 1), 0.0)
        n_bins[a:b] = np.minimum(nuse, 255).astype(np.uint8)
    return {"diffuse": diffuse, "excess": excess, "spec_max": spec_max, "spec_mean": spec_mean, "n_bins": n_bins}


def decompose_chart(geom: ChartGeometry, passes: list[PassFrames], reference_rgb: np.ndarray, reference_seen: np.ndarray,
                    model=None, stride: int = 2, align: bool = True, coarse_mm: float = 0.6, rounds: int = 2,
                    min_frame_share: float = 0.002, log=None) -> ChartDecomposition:
    """Steps 2-4 for one chart. ``reference_rgb`` (N, 3) and ``reference_seen`` (N,) bool: the
    keyframe median and where it is valid (``surface.maps.compute_stats``). ``model``: the device
    model the chart belongs to (for the coarse grid of the slot scan).

    ``rounds``: the first aligns every view to the keyframe median, which is blurred; the second
    aligns them again to the first round's surface colour. Frames that own fewer than
    ``min_frame_share`` of the slots are left out (each costs a decode, a resampling and an
    alignment for a few texels)."""
    from ..surface.maps import chart_geometry

    log = log or (lambda m: None)
    t0 = time.time()
    rows, cols = geom.shape
    k = max(1, int(round(coarse_mm / geom.res)))
    geom_c = chart_geometry(model, geom.chart, res=geom.res * k) if (model is not None and k > 1) else geom
    k = k if geom_c is not geom else 1
    cmap = coarse_map(geom, geom_c, k) if k > 1 else np.arange(len(geom.point))
    best_c, own_p, own_f, scan_info = scan_slots(geom_c, passes, stride=stride)
    # frames owning almost nothing are dropped
    key = own_p.astype(np.int64) * 1_000_000 + own_f
    ok_slot = own_f >= 0
    uk, cnt = np.unique(key[ok_slot], return_counts=True)
    small = uk[cnt < min_frame_share * ok_slot.sum()]
    if len(small):
        drop = np.isin(key, small)
        own_p[drop], own_f[drop] = -1, -1
        best_c[drop] = np.inf
    scan_info["frames_owning"] = int(len(uk))
    scan_info["frames_dropped"] = int(len(small))
    scan_info["slots_dropped_share"] = round(float(cnt[cnt < min_frame_share * ok_slot.sum()].sum() / max(ok_slot.sum(), 1)), 4)
    t1 = time.time()
    ref = np.zeros((rows * cols, 3), np.float32)
    ref[geom.flat_index] = np.nan_to_num(reference_rgb.astype(np.float32))
    ref = ref.reshape(rows, cols, 3)
    rv = np.zeros(rows * cols, bool)
    rv[geom.flat_index] = reference_seen
    rv = rv.reshape(rows, cols)
    cm_safe = np.where(cmap >= 0, cmap, 0)
    res_f = best_c[:, cm_safe]                                    # (B, N) resolution of each texel's slot
    res_f[:, cmap < 0] = np.inf
    rinfo = []
    for rnd in range(max(1, rounds)):
        samples, have, sinfo = collect_slot_samples(geom, geom_c, cmap, passes, own_p, own_f, ref, rv, align=align)
        out = fuse(samples, res_f, have)
        rinfo.append(sinfo)
        log(f"  decompose {geom.chart} round {rnd}: {sinfo['decoded']} frames, {sinfo['aligned']} aligned "
            f"(shift p50 {sinfo['shift_mm_p50']} mm, p90 {sinfo['shift_mm_p90']} mm)")
        if rnd + 1 < rounds and geom.chart in ("screen", "back") and align:
            # the next round aligns to this round's surface colour (sharp), keeping the keyframe
            # median where no slot was filled
            d_img = np.zeros((rows * cols, 3), np.float32)
            got = out["n_bins"] > 0
            d_img[geom.flat_index[got]] = out["diffuse"][got]
            d_img = d_img.reshape(rows, cols, 3)
            gm = np.zeros(rows * cols, bool)
            gm[geom.flat_index[got]] = True
            gm = gm.reshape(rows, cols)
            ref = np.where(gm[..., None], d_img, ref)
            rv = rv | gm
            del samples
        else:
            break
    t2 = time.time()
    rbest = np.where(have, res_f, np.inf).min(0)
    owner_p = own_p[:, cm_safe]
    owner_f = own_f[:, cm_safe].astype(np.int16 if own_f.max(initial=0) < 32767 else np.int32)
    owner_p[:, cmap < 0] = -1
    owner_f[:, cmap < 0] = -1
    info = {"chart": geom.chart, "texels": int(len(geom.point)), "res_mm": geom.res, "slot_grid_mm": round(geom_c.res, 3),
            "bins": N_BINS, "stride": stride, "scan": scan_info, "rounds": rinfo, "slots": int(have.sum()),
            "bins_per_texel_p50": float(np.median(out["n_bins"][out["n_bins"] > 0])) if (out["n_bins"] > 0).any() else 0.0,
            "best_res_mm_p50": round(float(np.median(rbest[np.isfinite(rbest)])), 3) if np.isfinite(rbest).any() else None,
            "seconds": {"scan": round(t1 - t0, 1), "samples_and_fuse": round(t2 - t1, 1)}}
    log(f"  decompose {geom.chart}: {info['slots']} slots ({info['bins_per_texel_p50']:.0f} bins per texel), "
        f"{scan_info['frames_owning'] - scan_info['frames_dropped']} frames (dropped {scan_info['frames_dropped']} owning "
        f"{scan_info['slots_dropped_share']:.1%}), finest {info['best_res_mm_p50']} mm; scan {info['seconds']['scan']} s, "
        f"samples+fuse {info['seconds']['samples_and_fuse']} s")
    return ChartDecomposition(geom.chart, out["diffuse"], out["excess"], out["spec_max"], out["spec_mean"], out["n_bins"], rbest,
                              owner_p, owner_f, res_f.astype(np.float16), info)


# ---------------------------------------------------------------------------- quality measures
def gradient_energy(img: np.ndarray, mask: np.ndarray | None = None) -> float:
    g = img.mean(-1).astype(np.float32) if img.ndim == 3 else img.astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    e = gx * gx + gy * gy
    if mask is None:
        return float(e.mean())
    m = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    return float(e[m].mean()) if m.any() else float("nan")


def flatten_shading(img: np.ndarray, valid: np.ndarray, res: float, scale_mm: float = 25.0, feature_dl: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    """Albedo-like colour: the slowly varying illumination over the face (the photographer's shadow,
    window light) divided out. The illumination is a masked median at ``scale_mm`` (on a 1 mm grid)
    of the lightness, leaving out features that stand ``feature_dl`` above or below their 10 mm
    surroundings (logo, print, camera) so they keep their contrast, then smoothed. Returns the
    flattened image (uint8 RGB) and the gain applied."""
    h, w = valid.shape
    k = max(1, int(round(1.0 / res)))
    small = cv2.resize(img.astype(np.float32), (max(w // k, 1), max(h // k, 1)), interpolation=cv2.INTER_AREA)
    vs = cv2.resize(valid.astype(np.float32), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA) > 0.99
    L = lab_of(np.clip(small, 0, 255).astype(np.uint8))[..., 0]
    from scipy import ndimage as ndi

    fill = float(np.median(L[vs])) if vs.any() else 50.0
    Lm = np.where(vs, L, fill)
    loc = ndi.median_filter(Lm, size=11)
    keep = vs & (np.abs(L - loc) < feature_dl)
    Lk = np.where(keep, L, np.nan)
    # masked median: fill the left-out cells from their surroundings, then a large median
    idx = ndi.distance_transform_edt(~keep, return_distances=False, return_indices=True)
    Lf = Lk[tuple(idx)] if keep.any() else Lm
    illum = ndi.median_filter(Lf, size=max(3, int(round(scale_mm))) | 1)
    illum = cv2.GaussianBlur(illum.astype(np.float32), (0, 0), scale_mm / 3.0)
    target = float(np.median(L[keep])) if keep.any() else fill
    gain_l = target / np.maximum(illum, 1.0)
    gain = cv2.resize(gain_l.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    # lightness gain applied to RGB as a gain on (approximately) linear light
    lin = (img.astype(np.float32) / 255.0) ** 2.2
    g_lin = np.clip(gain, 0.5, 2.0) ** 2.2
    out = np.clip((lin * g_lin[..., None]) ** (1 / 2.2) * 255.0, 0, 255).astype(np.uint8)
    out[~valid] = img[~valid]
    return out, gain


def streak_band(img: np.ndarray, mask: np.ndarray, res: float, lo_mm: float = 2.0, hi_mm: float = 10.0,
                feature_dl: float = 6.0) -> float:
    """Variation at the width of reflection streaks: std of lightness band-passed between ``lo_mm``
    and ``hi_mm`` (difference of Gaussians), over the face without design features (cells
    ``feature_dl`` away from their 10 mm median). Slow shading and fine print hardly count."""
    from scipy import ndimage as ndi

    L = lab_of(img.astype(np.uint8))[..., 0]
    m = mask.astype(np.float32)

    def blur(x, s):
        num = cv2.GaussianBlur(x * m, (0, 0), s / res)
        den = cv2.GaussianBlur(m, (0, 0), s / res)
        return num / np.maximum(den, 1e-6)

    bp = blur(L, lo_mm) - blur(L, hi_mm)
    k = max(1, int(round(1.0 / res)))
    loc = ndi.median_filter(np.where(mask, L, np.median(L[mask]) if mask.any() else 0)[::k, ::k], size=11)
    loc = cv2.resize(loc.astype(np.float32), (L.shape[1], L.shape[0]), interpolation=cv2.INTER_LINEAR)
    feat = np.abs(L - loc) >= feature_dl
    feat = ndi.binary_dilation(feat, iterations=int(round(hi_mm / res)))
    inner = cv2.erode(mask.astype(np.uint8), np.ones((int(2 * hi_mm / res) | 1,) * 2, np.uint8)) > 0
    sel = inner & ~feat
    return float(np.std(bp[sel])) if sel.sum() > 1000 else float("nan")


def streak_index(img: np.ndarray, mask: np.ndarray, sigma_mm: float, res: float) -> float:
    """Low-frequency lightness variation (std of a ``sigma_mm`` blur, in Lab L units) over ``mask``:
    broad reflection streaks and shading patches raise it; a uniform surface keeps it low."""
    L = lab_of(img.astype(np.uint8))[..., 0]
    m = mask.astype(np.float32)
    s = sigma_mm / res
    num = cv2.GaussianBlur(L * m, (0, 0), s)
    den = cv2.GaussianBlur(m, (0, 0), s)
    lo = num / np.maximum(den, 1e-6)
    inner = cv2.erode(mask.astype(np.uint8), np.ones((int(3 * s) | 1, int(3 * s) | 1), np.uint8)) > 0
    return float(np.std(lo[inner])) if inner.any() else float("nan")


def pass_frames_from_prepared(pp, T_world_cad: np.ndarray | None = None) -> PassFrames | None:
    """A ``PassFrames`` from a prepared Stray pass (``pipeline.PreparedPass`` with ``_poses``/``_reg``)."""
    poses = getattr(pp, "_poses", None)
    reg = getattr(pp, "_reg", None)
    if poses is None or reg is None or pp.video is None:
        return None
    from ..io.stray import read_stray

    cap = read_stray(Path(pp.report["root"]))
    if cap.T_ark_cam is None:
        return None
    key_T = {int(i): np.asarray(T) for i, T in poses.T_world_cam.items()}
    allT = all_frame_poses(np.asarray(cap.T_ark_cam), key_T, len(cap))      # ARKit corrected like the keyframes
    return PassFrames(pp.name, cap, Path(pp.video), allT, reg.T_world_cad if T_world_cad is None else T_world_cad, pp.charts,
                      pp.report.get("undistortion", "none"), frozenset(getattr(pp, "exclude", frozenset())))


# ---------------------------------------------------------------------------- command line (experiments on a finished run)
def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    from ..geometry.transforms import inv_T as _inv
    from ..io.stray import read_stray
    from ..surface.maps import chart_geometry, collect_samples, compute_stats
    from ..utils.io import read_json, write_json

    ap = argparse.ArgumentParser(prog="python -m flawmap.texture.decompose")
    ap.add_argument("run", help="a finished run folder (run.json, passes/<pass>/poses.json, registration.json)")
    ap.add_argument("--pass", dest="passes", action="append", required=True, help="NAME=STRAY_FOLDER (repeat)")
    ap.add_argument("--chart", action="append", help="charts to decompose (default: every chart the passes saw)")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--no-align", action="store_true")
    ap.add_argument("--out", help="default <run>/decompose")
    a = ap.parse_args(argv)
    run = Path(a.run)
    out = Path(a.out) if a.out else run / "decompose"
    out.mkdir(parents=True, exist_ok=True)
    rep = read_json(run / "run.json")
    from ..anchor.template import model_of_run

    model = model_of_run(rep)
    pfs, key = [], []
    for spec in a.passes:
        name, _, root = spec.partition("=")
        root = Path(root)
        pr = read_json(run / "passes" / name / "poses.json")
        key_T = {int(k): np.asarray(v["T_world_cam"]) for k, v in pr["poses"].items()}
        T_wcad = np.asarray(read_json(run / "passes" / name / "registration.json")["T_world_cad"])
        cap = read_stray(root)
        face = next(p["face_up"] for p in rep["passes"] if p["name"] == name)
        undist = next(p.get("undistortion", "none") for p in rep["passes"] if p["name"] == name)
        charts = ["screen", "frame"] if face == "screen" else ["back", "frame"]
        pfs.append(PassFrames(name, cap, root / "rgb.mp4", all_frame_poses(np.asarray(cap.T_ark_cam), key_T, len(cap)), T_wcad,
                              charts, undist))
        key.append((name, cap, key_T, T_wcad, charts))
    charts = a.chart or sorted({c for pf in pfs for c in pf.charts}, key=["screen", "back", "frame"].index)
    report = {"run": str(run), "charts": {}}
    for c in charts:
        geom = chart_geometry(model, c)
        parts = []
        for name, cap, key_T, T_wcad, cs in key:
            if c not in cs:
                continue
            fr = dict(iter_frames(Path(next(pf.video for pf in pfs if pf.name == name)), sorted(key_T)))
            cams = {i: cap.camera(i) for i in fr}
            Tcad = {i: _inv(T_wcad) @ key_T[i] for i in fr}
            parts.append(collect_samples(model, {c: geom}, fr, cams, Tcad, np.eye(4))[c])
            del fr
        from ..surface.maps import ChartMaps, ViewSamples

        cm = ChartMaps(geom, ViewSamples(sum((x.samples.frames for x in parts), []), np.concatenate([x.samples.rgb for x in parts]),
                                         np.concatenate([x.samples.cos for x in parts])))
        compute_stats(cm)
        med = np.nan_to_num(cm.stats["med_rgb"]).astype(np.float32)
        dec = decompose_chart(geom, pfs, med, cm.stats["n_good"] >= 1, model=model, stride=a.stride, align=not a.no_align, log=print)
        img_d = geom.to_image(dec.diffuse)
        img_m = geom.to_image(np.clip(med, 0, 255).astype(np.uint8))
        img_a, _ = flatten_shading(img_d, geom.valid & (geom.to_image(dec.n_bins, fill=0) > 0), geom.res) if c != "frame" else (img_d, None)
        cv2.imwrite(str(out / f"{c}_albedo.png"), cv2.cvtColor(img_a, cv2.COLOR_RGB2BGR))
        smax = geom.to_image(np.clip(dec.spec_max * 4, 0, 255).astype(np.uint8))
        valid = geom.valid
        cv2.imwrite(str(out / f"{c}_diffuse.png"), cv2.cvtColor(img_d, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / f"{c}_keyframe_median.png"), cv2.cvtColor(img_m, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / f"{c}_spec_max.png"), smax)
        rec = {**dec.info, "gradient_energy": {"diffuse": round(gradient_energy(img_d, valid), 2), "median": round(gradient_energy(img_m, valid), 2)}}
        if c in ("screen", "back"):
            rec["streak_index_8mm"] = {"diffuse": round(streak_index(img_d, valid, 8.0, geom.res), 3),
                                       "albedo": round(streak_index(img_a, valid, 8.0, geom.res), 3),
                                       "median": round(streak_index(img_m, valid, 8.0, geom.res), 3)}
            rec["streak_band_2_10mm"] = {"diffuse": round(streak_band(img_d, valid, geom.res), 3),
                                         "albedo": round(streak_band(img_a, valid, geom.res), 3),
                                         "median": round(streak_band(img_m, valid, geom.res), 3)}
        if c == "back":
            # the window used on 11-17 for the texture sharpness: 75 x 80 mm around the logo
            u0, u1, v0, v1 = [int(round(x / geom.res)) for x in (108 - 37.5, 108 + 37.5, 140 - 40, 140 + 40)]
            rec["logo_window_energy"] = {"diffuse": round(gradient_energy(img_d[v0:v1, u0:u1]), 1),
                                         "albedo": round(gradient_energy(img_a[v0:v1, u0:u1]), 1),
                                         "median": round(gradient_energy(img_m[v0:v1, u0:u1]), 1)}
        report["charts"][c] = rec
        np.savez_compressed(out / f"{c}_decomposition.npz", diffuse=dec.diffuse, spec_max=dec.spec_max, spec_mean=dec.spec_mean,
                            n_bins=dec.n_bins, best_res=dec.best_res, excess=dec.excess)
        print(json.dumps({k: v for k, v in rec.items() if k not in ("scan",)}, ensure_ascii=False), flush=True)
    write_json(out / "decompose.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
