"""Every frame of a pass, per texel: how fine the video resolves the surface, and the sharpest texture.

The pipeline samples 100-140 keyframes and takes the median over them. On a fast hand-held orbit
most frames are motion-blurred and the median washes detail out (the iPad's logo: gradient energy
17 against 509 in the sharpest frame). This module uses all frames (60 fps) instead and decides per
texel, not per frame, so it does not depend on how the capture was filmed: a slow, close capture
simply resolves more.

1. **Poses for all frames.** The final keyframe poses (board, boardless or COLMAP) differ from the
   raw ARKit poses by a correction that varies slowly; it is interpolated between keyframes (rotation
   by slerp, translation linearly) and applied to every frame's ARKit pose.
2. **Effective resolution** of a texel in a frame, in mm:
   ``footprint * sqrt(1 + blur^2)`` with ``footprint = depth / f / cos`` (surface mm per pixel) and
   ``blur = exposure * |image motion of the texel per frame|`` in pixels, predicted from the poses
   of the neighbouring frames (``exposure`` = exposure time / frame interval). The faces of a
   plate-like device are convex, so a texel is seen when it faces the camera and projects inside
   the image. ``resolution_map`` keeps the best and the k-th best value per texel: what the video
   can show at all, and what at least k frames agree on.
3. **Texture.** Per texel the k frames with the best effective resolution are kept (streaming, so
   memory does not grow with the number of frames). Each contributing frame is rectified into the
   chart, aligned to a first fused texture by ECC (affine, on gradient magnitude, in the chart), and
   the aligned samples are fused (weighted by 1/resolution^2 after dropping samples far from the
   median, i.e. glare that moves with the view).

    python -m flawmap.texture.allframes runs/<run> --pass back=<stray folder> [--chart back] [--k 5]
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from ..geometry.camera import Camera
from ..geometry.transforms import inv_T
from ..io.stray import iter_frames, read_stray
from ..surface.maps import ChartGeometry, chart_geometry
from ..utils.io import write_json
from ..utils.logging import get_logger

log = get_logger("flawmap.texture")


# ---------------------------------------------------------------------------- poses of every frame
def all_frame_poses(T_ark: np.ndarray, key_T: dict[int, np.ndarray], n: int) -> dict[int, np.ndarray]:
    """``T_world_cam`` for frames 0..n-1 from the keyframes' final poses and every frame's ARKit
    pose ``T_ark`` (same camera axes): the correction ``key_T[k] @ inv(T_ark[k])`` is interpolated
    between keyframes and held constant before the first and after the last."""
    keys = sorted(k for k in key_T if k < n)
    C = np.stack([key_T[k] @ inv_T(T_ark[k]) for k in keys])
    t = np.array(keys, float)
    slerp = Slerp(t, Rotation.from_matrix(C[:, :3, :3]))
    q = np.clip(np.arange(n, dtype=float), t[0], t[-1])
    R = slerp(q).as_matrix()
    tr = np.stack([np.interp(q, t, C[:, k, 3]) for k in range(3)], 1)
    out = {}
    for i in range(n):
        Ci = np.eye(4)
        Ci[:3, :3], Ci[:3, 3] = R[i], tr[i]
        out[i] = Ci @ T_ark[i]
    return out


# ---------------------------------------------------------------------------- effective resolution
@dataclass
class FrameRes:
    eff_mm: np.ndarray        # (N,) effective resolution, inf where not seen
    u: np.ndarray             # (N,) pixel coordinates in the frame
    v: np.ndarray
    blur_px: np.ndarray


def frame_resolution(geom: ChartGeometry, Pw: np.ndarray, Nw: np.ndarray, cam: Camera, T: np.ndarray,
                     T_next: np.ndarray | None, exposure: float, min_cos: float = 0.3, margin: int = 2) -> FrameRes:
    Tcw = inv_T(T)
    pc = Pw @ Tcw[:3, :3].T + Tcw[:3, 3]
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cam.fx * pc[:, 0] / z + cam.cx
        v = cam.fy * pc[:, 1] / z + cam.cy
    view = T[:3, 3] - Pw
    dist = np.linalg.norm(view, axis=1)
    cosv = np.einsum("ij,ij->i", Nw, view) / np.maximum(dist, 1e-9)
    ok = (z > 1.0) & (cosv > min_cos) & (u >= margin) & (v >= margin) & (u < cam.width - margin - 1) & (v < cam.height - margin - 1)
    foot = z / cam.fx / np.maximum(cosv, 1e-3)
    if T_next is not None:
        Tn = inv_T(T_next)
        pn = Pw @ Tn[:3, :3].T + Tn[:3, 3]
        with np.errstate(divide="ignore", invalid="ignore"):
            un = cam.fx * pn[:, 0] / pn[:, 2] + cam.cx
            vn = cam.fy * pn[:, 1] / pn[:, 2] + cam.cy
        blur = exposure * np.hypot(un - u, vn - v)
    else:
        blur = np.zeros_like(u)
    eff = np.where(ok, foot * np.sqrt(1.0 + blur ** 2), np.inf).astype(np.float32)
    return FrameRes(eff, u.astype(np.float32), v.astype(np.float32), blur.astype(np.float32))


class TopK:
    """Per texel, the k smallest scores seen so far and the frames they came from."""

    def __init__(self, n: int, k: int):
        self.score = np.full((k, n), np.inf, np.float32)
        self.frame = np.full((k, n), -1, np.int32)

    def add(self, score: np.ndarray, frame: int) -> None:
        worst = np.argmax(self.score, 0)
        cols = np.flatnonzero(score < self.score[worst, np.arange(len(score))])
        if len(cols):
            self.score[worst[cols], cols] = score[cols]
            self.frame[worst[cols], cols] = frame

    def sorted(self) -> tuple[np.ndarray, np.ndarray]:
        o = np.argsort(self.score, 0)
        return np.take_along_axis(self.score, o, 0), np.take_along_axis(self.frame, o, 0)


def scan(geom: ChartGeometry, T_world_cad: np.ndarray, cap, poses: dict[int, np.ndarray], k: int, exposure: float,
         frames: list[int] | None = None) -> tuple[TopK, dict]:
    """Pass over the poses only (no images): best-k effective resolution per texel."""
    Pw = (geom.point @ T_world_cad[:3, :3].T + T_world_cad[:3, 3]).astype(np.float64)
    Nw = geom.normal @ T_world_cad[:3, :3].T
    ids = sorted(poses) if frames is None else frames
    top = TopK(len(Pw), k)
    seen_frames = 0
    blur_all = []
    for j, i in enumerate(ids):
        nxt = poses.get(i + 1)
        fr = frame_resolution(geom, Pw, Nw, cap.camera(i), poses[i], nxt, exposure)
        if np.isfinite(fr.eff_mm).any():
            seen_frames += 1
            top.add(fr.eff_mm, i)
            b = fr.blur_px[np.isfinite(fr.eff_mm)]
            blur_all.append(float(np.median(b)))
    return top, {"frames_seen": seen_frames, "frames": len(ids), "median_blur_px_per_frame": blur_all}


# ---------------------------------------------------------------------------- texture
def _rectify(img: np.ndarray, geom: ChartGeometry, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The frame resampled onto the full chart grid (float32 RGB), NaN where the texel is not valid."""
    rows, cols = geom.shape
    mu = np.full(rows * cols, -10.0, np.float32)
    mv = np.full(rows * cols, -10.0, np.float32)
    mu[geom.flat_index], mv[geom.flat_index] = u, v
    out = cv2.remap(img, mu.reshape(rows, cols), mv.reshape(rows, cols), cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0).astype(np.float32)
    return out


def _grad_mag(g: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)), (0, 0), 1.5)


def _align(img: np.ndarray, mask: np.ndarray, ref: np.ndarray, ref_mask: np.ndarray, down: int) -> np.ndarray | None:
    """Affine (chart pixels) taking ``img`` onto ``ref`` by ECC on gradient magnitude at 1/down
    resolution, or None when ECC fails or the motion is implausible."""
    if down > 1:
        sz = (img.shape[1] // down, img.shape[0] // down)
        a = cv2.resize(img, sz, interpolation=cv2.INTER_AREA)
        b = cv2.resize(ref, sz, interpolation=cv2.INTER_AREA)
        m = cv2.resize(mask.astype(np.uint8), sz, interpolation=cv2.INTER_NEAREST) & \
            cv2.resize(ref_mask.astype(np.uint8), sz, interpolation=cv2.INTER_NEAREST)
    else:
        a, b, m = img, ref, (mask & ref_mask).astype(np.uint8)
    m = cv2.erode(m, np.ones((5, 5), np.uint8))
    if m.sum() < 2000:
        return None
    ga, gb = _grad_mag(a), _grad_mag(b)
    for motion in (cv2.MOTION_AFFINE, cv2.MOTION_TRANSLATION):     # affine first; a weak overlap still allows a shift
        M = np.eye(2, 3, dtype=np.float32)
        try:
            _, M = cv2.findTransformECC(gb, ga, M, motion, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), m, 5)
        except cv2.error:
            continue
        if abs(np.linalg.det(M[:, :2]) - 1.0) > 0.03 or np.hypot(M[0, 2], M[1, 2]) * down > 200:
            continue
        M[:, 2] *= down
        return M
    return None


def fuse_texture(geom: ChartGeometry, T_world_cad: np.ndarray, cap, video: Path, poses: dict[int, np.ndarray],
                 top: TopK, align_down: int = 4, rounds: int = 1) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fused texture (rows, cols, 3) uint8 and its valid mask from the best-k frames per texel."""
    score, frame = top.sorted()
    k, N = frame.shape
    contrib = sorted(set(int(f) for f in np.unique(frame) if f >= 0))
    Pw = (geom.point @ T_world_cad[:3, :3].T + T_world_cad[:3, 3]).astype(np.float64)
    Nw = geom.normal @ T_world_cad[:3, :3].T
    slot = {f: (frame == f) for f in contrib}                  # (k, N) where frame f holds a slot
    rows, cols = geom.shape
    valid = np.zeros(rows * cols, bool)
    valid[geom.flat_index] = True
    valid = valid.reshape(rows, cols)
    affines: dict[int, np.ndarray] = {}
    stats = {"contributing_frames": len(contrib), "k": k}
    ref = ref_mask = None
    fused = None
    # shading: each view sees its own reflections and exposure, so texels fused from different views
    # left bright and dark patches. A coarse (1 mm) median over every contributing view gives the
    # reference shading; each view's own coarse shading is divided out and replaced by it before
    # fusing, which keeps the fine detail and removes the seams.
    cdown = max(1, int(round(1.0 / geom.res)))
    csz = (cols // cdown, rows // cdown)
    coarse = []
    for i, img in iter_frames(video, contrib):
        fr = frame_resolution(geom, Pw, Nw, cap.camera(i), poses[i], None, 0.0)
        rect = _rectify(img, geom, fr.u, fr.v)
        seen = np.zeros(rows * cols, np.float32)
        seen[geom.flat_index] = np.isfinite(fr.eff_mm)
        seen = seen.reshape(rows, cols)
        cs = cv2.resize(seen, csz, interpolation=cv2.INTER_AREA)
        cr = cv2.resize(rect * seen[..., None], csz, interpolation=cv2.INTER_AREA) / np.maximum(cs, 1e-6)[..., None]
        cr[cs < 0.9] = np.nan
        coarse.append(cr.astype(np.float16))
    with np.errstate(all="ignore"):
        base = np.nanmedian(np.stack(coarse).astype(np.float32), 0)
    base = np.where(np.isfinite(base), base, np.nanmedian(base))
    base = cv2.GaussianBlur(base, (0, 0), 2.0)
    del coarse

    def shade(rect: np.ndarray, seen: np.ndarray) -> np.ndarray:
        cs = cv2.resize(seen.astype(np.float32), csz, interpolation=cv2.INTER_AREA)
        cr = cv2.resize(rect * seen[..., None], csz, interpolation=cv2.INTER_AREA) / np.maximum(cs, 1e-6)[..., None]
        own = cv2.GaussianBlur(np.where(cs[..., None] > 0.5, cr, base), (0, 0), 2.0)
        gain = np.clip(base / np.maximum(own, 1.0), 0.4, 2.5)
        return rect * cv2.resize(gain, (cols, rows), interpolation=cv2.INTER_LINEAR)

    for rnd in range(rounds + 1):
        samples = np.zeros((k, N, 3), np.uint8)
        have = np.zeros((k, N), bool)
        moved = []
        for i, img in iter_frames(video, contrib):
            fr = frame_resolution(geom, Pw, Nw, cap.camera(i), poses[i], None, 0.0)
            rect = _rectify(img, geom, fr.u, fr.v)
            seen = np.zeros(rows * cols, bool)
            seen[geom.flat_index] = np.isfinite(fr.eff_mm)
            seen = seen.reshape(rows, cols)
            rect = shade(rect, seen)
            if ref is not None:
                M = _align(rect.mean(-1), seen, ref, ref_mask, align_down)
                if M is not None:
                    rect = cv2.warpAffine(rect, M, (cols, rows), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
                    seen = cv2.warpAffine(seen.astype(np.uint8), M, (cols, rows), flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP) > 0
                    affines[i] = M
                    moved.append(float(np.hypot(M[0, 2], M[1, 2]) * geom.res))
            vals = rect.reshape(-1, 3)[geom.flat_index]
            ok = seen.reshape(-1)[geom.flat_index]
            sl = slot[i] & ok[None, :]
            for s in range(k):
                c = np.flatnonzero(sl[s])
                samples[s, c] = np.clip(vals[c], 0, 255).astype(np.uint8)
                have[s, c] = True
        # fuse: drop samples far from the per-texel median (glare), weight the rest by 1/res^2
        f = samples.astype(np.float32)
        f[~have] = np.nan
        with np.errstate(all="ignore"):
            med = np.nanmedian(f, 0)
            dev = np.nanmax(np.abs(f - med[None]), -1)
            keep = have & (dev < 40.0)
            w = np.where(keep, 1.0 / np.maximum(score, 1e-3) ** 2, 0.0)
            fused_v = np.nansum(np.where(keep[..., None], f, 0.0) * w[..., None], 0) / np.maximum(w.sum(0), 1e-9)[:, None]
            fused_v = np.where((w.sum(0) > 0)[:, None], fused_v, med)
        fused = np.zeros((rows * cols, 3), np.float32)
        fused[geom.flat_index] = np.nan_to_num(fused_v)
        fused = fused.reshape(rows, cols, 3)
        ref = fused.mean(-1)
        mm = np.zeros(rows * cols, bool)
        mm[geom.flat_index] = have.any(0)
        ref_mask = mm.reshape(rows, cols)
        stats[f"round{rnd}"] = {"aligned": len(moved), "align_move_mm_p50": round(float(np.median(moved)), 3) if moved else None,
                                "align_move_mm_p90": round(float(np.percentile(moved, 90)), 3) if moved else None}
        log.info(f"  fuse round {rnd}: {len(contrib)} frames" + (f", aligned {len(moved)}, move p50 {np.median(moved):.2f} mm" if moved else ""))
    return np.clip(fused, 0, 255).astype(np.uint8), ref_mask, stats


# ---------------------------------------------------------------------------- reports
def sharpness(img: np.ndarray, mask: np.ndarray) -> float:
    g = img.mean(-1).astype(np.float32) if img.ndim == 3 else img.astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    m = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    return float((gx * gx + gy * gy)[m].mean()) if m.any() else float("nan")


def resolution_image(geom: ChartGeometry, eff: np.ndarray, lo: float = 0.15, hi: float = 1.5) -> np.ndarray:
    """Colour map of a resolution array (log scale, turbo: blue = fine, red = coarse, grey = unseen)."""
    img = np.full(geom.shape[0] * geom.shape[1], np.nan, np.float32)
    img[geom.flat_index] = eff
    x = np.clip((np.log(np.where(np.isfinite(img), img, hi)) - np.log(lo)) / (np.log(hi) - np.log(lo)), 0, 1)
    col = cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_TURBO).reshape(-1, 3)
    col[~np.isfinite(img)] = (90, 90, 90)
    return col.reshape(geom.shape[0], geom.shape[1], 3)


def share_at(eff: np.ndarray, levels=(0.2, 0.3, 0.5, 1.0)) -> dict:
    return {f"<= {l} mm": round(float((eff <= l).mean()), 3) for l in levels}


# ---------------------------------------------------------------------------- command line
def main(argv: list[str] | None = None) -> int:
    from ..anchor.parametric import DeviceSpec, build_device_model
    from ..anchor.template import template_spec

    ap = argparse.ArgumentParser(prog="python -m flawmap.texture.allframes")
    ap.add_argument("run", help="a finished run folder (passes/<pass>/poses.json, registration.json, run.json)")
    ap.add_argument("--pass", dest="pas", required=True, help="NAME=STRAY_FOLDER, e.g. back=data/captures/x/back_orbit")
    ap.add_argument("--chart", help="chart to texture (default: the pass's face)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--res", type=float, default=0.1, help="texel size in mm")
    ap.add_argument("--exposure", type=float, default=0.5, help="exposure time / frame interval")
    ap.add_argument("--out", help="output folder (default <run>/allframes/<pass>)")
    ap.add_argument("--reuse", action="store_true", help="reuse the best-k scan saved in the output folder")
    a = ap.parse_args(argv)
    run = Path(a.run)
    name, _, root = a.pas.partition("=")
    root = Path(root)
    out = Path(a.out) if a.out else run / "allframes" / name
    out.mkdir(parents=True, exist_ok=True)
    rep = json.loads((run / "run.json").read_text(encoding="utf-8"))
    if rep.get("device") == "template":
        est = rep["processing"]["device_estimate"]
        model = build_device_model(template_spec(*est["size_mm"], est["corner_radius"]))
    else:
        model = build_device_model(DeviceSpec.by_slug(rep["device"]))
    pr = json.loads((run / "passes" / name / "poses.json").read_text(encoding="utf-8"))
    key_T = {int(k): np.array(v["T_world_cam"]) for k, v in pr["poses"].items()}
    T_world_cad = np.array(json.loads((run / "passes" / name / "registration.json").read_text(encoding="utf-8"))["T_world_cad"])
    cap = read_stray(root)
    n = len(cap)
    chart = a.chart or next(p["face_up"] for p in rep["passes"] if p["name"] == name)
    t0 = time.time()
    poses = all_frame_poses(cap.T_ark_cam, key_T, n)
    geom = chart_geometry(model, chart, res=a.res)
    log.info(f"[{name}] {n} frames, chart {chart} {geom.shape[1]} x {geom.shape[0]} at {a.res} mm ({len(geom.point) / 1e6:.1f} M texels)")
    keys = sorted(key_T)
    cache = out / "topk.npz"
    if a.reuse and cache.exists():
        z = np.load(cache)
        top = TopK(len(geom.point), a.k)
        top.score, top.frame = z["score"], z["frame"]
        scan_info, sk = {"median_blur_px_per_frame": list(z["blur"])}, z["score_keys"]
        log.info(f"[{name}] reusing {cache}")
    else:
        top, scan_info = scan(geom, T_world_cad, cap, poses, a.k, a.exposure)
        top_k, _ = scan(geom, T_world_cad, cap, poses, min(a.k, 3), a.exposure, frames=keys)   # keyframes only
        sk, _ = top_k.sorted()
        np.savez_compressed(cache, score=top.score, frame=top.frame, score_keys=sk,
                            blur=np.array(scan_info["median_blur_px_per_frame"], np.float32))
    score, _ = top.sorted()
    best, kth = score[0], score[min(2, a.k - 1)]
    log.info(f"[{name}] scanned in {time.time() - t0:.0f} s: best resolution {share_at(best)}, 3rd best {share_at(kth)}")
    cv2.imwrite(str(out / f"resolution_best_{chart}.png"), resolution_image(geom, best))
    cv2.imwrite(str(out / f"resolution_3rd_{chart}.png"), resolution_image(geom, kth))
    fused, fmask, fuse_info = fuse_texture(geom, T_world_cad, cap, root / "rgb.mp4", poses, top)
    cv2.imwrite(str(out / f"texture_{chart}.png"), cv2.cvtColor(fused, cv2.COLOR_RGB2BGR))
    info = {"pass": name, "chart": chart, "res_mm": a.res, "k": a.k, "exposure": a.exposure, "frames": n,
            "keyframes": len(keys), "scan": {**scan_info, "median_blur_px_per_frame": None},
            "median_blur_px": round(float(np.median(scan_info["median_blur_px_per_frame"])), 2),
            "best_all_frames": share_at(best), "third_best_all_frames": share_at(kth),
            "best_keyframes_only": share_at(sk[0]),
            "best_median_mm": round(float(np.median(best[np.isfinite(best)])), 3),
            "third_median_mm": round(float(np.median(kth[np.isfinite(kth)])), 3),
            "fuse": fuse_info, "texture_sharpness": round(sharpness(fused, fmask), 2), "seconds": round(time.time() - t0, 1)}
    write_json(out / "allframes.json", info)
    print(json.dumps({k: v for k, v in info.items() if k not in ("scan",)}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
