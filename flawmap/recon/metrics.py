"""How the candidates are judged, the same way for all.

Fidelity on held-out frames (``heldout_scores``): the candidate's render at a test frame's camera
against that real frame, inside the plate's silhouette shrunk by 3 px, saturated pixels left out.
Before scoring, the render is aligned to the frame by a 2D shift of at most ``max_shift_px`` (ECC on
gradient magnitude): every candidate shares the same cameras, whose in-plane error is 1-2 mm, so
the same small alignment is applied to all. Scores are given with one fitted RGB gain per frame
(exposure; ``*_g``) and without. LPIPS (AlexNet, Zhang et al. 2018) is computed on the masked
bounding box with the real frame outside the mask.

Robustness on stress views (``stress_scores``), no ground truth: the fitted plate says where the
device is, not what it looks like:
- ``float``: share of the rendered opacity outside the silhouette grown by 1 mm-ish (6 px)
- ``hole``: share of the silhouette shrunk by 4 px that renders with alpha < 0.5 (filmed-face views),
  and the same on the unfilmed side (``hole_under``)
- ``flicker``: mean absolute change between consecutive 1-degree turntable frames inside the
  silhouette (0-1 scale), and its 95th percentile over the turn
- ``jump`` (in ``flicker``): the largest 1-degree change of the turn over the median one, the same
  for the 99th percentile (``jump_p99``), and where the turn jumps (``jump_deg``: the start of the
  steps over twice the median, largest first; every step is kept in ``steps``). A smooth model changes a little every degree; a
  swap of blended views or a blotch shows as a spike. The user's blind ratings of round 4 put the
  candidates that jumped this way below the smooth ones although the held-out frames scored them
  higher (2026-11-19)
- ``blowup``: share of silhouette pixels whose colour leaves the range of the training frames'
  device pixels (0.5-99.5 percentile per channel, +-0.05)
"""
from __future__ import annotations

import math

import cv2
import numpy as np

_LPIPS = None


def _lpips():
    global _LPIPS
    if _LPIPS is None:
        try:
            import lpips
            import torch

            m = lpips.LPIPS(net="alex", verbose=False)
            _LPIPS = (m.cuda() if torch.cuda.is_available() else m).eval()
        except Exception:      # pragma: no cover - LPIPS is optional
            _LPIPS = False
    return _LPIPS


def lpips_score(a: np.ndarray, b: np.ndarray) -> float | None:
    m = _lpips()
    if not m:
        return None
    import torch

    dev = next(m.parameters()).device
    ta = torch.from_numpy(a.transpose(2, 0, 1)[None]).float().to(dev) * 2 - 1
    tb = torch.from_numpy(b.transpose(2, 0, 1)[None]).float().to(dev) * 2 - 1
    with torch.no_grad():
        return float(m(ta, tb).item())


def _grad(img: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(img.mean(-1).astype(np.float32), (0, 0), 1.0)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return np.sqrt(gx * gx + gy * gy)


def align_shift(render: np.ndarray, real: np.ndarray, mask: np.ndarray, max_shift_px: float = 6.0) -> tuple[np.ndarray, tuple[float, float]]:
    """Shift ``render`` (a 2D translation, ECC on gradient magnitude) to match ``real`` inside ``mask``."""
    a, b = _grad(real), _grad(render)
    W = np.eye(2, 3, dtype=np.float32)
    try:
        _, W = cv2.findTransformECC(a, b, W, cv2.MOTION_TRANSLATION,
                                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5), mask.astype(np.uint8), 5)
    except cv2.error:
        return render, (0.0, 0.0)
    dx, dy = float(W[0, 2]), float(W[1, 2])
    if math.hypot(dx, dy) > max_shift_px:
        return render, (0.0, 0.0)
    out = cv2.warpAffine(render, W, (render.shape[1], render.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_REPLICATE)
    return out, (dx, dy)


def psnr(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    e = ((a - b) ** 2)[m].mean()
    return float(10 * np.log10(1.0 / max(e, 1e-10)))


def ssim_masked(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    ga, gb = a[y0:y1, x0:x1].mean(-1), b[y0:y1, x0:x1].mean(-1)
    _, S = structural_similarity(ga, gb, data_range=1.0, full=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    return float(S[m[y0:y1, x0:x1]].mean())


def hf_ncc(a: np.ndarray, b: np.ndarray, m: np.ndarray, s1: float = 1.0, s2: float = 4.0) -> float:
    """Normalised correlation of the band-passed (difference of Gaussians, 1-4 px) grey images inside
    ``m``: does the render reproduce the detail (print, logo, edges, sharp reflections)? A uniform or
    blurred render scores near 0 even when its mean colour, and so its PSNR, is right (the iPad back is
    nearly uniform: an almost empty splat model got a fitted-gain PSNR of 20.5 dB, 2026-11-18)."""
    ga, gb = a.mean(-1).astype(np.float32), b.mean(-1).astype(np.float32)
    da = cv2.GaussianBlur(ga, (0, 0), s1) - cv2.GaussianBlur(ga, (0, 0), s2)
    db = cv2.GaussianBlur(gb, (0, 0), s1) - cv2.GaussianBlur(gb, (0, 0), s2)
    x, y = da[m] - da[m].mean(), db[m] - db[m].mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den > 1e-12 else 0.0


def fit_gain(render: np.ndarray, real: np.ndarray, m: np.ndarray) -> np.ndarray:
    r, t = render[m], real[m]
    g = (r * t).sum(0) / np.maximum((r * r).sum(0), 1e-9)
    return np.clip(g, 0.25, 4.0).astype(np.float32)


def heldout_scores(ctx, render_fn, frames: list[int] | None = None, save_dir=None, max_shift_px: float = 6.0) -> dict:
    """Scores of ``render_fn(view) -> (rgb (H,W,3) float [0,1] sRGB, alpha (H,W))`` on the test frames."""
    rows = []
    frames = frames if frames is not None else ctx.test_frames
    for i in frames:
        view = ctx.view(i)
        real = ctx.image(i).astype(np.float32) / 255.0
        rgb, alpha = render_fn(view)
        # a candidate that leaves holes in the device is scored on them (grey), not excused from them
        rgb = rgb * alpha[..., None] + (1.0 - alpha[..., None]) * 0.5
        sil = ctx.silhouette(view, -3.0)
        m = sil & ~ctx.saturated((real * 255).astype(np.uint8))
        if m.sum() < 500:
            continue
        rgb_a, (dx, dy) = align_shift(rgb, real, m, max_shift_px)
        rec = {"frame": int(i), "shift_px": [round(dx, 2), round(dy, 2)], "coverage": float((alpha > 0.5)[sil].mean())}
        rec["psnr"], rec["ssim"] = psnr(rgb_a, real, m), ssim_masked(rgb_a, real, m)
        g = fit_gain(rgb_a, real, m)
        rg = np.clip(rgb_a * g, 0, 1)
        rec["psnr_g"], rec["ssim_g"] = psnr(rg, real, m), ssim_masked(rg, real, m)
        rec["hf_ncc"] = hf_ncc(rgb_a, real, cv2.erode(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        comp = np.where(m[..., None], rg, real)[y0:y1, x0:x1]
        lp = lpips_score(comp, real[y0:y1, x0:x1])
        if lp is not None:
            rec["lpips_g"] = lp
        rows.append(rec)
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            pair = np.concatenate([real, rg], 1)
            cv2.imwrite(str(save_dir / f"{i:05d}.jpg"), cv2.cvtColor((np.clip(pair, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not rows:
        return {"n": 0}
    agg = {"n": len(rows)}
    for k in ("psnr", "ssim", "psnr_g", "ssim_g", "lpips_g", "hf_ncc", "coverage"):
        v = [r[k] for r in rows if k in r]
        if v:
            agg[k] = round(float(np.mean(v)), 4)
    agg["per_frame"] = rows
    return agg


def train_colour_range(ctx, frames: list[int]) -> tuple[np.ndarray, np.ndarray]:
    vals = []
    for i in frames:
        img = ctx.image(i).astype(np.float32) / 255.0
        m = ctx.silhouette(ctx.view(i), -3.0)
        vals.append(img[m][:: 17])
    v = np.concatenate(vals)
    return np.percentile(v, 0.5, axis=0) - 0.05, np.percentile(v, 99.5, axis=0) + 0.05


def stress_scores(ctx, render_fn, views, colour_range, save_dir=None, video_path=None) -> dict:
    lo, hi = colour_range
    by: dict[str, list] = {}
    prev = None
    flick, flick_az = [], []
    frames_turn = []
    for v in views:
        rgb, alpha = render_fn(v)
        sil = ctx.silhouette(v, 0.0)
        out_zone = ~ctx.silhouette(v, 6.0)
        inner = ctx.silhouette(v, -4.0)
        group = v.name.split()[0]
        rec = {"float": float(alpha[out_zone].sum() / max(alpha.sum(), 1e-6))}
        if inner.sum() > 200:
            rec["hole"] = float((alpha[inner] < 0.5).mean())
            px = rgb[inner & (alpha > 0.5)]
            rec["blowup"] = float(((px < lo) | (px > hi)).any(-1).mean()) if len(px) else 0.0
        by.setdefault(group, []).append(rec)
        if group == "turn30":
            if prev is not None:
                m = sil & prev[1]
                if m.sum() > 200:
                    flick.append(float(np.abs(rgb - prev[0])[m].mean()))
                    flick_az.append(prev[2])
            prev = (rgb, sil, _azimuth(v.name))
            if video_path is not None:
                frames_turn.append((np.clip(rgb * alpha[..., None] + (1 - alpha[..., None]) * 0.94, 0, 1) * 255).astype(np.uint8))
        if save_dir is not None and group != "turn30":
            save_dir.mkdir(parents=True, exist_ok=True)
            img = np.clip(rgb * alpha[..., None] + (1 - alpha[..., None]) * 0.94, 0, 1)
            cv2.imwrite(str(save_dir / (v.name.replace(" ", "_").replace("#", "d") + ".jpg")),
                        cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    agg = {}
    for g, recs in by.items():
        agg[g] = {k: round(float(np.mean([r[k] for r in recs if k in r])), 4) for k in ("float", "hole", "blowup")
                  if any(k in r for r in recs)}
    if flick:
        agg["flicker"] = {"mean": round(float(np.mean(flick)), 4), "p95": round(float(np.percentile(flick, 95)), 4),
                          **turn_jumps(flick, flick_az), "steps": [round(x, 5) for x in flick], "steps_deg": flick_az}
    if video_path is not None and frames_turn:
        write_video(video_path, frames_turn, fps=30)
    return agg


def turn_jumps(steps, az) -> dict:
    """How far the largest 1-degree steps of a turn stand out from the usual one (``jump``,
    ``jump_p99``: over the median step) and where they are (``jump_deg``: the azimuth each step
    starts from, steps over twice the median, the 5 largest)."""
    s = np.asarray(steps, float)
    if len(s) < 3:
        return {}
    med = float(np.median(s))
    if med <= 0:
        return {}
    top = np.argsort(-s, kind="stable")
    return {"jump": round(float(s.max() / med), 2), "jump_p99": round(float(np.percentile(s, 99) / med), 2),
            "jump_deg": [int(round(float(az[k]))) for k in top[:5] if s[k] > 2 * med]}


def _azimuth(name: str) -> float:
    """The azimuth in a stress view's name (``turn30 el30 az77`` -> 77)."""
    for part in name.split():
        if part.startswith("az"):
            try:
                return float(part[2:])
            except ValueError:
                break
    return float("nan")


def write_video(path, frames: list[np.ndarray], fps: int = 30) -> None:
    """H.264 MP4 through the bundled ffmpeg (browser-playable)."""
    import subprocess

    import imageio_ffmpeg

    h, w = frames[0].shape[:2]
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    p.wait()
