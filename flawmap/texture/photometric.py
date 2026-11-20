"""Exposure and white balance of every frame, from the room around the device.

Stray Scanner records no exposure. A per-frame gain fitted on the device itself swallows the
low-frequency part of the device's own sheen (the window of mirror directions a frame sees moves
with the gain): on 29d760c369 held-out 1-second blocks dropped from 17.1 to 12.3 dB that way
(2026-11-18). So the gains come from outside the device, as in global colour correction of photo
collections (Moulon et al. 2013) and photometric calibration from video (Goldman 2010; Engel et al.
2016):

1. LiDAR points (confidence 2) of a frame, outside the device box grown by 30 mm, are projected into
   frames k and k+d (d = 1, 2, 4 training steps); points where both frames are well exposed
   (code 20-245) and the local gradient is low give log ratios per channel.
2. Each pair's ratio is a robust centre of those (median of the central 50 %); the pair is weighted
   by its point count.
3. log g is the least-squares solution of the chain of pair ratios, median 0 (a gauge). Floors are
   often glossy (wood), so only short baselines are used.

Values are linear light: 8-bit codes are decoded with the BT.709 transfer (``to_linear``).
"""
from __future__ import annotations

import numpy as np

_LUT = None


def to_linear(img_u8: np.ndarray) -> np.ndarray:
    """8-bit BT.709/sRGB-style code values to linear light in [0, 1] (inverse BT.709 OETF)."""
    global _LUT
    if _LUT is None:
        v = np.arange(256) / 255.0
        _LUT = np.where(v < 0.081, v / 4.5, ((v + 0.099) / 1.099) ** (1 / 0.45)).astype(np.float32)
    return _LUT[img_u8]


def to_code(lin: np.ndarray) -> np.ndarray:
    """Linear light to BT.709 code values in [0, 1] (the forward OETF)."""
    lin = np.clip(lin, 0, None)
    return np.where(lin < 0.018, 4.5 * lin, 1.099 * np.power(lin, 0.45) - 0.099).clip(0, 1).astype(np.float32)


def background_gains(ctx, frames: list[int], steps=(1, 2, 4), min_points: int = 200) -> dict[int, np.ndarray]:
    """Per-frame RGB gains (linear light, median 1) from the background of ``ctx`` (recon context)."""
    import cv2

    s = ctx.model.spec
    box = np.array([s.width / 2, s.length / 2, max(abs(ctx.model.z_front), abs(ctx.model.z_back))]) + 30.0
    lin, grad, views = {}, {}, {}
    for i in frames:
        img = ctx.image(i)
        lin[i] = to_linear(img)
        g = cv2.GaussianBlur(img.mean(-1).astype(np.float32), (0, 0), 1.0)
        grad[i] = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
        views[i] = ctx.view(i)
    rows, rhs, wts = [], [], []
    idx = {f: k for k, f in enumerate(frames)}
    for a_i, a in enumerate(frames):
        d, conf = ctx.depth(a)
        if d is None:
            continue
        cam = ctx.camera(a)
        h, w = d.shape
        sc = w / cam.width
        fx, fy, cx, cy = cam.fx * sc, cam.fy * sc, (cam.cx + 0.5) * sc - 0.5, (cam.cy + 0.5) * sc - 0.5
        v, u = np.nonzero((conf >= 2) & (d > 50))
        v, u = v[::2], u[::2]
        z = d[v, u]
        pc = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], 1)
        T = ctx.T_cad_cam[a]
        P = pc @ T[:3, :3].T + T[:3, 3]
        P = P[~np.all(np.abs(P) < box, axis=1)]
        if len(P) < min_points:
            continue
        sa = _sample(lin[a], grad[a], views[a], P)
        for st in steps:
            if a_i + st >= len(frames):
                continue
            b = frames[a_i + st]
            sb = _sample(lin[b], grad[b], views[b], P)
            ok = sa[1] & sb[1]
            if ok.sum() < min_points:
                continue
            r = np.log(np.maximum(sb[0][ok], 1e-4)) - np.log(np.maximum(sa[0][ok], 1e-4))       # (n, 3)
            lo, hi = np.percentile(r, 25, axis=0), np.percentile(r, 75, axis=0)
            centre = np.array([np.median(r[(r[:, c] >= lo[c]) & (r[:, c] <= hi[c]), c]) for c in range(3)])
            rows.append((idx[a], idx[b]))
            rhs.append(centre)
            wts.append(min(ok.sum(), 2000) / st)
    n = len(frames)
    if not rows:
        return {f: np.ones(3, np.float32) for f in frames}
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import lsqr

    m = len(rows)
    A = coo_matrix((np.concatenate([-np.ones(m), np.ones(m)]),
                    (np.concatenate([np.arange(m), np.arange(m)]), np.concatenate([[r[0] for r in rows], [r[1] for r in rows]]))),
                   shape=(m, n)).tocsr()
    W = np.sqrt(np.asarray(wts))[:, None]
    out = np.zeros((n, 3))
    for c in range(3):
        sol = lsqr(A.multiply(W), (np.asarray(rhs)[:, c:c + 1] * W).ravel(), damp=1e-3)[0]
        out[:, c] = sol - np.median(sol)
    return {f: np.exp(out[k]).astype(np.float32) for k, f in enumerate(frames)}


def _sample(lin: np.ndarray, grad: np.ndarray, view, P: np.ndarray):
    """Linear RGB at the projections of device-frame points and whether each is usable."""
    import cv2

    T = np.linalg.inv(np.asarray(view.T_cad_cam, float))
    pc = P @ T[:3, :3].T + T[:3, 3]
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = view.K[0, 0] * pc[:, 0] / z + view.K[0, 2]
        y = view.K[1, 1] * pc[:, 1] / z + view.K[1, 2]
    H, W = lin.shape[:2]
    inb = (z > 50) & (x > 2) & (y > 2) & (x < W - 3) & (y < H - 3)
    xs, ys = np.where(inb, x, 0).astype(np.float32), np.where(inb, y, 0).astype(np.float32)
    val = cv2.remap(lin, xs[None], ys[None], cv2.INTER_LINEAR)[0] if len(xs) < 32000 else np.concatenate(
        [cv2.remap(lin, xs[None, k:k + 30000], ys[None, k:k + 30000], cv2.INTER_LINEAR)[0] for k in range(0, len(xs), 30000)])
    gr = cv2.remap(grad, xs[None], ys[None], cv2.INTER_LINEAR)[0] if len(xs) < 32000 else np.concatenate(
        [cv2.remap(grad, xs[None, k:k + 30000], ys[None, k:k + 30000], cv2.INTER_LINEAR)[0] for k in range(0, len(xs), 30000)])
    code_ok = (val > to_linear(np.array([20], np.uint8))[0]).all(1) & (val < to_linear(np.array([245], np.uint8))[0]).all(1)
    thr = np.percentile(gr[inb], 40) if inb.any() else 0
    return val, inb & code_ok & (gr <= thr)
