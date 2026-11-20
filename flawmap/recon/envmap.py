"""The measured environment map shared by the reflection candidates (C1, C4b, C7).

One recording sees the room's reflection in the device's face only in the directions the camera
went through: a flat device filmed from above reflects the ceiling and the photographer, a device
leaning on a cabinet filmed along a 90-degree arc reflects half of the room in front of it. The
map is learned only there. Round 3 left the other directions at the dark starting value, so turning
the model past the filmed arc showed a sharp bright-to-dark edge on the face (6c417d24a0, 60 degrees
from the filmed arc, 2026-11-18).

- ``support``: how well each texel was measured: the number of distinct training frames whose
  face pixels looked it up (one frame's pixels are strongly correlated, so frames are counted,
  not pixels);
- ``pull_push``: the unmeasured texels filled from the measured ones with the pull-push algorithm
  of the Lumigraph (Gortler et al. 1996; Grossman & Dally 1998): a pyramid of confidence-weighted
  averages (solid angle weighted on the sphere, wrapping in azimuth), pushed back down so that a
  texel next to measured ones takes their value and one far from them a broad average of the room.
  Partly measured texels keep their share (confidence 0-1);
- ``proxy_dirs``: the lookup direction for a finite room (a parallax-corrected environment map,
  as local reflection probes do in real-time rendering; Lagarde & Zanuttini 2012): the mirror ray
  from the face point ``p`` meets a sphere of radius R about the device centre at ``q``, and the map
  is read in the direction of ``q``. R -> infinity is the distant-environment map. The room is a
  few metres away while the device is 28 cm long, so a distant map reads the same ceiling lamp up
  to 8 degrees apart across the face.

- ``pole_soften``: near the poles (the faces' normals) the map is blurred on the sphere. There all
  the columns meet; filmed from above, the face reflects the photographer there, in a different
  place in every frame, so the measured columns disagree, and a view straight at the face (whose
  mirror direction is the pole) showed them as a pinwheel of wedges meeting in a point (the iPad's
  back in the service viewer, 2026-11-19). Averaging each row instead (``pole_blend``) turned a lamp
  measured near the pole into a ring.

Equirectangular in the device frame: u = atan2(y, x) / 2 pi + 0.5, v = acos(z) / pi.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def texel_of(w: np.ndarray, H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    """Row and column of the texel holding unit directions ``w`` (..., 3)."""
    u = np.arctan2(w[..., 1], w[..., 0]) / (2 * math.pi) + 0.5
    v = np.arccos(np.clip(w[..., 2], -1.0, 1.0)) / math.pi
    c = np.clip((u * W).astype(np.int64), 0, W - 1)
    r = np.clip((v * H).astype(np.int64), 0, H - 1)
    return r, c


def support(w: np.ndarray, fid: np.ndarray, H: int, W: int, min_px: int = 20) -> np.ndarray:
    """Distinct frames (of ids ``fid``) with at least ``min_px`` lookups in each texel (H, W)."""
    r, c = texel_of(np.asarray(w, np.float32), H, W)
    key = (np.asarray(fid, np.int64) * H + r) * W + c
    k, n = np.unique(key, return_counts=True)
    k = k[n >= min_px]
    cell = k % (H * W)
    out = np.zeros(H * W, np.int64)
    np.add.at(out, cell, 1)
    return out.reshape(H, W)


def confidence(n_frames: np.ndarray, full: int = 5) -> np.ndarray:
    """0 for texels no frame looked up, 1 from ``full`` frames on, linear between."""
    return np.clip((n_frames.astype(np.float32)) / float(full), 0.0, 1.0)


def _down(x: np.ndarray) -> np.ndarray:
    """2 x 2 block sums (rows and columns even)."""
    return x[0::2, 0::2] + x[1::2, 0::2] + x[0::2, 1::2] + x[1::2, 1::2]


def _up(x: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bilinear upsampling by 2 that wraps in azimuth (columns) and clamps at the poles (rows)."""
    pad = np.concatenate([x[:, -1:], x, x[:, :1]], axis=1)
    big = cv2.resize(pad, (pad.shape[1] * 2, pad.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
    big = big[:, 2:-2]
    if big.ndim == 2 and len(shape) == 3:
        big = big[..., None]
    return big.reshape(shape[0], shape[1], *x.shape[2:])


def pull_push(L: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """``L`` (H, W, C) with confidence ``conf`` (H, W) in [0, 1], filled where conf < 1 (see above).
    H and W must be powers of two with W = 2 H."""
    H, W = conf.shape
    L = np.asarray(L, np.float32)
    theta = (np.arange(H) + 0.5) / H * math.pi
    area = np.broadcast_to(np.sin(theta)[:, None], (H, W)).astype(np.float32)
    levels = [(L, conf.astype(np.float32), area)]
    while levels[-1][1].shape[0] > 1:
        v, c, a = levels[-1]
        wa = c * a
        num = _down(v * wa[..., None])
        den = _down(wa)
        a2 = _down(a)
        v2 = num / np.maximum(den, 1e-12)[..., None]
        # a parent is as sure as its measured area (Gortler's clamped sum, area weighted)
        c2 = np.clip(4.0 * den / np.maximum(a2, 1e-12), 0.0, 1.0)
        levels.append((v2, c2, a2 / 4.0))
    filled = levels[-1][0]
    for v, c, _ in reversed(levels[:-1]):
        up = _up(filled, v.shape)
        filled = c[..., None] * v + (1.0 - c[..., None]) * up
    return filled.astype(np.float32)


def proxy_dirs(p, w, radius_mm):
    """Lookup directions for face points ``p`` (mm, device frame) and mirror directions ``w`` (unit):
    where the mirror ray meets the sphere of ``radius_mm`` about the device centre. numpy or torch;
    ``radius_mm`` None or inf: ``w`` itself."""
    if radius_mm is None:
        return w
    xp = np if isinstance(w, np.ndarray) else __import__("torch")
    if isinstance(radius_mm, float) and not math.isfinite(radius_mm):
        return w
    pw = (p * w).sum(-1, keepdims=True)
    pp = (p * p).sum(-1, keepdims=True)
    t = -pw + xp.sqrt(xp.clip(pw * pw - pp + radius_mm * radius_mm, 1e-6, None))
    q = p + t * w
    return q / xp.linalg.norm(q, axis=-1, keepdims=True) if xp is np else q / q.norm(dim=-1, keepdim=True)


def face_points(uv, chart: str, size_mm: tuple[float, float], z_face: float):
    """Device-frame points (mm) of chart coordinates ``uv`` (mm) on a face chart (``recon.render.chart_grid``):
    back u = W/2 - x, v = L/2 - y; screen u = x + W/2, v = L/2 - y. numpy or torch."""
    w, l = size_mm
    xp = np if isinstance(uv, np.ndarray) else __import__("torch")
    if chart == "back":
        x = w / 2 - uv[..., 0]
    else:
        x = uv[..., 0] - w / 2
    y = l / 2 - uv[..., 1]
    z = xp.full_like(x, z_face) if xp is np else xp.full_like(x, float(z_face))
    return xp.stack([x, y, z], -1)


POLE_DEG = 30.0        # pole_soften: the map is softened within this angle of the poles ...
POLE_SIGMA_DEG = 8.0   # ... by a Gaussian of this many degrees on the sphere


def pole_blend(L: np.ndarray, pole_deg: float = POLE_DEG) -> np.ndarray:
    """``L`` (H, W, C) with each row's variation along the azimuth scaled by smoothstep(d / pole_deg),
    d the row's angle to the nearer pole: the row's mean at the pole, unchanged from ``pole_deg`` on.
    Not used: a lamp measured near the pole (the screen of 87cb206a14) became a ring."""
    L = np.asarray(L, np.float32)
    H = L.shape[0]
    theta = (np.arange(H) + 0.5) / H * 180.0
    k = np.clip(np.minimum(theta, 180.0 - theta) / pole_deg, 0.0, 1.0)
    k = (k * k * (3.0 - 2.0 * k)).astype(np.float32)
    mean = L.mean(axis=1, keepdims=True)
    return (mean + (L - mean) * k[:, None, None]).astype(np.float32)


def pole_soften(L: np.ndarray, pole_deg: float = POLE_DEG, sigma_deg: float = POLE_SIGMA_DEG) -> np.ndarray:
    """``L`` (H, W, C) blurred on the sphere (Gaussian of ``sigma_deg``, solid angle weighted) near the
    poles: fully at a pole, not from ``pole_deg`` on (smoothstep). The same angle everywhere, so the
    columns no longer meet in a point at the pole, and a lamp near it stays a spot."""
    L = np.asarray(L, np.float32)
    H, W = L.shape[:2]
    th = (np.arange(H) + 0.5) / H * math.pi
    ph = ((np.arange(W) + 0.5) / W - 0.5) * 2.0 * math.pi          # u = atan2(y, x) / 2 pi + 0.5
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    D = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], -1).reshape(-1, 3)
    ang = np.arccos(np.clip(D @ D.T, -1.0, 1.0))
    wt = np.exp(-0.5 * (ang / math.radians(sigma_deg)) ** 2) * np.sin(TH).reshape(1, -1)
    blur = (wt @ L.reshape(H * W, -1).astype(np.float64)) / wt.sum(1, keepdims=True)
    d = np.degrees(np.minimum(TH, math.pi - TH)).reshape(-1, 1)
    k = np.clip(d / pole_deg, 0.0, 1.0)
    k = k * k * (3.0 - 2.0 * k)
    return (k * L.reshape(H * W, -1) + (1.0 - k) * blur).reshape(L.shape).astype(np.float32)
