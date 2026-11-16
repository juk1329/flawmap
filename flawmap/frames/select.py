"""Keyframe selection: the sharpest, well-exposed frame in each time window.

Defects of 1 mm span a few pixels, so motion blur matters more than frame count. Sharpness is the
variance of the Laplacian on a half-resolution grey image (blur halves it quickly); exposure is the
share of blown-out pixels (dark pixels are not counted: on 2026-11-15 counting them dropped every
low synthetic view, whose background above the horizon was black).
"""
from __future__ import annotations

import cv2
import numpy as np


def frame_quality(rgb: np.ndarray) -> dict:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    g2 = cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2), interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(g2, cv2.CV_32F)
    # only blown highlights count as bad exposure: dark areas are normal (a black phone, a dark room)
    return {"sharpness": float(lap.var()), "clipped": float((g >= 250).mean()), "mean": float(g.mean())}


def select_keyframes(quality: dict[int, dict], timestamps: np.ndarray, target: int = 160, window_s: float | None = None,
                     max_clipped: float = 0.25) -> list[int]:
    """One frame per time window (the sharpest that is not badly exposed)."""
    idx = np.array(sorted(quality))
    if len(idx) <= target:
        return [int(i) for i in idx if quality[i]["clipped"] <= max_clipped]
    t = timestamps[idx]
    window_s = window_s or (t[-1] - t[0]) / target
    bins = np.floor((t - t[0]) / max(window_s, 1e-6)).astype(int)
    keep = []
    for b in np.unique(bins):
        cand = idx[bins == b]
        cand = [i for i in cand if quality[i]["clipped"] <= max_clipped] or list(cand)
        keep.append(int(max(cand, key=lambda i: quality[i]["sharpness"])))
    return keep
