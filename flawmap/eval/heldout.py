"""Held-out frames for judging a model against real frames it never saw.

Frames next to each other in a 60 fps video are nearly the same picture (and HEVC codes them from
each other), so holding out every 20th frame measures little more than interpolation: on the iPad
back the measured sheen gained 6.8 dB that way but only 1.9 dB on held-out 1-second blocks
(2026-11-18). The split is therefore by time blocks, with guard bands that no model may train on:

    block k = floor((t - t0) / block_s);  k % period == test_at -> test, == val_at -> validation
    frames within ``guard_s`` of a test or validation block -> guard (used by nothing)
    the first and last ``skip_s`` seconds are left in training (shaky starts and stops)

Everything that learns from frames (keyframe choice, texture slot owners, every reconstruction
candidate) must skip ``excluded`` = test + validation + guard.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Split:
    train: list[int]
    val: list[int]
    test: list[int]
    guard: list[int]
    params: dict = field(default_factory=dict)

    @property
    def excluded(self) -> frozenset:
        return frozenset(self.val) | frozenset(self.test) | frozenset(self.guard)

    def to_json(self) -> dict:
        return {"params": self.params, "train": len(self.train), "val": self.val, "test": self.test, "guard": self.guard}

    @classmethod
    def from_json(cls, d: dict, n_frames: int) -> "Split":
        ex = set(d["val"]) | set(d["test"]) | set(d["guard"])
        return cls([i for i in range(n_frames) if i not in ex], list(d["val"]), list(d["test"]), list(d["guard"]), d.get("params", {}))


def block_split(timestamps: np.ndarray, block_s: float = 1.0, period: int = 6, test_at: int = 2, val_at: int = 5,
                guard_s: float = 0.25, skip_s: float = 1.0) -> Split:
    """Time-block split of one recording (``timestamps`` in seconds, one per frame)."""
    t = np.asarray(timestamps, np.float64)
    t = t - t[0]
    n = len(t)
    k = np.floor(t / block_s).astype(int)
    role = np.full(n, "train", object)
    inner = (t >= skip_s) & (t <= t[-1] - skip_s)
    role[inner & (k % period == test_at)] = "test"
    role[inner & (k % period == val_at)] = "val"
    held = np.flatnonzero((role == "test") | (role == "val"))
    if len(held):
        th = t[held]
        # distance of every frame to the nearest held-out frame
        j = np.searchsorted(th, t)
        d = np.minimum(np.abs(t - th[np.clip(j, 0, len(th) - 1)]), np.abs(t - th[np.clip(j - 1, 0, len(th) - 1)]))
        role[(role == "train") & (d <= guard_s)] = "guard"
    pick = lambda r: [int(i) for i in np.flatnonzero(role == r)]
    return Split(pick("train"), pick("val"), pick("test"), pick("guard"),
                 {"block_s": block_s, "period": period, "test_at": test_at, "val_at": val_at, "guard_s": guard_s, "skip_s": skip_s})
