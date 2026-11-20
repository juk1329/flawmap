"""Rendering helpers shared by the candidates that draw the fitted plate (C0-C3).

``PlateRaster`` casts one ray per pixel at the device model (``surface.raycast``) and returns, per
pixel: which chart and where on it (mm), the 3D point and normal in the device frame, and the unit
vector from the point to the camera. Candidates turn that into colour (texture lookups, blends of
real frames, learned view-dependent terms). Colours are sRGB in [0, 1]; alpha is 1 where the plate is.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..anchor.parametric import CHARTS
from ..surface.raycast import RaycastScene
from .context import View


@dataclass
class PlateHits:
    hit: np.ndarray          # (H, W) bool
    chart: np.ndarray        # (H, W) int chart index (CHARTS), -1 where missed
    uv: np.ndarray           # (H, W, 2) chart coordinates, mm
    point: np.ndarray        # (H, W, 3) device frame, mm
    normal: np.ndarray       # (H, W, 3) outward face normal, device frame
    to_cam: np.ndarray       # (H, W, 3) unit vector point -> camera


class PlateRaster:
    def __init__(self, model):
        self.model = model
        self.scene = RaycastScene(model.mesh, None)

    def hits(self, view: View) -> PlateHits:
        r = self.scene.cast(view.cam, np.asarray(view.T_cad_cam, float))
        hit = r.face >= 0
        f = np.where(hit, r.face, 0)
        ch, uv = self.model.surface_coords(f, r.bary)
        ch = np.where(hit, ch, -1)
        n = self.scene.face_normals_obj[f]
        c = np.asarray(view.T_cad_cam)[:3, 3]
        d = c[None, None, :] - r.point
        d = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
        # outward normals: flip any that face away from the camera (the plate is closed, so a visible
        # face's outward normal points to the camera)
        flip = (np.einsum("hwk,hwk->hw", n, d) < 0)[..., None]
        n = np.where(flip, -n, n)
        return PlateHits(hit, ch, uv, r.point, n, d)


def atlas_sampler(atlas: np.ndarray, layout: dict, model):
    """A function (chart index array, uv mm array) -> sRGB float colours from a bundle atlas
    (``export.bundle.atlas_layout``); bilinear."""
    img = atlas.astype(np.float32) / 255.0
    H, W = img.shape[:2]

    def sample(ch: np.ndarray, uv: np.ndarray) -> np.ndarray:
        out = np.zeros(ch.shape + (img.shape[2],), np.float32)
        mx = np.zeros(ch.shape, np.float32)
        my = np.zeros(ch.shape, np.float32)
        for ci, c in enumerate(CHARTS):
            sel = ch == ci
            if not sel.any():
                continue
            x0, y0, res = layout[c]
            mx[sel] = x0 + uv[sel, 0] / res - 0.5
            my[sel] = y0 + uv[sel, 1] / res - 0.5
        rem = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        out[ch >= 0] = rem[ch >= 0]
        return out

    return sample


def chart_grid(model, chart: str, res: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points (device frame, mm) and outward normals of a face chart sampled every ``res`` mm, and
    the grid shape. Face charts are planes: back u = W/2 - x, v = L/2 - y at z_back; screen u = x + W/2,
    v = L/2 - y at z_front (``anchor.parametric``)."""
    w, l = model.chart_size(chart)
    cols, rows = int(np.ceil(w / res)), int(np.ceil(l / res))
    u = (np.arange(cols) + 0.5) * res
    v = (np.arange(rows) + 0.5) * res
    U, V = np.meshgrid(u, v)
    if chart == "back":
        X, Y, Z, n = w / 2 - U, l / 2 - V, np.full_like(U, model.z_back), np.array([0, 0, -1.0])
    elif chart == "screen":
        X, Y, Z, n = U - w / 2, l / 2 - V, np.full_like(U, model.z_front), np.array([0, 0, 1.0])
    else:
        raise ValueError(chart)
    P = np.stack([X, Y, Z], -1)
    return P, np.broadcast_to(n, P.shape).copy(), (rows, cols)


def chart_outline_mask(model, chart: str, res: float, shape: tuple[int, int], margin_mm: float = 0.0) -> np.ndarray:
    """Texels of a face chart grid (``chart_grid``) inside the device outline shrunk by ``margin_mm``."""
    w, l = model.chart_size(chart)
    xy = np.asarray(model.outline_xy, float)
    if chart == "back":
        uv = np.stack([w / 2 - xy[:, 0], l / 2 - xy[:, 1]], 1)
    else:
        uv = np.stack([xy[:, 0] + w / 2, l / 2 - xy[:, 1]], 1)
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round((uv / res - 0.5) * 4).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=2)
    if margin_mm > 0:
        r = max(1, int(round(margin_mm / res)))
        m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return m > 0
