"""C1 and C3: view-dependent appearance fitted on the fitted plate's filmed face (``plate_fit``).

- C3 (``C3SHTexture``): the face colour plus per-texel SH (degree 2, 2 mm) of the view direction,
  fitted to all training pixels: texture-map optimisation with view dependence (TMO; Mesh-Learner).
- C1 (``C1SRM``): the face colour plus s(u) x L(mirror direction): a measured environment map (one
  64 x 32 map, device frame) times a per-texel specular scale (Park et al. 2020's analytic part).

The other charts (the band, the face that was not filmed) keep the inspection texture. Both are
drawn in the browser by a small shader (``web/recon/mesh_shtex.js``, ``mesh_srm.js``) that decodes
the face colour, adds the view-dependent term in linear light and encodes with BT.709, like here.
"""
from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from ...anchor.parametric import CHARTS
from ..envmap import pole_soften
from ...export.bundle import atlas_layout
from ..render import PlateRaster, atlas_sampler
from .base import Candidate
from .plate_fit import ENV_H, ENV_W, SH_RES, FaceSamples, PlateAppearance

STEPS = 4000


class _PlateVD(Candidate):
    family = "plate"
    kind = ""
    # C1: read the environment in a finite room of learned radius (envmap.proxy_dirs)? Off: on 6c417d24a0,
    # cc53da7c23 (C1) and both C4b fits the radius ran to its 50 m bound, the distant map (2026-11-19)
    proxy = False

    def _face_rect(self):
        lay = atlas_layout(self.ctx.model)
        x0, y0, res = lay[self.ctx.face]
        w, l = self.ctx.model.chart_size(self.ctx.face)
        return int(x0), int(y0), int(np.ceil(w / res)), int(np.ceil(l / res)), res

    def fit(self) -> dict:
        ctx = self.ctx
        t0 = time.time()
        atlas = cv2.cvtColor(cv2.imread(str(ctx.base_run / "bundle" / "texture.jpg")), cv2.COLOR_BGR2RGB)
        x0, y0, w, h, res = self._face_rect()
        base_img = atlas[y0:y0 + h, x0:x0 + w].copy()
        app = PlateAppearance(ctx, self.kind, base_img, res, proxy=self.proxy)
        samples = FaceSamples(ctx, ctx.train_frames)
        losses = app.fit(samples, steps=STEPS)
        new = atlas.copy()
        new[y0:y0 + h, x0:x0 + w] = app.base_image()
        cv2.imwrite(str(self.art / "texture.jpg"), cv2.cvtColor(new, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 93])
        st = app.state()
        np.savez_compressed(self.art / "state.npz", **{k: v for k, v in st.items() if isinstance(v, np.ndarray)})
        (self.art / "state.json").write_text(json.dumps({k: v for k, v in st.items() if not isinstance(v, np.ndarray)}))
        shutil.copy2(ctx.base_run / "bundle" / "model.glb", self.art / "model.glb")
        d = app.delta.detach().cpu().numpy()
        return {"pixels": int(len(samples.uv)), "frames": samples.n_frames, "steps": STEPS, "loss": losses,
                "frame_shift_mm_p50": round(float(np.median(np.linalg.norm(3 * np.tanh(d / 3), axis=1))), 3), "train_s": round(time.time() - t0, 1),
                **({"environment": app.env_info} if getattr(app, "env_info", None) else {})}

    def load(self) -> None:
        ctx = self.ctx
        tex = cv2.cvtColor(cv2.imread(str(self.art / "texture.jpg")), cv2.COLOR_BGR2RGB)
        self.sample = atlas_sampler(tex, atlas_layout(ctx.model), ctx.model)
        x0, y0, w, h, res = self._face_rect()
        js = json.loads((self.art / "state.json").read_text())
        R = js.get("room_radius_mm")
        self.app = PlateAppearance(ctx, self.kind, tex[y0:y0 + h, x0:x0 + w], res, proxy=R is not None)
        st = np.load(self.art / "state.npz")
        import torch

        with torch.no_grad():
            if self.kind == "sh":
                self.app.C.copy_(torch.from_numpy(st["C"]).permute(2, 0, 1)[None])
            else:
                S = torch.from_numpy(st["S"])[None, None]
                self.app.S.copy_(torch.log(torch.expm1(S.clamp(min=1e-6))) - 1.0)
                # the stored map is the filled one (round 4); earlier artifacts: as fitted, a distant map.
                # Softened near the poles where it is read (here and in the browser, _export_terms):
                # straight at the face the mirror direction is the pole (envmap.pole_soften)
                self.app.L_filled = torch.from_numpy(pole_soften(st["L"])).float().to(self.app.dev)
                if R is not None:
                    self.app.rho.copy_(torch.tensor(math.log(R)))
        self.raster = PlateRaster(ctx.model)
        self.ci = CHARTS.index(ctx.face)

    def render(self, view):
        h = self.raster.hits(view)
        rgb = self.sample(h.chart, h.uv)
        m = h.chart == self.ci
        if m.any():
            d = h.to_cam[m]
            n = h.normal[m]
            w = 2 * (d * n).sum(1, keepdims=True) * n - d
            rgb[m] = self.app.render_face(h.uv[m].astype(np.float32), d.astype(np.float32), w.astype(np.float32))
        return rgb, h.hit.astype(np.float32)

    def export_web(self, web: Path) -> dict:
        return self.write_manifest(web, self.web_terms(web))

    def web_terms(self, web: Path) -> dict:
        """The mesh, the atlas and the view-dependent term's files in ``web``, and their manifest entries."""
        web.mkdir(parents=True, exist_ok=True)
        for f in ("texture.jpg", "model.glb"):
            shutil.copy2(self.art / f, web / f)
        st = np.load(self.art / "state.npz")
        x0, y0, w, h, res = self._face_rect()
        cw, cl = self.ctx.model.chart_size(self.ctx.face)
        face = {"chart": self.ctx.face, "rect": [x0, y0, w, h], "res_mm": res, "size_mm": [cw, cl], "atlas": [4096, 2048],
                "normal": [0, 0, -1] if self.ctx.face == "back" else [0, 0, 1]}
        extra = self._export_terms(web, st)
        return {"files": {"model": "model.glb", "texture": "texture.jpg"}, "face": face, **extra}


def _f16(path: Path, a: np.ndarray) -> None:
    path.write_bytes(np.ascontiguousarray(a, dtype="<f2").tobytes())


class C3SHTexture(_PlateVD):
    id = "c3"
    kind = "sh"
    label_ko = "판 위 학습형 시점 텍스처 (텍셀별 구면 조화)"
    renderer = "mesh-shtex"
    research = ["TMO 텍스처 최적화 (CVPR 2023)", "Mesh-Learner SH 텍스처 (IROS 2025)", "Zhou·Koltun 2014 (프레임별 보정)"]

    def _export_terms(self, web: Path, st) -> dict:
        C = st["C"]                                        # (gh, gw, 27): coefficient k of channel c at 3k + c
        gh, gw, _ = C.shape
        tiles = np.zeros((gh, gw * 7, 4), np.float32)
        flat = np.concatenate([C, np.zeros((gh, gw, 1), np.float32)], -1)    # 28 = 7 x RGBA
        for j in range(7):
            tiles[:, j * gw:(j + 1) * gw] = flat[..., 4 * j:4 * j + 4]
        _f16(web / "sh.f16", tiles)
        return {"sh": {"file": "sh.f16", "w": gw * 7, "h": gh, "tile_w": gw, "res_mm": SH_RES, "layout": "coef k, channel c at index 3k+c"}}


class C1SRM(_PlateVD):
    id = "c1"
    kind = "srm"
    label_ko = "잰 반사 지도 (맞춘 판 + 같은 녹화에서 잰 방의 반사)"
    renderer = "mesh-srm"
    research = ["Bag of Chips 반사 지도 (CVPR 2020, 해석 부분)", "Szeliski 2000 층 분리", "Zhou·Koltun 2014"]

    def _export_terms(self, web: Path, st) -> dict:
        S = st["S"].astype(np.float32)
        L = pole_soften(st["L"].astype(np.float32))       # as C1SRM.load reads it
        _f16(web / "s.f16", np.dstack([S, S, S, np.ones_like(S)]))
        _f16(web / "env.f16", np.dstack([L, np.ones(L.shape[:2], np.float32)]))
        js = json.loads((self.art / "state.json").read_text())
        return {"srm": {"s": "s.f16", "s_w": S.shape[1], "s_h": S.shape[0], "s_res_mm": SH_RES, "env": "env.f16",
                        "env_w": L.shape[1], "env_h": L.shape[0], "equirect": "u = atan2(y, x)/2pi + 0.5, v = acos(z)/pi (device frame)",
                        "room_radius_mm": js.get("room_radius_mm"),
                        "lookup": "the mirror ray from the face point meets a sphere of room_radius_mm about the device centre; "
                                  "the map is read in that point's direction (null: the mirror direction)"}}
