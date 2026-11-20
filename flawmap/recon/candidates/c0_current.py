"""C0, today's model: the fitted plate with the inspection run's texture (the lowest well-supported
lightness per texel over 24 view directions, shading-flattened), lit in the viewer by a synthetic
room (three.js RoomEnvironment) and a roughness map. The baseline every other candidate is judged
against."""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from ...export.bundle import atlas_layout
from ..render import PlateRaster, atlas_sampler
from .base import Candidate


class C0Current(Candidate):
    id = "c0"
    label_ko = "지금 방식 (맞춘 판 + 검사용 텍스처)"
    family = "plate"
    renderer = "mesh-standard"
    research = ["지금 파이프라인: 24방향 칸 + 지지받는 하한 (Waechter 2014 계열)"]

    def fit(self) -> dict:
        b = self.ctx.base_run / "bundle"
        for f in ("texture.jpg", "model.glb", "roughness.png"):
            if (b / f).exists():
                shutil.copy2(b / f, self.art / f)
        return {"source": str(b)}

    def load(self) -> None:
        tex = cv2.cvtColor(cv2.imread(str(self.art / "texture.jpg")), cv2.COLOR_BGR2RGB)
        self.sample = atlas_sampler(tex, atlas_layout(self.ctx.model), self.ctx.model)
        self.raster = PlateRaster(self.ctx.model)

    def render(self, view):
        h = self.raster.hits(view)
        rgb = self.sample(h.chart, h.uv)
        return rgb, h.hit.astype(np.float32)

    def export_web(self, web: Path) -> dict:
        web.mkdir(parents=True, exist_ok=True)
        files = {}
        for f in ("texture.jpg", "model.glb", "roughness.png"):
            if (self.art / f).exists():
                shutil.copy2(self.art / f, web / f)
                files[f.split(".")[0]] = f
        return self.write_manifest(web, {"files": files})
