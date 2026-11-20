"""What every reconstruction candidate provides (``recon.registry``)."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from ..context import CaptureContext, View


class Candidate:
    id = "c?"
    label_ko = ""                  # shown on the comparison page (hidden in blind mode)
    family = ""                    # "plate" (fitted plate geometry) | "free" (geometry learned from the capture)
    renderer = "mesh-standard"     # which renderer the comparison page uses (web/recon/*.js)
    research: list[str] = []       # the methods it follows, short citations

    def __init__(self, ctx: CaptureContext, out: Path | None = None):
        self.ctx = ctx
        self.out = Path(out) if out is not None else ctx.out / self.id
        self.art = self.out / "artifact"

    # ------------------------------------------------------------------ to implement
    def fit(self) -> dict:
        """Build the model from the training frames of ``ctx`` into ``self.art``; return stats."""
        raise NotImplementedError

    def load(self) -> None:
        """Load the artifact for rendering."""

    def render(self, view: View) -> tuple[np.ndarray, np.ndarray]:
        """sRGB float (H, W, 3) in [0, 1] and alpha (H, W) for ``view``."""
        raise NotImplementedError

    def export_web(self, web: Path) -> dict:
        """Files for the comparison page in ``web`` and the manifest (also written to ``web/manifest.json``)."""
        raise NotImplementedError

    # ------------------------------------------------------------------ shared
    def fitted(self) -> bool:
        return (self.art / "fit.json").exists()

    def run_fit(self) -> dict:
        if self.art.exists():
            shutil.rmtree(self.art)
        self.art.mkdir(parents=True)
        t0 = time.time()
        stats = self.fit() or {}
        stats["seconds"] = round(time.time() - t0, 1)
        stats["artifact_mb"] = round(sum(p.stat().st_size for p in self.art.rglob("*") if p.is_file()) / 1e6, 2)
        (self.art / "fit.json").write_text(json.dumps(stats))
        return stats

    def fit_stats(self) -> dict:
        p = self.art / "fit.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def write_manifest(self, web: Path, manifest: dict) -> dict:
        web.mkdir(parents=True, exist_ok=True)
        size = sum(p.stat().st_size for p in web.rglob("*") if p.is_file() and p.name != "manifest.json")
        manifest = {"id": self.id, "label": self.label_ko, "family": self.family, "renderer": self.renderer,
                    "research": self.research, "web_mb": round(size / 1e6, 2), **manifest}
        (web / "manifest.json").write_text(json.dumps(manifest))
        return manifest
