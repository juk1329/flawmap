"""C7, C1 and C2 together: the measured reflection everywhere, the real frames' own detail near the
directions they were taken from.

C1 explains a pixel of the face as surface colour plus a measured environment map reflected by the
face; C2 shows the real frames blended by direction. C2 is the most photographic where the eye is
near a captured ray but falls back to the flat inspection texture beyond, and the views it blends
carry each one's own reflection, so a change of view shows as a line where the blend changes
(cc53da7c23, held-out frame 165, the bottom quarter, round 3). C1 turns smoothly everywhere but its
64 x 32 map cannot hold the sharp detail of the reflections and the print.

C7 keeps C1's model as the base and blends only what C1 did not explain: each stored view holds
the residual of its frame against C1's prediction from that frame's own camera,

    r_k(u) = y_k(u) / g_k - C1(u, from the camera of k),      g_k: the frame's exposure against C1 (median),

and the render is C1 from the eye plus the residuals blended with C2's weights (the 4 views with
the smallest angle to the eye's ray among 12: 4 kept for coverage for the whole recording and the
nearest in direction, NLR-RAS weights, feathered alphas), faded out by the blend's weighted angle (chosen on the validation frames):

    out(u) = C1(u, eye) + (1 - fade) cover sum_k w_k r_k / sum_k w_k.

Near a captured ray this is the real frame (C1 cancels), past the fade it is C1 alone, and in
between the blend moves residuals, which are smaller than the frames, so a change of views shows
less.

How much of C1's predicted change of view to trust, beta, is chosen on the validation frames with
the fade (``BETAS``): with C1 seen from each blended view, C1_k,

    out(u) = C1(u, eye) + share (r_blend - (1 - beta) (C1(u, eye) - C1_blend)),

beta 1 is the formula above, beta 0 blends the frames themselves (r_k + C1_k = y_k / g_k) with C1 as
the fallback past the fade, that is C2 falling back to C1 instead of the flat texture. On 6c417d24a0
C1's change of view was right and C7 beat C1 and C2 (24.26 dB / LPIPS 0.147); on cc53da7c23 the
residuals against C1 were 3.6 times larger (median 0.029 against 0.008), C1's change of view was
wrong where the frames' own was right, and with beta 1 C7 came out below both (22.14 / 0.175 against
C1 22.58 / 0.159 and C2 22.59 / 0.157; 2026-11-19). This is the decomposition of surface light fields (Wood et al. 2000: a reflection model plus a
per-view remainder; Chen et al. 2002) with image-based rendering for the remainder (Debevec et al.
1996/1998, Buehler et al. 2001). C1 is fitted first when this recording has none, and its files travel
with C7 (``artifact/state.npz`` ...), so C7 stands on its own. Stored as WebP: 0.5 + 0.5 r, alpha
as C2.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .c2_ulr import (BETA_NO_VAL, FADE, FADE_NO_VAL, JUMP_LIMIT, MIN_VIEW_SEP_DEG, N_VIEWS, RES, C2ULR, feather_mm, FaceGrid, feathered_alpha, pick_frames, rectified_frames,
                     choose_sep, stored_views_choice, turn_jump, views_for_sep, write_view)
from .plate_vd import C1SRM

GAIN_RANGE = (0.7, 1.4)
FADES = ((12.0, 25.0), (20.0, 35.0), (30.0, 50.0))
BETAS = (1.0, 0.5, 0.0)


class C7Hybrid(C2ULR):
    id = "c7"
    label_ko = "잰 반사 + 실제 프레임 잔차 (C1 위에 C2를 섞음)"
    family = "plate"
    renderer = "mesh-hybrid"
    research = ["표면 광장: 반사 모형 + 시점별 나머지 (Wood 2000)", "ULR (Buehler 2001)", "Bag of Chips 반사 지도 (CVPR 2020, 해석 부분)",
                "당겨 밀어 채우기 (Gortler 1996)"]

    # ------------------------------------------------------------------ fit
    def fit(self) -> dict:
        ctx = self.ctx
        c1 = C1SRM(ctx)
        c1_fit = c1.run_fit() if not c1.fitted() else c1.fit_stats()
        c1.load()
        for f in ("state.npz", "state.json", "texture.jpg", "model.glb"):
            shutil.copy2(c1.art / f, self.art / f)
        fg = FaceGrid(ctx, RES)
        pick, n_cands = pick_frames(ctx, fg.n, N_VIEWS)
        sel = fg.inside
        uv, P = fg.uv[sel], fg.P[sel]
        meta = []
        for i, rect, valid, shift in rectified_frames(ctx, fg, pick):
            c = ctx.T_cad_cam[i][:3, 3]
            d = c[None, :] - P
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            w = 2 * (d @ fg.n)[:, None] * fg.n[None, :] - d
            pred = np.zeros(fg.shape + (3,), np.float32)
            pred[sel] = c1.app.render_face(uv, d.astype(np.float32), w.astype(np.float32))
            g = np.clip(np.median(rect[valid] / np.maximum(pred[valid], 1e-3), axis=0), *GAIN_RANGE)
            res = rect / g - pred
            m = {"frame": int(i), "center_mm": c.tolist(), "gain": [round(float(x), 4) for x in g], "shift_mm": shift,
                 "valid": round(float(valid.sum() / fg.inside.sum()), 3), "residual_mad": round(float(np.median(np.abs(res[valid]))), 4),
                 "file": f"views/{len(meta):03d}.webp"}
            write_view(self.art / m["file"], 0.5 + 0.5 * res, feathered_alpha(valid, fg.inside, feather_mm=feather_mm(ctx.model)))
            meta.append(m)
        info = {"chart": ctx.face, "res_mm": RES, "shape": list(fg.shape), "views": meta, "fade_deg": list(FADE), "feather_mm": round(feather_mm(ctx.model), 2),
                "residual": {"offset": 0.5, "scale": 0.5, "base": "c1"}}
        info["active_views"], info["cover_views"] = stored_views_choice(self.art, meta)
        (self.art / "views.json").write_text(json.dumps(info))
        tune, best = self.tune(info)
        mad = [m["residual_mad"] for m in meta]
        return {"views": len(meta), "picked": len(pick), "candidates": n_cands, "val_psnr_by_fade_beta": tune, "fade_deg": list(best[0]),
                "beta": best[1], "turn_jump_every_view": info["turn_jump_every_view"], "min_view_sep_deg": info["min_view_sep_deg"],
                "active_views": len(info["active_views"]),
                "cover_views": info["cover_views"],
                "residual_mad_p50": round(float(np.median(mad)), 4), "c1_seconds": c1_fit.get("seconds"),
                "ecc_shift_mm_p50": float(np.median([m["shift_mm"] for m in meta if m["shift_mm"] is not None] or [np.nan]))}

    def tune(self, info: dict):
        """The fade and beta with the best validation PSNR (gain fitted) with every stored view
        bindable; and, only if the turntable then jumps, whether to bind views at least
        MIN_VIEW_SEP_DEG apart instead (``C2ULR.tune_fade``, ``choose_sep``). Written to views.json."""
        from ..metrics import heldout_scores

        self.load()
        quality = [m.get("valid", 1.0) for m in info["views"]]
        tune, best = {}, {}

        def grid(sep):
            self.active, self.fixed = views_for_sep(self.vdirs, self.cover, quality, sep)
            for fd in (FADES if self.ctx.val_frames else ()):
                for b in BETAS:
                    self.fade, self.beta = fd, b
                    v = heldout_scores(self.ctx, self.render, frames=self.ctx.val_frames).get("psnr_g")
                    tune[f"{sep:g} deg apart, {fd[0]:g}-{fd[1]:g} deg, beta {b:g}"] = v
                    if v is not None and v > best.get(sep, (None, -1.0))[1]:
                        best[sep] = ((fd, b), v)
            if sep not in best:                  # too short for validation blocks: the middle settings
                best[sep] = ((FADE_NO_VAL, BETA_NO_VAL), None)
            self.fade, self.beta = best[sep][0]

        grid(0.0)
        jump = turn_jump(self)
        sep = 0.0
        if jump > JUMP_LIMIT:
            grid(MIN_VIEW_SEP_DEG)
            # without validation scores the smoother set is kept (the jump is what the eye sees)
            sep = choose_sep({k: v for k, (_, v) in best.items()}) if self.ctx.val_frames else MIN_VIEW_SEP_DEG
        self.active, self.fixed = views_for_sep(self.vdirs, self.cover, quality, sep)
        self.fade, self.beta = best[sep][0]
        info.update({"turn_jump_every_view": jump, "min_view_sep_deg": sep, "active_views": self.active, "cover_views": self.fixed,
                     "fade_deg": list(self.fade), "beta": self.beta})
        (self.art / "views.json").write_text(json.dumps(info))
        return tune, best[sep][0]

    # ------------------------------------------------------------------ render
    def load(self) -> None:
        super().load()
        self.c1 = C1SRM(self.ctx, out=self.out)
        self.c1.load()
        off = self.meta.get("residual", {})
        self.r_off, self.r_scale = off.get("offset", 0.5), off.get("scale", 0.5)
        self.beta = float(self.meta.get("beta", 1.0))

    def render(self, view):
        h = self.raster.hits(view)
        rgb = self.sample(h.chart, h.uv)
        m = h.chart == self.chart_i
        if m.any():
            d = h.to_cam[m]
            n = h.normal[m]
            uv = h.uv[m].astype(np.float32)
            w = 2 * (d * n).sum(1, keepdims=True) * n - d
            base = self.c1.app.render_face(uv, d.astype(np.float32), w.astype(np.float32))
            if self.beta < 1.0:
                def c1_from(k, dk):
                    wk = 2 * (dk * n).sum(1, keepdims=True) * n - dk
                    return self.c1.app.render_face(uv, dk.astype(np.float32), wk.astype(np.float32))

                ulr, share, c1v = self.blend_views(h.point[m], d, uv, view.center, c1_from)
                corr = (1.0 - self.beta) * (base - c1v)
            else:
                ulr, share = self.blend_views(h.point[m], d, uv, view.center)
                corr = 0.0
            rgb[m] = np.clip(base + share[:, None] * ((ulr - self.r_off) / self.r_scale - corr), 0, 1)
        return rgb, h.hit.astype(np.float32)

    # ------------------------------------------------------------------ web
    def export_web(self, web: Path) -> dict:
        terms = C1SRM(self.ctx, out=self.out).web_terms(web)
        terms["ulr"] = self.ulr_terms(web)
        terms["ulr"]["beta"] = float(json.loads((self.art / "views.json").read_text()).get("beta", 1.0))
        return self.write_manifest(web, terms)
