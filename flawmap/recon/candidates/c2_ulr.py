"""C2, real frames blended on the plate: view-dependent texture mapping / unstructured lumigraph
rendering with the fitted plate as the proxy geometry.

Debevec et al. 1996/1998 (VDTM: per view direction, the nearest photographs blended), Buehler et al.
2001 (ULR: weights from the angle between the desired and the captured ray), Kellnhofer et al. 2021
(NLR-RAS weights w_k = (1/theta_k)(1 - theta_k/theta_{K+1})), Eisemann et al. 2008 (floating textures,
for later if blends ghost).

Fit: about 96 training frames spread over view directions (farthest-point sampling over the
camera direction from the device centre, among the sharper 70 %) are rectified onto the filmed face
chart at 0.25 mm, aligned to the inspection texture by ECC (affine, at most 4 mm: the cameras'
in-plane error), and brought to a common exposure (median ratio to the texture). Stored as WebP
with an alpha channel for where the frame saw the face.

Render: like the comparison page's shader, 12 stored views are bound (``select_views``: 4 kept for
coverage, the rest nearest to the eye's direction from the device centre); per pixel they are
blended with weights falling with the angle between the eye's ray and the view's ray to that point
(the unstructured lumigraph's angle penalty; round 4 blended the 4 nearest with NLR-RAS weights).
Past 12-25 degrees from the captured rays the blend fades to the inspection texture, so turning the
model never shows holes; the other charts (the band, the unfilmed face) use the inspection texture.
All of it changes continuously as the eye turns (``blend_views``).

Each view's alpha falls off over FEATHER_MM (20 mm) towards where that frame stopped seeing the face (the
frame's border, too grazing), and the weights and the blend's share follow it: with hard alpha and 8
candidates the blend jumped to the texture along a line where the nearest views' frames ended
(cc53da7c23, held-out frame 165, 2026-11-18).

The 12 candidates are chosen for coverage too: on cc53da7c23 the 12 nearest to held-out frame 165
were all close-ups (frames 620-642, 4.5-8 degrees away) and none saw the bottom quarter of the face.
Up to 4 views are kept for coverage in every bound set (``cover_views``, chosen once per recording),
the rest are the nearest in direction (``select_views``), among views at least 3 degrees apart
(``distinct_views``). Choosing the coverage views again for each
eye direction (round 4) swapped them in steps: on cc53da7c23 at 77 -> 78 degrees of the turntable
the one view that alone covered about 40 % of the face left the set and that part of the face
snapped to the fallback in one degree, the jumps the user saw (2026-11-19). With the coverage views
fixed, the bound set changes only by direction rank, one view at a time. The share of the
inspection texture follows the blend's weighted angle, not the nearest view's, so it changes
smoothly where the views change.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from ...anchor.parametric import CHARTS
from ...export.bundle import atlas_layout
from ..render import PlateRaster, atlas_sampler, chart_grid, chart_outline_mask
from .base import Candidate

RES = 0.25          # mm per texel of the stored views
N_VIEWS = 192
PRESELECT = 12
FEATHER_MM = 20.0    # alpha ramp at the edge of what a view saw on the 13" iPad (6 mm still showed the change of view as a line, cc53 frame 165) ...
FEATHER_FRAC = 20.0 / 215.0   # ... as a share of the device's width, so a phone's ramp is as wide for its size (feather_mm)
FADE_NO_VAL = (20.0, 35.0)    # the settings when a recording is too short for validation blocks (under about 6 s) ...
BETA_NO_VAL = 0.5             # ... the middle of the grids, chosen most often on the recordings so far (C7: beta)
N_COVER = 4          # views kept for coverage in every bound set (cover_views); the rest of PRESELECT by direction
MIN_VIEW_SEP_DEG = 3.0  # when the turntable jumps with every view bound, only views this far apart in direction are bound ...
SEP_TOLERANCE_DB = 0.2  # ... unless that costs more than this on the validation frames (choose_sep)
JUMP_LIMIT = 2.0        # a turntable step over this many times the usual one is a jump (metrics.turn_jumps)
COVER_GRID = (12, 9)  # rows x cols over the face for the coverage choice
COVER_REACH_DEG = 30.0  # a coverage view counts for eye directions this close (the blend's own angle cap)
ANGLE_CAP = 30.0     # degrees: a view's weight falls to 0 at this angle from the eye's ray
HELD_REF_DEG = 15.0  # the share is full while the blend's weight is at least that of one full view at this angle
CAP_RAMP_DEG = 5.0   # the blend's share falls to 0 over this much before the cap
RAMP_FRAC = 0.5      # views bound by direction fade out over the outer half of their angles (rank_ramp) ...
RAMP_MIN_DEG = 5.0   # ... and over at least this many degrees
FADE = (12.0, 25.0)  # degrees: full ULR below, inspection texture above (default; chosen on validation frames)
FADES = ((12.0, 25.0), (20.0, 35.0), (30.0, 50.0), (60.0, 90.0))


def fps_directions(dirs: np.ndarray, n: int, start: int = 0) -> list[int]:
    """Farthest-point sampling on unit vectors (greedy max-min angle)."""
    chosen = [start]
    d = dirs @ dirs[start]
    best = d.copy()
    for _ in range(min(n, len(dirs)) - 1):
        k = int(np.argmin(best))
        chosen.append(k)
        best = np.maximum(best, dirs @ dirs[k])
    return chosen


class C2ULR(Candidate):
    id = "c2"
    label_ko = "실제 프레임 섞기 (맞춘 판 + 시점별 실제 사진)"
    family = "plate"
    renderer = "mesh-ulr"
    research = ["VDTM (Debevec 1996)", "ULR (Buehler 2001)", "NLR-RAS 가중치 (Kellnhofer 2021)"]

    # ------------------------------------------------------------------ fit
    def fit(self) -> dict:
        ctx = self.ctx
        fg = FaceGrid(ctx, RES)
        pick, n_cands = pick_frames(ctx, fg.n, N_VIEWS)
        views_dir = self.art / "views"
        views_dir.mkdir(parents=True)
        meta, ratios, rects = [], [], {}
        for i, rect, valid, shift in rectified_frames(ctx, fg, pick):
            r = np.median(rect[valid] / np.maximum(fg.base[valid], 1e-3), axis=0)
            ratios.append(r)
            rects[i] = (rect, valid)
            meta.append({"frame": int(i), "center_mm": ctx.T_cad_cam[i][:3, 3].tolist(), "ratio": r.tolist(), "shift_mm": shift,
                         "valid": round(float(valid.sum() / fg.inside.sum()), 3)})
        ref = np.median(np.stack(ratios), axis=0)
        for k, m in enumerate(meta):
            rect, valid = rects[m["frame"]]
            g = ref / np.asarray(m["ratio"])
            m["file"] = f"views/{k:03d}.webp"
            write_view(self.art / m["file"], np.clip(rect * g, 0, 1), feathered_alpha(valid, fg.inside, feather_mm=feather_mm(ctx.model)))
        info = {"chart": ctx.face, "res_mm": RES, "shape": list(fg.shape), "views": meta, "fade_deg": list(FADE),
                "feather_mm": round(feather_mm(ctx.model), 2)}
        info["active_views"], info["cover_views"] = stored_views_choice(self.art, meta)
        (self.art / "views.json").write_text(json.dumps(info))
        shutil.copy2(ctx.base_run / "bundle" / "texture.jpg", self.art / "texture.jpg")
        shutil.copy2(ctx.base_run / "bundle" / "model.glb", self.art / "model.glb")
        tune, best = self.tune_fade(info)
        return {"views": len(meta), "picked": len(pick), "candidates": n_cands, "val_psnr_by_fade": tune, "fade_deg": list(best),
                "turn_jump_every_view": info["turn_jump_every_view"], "min_view_sep_deg": info["min_view_sep_deg"],
                "active_views": len(info["active_views"]), "cover_views": info["cover_views"],
                "ecc_shift_mm_p50": float(np.median([m["shift_mm"] for m in meta if m["shift_mm"] is not None] or [np.nan]))}

    def tune_fade(self, info: dict):
        """How far from a captured ray to keep blending real frames: chosen on the validation frames
        with every stored view bindable; and, only if the turntable then jumps (a step over
        JUMP_LIMIT times the usual), whether to bind views at least MIN_VIEW_SEP_DEG apart instead
        (``choose_sep``). Written to views.json; returns the scores and the fade."""
        from ..metrics import heldout_scores

        self.load()
        quality = [m.get("valid", 1.0) for m in info["views"]]
        tune, best = {}, {}

        def grid(sep):
            self.active, self.fixed = views_for_sep(self.vdirs, self.cover, quality, sep)
            if not self.ctx.val_frames:          # too short for validation blocks: the middle settings
                best[sep] = (FADE_NO_VAL, None)
            for fd in (FADES if self.ctx.val_frames else ()):
                self.fade = fd
                v = heldout_scores(self.ctx, self.render, frames=self.ctx.val_frames).get("psnr_g")
                tune[f"{sep:g} deg apart, {fd[0]:g}-{fd[1]:g} deg"] = v
                if v is not None and v > best.get(sep, (None, -1.0))[1]:
                    best[sep] = (fd, v)
            if sep not in best:
                best[sep] = (FADE_NO_VAL, None)
            self.fade = best[sep][0]

        grid(0.0)
        jump = turn_jump(self)
        sep = 0.0
        if jump > JUMP_LIMIT:
            grid(MIN_VIEW_SEP_DEG)
            # without validation scores the smoother set is kept (the jump is what the eye sees)
            sep = choose_sep({k: v for k, (_, v) in best.items()}) if self.ctx.val_frames else MIN_VIEW_SEP_DEG
        self.active, self.fixed = views_for_sep(self.vdirs, self.cover, quality, sep)
        self.fade = best[sep][0]
        info.update({"turn_jump_every_view": jump, "min_view_sep_deg": sep, "active_views": self.active, "cover_views": self.fixed,
                     "fade_deg": list(self.fade)})
        (self.art / "views.json").write_text(json.dumps(info))
        return tune, self.fade

    # ------------------------------------------------------------------ render
    def load(self) -> None:
        meta = json.loads((self.art / "views.json").read_text())
        self.meta = meta
        self.fade = tuple(meta.get("fade_deg", FADE))
        self.chart_i = CHARTS.index(meta["chart"])
        self.views = []
        for m in meta["views"]:
            im = cv2.imread(str(self.art / m["file"]), cv2.IMREAD_UNCHANGED)
            self.views.append(cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA))
        self.centers = np.array([m["center_mm"] for m in meta["views"]], float)
        self.vdirs = self.centers / np.linalg.norm(self.centers, axis=1, keepdims=True)
        self.cover = np.stack([coverage_grid(v[..., 3]) for v in self.views])
        if "active_views" in meta:
            self.active, self.fixed = meta["active_views"], meta["cover_views"]
        else:                           # fitted before round 5: the same choice
            self.active = distinct_views(self.vdirs, [m.get("valid", 1.0) for m in meta["views"]])
            self.fixed = cover_views(self.vdirs, self.cover, active=self.active)
        tex = cv2.cvtColor(cv2.imread(str(self.art / "texture.jpg")), cv2.COLOR_BGR2RGB)
        self.sample = atlas_sampler(tex, atlas_layout(self.ctx.model), self.ctx.model)
        self.raster = PlateRaster(self.ctx.model)

    def render(self, view):
        h = self.raster.hits(view)
        rgb = self.sample(h.chart, h.uv)
        sel = h.chart == self.chart_i
        if sel.any():
            rgb[sel] = self.blend(h.point[sel], h.to_cam[sel], h.uv[sel], view.center, rgb[sel])
        return rgb, h.hit.astype(np.float32)

    def blend(self, P: np.ndarray, d: np.ndarray, uv: np.ndarray, eye: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        ulr, share = self.blend_views(P, d, uv, eye)
        return share[:, None] * ulr + (1 - share)[:, None] * fallback

    def blend_views(self, P: np.ndarray, d: np.ndarray, uv: np.ndarray, eye: np.ndarray, extra=None):
        """The stored views blended for face points ``P`` seen along ``d`` from ``eye`` (values 0-1, as
        stored), and the blend's share per point (0 where no view saw it).
        ``extra(k, dk)``: another value per view (dk: unit vectors from the points to view k's camera),
        blended with the same weights and returned third (C7 uses C1 seen from each view).

        Every bound view k is weighted, w_k = ramp_k x alpha_k x (1/theta_k)(1 - theta_k/ANGLE_CAP)
        (theta_k: the angle between the eye's ray and the view's ray to the point), so every term
        changes continuously as the eye turns (round 5): a view bound by direction joins and leaves the
        set with ramp 0 (``rank_ramp``); the share is (1 - fade) x cover x held x near_cap, with cover
        the alphas averaged with the weights, held the blend's weight against that of one full view at
        HELD_REF_DEG (at most 1: a point seen only by views about to leave fades to the fallback with
        them) and near_cap falling to 0 over CAP_RAMP_DEG before the cap.

        Round 4 blended the 4 views with the smallest angle (NLR-RAS weights, normalised by the 5th)
        out of a hard set: on 509e5ece1b at 139 -> 140 degrees of the turntable 2 of about 14 views at
        nearly the same angle from the eye left at once and 45 % of the blend's weight changed hands in
        one degree. Penalising the leaving views' angles instead made the dense 29d760c369 jump (a
        5-degree ramp moved the angles by up to 6 degrees a step); held as the part of the weight the
        ramps kept took the share down where many views sit at nearly the same angle (cc53da7c23 test
        frames: held 0.41 on average, C7's detail 0.28 -> 0.23, C2 22.6 -> 20.7 dB); weights falling as
        1/theta^2 changed little (2026-11-19, runs/compare/r5_figures/variants)."""
        e = eye / np.linalg.norm(eye)
        active = getattr(self, "active", None)
        near = select_views(self.vdirs, e, self.fixed, active=active)
        sel = rank_ramp(self.vdirs, e, near, self.fixed, getattr(self, "ramp", RAMP_FRAC), active=active)
        res = self.meta["res_mm"]
        mx = (uv[:, 0] / res - 0.5).astype(np.float32)
        my = (uv[:, 1] / res - 0.5).astype(np.float32)
        n = len(P)
        out = np.zeros((n, 3), np.float32)
        out_ex = np.zeros((n, 3), np.float32) if extra is not None else None
        wsum = np.zeros(n, np.float32)
        asum = np.zeros(n, np.float32)
        tsum = np.zeros(n, np.float32)
        for j, k in enumerate(near):
            s = remap_points(self.views[k], mx, my).astype(np.float32) / 255.0
            dk = self.centers[k][None, :] - P
            dk /= np.linalg.norm(dk, axis=1, keepdims=True)
            th = np.maximum(np.degrees(np.arccos(np.clip((dk * d).sum(1), -1, 1))), 0.05)
            w0 = (1.0 / th) * np.clip(1.0 - th / ANGLE_CAP, 0, None) * s[:, 3]
            w = sel[j] * w0
            out += w[:, None] * s[:, :3]
            if extra is not None and sel[j] > 0:
                out_ex += w[:, None] * extra(k, dk)
            wsum += w
            asum += w * s[:, 3]
            tsum += w * th
        ok = wsum > 1e-9
        ulr = out / np.maximum(wsum, 1e-9)[:, None]
        cover = asum / np.maximum(wsum, 1e-9)
        held = np.clip(wsum / ((1.0 - HELD_REF_DEG / ANGLE_CAP) / HELD_REF_DEG), 0, 1)
        t_eff = np.where(ok, tsum / np.maximum(wsum, 1e-9), ANGLE_CAP)
        fade = np.clip((t_eff - self.fade[0]) / (self.fade[1] - self.fade[0]), 0, 1)
        near_cap = np.clip((ANGLE_CAP - t_eff) / CAP_RAMP_DEG, 0, 1)
        share = np.where(ok, (1.0 - fade) * cover * held * near_cap, 0.0)
        if extra is not None:
            return ulr, share, out_ex / np.maximum(wsum, 1e-9)[:, None]
        return ulr, share

    # ------------------------------------------------------------------ web
    def export_web(self, web: Path) -> dict:
        web.mkdir(parents=True, exist_ok=True)
        for f in ("texture.jpg", "model.glb"):
            shutil.copy2(self.art / f, web / f)
        return self.write_manifest(web, {"files": {"model": "model.glb", "texture": "texture.jpg"}, "ulr": self.ulr_terms(web)})

    def ulr_terms(self, web: Path) -> dict:
        """The stored views copied to ``web/views`` and their manifest entry."""
        if (web / "views").exists():
            shutil.rmtree(web / "views")
        shutil.copytree(self.art / "views", web / "views")
        meta = json.loads((self.art / "views.json").read_text())
        lay = atlas_layout(self.ctx.model)
        return {
            "chart": meta["chart"], "res_mm": meta["res_mm"], "shape": meta["shape"],
            "chart_size_mm": list(self.ctx.model.chart_size(meta["chart"])), "atlas": {"w": 4096, "h": 2048, "layout": lay},
            "views": [{"file": m["file"], "center_mm": m["center_mm"]} for m in meta["views"]],
            **dict(zip(("active_views", "cover_views"), ((meta["active_views"], meta["cover_views"]) if "active_views" in meta
                                                          else stored_views_choice(self.art, meta["views"])))),
            "preselect": PRESELECT, "fade_deg": list(meta.get("fade_deg", FADE)),
            "angle_cap": ANGLE_CAP, "cap_ramp_deg": CAP_RAMP_DEG, "held_ref_deg": HELD_REF_DEG, "ramp_frac": RAMP_FRAC,
            "ramp_min_deg": RAMP_MIN_DEG,
            **({"residual": meta["residual"]} if "residual" in meta else {})}


class FaceGrid:
    """The filmed face chart sampled every ``res`` mm: points and normal (device frame), the inspection
    texture on it (alignment target, exposure reference, fallback) and the texels inside the outline."""

    def __init__(self, ctx, res: float):
        model, chart = ctx.model, ctx.face
        self.res = res
        tex = cv2.cvtColor(cv2.imread(str(ctx.base_run / "bundle" / "texture.jpg")), cv2.COLOR_BGR2RGB)
        sample = atlas_sampler(tex, atlas_layout(model), model)
        self.P, _, self.shape = chart_grid(model, chart, res)
        rows, cols = self.shape
        U, V = np.meshgrid((np.arange(cols) + 0.5) * res, (np.arange(rows) + 0.5) * res)
        self.uv = np.stack([U, V], -1).astype(np.float32)
        self.base = sample(np.full(self.shape, CHARTS.index(chart)), self.uv)
        self.inside = chart_outline_mask(model, chart, res, self.shape, 0.3)
        self.n = np.array([0, 0, -1.0]) if chart == "back" else np.array([0, 0, 1.0])
        self.g_base = _grad(self.base.mean(-1))


def pick_frames(ctx, n: np.ndarray, n_views: int) -> tuple[list[int], int]:
    """Training frames that look at the face, the sharper 70 %, spread over view direction (farthest
    point sampling from straight above)."""
    cands = []
    for i in ctx.train_frames:
        c = ctx.T_cad_cam[i][:3, 3]
        d = c / np.linalg.norm(c)
        if d @ n < 0.3:
            continue
        img = ctx.image(i)
        sil = ctx.silhouette(ctx.view(i), -6.0)
        if sil.sum() < 2000:
            continue
        lap = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32), cv2.CV_32F)
        cands.append((i, d, float(lap[sil].var())))
    sharp = np.array([s for _, _, s in cands])
    keep = [k for k, s in enumerate(sharp) if s >= np.percentile(sharp, 30)]
    dirs = np.stack([cands[k][1] for k in keep])
    start = int(np.argmax(dirs @ n))
    return sorted(cands[keep[k]][0] for k in fps_directions(dirs, n_views, start)), len(cands)


def rectified_frames(ctx, fg: FaceGrid, frames: list[int]):
    """Each frame (full resolution) rectified onto the face grid, aligned to the inspection texture by
    ECC (affine, at most 4 mm: the cameras' in-plane error): yields (frame, rgb 0-1, valid, shift mm).
    Valid: inside the outline, in the frame, not too grazing, nothing in front (``context.occluded``)."""
    P, n, inside = fg.P, fg.n, fg.inside
    rows, cols = fg.shape
    for i, img in ctx.full_frames(frames):
        cam = ctx.camera(i)
        T = np.linalg.inv(ctx.T_cad_cam[i])
        pc = P @ T[:3, :3].T + T[:3, 3]
        z = pc[..., 2]
        mx = (cam.fx * pc[..., 0] / z + cam.cx).astype(np.float32)
        my = (cam.fy * pc[..., 1] / z + cam.cy).astype(np.float32)
        to_cam = ctx.T_cad_cam[i][:3, 3][None, None, :] - P
        cosv = (to_cam @ n) / np.linalg.norm(to_cam, axis=-1)
        valid = inside & (z > 1) & (cosv > 0.25) & (mx > 2) & (my > 2) & (mx < cam.width - 3) & (my < cam.height - 3)
        occ = ctx.occluded(i)
        if occ.any():           # a stand or a hand in front of the plate (context.occluded)
            sc = ctx.scale
            valid &= cv2.remap(occ.astype(np.uint8), (mx + 0.5) * sc - 0.5, (my + 0.5) * sc - 0.5, cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0) == 0
        rect = cv2.remap(img, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE).astype(np.float32) / 255.0
        M = _ecc(_grad(rect.mean(-1)), fg.g_base, valid & inside)
        if M is not None and np.hypot(M[0, 2], M[1, 2]) * fg.res <= 4.0:
            rect = cv2.warpAffine(rect, M, (cols, rows), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
            valid = cv2.warpAffine(valid.astype(np.uint8), M, (cols, rows), flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP) > 0
            shift = float(np.hypot(M[0, 2], M[1, 2]) * fg.res)
        else:
            shift = None
        if valid.sum() < 1000:
            continue
        yield i, rect, valid, shift


def feather_mm(model) -> float:
    """The alpha ramp of a device: FEATHER_FRAC of its width (20 mm on the 13" iPad, 6.6 mm on a phone 71.5 mm wide)."""
    return float(FEATHER_FRAC * min(model.spec.width, model.spec.length))


def feathered_alpha(valid: np.ndarray, inside: np.ndarray, res: float = RES, feather_mm: float = FEATHER_MM) -> np.ndarray:
    """1 where the view saw the face, falling to 0 over ``feather_mm`` towards the texels it did not see
    (the outline itself is no edge)."""
    dist = cv2.distanceTransform((valid | ~inside).astype(np.uint8), cv2.DIST_L2, 5) * res
    return np.clip(dist / feather_mm, 0, 1) * valid


def write_view(path: Path, rgb01: np.ndarray, alpha: np.ndarray) -> None:
    rgba = np.dstack([np.clip(rgb01, 0, 1) * 255, np.clip(alpha, 0, 1) * 255]).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA), [cv2.IMWRITE_WEBP_QUALITY, 90])


def coverage_grid(alpha: np.ndarray, grid=COVER_GRID) -> np.ndarray:
    """Which cells of a ``grid`` over the face a view saw (at least half of the cell; alpha 0-255)."""
    a = cv2.resize((np.asarray(alpha) > 127).astype(np.float32), (grid[1], grid[0]), interpolation=cv2.INTER_AREA)
    return a >= 0.5


def distinct_views(vdirs: np.ndarray, quality, min_deg: float = MIN_VIEW_SEP_DEG, dist=None, max_ratio: float = 1.15) -> list[int]:
    """The stored views kept for binding: going from the one that saw most of the face (``quality``),
    a view is kept unless its direction is within ``min_deg`` of one kept already. Near repeats
    (often consecutive frames) filled the bound set together and made it flip from one cluster to
    another as the eye turned: on 509e5ece1b (87 views, half of them within 0.7 degrees of another)
    the largest 1-degree change of the turntable fell from 1.97 to 1.23 times the usual one with 3
    degrees, and the detail on the test frames rose (0.122 -> 0.160); on the four recordings of
    round 5 the test PSNR and LPIPS were as good or better on three (2026-11-19)."""
    order = sorted(range(len(vdirs)), key=lambda k: (-float(quality[k]), k))
    lim = np.cos(np.radians(min_deg))
    near = (lambda a, b: True) if dist is None else (lambda a, b: max(dist[a], dist[b]) <= max_ratio * min(dist[a], dist[b]))
    keep: list[int] = []
    for k in order:
        if all(float(vdirs[k] @ vdirs[j]) < lim or not near(k, j) for j in keep):
            keep.append(k)
    return sorted(keep)


def cover_views(vdirs: np.ndarray, cover: np.ndarray, n_fixed: int = N_COVER, n_dir: int = PRESELECT - N_COVER,
                reach_deg: float = COVER_REACH_DEG, active=None) -> list[int]:
    """The views kept for coverage in every bound set, chosen once per recording among the ``active``
    views (all if None). Over eye directions at the views' own directions (where the capture looked
    from), the cells of the face that the ``n_dir`` nearest views leave unseen are the holes;
    greedily, the view that fills the most holes of the eye directions within ``reach_deg`` of it is
    kept, until ``n_fixed`` are kept or no hole is left. ``cover``: (views, cells) bool. Mirrored in
    ``web/recon/mesh_ulr.js`` through the manifest (``ulr.cover_views``)."""
    idx = np.arange(len(vdirs)) if active is None else np.asarray(sorted(active), int)
    m = len(idx)
    if m == 0:
        return []
    vd = vdirs[idx]
    cover = cover[idx].reshape(m, -1).astype(bool)
    target = cover.any(0)
    cosang = vd @ vd.T                                         # (eyes, views): the eyes at the views' directions
    near = np.argsort(-cosang, axis=1, kind="stable")[:, :n_dir]
    holes = target[None, :] & ~cover[near].any(1)              # (eyes, cells)
    reach = cosang >= np.cos(np.radians(reach_deg))
    chosen: list[int] = []
    for _ in range(min(n_fixed, m)):
        gain = (reach * (holes.astype(np.float64) @ cover.T.astype(np.float64))).sum(0)
        gain[chosen] = -1
        k = int(np.argmax(gain))
        if gain[k] <= 0:
            break
        chosen.append(k)
        holes &= ~(cover[k][None, :] & reach[:, k][:, None])
    return [int(idx[k]) for k in chosen]


def views_for_sep(vdirs: np.ndarray, cover: np.ndarray, quality, sep: float) -> tuple[list[int], list[int]]:
    """The views that may be bound (at least ``sep`` degrees apart; all with 0) and the coverage
    views among them."""
    active = distinct_views(vdirs, quality, sep) if sep > 0 else list(range(len(vdirs)))
    return active, cover_views(vdirs, cover, active=active)


def turn_jump(cand) -> float:
    """The largest 1-degree step of ``cand``'s turntable (30 degrees, the stress views) over the usual
    one, as scored (``metrics.turn_jumps``)."""
    from ..metrics import stress_scores
    from ..stress import stress_views

    views = [v for v in stress_views(cand.ctx.model, cand.ctx.face) if v.name.split()[0] == "turn30"]
    st = stress_scores(cand.ctx, cand.render, views, (np.full(3, -1.0), np.full(3, 2.0)))
    return float(st.get("flicker", {}).get("jump", 0.0))


def choose_sep(best_by_sep: dict, tol: float = SEP_TOLERANCE_DB) -> float:
    """The largest separation of bound views whose best validation PSNR is within ``tol`` dB of the
    best of all: fewer near repeats turn more smoothly, which the validation frames do not see. Asked
    only when the turntable jumps with every view bound (``tune_fade``): binding fewer views also
    blends views farther from the eye, and on 6c417d24a0 (131 -> 15 views) the logo of test frame 480
    came out as two reflections mixed and the detail of the test frames fell (0.244 -> 0.195), though
    that recording never jumped and its validation PSNR rose; on cc53da7c23 the 3-degree set lost
    1.0 dB on the validation frames (19.12 -> 18.10) (2026-11-19)."""
    vals = {s: v for s, v in best_by_sep.items() if v is not None}
    if not vals:
        return max(best_by_sep)
    top = max(vals.values())
    return max(s for s, v in vals.items() if v >= top - tol)


def stored_views_choice(art: Path, views: list[dict]) -> tuple[list[int], list[int]]:
    """The views kept for binding (``distinct_views``) and the coverage views among them
    (``cover_views``), from the stored views' directions and their files' alpha (before the
    validation choice of ``tune_fade``, which may bind every view)."""
    c = np.array([v["center_mm"] for v in views], float)
    vdirs = c / np.linalg.norm(c, axis=1, keepdims=True)
    active = distinct_views(vdirs, [v.get("valid", 1.0) for v in views])
    cov = np.stack([coverage_grid(cv2.imread(str(art / v["file"]), cv2.IMREAD_UNCHANGED)[..., 3]) for v in views])
    return active, cover_views(vdirs, cov, active=active)


def select_views(vdirs: np.ndarray, e: np.ndarray, fixed=(), n: int = PRESELECT, active=None) -> list[int]:
    """The bound set for the eye's direction ``e``: the coverage views (``cover_views``, the same for
    every eye), then the ``active`` views (all if None) nearest to ``e`` in direction. As the eye turns
    the set changes only by direction rank. Mirrored in ``web/recon/mesh_ulr.js``."""
    fixed = [int(k) for k in fixed][:n]
    keep = set(fixed)
    ok = None if active is None else set(int(k) for k in active)
    order = [int(k) for k in np.argsort(-(vdirs @ e), kind="stable") if int(k) not in keep and (ok is None or int(k) in ok)]
    return fixed + order[: n - len(fixed)]


def rank_ramp(vdirs: np.ndarray, e: np.ndarray, bound: list[int], fixed=(), frac: float = RAMP_FRAC,
              min_deg: float = RAMP_MIN_DEG, active=None) -> np.ndarray:
    """A factor per bound view on its weight, 1 for the coverage views; for the views bound by
    direction, 1 near the eye falling to 0 at the angle (from the device centre) of the nearest view
    left out, over the outer ``frac`` of the bound views' angles and at least ``min_deg``, so a view
    has no weight when it joins or leaves the set; only ``active`` views (all if None) are left out.
    ``frac`` 0: all 1."""
    out = np.ones(len(bound), np.float32)
    if frac <= 0:
        return out
    keep = set(int(k) for k in fixed)
    ang = np.degrees(np.arccos(np.clip(vdirs @ e, -1, 1)))
    left = np.setdiff1d(np.arange(len(vdirs)) if active is None else np.asarray(sorted(active), int), bound)
    dir_j = [j for j, k in enumerate(bound) if k not in keep]
    if not len(left) or not dir_j:
        return out
    cut = ang[left].min()
    first = min(ang[bound[j]] for j in dir_j)
    width = max(frac * (cut - first), min_deg)
    for j in dir_j:
        out[j] = np.clip((cut - ang[bound[j]]) / width, 0, 1)
    return out


def remap_points(img: np.ndarray, x: np.ndarray, y: np.ndarray, width: int = 4096) -> np.ndarray:
    """Bilinear samples of ``img`` at float pixel positions (any count; cv2.remap needs maps narrower
    than 32767, so the points are laid out in rows). Outside the image: 0."""
    n = len(x)
    rows = max(1, int(np.ceil(n / width)))
    pad = rows * width - n
    mx = np.concatenate([x, np.full(pad, -10, np.float32)]).reshape(rows, width)
    my = np.concatenate([y, np.full(pad, -10, np.float32)]).reshape(rows, width)
    out = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out.reshape(rows * width, -1)[:n]


def _grad(g: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.5)
    return np.sqrt(cv2.Sobel(g, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(g, cv2.CV_32F, 0, 1) ** 2)


def _ecc(src: np.ndarray, dst: np.ndarray, mask: np.ndarray, down: int = 2):
    """Affine ``M`` with ``src(M x) ~ dst(x)`` (cv2 convention), estimated at 1/``down`` resolution."""
    s = cv2.resize(src, None, fx=1 / down, fy=1 / down, interpolation=cv2.INTER_AREA)
    d = cv2.resize(dst, None, fx=1 / down, fy=1 / down, interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask.astype(np.uint8), (s.shape[1], s.shape[0]), interpolation=cv2.INTER_NEAREST)
    W = np.eye(2, 3, dtype=np.float32)
    try:
        _, W = cv2.findTransformECC(d, s, W, cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), m, 5)
    except cv2.error:
        return None
    W[:, 2] *= down
    return W
