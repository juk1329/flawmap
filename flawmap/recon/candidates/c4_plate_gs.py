"""C4, Gaussians bound to the fitted plate: the look of splatting, the geometry of the plate.

Flat Gaussians (surfels: two in-plane scales, the third tiny) whose centres are parameterised on the
filmed face of the fitted plate, in its chart coordinates (u, v mm) plus a small outward offset
(0-0.3 mm), with an in-plane rotation and SH colour (degree 3). The surface-binding idea follows 2DGS
(Huang et al. 2024: surfels), SuGaR (Guedon & Lepetit 2024: Gaussians aligned to the surface),
GaMeS (Waczynska et al. 2024: Gaussians parameterised on a mesh) and Gaussian surfels (Dai et al.
2024); here the "mesh" is the plate fitted from the capture, so nothing can float off it.

No densification: a fixed grid every 0.35 mm over the face inside the outline (about 480k on an
iPad back), initialised from the inspection texture, so the strategies that move splats in 3D are not
needed. Everything off the filmed face (the band, the face that was not filmed) is the plate mesh
with the inspection texture underneath (C0's look); splats are composited over it with a depth test.

Trained on the context's training frames like C5 (L1 + D-SSIM inside the face, the mesh render
underneath), 15k steps. Export: the splats as a 3DGS PLY (device frame, metres) plus C0's mesh files.

Round 2 (2026-11-18): the first fit came out grainy and blurred (cc53da7c23: LPIPS 0.369 against
0.190 for C0; the logo and the print smeared). Each training frame now gets its own small shift of
the face (at most 3 mm, the cameras' in-plane error; C1/C3 and C2's ECC do the same) and its own RGB
gain, so pose and exposure errors are not averaged into the colours; the surfels move at most
0.2 mm, start opaque and are kept so, and the view-dependent part is SH degree 2 with a penalty.

The grain that stayed (cc53da7c23, LPIPS 0.326) is colour at finer than the training pixels: the
surfels sit every 0.35 mm, the half-resolution frames sample about 0.5 mm, and what a frame cannot see
Adam lets drift. A total-variation term on the surfel grid's colours (TV_WEIGHT; neighbours on the
0.35 mm grid) holds that part still.
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
from ...export.bundle import atlas_layout
from ...utils.logging import get_logger
from ..context import View
from ..render import PlateRaster, atlas_sampler, chart_grid, chart_outline_mask
from .base import Candidate
from .c5_free_gs import C0 as SH_C0, _ssim, read_ply, write_ply

log = get_logger("flawmap.recon")
SPACING = 0.35       # mm between initial splats
OFFSET_MAX = 0.3     # mm, outward
SCALE_MM = (0.05, 1.2)   # in-plane size of a surfel: without a bound, edge surfels grew into long needles
STEPS = 15000
SH_DEGREE = 2
SHIFT_MM = 3.0       # per training frame, in the face's plane
DUV_MM = 0.2         # per surfel
TV_WEIGHT = 0.02     # colour total variation between grid neighbours


def _quat_mul(a, b):
    import torch

    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def _scale(x):
    """In-plane surfel size in metres, bounded to SCALE_MM."""
    import torch

    lo, hi = SCALE_MM
    return (lo + (hi - lo) * torch.sigmoid(x)) * 1e-3


def _inv_scale(mm: float) -> float:
    lo, hi = SCALE_MM
    t = (mm - lo) / (hi - lo)
    return float(math.log(t / (1 - t)))


class C4PlateGS(Candidate):
    id = "c4"
    label_ko = "판에 묶은 가우시안 (맞춘 판 위 서펠 + SH)"
    family = "plate"
    renderer = "splat-mesh"
    research = ["2DGS 서펠 (SIGGRAPH 2024)", "SuGaR (CVPR 2024)", "GaMeS (2024)", "Gaussian surfels (SIGGRAPH 2024)", "gsplat (2024)"]

    # ------------------------------------------------------------------ geometry of the face
    def _face(self):
        model, chart = self.ctx.model, self.ctx.face
        w, l = model.chart_size(chart)
        if chart == "back":
            z, n, sx = model.z_back, np.array([0, 0, -1.0]), -1.0       # x = W/2 - u
        else:
            z, n, sx = model.z_front, np.array([0, 0, 1.0]), 1.0        # x = u - W/2
        return w, l, z, n, sx

    def _means(self, uv, off):
        """Centres (metres) from chart mm and outward offsets (mm)."""
        import torch

        w, l, z, n, sx = self._face()
        x = (w / 2 - uv[:, 0]) if sx < 0 else (uv[:, 0] - w / 2)
        y = l / 2 - uv[:, 1]
        zz = z + float(n[2]) * off
        return torch.stack([x, y, zz], -1) * 1e-3

    def _plane_quat(self):
        """Rotation taking the local z axis to the face normal (back: 180 degrees about x)."""
        import torch

        _, _, _, n, _ = self._face()
        return torch.tensor([0.0, 1.0, 0.0, 0.0]) if n[2] < 0 else torch.tensor([1.0, 0.0, 0.0, 0.0])

    # ------------------------------------------------------------------ fit
    def fit(self) -> dict:
        import torch
        from gsplat import rasterization

        ctx, model, chart = self.ctx, self.ctx.model, self.ctx.face
        dev = "cuda"
        t0 = time.time()
        tex = cv2.cvtColor(cv2.imread(str(ctx.base_run / "bundle" / "texture.jpg")), cv2.COLOR_BGR2RGB)
        lay = atlas_layout(model)
        sample = atlas_sampler(tex, lay, model)
        raster = PlateRaster(model)
        # initial grid on the face
        P, _, shape = chart_grid(model, chart, SPACING)
        rows, cols = shape
        inside = chart_outline_mask(model, chart, SPACING, shape, 0.2)
        U, V = np.meshgrid((np.arange(cols) + 0.5) * SPACING, (np.arange(rows) + 0.5) * SPACING)
        uv0 = np.stack([U[inside], V[inside]], 1).astype(np.float32)
        N = len(uv0)
        # grid neighbours (right, down) for the colour total variation
        gidx = np.full(inside.shape, -1, np.int64)
        gidx[inside] = np.arange(N)
        pairs = []
        for a, b in ((gidx[:, :-1], gidx[:, 1:]), (gidx[:-1, :], gidx[1:, :])):
            ok = (a >= 0) & (b >= 0)
            pairs.append(np.stack([a[ok], b[ok]], 1))
        pairs = np.concatenate(pairs)
        # colours of the inspection texture at the centres (laid out in rows: cv2.remap maps must be narrower than 32767)
        R = int(np.ceil(N / 4096))
        uv_pad = np.concatenate([uv0, np.repeat(uv0[:1], R * 4096 - N, 0)]).reshape(R, 4096, 2)
        col0 = sample(np.full((R, 4096), CHARTS.index(chart)), uv_pad).reshape(-1, 3)[:N]
        log.info(f"[{ctx.rec}] c4: {N} surfels on the {chart}")
        # training data: frames, the mesh underneath (C0 look, unlit texture), the face region
        frames = ctx.train_frames
        imgs, bases, faces, Ks, vms = [], [], [], [], []
        for i in frames:
            v = ctx.view(i)
            h = raster.hits(v)
            base = sample(h.chart, h.uv)
            face = (h.chart == CHARTS.index(chart)) & ctx.silhouette(v, -3.0)
            imgs.append(torch.from_numpy(ctx.image(i)))
            bases.append(torch.from_numpy((np.clip(base, 0, 1) * 255).astype(np.uint8)))
            faces.append(torch.from_numpy(face))
            Ks.append(torch.from_numpy(v.K.astype(np.float32)))
            T = np.asarray(v.T_cad_cam, float).copy()
            T[:3, 3] *= 1e-3
            vms.append(torch.from_numpy(np.linalg.inv(T).astype(np.float32)))
        imgs, bases, faces = torch.stack(imgs), torch.stack(bases), torch.stack(faces)
        Ks, vms = torch.stack(Ks).to(dev), torch.stack(vms).to(dev)
        H, W = imgs.shape[1:3]
        sh_n = (SH_DEGREE + 1) ** 2 - 1
        F = len(frames)
        p = torch.nn.ParameterDict({
            "duv": torch.nn.Parameter(torch.zeros(N, 2, device=dev)),
            "off": torch.nn.Parameter(torch.full((N,), -2.0, device=dev)),
            "phi": torch.nn.Parameter(torch.zeros(N, device=dev)),
            "lsc": torch.nn.Parameter(torch.full((N, 2), _inv_scale(SPACING * 0.7), device=dev)),
            "opa": torch.nn.Parameter(torch.logit(torch.full((N,), 0.98, device=dev))),
            "sh0": torch.nn.Parameter(((torch.from_numpy(col0).to(dev) - 0.5) / SH_C0)[:, None, :]),
            "shN": torch.nn.Parameter(torch.zeros(N, sh_n, 3, device=dev)),
            "fsh": torch.nn.Parameter(torch.zeros(F, 2, device=dev)),       # per-frame face shift (atanh of mm / SHIFT_MM)
            "fgain": torch.nn.Parameter(torch.zeros(F, 3, device=dev)),     # per-frame log RGB gain
        })
        lrs = {"duv": 2e-3, "off": 1e-2, "phi": 1e-2, "lsc": 5e-3, "opa": 5e-3, "sh0": 2.5e-3, "shN": 2.5e-3 / 20,
               "fsh": 2e-3, "fgain": 2e-3}
        opt = torch.optim.Adam([{"params": [p[k]], "lr": lr} for k, lr in lrs.items()], eps=1e-15)
        uv_t = torch.from_numpy(uv0).to(dev)
        pa, pb = (torch.from_numpy(pairs[:, k]).to(dev) for k in (0, 1))
        qp = self._plane_quat().to(dev)
        rng = np.random.default_rng(0)
        losses = []
        for step in range(STEPS):
            k = int(rng.integers(len(frames)))
            gt = imgs[k].to(dev).float()[None] / 255.0
            base = bases[k].to(dev).float()[None] / 255.0
            face = faces[k].to(dev)[None, ..., None].float()
            shift = SHIFT_MM * torch.tanh(p["fsh"][k])
            means = self._means(uv_t + DUV_MM * torch.tanh(p["duv"]) + shift[None, :], OFFSET_MAX * torch.sigmoid(p["off"]))
            half = 0.5 * p["phi"]
            qz = torch.stack([torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], -1)
            quats = _quat_mul(qp.expand_as(qz), qz)
            scales = torch.cat([_scale(p["lsc"]), torch.full((N, 1), 1e-6, device=dev)], 1)
            colors = torch.cat([p["sh0"], p["shN"]], 1)
            rgb, alpha, _ = rasterization(means, quats, scales, torch.sigmoid(p["opa"]), colors, vms[k:k + 1], Ks[k:k + 1], W, H,
                                          sh_degree=min(step // 1000, SH_DEGREE), packed=False)
            pred = (rgb + (1 - alpha) * base) * torch.exp(p["fgain"][k])[None, None, None, :]
            l1 = (torch.abs(pred - gt) * face).sum() / (face.sum() * 3 + 1)
            ss = _ssim((pred * face).permute(0, 3, 1, 2), (gt * face).permute(0, 3, 1, 2)).mean()
            # the splats belong to the filmed face: nothing of them may show off it (the band, the floor)
            off_face = (alpha * (1 - face)).sum() / ((1 - face).sum() + 1)
            # the face is opaque; the view-dependent colour stays small; the gains have no common drift
            reg = 0.05 * (1 - torch.sigmoid(p["opa"])).mean() + 0.1 * (p["shN"] ** 2).mean() + 1.0 * p["fgain"].mean(0).pow(2).sum()
            reg = reg + TV_WEIGHT * SH_C0 * (p["sh0"][pa] - p["sh0"][pb]).abs().mean()
            loss = 0.8 * l1 + 0.2 * (1 - ss) + 1.0 * off_face + reg
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            if step % 2500 == 0:
                losses.append(float(loss.item()))
                log.info(f"[{ctx.rec}] c4 step {step}: loss {loss.item():.4f}")
        with torch.no_grad():
            shifts = (SHIFT_MM * torch.tanh(p["fsh"])).norm(dim=1).cpu().numpy()
            gains = torch.exp(p["fgain"]).cpu().numpy()
            means = self._means(uv_t + DUV_MM * torch.tanh(p["duv"]), OFFSET_MAX * torch.sigmoid(p["off"]))
            half = 0.5 * p["phi"]
            qz = torch.stack([torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], -1)
            quats = _quat_mul(qp.expand_as(qz), qz)
            scales = torch.cat([torch.log(_scale(p["lsc"])), torch.full((N, 1), math.log(1e-6), device=dev)], 1)
            keep = torch.sigmoid(p["opa"]) > 0.02
            data = {"means": means[keep], "quats": quats[keep], "scales": scales[keep], "opacities": p["opa"][keep],
                    "sh0": p["sh0"][keep], "shN": p["shN"][keep]}
            data = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in data.items()}
        write_ply(self.art / "splat.ply", data)
        for f in ("texture.jpg", "model.glb", "roughness.png"):
            if (ctx.base_run / "bundle" / f).exists():
                shutil.copy2(ctx.base_run / "bundle" / f, self.art / f)
        return {"surfels": int(keep.sum()), "spacing_mm": SPACING, "steps": STEPS, "sh_degree": SH_DEGREE, "loss": losses,
                "frame_shift_mm_p50": round(float(np.median(shifts)), 3), "frame_shift_mm_p90": round(float(np.percentile(shifts, 90)), 3),
                "frame_gain_range": [round(float(gains.min()), 3), round(float(gains.max()), 3)], "train_s": round(time.time() - t0, 1)}

    # ------------------------------------------------------------------ render: mesh underneath, splats over it with a depth test
    def load(self) -> None:
        import torch

        d = read_ply(self.art / "splat.ply")
        self.g = {k: torch.from_numpy(v).cuda() for k, v in d.items()}
        tex = cv2.cvtColor(cv2.imread(str(self.art / "texture.jpg")), cv2.COLOR_BGR2RGB)
        self.sample = atlas_sampler(tex, atlas_layout(self.ctx.model), self.ctx.model)
        self.raster = PlateRaster(self.ctx.model)

    def render(self, view: View):
        import torch
        from gsplat import rasterization

        h = self.raster.hits(view)
        base = self.sample(h.chart, h.uv)
        T = np.asarray(view.T_cad_cam, float).copy()
        T[:3, 3] *= 1e-3
        vm = torch.from_numpy(np.linalg.inv(T).astype(np.float32)).cuda()[None]
        K = torch.from_numpy(np.asarray(view.K, np.float32)).cuda()[None]
        g = self.g
        deg = int(round(math.sqrt(1 + g["shN"].shape[1]))) - 1
        with torch.no_grad():
            rgbd, alpha, _ = rasterization(g["means"], g["quats"], torch.exp(g["scales"]), torch.sigmoid(g["opacities"]),
                                           torch.cat([g["sh0"], g["shN"]], 1), vm, K, view.W, view.H, sh_degree=deg, render_mode="RGB+ED")
        a = alpha[0, ..., 0].cpu().numpy()
        c = rgbd[0, ..., :3].cpu().numpy()
        ed = rgbd[0, ..., 3].cpu().numpy() * 1e3                        # mm along the ray axis
        # the plate hides splats behind it (seen from the unfilmed side)
        zmesh = np.where(h.hit, np.einsum("hwk,k->hw", h.point - np.asarray(view.T_cad_cam)[:3, 3], np.asarray(view.T_cad_cam)[:3, 2]), np.inf)
        vis = ~(h.hit & (ed > zmesh + 1.0))
        a = np.where(vis, a, 0.0)
        c = np.where(vis[..., None], c, 0.0)
        out = c + (1 - a[..., None]) * base
        alpha_out = np.maximum(h.hit.astype(np.float32), a)
        return np.clip(out, 0, 1).astype(np.float32), alpha_out.astype(np.float32)

    def export_web(self, web: Path) -> dict:
        web.mkdir(parents=True, exist_ok=True)
        files = {"splat": "splat.ply"}
        shutil.copy2(self.art / "splat.ply", web / "splat.ply")
        for f in ("texture.jpg", "model.glb", "roughness.png"):
            if (self.art / f).exists():
                shutil.copy2(self.art / f, web / f)
                files[f.split(".")[0]] = f
        return self.write_manifest(web, {"files": files, "surfels": self.fit_stats().get("surfels")})
