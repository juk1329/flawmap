"""C4b, Gaussians bound to the fitted plate, round 4: surface colour held to the inspection texture and a
measured reflection shared by the whole face.

What C4 left (rounds 2-3): close views blotched and grainy, the print and the camera island
smeared (cc53da7c23 held-out frame 489, a close-up of the bottom corner: LPIPS 0.54 against 0.17
for C0 and C2; the "iPad Pro" print gone). Each surfel carried its own colour and its own SH of
the view direction, fitted to the frames that happened to see it: where few frames saw a part of
the face, or saw it only from afar, the colour and its view-dependent part drifted (Adam keeps
moving what the loss does not pin), and the sheen of a moving reflection, which is one smooth
thing for the whole face, was learned 480 000 times over.

C4b changes two things, each from a known method (a third, relief, was tried and taken out, below):
- **Colour held to the inspection texture** (as C1 and C3 since round 2): a surfel's colour is the
  inspection texture at its place times a smooth gain (2 mm grid, log, total variation) plus its own
  residual, which an L1 prior keeps at zero unless frames ask for it. Where frames say nothing the
  surfel shows the texture, which keeps each texel from its best-supported views.
- **Deferred reflection** (3DGS-DR, Ye et al. SIGGRAPH 2024; GaussianShader, Jiang et al. CVPR 2024;
  Ref-Gaussian 2025): the view-dependent part is no SH per surfel but one environment map shared by
  the face, read in the mirror direction per pixel after the splats are drawn, times a specular scale
  on a 2 mm grid. Those methods have to estimate the normals of free Gaussians, which is where they
  go wrong; here the normal is the fitted plate's. The map is read in a finite room (the parallax-
  corrected lookup of ``envmap.proxy_dirs``) and filled where no frame looked (``envmap.pull_push``),
  like C1. The reflection is added to the splats' code values, which is what the browser can do with
  an additive overlay on top of the splats.
- **Relief, tried and taken out:** surfels allowed to stand off the face by -0.3 to +2.5 mm (total
  variation and L1 to zero), for the camera island (about 1.5 mm off an iPad's back). In none of three
  fits did the island rise (cc53da7c23: at most 0.73 mm anywhere); on 6c417d24a0 only the mirror-polished
  logo rose, to 2.5 mm, imitating its reflection's parallax with height, and turned to 10-25 degrees the
  logo became a ragged smudge (2026-11-19). The offsets are C4's again (0-0.3 mm).

Design check (validation frames, gain-fitted PSNR / LPIPS; the variants differ in the residual prior
and the relief): the L1 + neighbour TV prior on the residual (``W_RES_L1``, ``W_COL_TV``) gave the best
LPIPS on both recordings; a prior on the residual's local mean only (``W_RES_LOW``, which frees detail)
gave sharper print but more grain (cc53da7c23 0.258 against 0.245). The reflection is read in the mirror
direction: a learned finite room (``room``) ran to its 50 m bound on both recordings.

Everything else is C4's: the grid every 0.35 mm, in-plane moves of at most 0.2 mm, near-opaque
surfels over the plate mesh with the inspection texture, per-frame face shift (at most 3 mm) and RGB
gain, L1 + D-SSIM inside the face, no alpha off the face, 15k steps.
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
from ..envmap import confidence, proxy_dirs, pull_push, support
from ..render import PlateRaster, atlas_sampler, chart_grid, chart_outline_mask
from .base import Candidate
from .c4_plate_gs import DUV_MM, SHIFT_MM, SPACING, C4PlateGS, _inv_scale, _quat_mul, _scale
from .c5_free_gs import C0 as SH_C0, _ssim, read_ply, write_ply
from .plate_fit import ENV_H, ENV_W, ROOM_BOUNDS, ROOM_MM

log = get_logger("flawmap.recon")
STEPS = 15000
OFFSET_RANGE = (0.0, 0.3)    # mm along the outward normal (C4's; relief to 2.5 mm was tried, see above)
GRID_MM = 2.0                # specular scale and colour gain grids
W_RES_L1 = 0.02              # L1 on the colour residual (code values)
W_COL_TV = 0.02              # colour residual total variation between grid neighbours
W_RES_LOW = 0.0              # L2 on the residual's local mean (RES_BLUR x RES_BLUR surfels): holds drift, frees detail
RES_BLUR = 5
W_OFF_TV = 0.05              # relief: total variation (mm) between grid neighbours
W_OFF_L1 = 0.005             # relief: L1 to the face (mm)


class C4bPlateGSReflect(C4PlateGS):
    id = "c4b"
    label_ko = "판에 묶은 가우시안 2판 (텍스처에 붙잡은 색 + 공통 반사 지도)"
    renderer = "splat-srm"
    research = ["2DGS 서펠 (SIGGRAPH 2024)", "3DGS-DR 지연 반사 (SIGGRAPH 2024)", "GaussianShader (CVPR 2024)",
                "Ref-Gaussian (2025)", "GaMeS (2024)", "당겨 밀어 채우기 (Gortler 1996)"]
    w_res_l1, w_col_tv, w_res_low, w_off_l1, offset_range = W_RES_L1, W_COL_TV, W_RES_LOW, W_OFF_L1, OFFSET_RANGE
    room = False             # read the environment in a finite room of learned radius (else the mirror direction)

    # ------------------------------------------------------------------ the reflection, per pixel
    def _face_hits(self, K: np.ndarray, T_cad_cam: np.ndarray, W: int, H: int, dev: str):
        """Per pixel of a pinhole view: the face plane's point (mm), the unit vector to the camera and
        the chart uv (mm); torch tensors (H, W, ...). Pixels whose ray misses the plane get NaN."""
        import torch

        w, l, z, n, sx = self._face()
        Kt = torch.as_tensor(K, dtype=torch.float32, device=dev)
        T = torch.as_tensor(np.asarray(T_cad_cam, float), dtype=torch.float32, device=dev)
        ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32),
                                indexing="ij")
        rc = torch.stack([(xs - Kt[0, 2]) / Kt[0, 0], (ys - Kt[1, 2]) / Kt[1, 1], torch.ones_like(xs)], -1)
        r = rc @ T[:3, :3].T
        c = T[:3, 3]
        t = (z - c[2]) / r[..., 2]
        p = c + t[..., None] * r
        p = torch.where((t > 0)[..., None], p, torch.full_like(p, float("nan")))
        d = c - p
        d = d / d.norm(dim=-1, keepdim=True)
        u = (w / 2 - p[..., 0]) if sx < 0 else (p[..., 0] - w / 2)
        v = l / 2 - p[..., 1]
        return p, d, torch.stack([u, v], -1)

    def _spec(self, p, d, uv, S, Lmap, rho, filled=None):
        """s(uv) L(lookup) in code values for points ``p`` (..., 3), unit vectors to the camera ``d``."""
        import torch
        import torch.nn.functional as F

        w_, l_, z, n, sx = self._face()
        nt = torch.tensor(n, dtype=p.dtype, device=p.device)
        mirror = 2 * (d * nt).sum(-1, keepdim=True) * nt - d
        if rho is None:
            q = mirror
        else:
            q = proxy_dirs(p, mirror, torch.exp(rho.clamp(math.log(ROOM_BOUNDS[0]), math.log(ROOM_BOUNDS[1]))))
        eu = torch.atan2(q[..., 1], q[..., 0]) / (2 * math.pi) + 0.5
        ev = torch.arccos(q[..., 2].clamp(-1 + 1e-6, 1 - 1e-6)) / math.pi        # finite slope at the poles
        shp = eu.shape
        ge = torch.stack([eu.reshape(-1) * 2 - 1, ev.reshape(-1) * 2 - 1], -1)[None, None]
        env = filled.permute(2, 0, 1)[None] if filled is not None else F.softplus(Lmap)
        L = F.grid_sample(env, ge, mode="bilinear", padding_mode="border", align_corners=False)[0, :, 0].T
        gs = torch.stack([uv[..., 0].reshape(-1) / w_ * 2 - 1, uv[..., 1].reshape(-1) / l_ * 2 - 1], -1)[None, None]
        s = F.softplus(F.grid_sample(S, gs, mode="bilinear", padding_mode="border", align_corners=False)[0, 0, 0])
        return (s[:, None] * L).reshape(*shp, 3), q

    # ------------------------------------------------------------------ fit
    def fit(self) -> dict:
        import torch
        import torch.nn.functional as F
        from gsplat import rasterization

        ctx, model, chart = self.ctx, self.ctx.model, self.ctx.face
        dev = "cuda"
        t0 = time.time()
        tex = cv2.cvtColor(cv2.imread(str(ctx.base_run / "bundle" / "texture.jpg")), cv2.COLOR_BGR2RGB)
        lay = atlas_layout(model)
        sample = atlas_sampler(tex, lay, model)
        raster = PlateRaster(model)
        w_mm, l_mm, _, _, _ = self._face()
        P, _, shape = chart_grid(model, chart, SPACING)
        rows, cols = shape
        inside = chart_outline_mask(model, chart, SPACING, shape, 0.2)
        U, V = np.meshgrid((np.arange(cols) + 0.5) * SPACING, (np.arange(rows) + 0.5) * SPACING)
        uv0 = np.stack([U[inside], V[inside]], 1).astype(np.float32)
        N = len(uv0)
        gidx = np.full(inside.shape, -1, np.int64)
        gidx[inside] = np.arange(N)
        pairs = []
        for a, b in ((gidx[:, :-1], gidx[:, 1:]), (gidx[:-1, :], gidx[1:, :])):
            ok = (a >= 0) & (b >= 0)
            pairs.append(np.stack([a[ok], b[ok]], 1))
        pairs = np.concatenate(pairs)
        R_ = int(np.ceil(N / 4096))
        uv_pad = np.concatenate([uv0, np.repeat(uv0[:1], R_ * 4096 - N, 0)]).reshape(R_, 4096, 2)
        tex0 = sample(np.full((R_, 4096), CHARTS.index(chart)), uv_pad).reshape(-1, 3)[:N]
        log.info(f"[{ctx.rec}] c4b: {N} surfels on the {chart}")
        frames = ctx.train_frames
        imgs, bases, faces, Ks, vms, Tcs = [], [], [], [], [], []
        for i in frames:
            v = ctx.view(i)
            h = raster.hits(v)
            base = sample(h.chart, h.uv)
            face = (h.chart == CHARTS.index(chart)) & ctx.silhouette(v, -3.0)
            imgs.append(torch.from_numpy(ctx.image(i)))
            bases.append(torch.from_numpy((np.clip(base, 0, 1) * 255).astype(np.uint8)))
            faces.append(torch.from_numpy(face))
            Ks.append(v.K.astype(np.float32))
            Tcs.append(np.asarray(v.T_cad_cam, float))
            T = np.asarray(v.T_cad_cam, float).copy()
            T[:3, 3] *= 1e-3
            vms.append(torch.from_numpy(np.linalg.inv(T).astype(np.float32)))
        imgs, bases, faces = torch.stack(imgs), torch.stack(bases), torch.stack(faces)
        Kt = torch.from_numpy(np.stack(Ks)).to(dev)
        vms = torch.stack(vms).to(dev)
        H, W = imgs.shape[1:3]
        F_ = len(frames)
        gh, gw = int(math.ceil(l_mm / GRID_MM)), int(math.ceil(w_mm / GRID_MM))
        p = torch.nn.ParameterDict({
            "duv": torch.nn.Parameter(torch.zeros(N, 2, device=dev)),
            "off": torch.nn.Parameter(torch.zeros(N, device=dev)),                      # mm
            "phi": torch.nn.Parameter(torch.zeros(N, device=dev)),
            "lsc": torch.nn.Parameter(torch.full((N, 2), _inv_scale(SPACING * 0.7), device=dev)),
            "opa": torch.nn.Parameter(torch.logit(torch.full((N,), 0.98, device=dev))),
            "res": torch.nn.Parameter(torch.zeros(N, 3, device=dev)),                   # colour residual (code)
            "G": torch.nn.Parameter(torch.zeros(1, 3, gh, gw, device=dev)),             # log colour gain
            "S": torch.nn.Parameter(torch.full((1, 1, gh, gw), -1.0, device=dev)),      # specular scale (softplus)
            "L": torch.nn.Parameter(torch.full((1, 3, ENV_H, ENV_W), -5.0, device=dev)),  # environment (softplus, code)
            "rho": torch.nn.Parameter(torch.tensor(math.log(ROOM_MM), device=dev), requires_grad=self.room),
            "fsh": torch.nn.Parameter(torch.zeros(F_, 2, device=dev)),
            "fgain": torch.nn.Parameter(torch.zeros(F_, 3, device=dev)),
        })
        lrs = {"duv": 2e-3, "off": 5e-3, "phi": 1e-2, "lsc": 5e-3, "opa": 5e-3, "res": 2.5e-3, "G": 1e-2, "S": 1e-2, "L": 1e-2,
               "rho": 1e-2, "fsh": 2e-3, "fgain": 2e-3}
        opt = torch.optim.Adam([{"params": [p[k]], "lr": lr} for k, lr in lrs.items()], eps=1e-15)
        uv_t = torch.from_numpy(uv0).to(dev)
        tex_t = torch.from_numpy(tex0).to(dev)
        gsl = torch.stack([uv_t[:, 0] / w_mm * 2 - 1, uv_t[:, 1] / l_mm * 2 - 1], -1)[None, None]
        pa, pb = (torch.from_numpy(pairs[:, k]).to(dev) for k in (0, 1))
        qp = self._plane_quat().to(dev)
        lo, hi = self.offset_range
        rng = np.random.default_rng(0)
        grid_idx = torch.from_numpy(gidx).to(dev)
        in_grid = grid_idx >= 0

        def low_pass(x):
            """Local mean over RES_BLUR x RES_BLUR grid neighbours inside the outline (N, C)."""
            g = torch.zeros(x.shape[1], *grid_idx.shape, device=dev)
            g[:, in_grid] = x.T
            w = in_grid.float()[None, None]
            k = torch.ones(1, 1, RES_BLUR, RES_BLUR, device=dev)
            num = F.conv2d(g[:, None], k, padding=RES_BLUR // 2)[:, 0]
            den = F.conv2d(w, k, padding=RES_BLUR // 2)[0, 0].clamp(min=1.0)
            return (num / den)[:, in_grid].T
        losses = []

        def colours():
            gain = torch.exp(F.grid_sample(p["G"], gsl, mode="bilinear", padding_mode="border", align_corners=False)[0, :, 0].T)
            return (tex_t * gain + p["res"]).clamp(0, 1)

        def splats(shift_mm):
            off = p["off"].clamp(lo, hi)
            means = self._means(uv_t + DUV_MM * torch.tanh(p["duv"]) + shift_mm[None, :], off)
            half = 0.5 * p["phi"]
            qz = torch.stack([torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], -1)
            quats = _quat_mul(qp.expand_as(qz), qz)
            scales = torch.cat([_scale(p["lsc"]), torch.full((N, 1), 1e-6, device=dev)], 1)
            return means, quats, scales

        for step in range(STEPS):
            k = int(rng.integers(F_))
            gt = imgs[k].to(dev).float()[None] / 255.0
            base = bases[k].to(dev).float()[None] / 255.0
            face = faces[k].to(dev)[None, ..., None].float()
            shift = SHIFT_MM * torch.tanh(p["fsh"][k])
            means, quats, scales = splats(shift)
            rgb, alpha, _ = rasterization(means, quats, scales, torch.sigmoid(p["opa"]), colours(), vms[k:k + 1], Kt[k:k + 1], W, H,
                                          sh_degree=None, packed=False)
            fm = faces[k].to(dev)
            with torch.no_grad():
                pt, dt, uvp = self._face_hits(Ks[k], Tcs[k], W, H, dev)
            spec_f, _ = self._spec(pt[fm], dt[fm], uvp[fm], p["S"], p["L"], p["rho"] if self.room else None)     # face pixels only (rays meet the plane)
            spec = torch.zeros(H, W, 3, device=dev).index_put((fm,), spec_f)
            pred = (rgb + (1 - alpha) * base + spec[None]) * torch.exp(p["fgain"][k])[None, None, None, :]
            l1 = (torch.abs(pred - gt) * face).sum() / (face.sum() * 3 + 1)
            ss = _ssim((pred * face).permute(0, 3, 1, 2), (gt * face).permute(0, 3, 1, 2)).mean()
            off_face = (alpha * (1 - face)).sum() / ((1 - face).sum() + 1)
            off = p["off"].clamp(lo, hi)

            def tv(X):
                return (X[..., 1:, :] - X[..., :-1, :]).abs().mean() + (X[..., :, 1:] - X[..., :, :-1]).abs().mean()

            reg = (0.05 * (1 - torch.sigmoid(p["opa"])).mean() + 1.0 * p["fgain"].mean(0).pow(2).sum()
                   + self.w_res_l1 * p["res"].abs().mean() + self.w_col_tv * (p["res"][pa] - p["res"][pb]).abs().mean()
                   + (self.w_res_low * low_pass(p["res"]).pow(2).mean() * 100.0 if self.w_res_low > 0 else 0.0)
                   + W_OFF_TV * (off[pa] - off[pb]).abs().mean() + self.w_off_l1 * off.abs().mean()
                   + 0.05 * tv(p["G"]) + 1e-2 * (p["G"] ** 2).mean() + 0.05 * tv(p["S"]) + 0.01 * tv(p["L"]))
            loss = 0.8 * l1 + 0.2 * (1 - ss) + 1.0 * off_face + reg
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            if step % 2500 == 0 or step == STEPS - 1:
                losses.append(round(float(loss.item()), 5))
                log.info(f"[{ctx.rec}] c4b step {step}: loss {loss.item():.4f}" + (f", room {float(torch.exp(p['rho'])):.0f} mm" if self.room else ""))
        # the environment: where the training frames looked, filled elsewhere
        with torch.no_grad():
            qs, fids = [], []
            for k in range(F_):
                pt, dt, uvp = self._face_hits(Ks[k], Tcs[k], W, H, dev)
                m = faces[k].to(dev)[::2, ::2]
                _, q = self._spec(pt[::2, ::2][m], dt[::2, ::2][m], uvp[::2, ::2][m], p["S"], p["L"], p["rho"] if self.room else None)
                qs.append(q.cpu().numpy().astype(np.float32))
                fids.append(np.full(len(qs[-1]), k, np.int32))
            Lraw = F.softplus(p["L"])[0].permute(1, 2, 0).cpu().numpy()
        conf = confidence(support(np.concatenate(qs), np.concatenate(fids), ENV_H, ENV_W, min_px=5))
        Lf = pull_push(Lraw, conf)
        with torch.no_grad():
            shifts = (SHIFT_MM * torch.tanh(p["fsh"])).norm(dim=1).cpu().numpy()
            gains = torch.exp(p["fgain"]).cpu().numpy()
            means, quats, scales = splats(torch.zeros(2, device=dev))
            col = colours()
            keep = torch.sigmoid(p["opa"]) > 0.02
            data = {"means": means[keep], "quats": quats[keep], "scales": torch.log(scales[keep]), "opacities": p["opa"][keep],
                    "sh0": ((col[keep] - 0.5) / SH_C0)[:, None, :], "shN": torch.zeros(int(keep.sum()), 0, 3, device=dev)}
            data = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in data.items()}
            off = p["off"].clamp(lo, hi).cpu().numpy()
            res = p["res"].detach().cpu().numpy()
            S = F.softplus(p["S"])[0, 0].cpu().numpy()
            room = float(torch.exp(p["rho"].clamp(math.log(ROOM_BOUNDS[0]), math.log(ROOM_BOUNDS[1])))) if self.room else None
        write_ply(self.art / "splat.ply", data)
        np.savez_compressed(self.art / "reflect.npz", S=S.astype(np.float32), L=Lf.astype(np.float32), L_raw=Lraw.astype(np.float32),
                            L_conf=conf.astype(np.float32))
        relief = np.full(inside.shape, np.nan, np.float32)
        relief[inside] = off
        np.save(self.art / "relief_mm.npy", relief)
        (self.art / "reflect.json").write_text(json.dumps({"room_radius_mm": room, "grid_mm": GRID_MM, "env_w": ENV_W, "env_h": ENV_H}))
        for f in ("texture.jpg", "model.glb", "roughness.png"):
            if (ctx.base_run / "bundle" / f).exists():
                shutil.copy2(ctx.base_run / "bundle" / f, self.art / f)
        return {"surfels": int(keep.sum()), "spacing_mm": SPACING, "steps": STEPS, "loss": losses,
                "relief_mm": {"p50": round(float(np.median(off)), 3), "p99": round(float(np.percentile(off, 99)), 3),
                              "max": round(float(off.max()), 3), "share_over_0.5": round(float((off > 0.5).mean()), 4)},
                "colour_residual_mad": round(float(np.median(np.abs(res))), 4), "room_radius_mm": None if room is None else round(room, 1),
                "environment": {"measured_share": round(float((conf > 0).mean()), 3), "confident_share": round(float((conf >= 1).mean()), 3)},
                "frame_shift_mm_p50": round(float(np.median(shifts)), 3), "frame_shift_mm_p90": round(float(np.percentile(shifts, 90)), 3),
                "frame_gain_range": [round(float(gains.min()), 3), round(float(gains.max()), 3)], "train_s": round(time.time() - t0, 1)}

    # ------------------------------------------------------------------ render: mesh, splats, then the reflection on the face
    def load(self) -> None:
        import torch

        super().load()
        r = np.load(self.art / "reflect.npz")
        js = json.loads((self.art / "reflect.json").read_text())
        self.S = torch.from_numpy(np.log(np.expm1(np.clip(r["S"], 1e-6, None))))[None, None].float().cuda()
        self.Lf = torch.from_numpy(r["L"]).float().cuda()
        self.rho = None if js.get("room_radius_mm") is None else torch.tensor(math.log(js["room_radius_mm"])).cuda()

    def render(self, view: View):
        import torch

        rgb, a = super().render(view)
        h = self.raster.hits(view)
        m = h.chart == CHARTS.index(self.ctx.face)
        if m.any():
            with torch.no_grad():
                pt = torch.from_numpy(h.point[m]).float().cuda()
                dt = torch.from_numpy(h.to_cam[m]).float().cuda()
                uv = torch.from_numpy(h.uv[m]).float().cuda()
                spec, _ = self._spec(pt, dt, uv, self.S, None, self.rho, filled=self.Lf)
            rgb[m] = np.clip(rgb[m] + spec.cpu().numpy(), 0, 1)
        return rgb.astype(np.float32), a

    def export_web(self, web: Path) -> dict:
        man = super().export_web(web)
        r = np.load(self.art / "reflect.npz")
        js = json.loads((self.art / "reflect.json").read_text())
        S, L = r["S"].astype(np.float32), r["L"].astype(np.float32)
        (web / "s.f16").write_bytes(np.ascontiguousarray(np.dstack([S, S, S, np.ones_like(S)]), dtype="<f2").tobytes())
        (web / "env.f16").write_bytes(np.ascontiguousarray(np.dstack([L, np.ones(L.shape[:2], np.float32)]), dtype="<f2").tobytes())
        lay = atlas_layout(self.ctx.model)
        x0, y0, res = lay[self.ctx.face]
        cw, cl = self.ctx.model.chart_size(self.ctx.face)
        man["face"] = {"chart": self.ctx.face, "rect": [int(x0), int(y0), int(np.ceil(cw / res)), int(np.ceil(cl / res))], "res_mm": res,
                       "size_mm": [cw, cl], "atlas": [4096, 2048], "normal": [0, 0, -1] if self.ctx.face == "back" else [0, 0, 1]}
        man["srm"] = {"s": "s.f16", "s_w": S.shape[1], "s_h": S.shape[0], "s_res_mm": GRID_MM, "env": "env.f16", "env_w": L.shape[1],
                      "env_h": L.shape[0], "room_radius_mm": js["room_radius_mm"], "space": "code values, added over the splats"}
        return self.write_manifest(web, {k: v for k, v in man.items() if k not in ("id", "label", "family", "renderer", "research", "web_mb")})
