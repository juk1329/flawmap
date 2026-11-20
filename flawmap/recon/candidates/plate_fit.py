"""Shared fitting of view-dependent appearance on the fitted plate's filmed face (C1 and C3).

Both candidates model a pixel of a training frame that sees the face at chart point u from view
direction d (device frame) as, in linear light,

    y = B(u + delta_k) + R(u + delta_k, d)

- B: the face's view-independent colour, at the atlas resolution (0.15 mm on an iPad): the inspection
  texture held fixed, times a smooth gain G (2 mm grid, log, total variation). When B itself was
  learned (round 1) it came out blurred and speckled: the camera bump stands 1.5 mm off the flat
  plate, so its print moves with the view and the average of 289 frames smeared it, and Adam on
  sparse random pixels left salt-and-pepper noise (29d760c369, 2026-11-18). The inspection texture
  keeps each texel from its best-supported views, which keeps the detail; ``base_mode="free"`` is
  the round-1 behaviour;
- delta_k: a per-frame shift on the chart (mm, at most 3): the cameras' in-plane error, refined
  jointly, as texture-map optimisation does (Zhou & Koltun 2014; TMO, Fu et al. 2023);
- R: the view-dependent term, which is what differs:
  - C3 (``sh``): per-texel spherical harmonics up to degree 2 of the local view direction, on a 2 mm
    grid (TMO-style refinement with view dependence; Mesh-Learner's SH textures, NeRF2Mesh's specular
    features);
  - C1 (``srm``): s(u) L(omega), a per-texel specular scale (2 mm) times one environment map indexed
    by the mirror direction omega of the view ray: the specular reflectance map of Park et al. 2020
    (only its analytic image formation; no neural renderer). L is 64 x 32 equirectangular in the
    device frame (z-up), non-negative. Round 4: L is read where the mirror ray meets a sphere of
    learned radius about the device (``envmap.proxy_dirs``, a finite room), and after the fit the
    directions no training frame looked up are filled from the measured ones (``envmap.pull_push``)
    instead of staying dark.

Fitted by Adam on random pixels of the training frames (L1 on BT.709 code values; saturated pixels
only penalised when the prediction is below them), with total-variation and size priors on R.
Exposure is held constant (gauge; per-frame gains fitted on the device would swallow the sheen).
"""
from __future__ import annotations

import math

import numpy as np

from ...anchor.parametric import CHARTS
from ...texture.photometric import to_code, to_linear
from ...utils.logging import get_logger
from ..envmap import confidence, face_points, proxy_dirs, pull_push, support
from ..render import PlateRaster

log = get_logger("flawmap.recon")
SH_RES = 2.0          # mm
ENV_W, ENV_H = 64, 32
ROOM_MM = 2000.0      # starting radius of the room for the environment lookup
ROOM_BOUNDS = (300.0, 50000.0)


def sh_basis(d):
    """Real SH up to degree 2 (9 values) of unit vectors (..., 3), torch or numpy."""
    xp = __import__("torch") if not isinstance(d, np.ndarray) else np
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    one = xp.ones_like(x)
    return xp.stack([0.282095 * one, 0.488603 * y, 0.488603 * z, 0.488603 * x, 1.092548 * x * y, 1.092548 * y * z,
                     0.315392 * (3 * z * z - 1), 1.092548 * x * z, 0.546274 * (x * x - y * y)], -1)


def equirect_uv(w):
    """Equirectangular coordinates in [0, 1] of unit directions (device frame, z up)."""
    xp = __import__("torch") if not isinstance(w, np.ndarray) else np
    u = (xp.arctan2(w[..., 1], w[..., 0]) / (2 * math.pi) + 0.5)
    # not quite +-1: arccos has an infinite slope at the poles, and a flat face filmed from straight
    # above reflects exactly there (the gradient to the room radius turned NaN, 2026-11-19)
    v = xp.arccos(xp.clip(w[..., 2], -1.0 + 1e-6, 1.0 - 1e-6)) / math.pi
    return u, v


class FaceSamples:
    """Every face pixel of the training frames: chart uv (mm), view dir and mirror dir (device frame),
    the frame's code value (0-1) and whether it was clipped."""

    def __init__(self, ctx, frames: list[int]):
        raster = PlateRaster(ctx.model)
        ci = CHARTS.index(ctx.face)
        uvs, ds, ws, ys, sats, fid = [], [], [], [], [], []
        for k, i in enumerate(frames):
            v = ctx.view(i)
            h = raster.hits(v)
            m = (h.chart == ci) & ctx.silhouette(v, -3.0)
            img = ctx.image(i)
            d = h.to_cam[m]
            n = h.normal[m]
            w = 2 * (d * n).sum(1, keepdims=True) * n - d             # mirror direction (where the light comes from)
            uvs.append(h.uv[m].astype(np.float32))
            ds.append(d.astype(np.float16))
            ws.append(w.astype(np.float16))
            ys.append(img[m])
            sats.append((img[m] >= 250).any(1))
            fid.append(np.full(m.sum(), k, np.int32))
        self.uv, self.d, self.w = np.concatenate(uvs), np.concatenate(ds), np.concatenate(ws)
        self.y, self.sat, self.fid = np.concatenate(ys), np.concatenate(sats), np.concatenate(fid)
        self.n_frames = len(frames)
        log.info(f"[{ctx.rec}] face samples: {len(self.uv)} pixels from {len(frames)} frames")


class PlateAppearance:
    """Parameters and forward model; ``kind`` is "sh" (C3) or "srm" (C1)."""

    def __init__(self, ctx, kind: str, base_img: np.ndarray, res: float, dev: str = "cuda", base_mode: str = "fixed",
                 proxy: bool = False):
        import torch

        self.ctx, self.kind, self.res, self.dev, self.base_mode = ctx, kind, res, dev, base_mode
        w, l = ctx.model.chart_size(ctx.face)
        self.size_mm = (w, l)
        lin = np.clip(to_linear(base_img), 1e-4, 1.2)
        B0 = torch.logit(torch.from_numpy(lin / 1.2).float().clamp(1e-4, 1 - 1e-4)).permute(2, 0, 1)[None].to(dev)
        gh, gw = int(math.ceil(l / SH_RES)), int(math.ceil(w / SH_RES))
        if base_mode == "free":
            self.B, self.G = torch.nn.Parameter(B0), None
        else:
            self.B, self.G = B0, torch.nn.Parameter(torch.zeros(1, 3, gh, gw, device=dev))
        if kind == "sh":
            self.C = torch.nn.Parameter(torch.zeros(1, 27, gh, gw, device=dev))
        else:
            self.S = torch.nn.Parameter(torch.zeros(1, 1, gh, gw, device=dev))
            self.L = torch.nn.Parameter(torch.full((1, 3, ENV_H, ENV_W), -5.0, device=dev))
            # log of the room's radius (mm) for the parallax-corrected lookup; None: a distant map
            self.rho = torch.nn.Parameter(torch.tensor(math.log(ROOM_MM), device=dev)) if proxy else None
            self.L_filled = None                 # (ENV_H, ENV_W, 3) after ``fill_env``
        self.z_face = ctx.model.z_back if ctx.face == "back" else ctx.model.z_front
        self.delta = None

    def _base_params(self) -> list:
        return [self.B] if self.G is None else [self.G]

    def params(self) -> list:
        return self._base_params() + self._view_params() + ([self.delta] if self.delta is not None else [])

    def _view_params(self) -> list:
        if self.kind == "sh":
            return [self.C]
        return [self.S, self.L] + ([self.rho] if self.rho is not None else [])

    def radius_mm(self):
        """The room's radius for the lookup (mm), None for a distant map."""
        if self.kind == "sh" or self.rho is None:
            return None
        import torch

        return torch.exp(self.rho.clamp(math.log(ROOM_BOUNDS[0]), math.log(ROOM_BOUNDS[1])))

    def _grid(self, uv):
        import torch

        w, l = self.size_mm
        return torch.stack([uv[:, 0] / w * 2 - 1, uv[:, 1] / l * 2 - 1], -1)[None, None]

    def base(self, uv):
        import torch
        import torch.nn.functional as F

        g = self._grid(uv)
        b = torch.sigmoid(F.grid_sample(self.B, g, mode="bilinear", align_corners=False))[0, :, 0].T * 1.2
        if self.G is not None:
            b = b * torch.exp(F.grid_sample(self.G, g, mode="bilinear", align_corners=False)[0, :, 0].T)
        return b

    def view_term(self, uv, d, w):
        import torch
        import torch.nn.functional as F

        g = self._grid(uv)
        if self.kind == "sh":
            c = F.grid_sample(self.C, g, mode="bilinear", align_corners=False)[0, :, 0].T.reshape(-1, 9, 3)
            return (sh_basis(d)[..., None] * c).sum(1)
        s = F.softplus(F.grid_sample(self.S, g, mode="bilinear", align_corners=False)[0, 0, 0] + 1.0)
        q = self.lookup_dirs(uv, w)
        eu, ev = equirect_uv(q)
        if self.L_filled is not None:
            env = self.L_filled.permute(2, 0, 1)[None]
            L = F.grid_sample(env, torch.stack([eu * 2 - 1, ev * 2 - 1], -1)[None, None], mode="bilinear",
                              padding_mode="border", align_corners=False)[0, :, 0].T
        else:
            L = F.softplus(F.grid_sample(self.L, torch.stack([eu * 2 - 1, ev * 2 - 1], -1)[None, None], mode="bilinear",
                                         padding_mode="border", align_corners=False)[0, :, 0].T)
        return s[:, None] * L

    def lookup_dirs(self, uv, w):
        """Where the environment map is read for face points ``uv`` (chart mm) and mirror directions ``w``."""
        R = self.radius_mm()
        if R is None:
            return w
        return proxy_dirs(face_points(uv, self.ctx.face, self.size_mm, self.z_face), w, R)

    def predict(self, uv, d, w):
        return self.base(uv) + self.view_term(uv, d, w)

    def regulariser(self):
        import torch

        def tv(X):
            return (X[..., 1:, :] - X[..., :-1, :]).abs().mean() + (X[..., :, 1:] - X[..., :, :-1]).abs().mean()

        rg = 0.0 if self.G is None else 0.05 * tv(self.G) + 1e-2 * (self.G ** 2).mean()
        if self.kind == "sh":
            C = self.C
            return rg + 0.05 * tv(C) + 1e-3 * (C[:, 3:] ** 2).mean()
        return rg + 0.05 * tv(self.S) + 0.01 * tv(self.L)

    # -------------------------------------------------------------- training
    def fit(self, samples: FaceSamples, steps: int = 4000, batch: int = 600_000, seed: int = 0) -> list[float]:
        import torch

        dev = self.dev
        self.delta = torch.nn.Parameter(torch.zeros(samples.n_frames, 2, device=dev))
        opt = torch.optim.Adam([{"params": self._base_params(), "lr": 2e-2 if self.G is None else 1e-2}, {"params": [self.delta], "lr": 1e-2},
                                {"params": self._view_params(), "lr": 1e-2}])
        uv, d, w = (torch.from_numpy(a) for a in (samples.uv, samples.d, samples.w))
        y = torch.from_numpy(samples.y)
        sat = torch.from_numpy(samples.sat)
        fid = torch.from_numpy(samples.fid)
        g = torch.Generator().manual_seed(seed)
        n = len(uv)
        losses = []
        for step in range(steps):
            idx = torch.randint(0, n, (min(batch, n),), generator=g)
            f = fid[idx].to(dev).long()
            u = uv[idx].to(dev) + 3.0 * torch.tanh(self.delta[f] / 3.0)
            pred = self.predict(u, d[idx].to(dev).float(), w[idx].to(dev).float())
            pc = _code(pred)
            tgt = y[idx].to(dev).float() / 255.0
            s = sat[idx].to(dev)[:, None]
            err = torch.where(s, torch.relu(tgt - pc), (pc - tgt).abs())
            loss = err.mean() + self.regulariser()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 500 == 0 or step == steps - 1:
                losses.append(round(float(loss.item()), 5))
        if self.kind == "srm":
            self.fill_env(samples)
        return losses

    def fill_env(self, samples: FaceSamples) -> dict:
        """The directions the training frames looked up (``envmap.support``, with the fitted per-frame
        shifts and room radius), and the map filled elsewhere (``envmap.pull_push``)."""
        import torch
        import torch.nn.functional as F

        qs = []
        with torch.no_grad():
            for k in range(0, len(samples.uv), 2_000_000):
                sl = slice(k, k + 2_000_000)
                uv = torch.from_numpy(samples.uv[sl]).to(self.dev)
                if self.delta is not None:
                    f = torch.from_numpy(samples.fid[sl]).to(self.dev).long()
                    uv = uv + 3.0 * torch.tanh(self.delta[f] / 3.0)
                q = self.lookup_dirs(uv, torch.from_numpy(samples.w[sl]).to(self.dev).float())
                qs.append(q.cpu().numpy().astype(np.float32))
            L = F.softplus(self.L.detach())[0].permute(1, 2, 0).cpu().numpy()
        n = support(np.concatenate(qs), samples.fid, ENV_H, ENV_W)
        conf = confidence(n)
        filled = pull_push(L, conf)
        self.L_filled = torch.from_numpy(filled).to(self.dev)
        self.env_info = {"measured_share": round(float((conf > 0).mean()), 3), "confident_share": round(float((conf >= 1).mean()), 3),
                         "room_radius_mm": None if self.radius_mm() is None else round(float(self.radius_mm()), 1)}
        self._env_raw, self._env_conf = L, conf
        log.info(f"[{self.ctx.rec}] environment map: {self.env_info}")
        return self.env_info

    # -------------------------------------------------------------- export
    def base_image(self) -> np.ndarray:
        """B (with the gain) as 8-bit code values (rows, cols, 3)."""
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            b = torch.sigmoid(self.B) * 1.2
            if self.G is not None:
                b = b * torch.exp(F.interpolate(self.G, size=b.shape[-2:], mode="bilinear", align_corners=False))
            lin = b[0].permute(1, 2, 0).cpu().numpy()
        return (to_code(lin) * 255).round().astype(np.uint8)

    def render_face(self, uv: np.ndarray, d: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Code values (n, 3) for face points (no per-frame shift: new views)."""
        import torch

        out = []
        with torch.no_grad():
            for k in range(0, len(uv), 400_000):
                t = (torch.from_numpy(np.ascontiguousarray(a[k:k + 400_000])).float().to(self.dev) for a in (uv, d, w))
                out.append(_code(self.predict(*t)).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 3), np.float32)

    def state(self) -> dict:
        import torch.nn.functional as F

        st = {"kind": self.kind, "size_mm": list(self.size_mm), "res_mm": self.res}
        if self.kind == "sh":
            st["C"] = self.C.detach()[0].permute(1, 2, 0).cpu().numpy()               # (gh, gw, 27): 9 SH x 3 channels
        else:
            st["S"] = F.softplus(self.S.detach() + 1.0)[0, 0].cpu().numpy()            # (gh, gw)
            raw = F.softplus(self.L.detach())[0].permute(1, 2, 0).cpu().numpy()        # (32, 64, 3) linear
            st["L"] = self.L_filled.cpu().numpy() if self.L_filled is not None else raw
            st["L_raw"] = raw
            if getattr(self, "_env_conf", None) is not None:
                st["L_conf"] = self._env_conf
            R = self.radius_mm()
            st["room_radius_mm"] = None if R is None else float(R)
        return st


def _code(lin):
    import torch

    lin = torch.clamp(lin, min=0.0)
    return torch.where(lin < 0.018, 4.5 * lin, 1.099 * torch.pow(lin + 1e-8, 0.45) - 0.099).clamp(0, 1)
